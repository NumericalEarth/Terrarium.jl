using Terrarium

using CUDA
using Dates
using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.SoilGrids
using Oceananigans.Units: day
using Rasters, NCDatasets
using Statistics

using CairoMakie, GeoMakie
using ProgressMeter

import RingGrids
import SpeedyWeather as Speedy

# Choose architecture based on available hardware
arch = CUDA.functional() ? GPU() : CPU()

speedy_arch = RingGrids.Architectures.architecture(arch)
spectral_grid = Speedy.SpectralGrid(truncation = 64, architecture = speedy_arch)
ring_grid = spectral_grid.grid

mean_annual_temperature(lat) = 20 - abs(40 * sin(lat)) # maximum at equator

function initial_soil_temperature(grid)
    _, grid_lat = RingGrids.get_lonlats(grid.rings) # in radians
    grid_lat_masked = grid_lat[on_architecture(CPU(), grid.mask)]
    function init(x, z)
        latᵢ = grid_lat_masked[round(Int, x)]
        T₀ = mean_annual_temperature(latᵢ)
        T = T₀ - 0.05 * z
        return T
    end
    return init
end

function Terrarium.InputSources(dataset::SoilGrids2, grid::ColumnRingGrid, horizons = (Symbol(:horizon, i) for i in 1:6); name = nameof(typeof(dataset)), verbose = true)
    soilgrids_vars = (:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)
    metadataset = MetadataSet(soilgrids_vars...; dataset)
    arch = RingGrids.architecture(grid.rings)
    soilgrids_inputs = []
    for (idx, horizon) in enumerate(horizons)
        layer_inputs = Dict()
        for var in soilgrids_vars
            verbose && @info "Loading input data for $var on $horizon"
            var_field = Field(getproperty(metadataset, var))
            ring_field = RingGrids.on_architecture(arch, RingGrids.FullClenshawField(interior(var_field)[:, (end - 1):-1:2, end - idx + 1], input_as = Matrix))
            target_field = RingGrids.Field(grid.rings)
            RingGrids.interpolate!(target_field, ring_field)
            layer_inputs[var] = InputSource(grid, Field(target_field, grid); name = horizon => var)
        end
        # Ensure that mineral texture components with each horizon sum to unity
        Terrarium.normalize_texture!(layer_inputs[:sand_fraction].field, layer_inputs[:silt_fraction].field, layer_inputs[:clay_fraction].field)
        append!(soilgrids_inputs, values(layer_inputs))
    end
    return InputSources(name, soilgrids_inputs...)
end

lai_asset = ERA5LandLeafAreaIndex(Terrarium.N72)
lai_highveg = Terrarium.load_asset(lai_asset, "lai_hv"; NF = Float32, fill_value = 0.0f0)
lai_highveg_fields = []
@showprogress for t in dims(lai_highveg, :dayofyear)
    LAI_t = lai_highveg[dayofyear = At(t)]
    data = reshape(Array(LAI_t), :, 1)
    src_field = RingGrids.Field(data, Terrarium.native_grid(lai_asset))
    dst_field = RingGrids.interpolate(ring_grid, on_architecture(arch, src_field))
    push!(lai_highveg_fields, dst_field)
end

Nz = 30
land_sea_mask = Speedy.EarthLandSeaMask(spectral_grid)
Speedy.load_mask!(land_sea_mask)
Makie.heatmap(on_architecture(CPU(), land_sea_mask.land_fraction))
land_grid = ColumnRingGrid(arch, Float32, ExponentialSpacing(; N = Nz), land_sea_mask.land_fraction .> 0)
porosity = SoilPorositySURFEX(eltype(land_grid))
strat = SoilGridsStratigraphy(eltype(land_grid); porosity)
soil = SoilEnergyWaterCarbon(eltype(land_grid); strat)
vegetation = PrescribedVegetation(eltype(land_grid))
surface_energy_balance = SurfaceEnergyBalance(eltype(land_grid); albedo = DiagnosticAlbedo(eltype(land_grid)))
terrarium_model = Terrarium.LandModel(land_grid; vegetation, soil, surface_energy_balance)

initial_date = DateTime(2024)
soilgrids_inputs = InputSources(SoilGrids2(), land_grid)
lai_highveg_fts = FieldTimeSeries(cat(lai_highveg_fields..., dims = 2), land_grid, 0.0:1day:365day)
lai_inputs = InputSource(lai_highveg_fts, name = :leaf_area_index, reftime = Speedy.DEFAULT_DATE)
inputs = InputSources(lai_inputs, soilgrids_inputs.sources...) # combine input sources

initializers = (
    temperature = initial_soil_temperature(land_grid),
    saturation_water_ice = 1.0, # fully saturated soil everywhere
)

land = Speedy.LandModel(
    spectral_grid,
    terrarium_model;
    inputs,
    initializers,
    Δt = Minute(5)
)

# Build the coupled PrimitiveWetModel
surface_heat_flux = Speedy.SurfaceHeatFlux(land.spectral_grid, land = Speedy.PrescribedLandHeatFlux())
surface_humidity_flux = Speedy.SurfaceHumidityFlux(land.spectral_grid, land = Speedy.PrescribedLandHumidityFlux())
albedo = Speedy.OceanLandAlbedo(land.spectral_grid, land = Speedy.PrescribedAlbedo(land.spectral_grid))
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveDryModel; interval = Hour(4), nlayers_soil = land.geometry.nlayers, path = "outputs/")
time_stepping = Speedy.Leapfrog(land.spectral_grid, Δt_at_T32 = Minute(15))
primitive_wet_coupled = Speedy.PrimitiveWetModel(
    land.spectral_grid;
    land,
    surface_heat_flux,
    surface_humidity_flux,
    albedo,
    land_sea_mask,
    time_stepping,
    output,
)
Speedy.add!(primitive_wet_coupled.output, Speedy.SoilTemperatureOutput())
Speedy.add!(primitive_wet_coupled.output, Speedy.SnowDepthOutput())
Speedy.add!(primitive_wet_coupled.output, Speedy.PrecipitationOutput())
Speedy.add!(primitive_wet_coupled.output, Speedy.SurfaceSensibleHeatFluxOutput())
Speedy.add!(primitive_wet_coupled.output, Speedy.SurfaceHumidityFluxOutput())
Speedy.add!(primitive_wet_coupled.output, Speedy.AlbedoOutput())

# Initialize the coupled simulation
@info "Initializing coupled simulation"
sim_coupled = @time Speedy.initialize!(primitive_wet_coupled; time = initial_date)

land_state = sim_coupled.variables.prognostic.land.terrarium
Terrarium.checkfinite!(land_state.prognostic)

period = Year(2)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled; period, output = true)
