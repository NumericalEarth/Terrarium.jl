using Terrarium

using CUDA
using Dates
using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.SoilGrids
using Oceananigans.Units: day
using ProgressMeter
using Rasters, NCDatasets
using Statistics

import RingGrids
import SpeedyWeather as Speedy

# Make output directory for simulation results
output_dir = joinpath(@__DIR__, "outputs")
mkpath(output_dir)

# Choose architecture based on available hardware
arch = CUDA.functional() ? GPU() : CPU()
speedy_arch = RingGrids.Architectures.architecture(arch)
spectral_grid = Speedy.SpectralGrid(truncation = 64, architecture = speedy_arch)
ring_grid = spectral_grid.grid
@info "Setting up simulation on grid $(ring_grid)"

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
land_grid = ColumnRingGrid(arch, Float32, ExponentialSpacing(; N = Nz), land_sea_mask.land_fraction .> 0)
porosity = SoilPorositySURFEX(eltype(land_grid))
strat = SoilGridsStratigraphy(eltype(land_grid); porosity)
soil = SoilEnergyWaterCarbon(eltype(land_grid); strat)
vegetation = PrescribedVegetation(eltype(land_grid))
surface_energy_balance = SurfaceEnergyBalance(eltype(land_grid); albedo = DiagnosticAlbedo(eltype(land_grid)))

# Initial soil temperature profiles are linear in depth via QuasiThermalSteadyState, with the surface
# temperature given by the mean annual land surface temperature from SpeedyWeather's monthly climatology (°C)
lst_climatology = RingGrids.get_asset(
    "data/boundary_conditions/land_surface_temperature.nc";
    from_assets = true,
    name = "lst",
    ArrayType = RingGrids.FullGaussianField,
    FileFormat = NCDataset,
    output_grid = on_architecture(CPU(), ring_grid),
)
mean_lst = RingGrids.Field(on_architecture(CPU(), ring_grid))
mean_lst.data .= vec(mean(lst_climatology.data, dims = 2)) .- 273.15f0 # K to °C

# Fall back to a latitude-dependent climatology for coastal columns where the climatology is undefined
_, ring_lat = RingGrids.get_lonlats(on_architecture(CPU(), ring_grid)) # in radians
fallback = LatitudinalClimatology(Float32)
missing_lst = .!isfinite.(mean_lst.data)
mean_lst.data[missing_lst] .= fallback.(rad2deg.(ring_lat[missing_lst]))

T₀_source = InputSource(land_grid, mean_lst; name = :initial_surface_temperature, domain = Terrarium.Ground(), units = u"°C")
energy_initializer = QuasiThermalSteadyState(Float32; Qgeo = 0.05f0)
initializer = SoilInitializer(eltype(land_grid); energy = energy_initializer, hydrology = DefaultInitializer(eltype(land_grid)))
terrarium_model = Terrarium.LandModel(land_grid; vegetation, soil, surface_energy_balance, initializer)

initial_date = DateTime(2024)
soilgrids_inputs = InputSources(SoilGrids2(), land_grid)
lai_highveg_fts = FieldTimeSeries(cat(lai_highveg_fields..., dims = 2), land_grid, 0.0:1day:365day)
lai_inputs = InputSource(lai_highveg_fts, name = :leaf_area_index, reftime = Speedy.DEFAULT_DATE; domain = Terrarium.Canopy())
inputs = InputSources(T₀_source, lai_inputs, soilgrids_inputs.sources...) # combine input sources

initializers = (
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
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveDryModel; interval = Hour(4), nlayers_soil = land.geometry.nlayers, path = output_dir)
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
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :evaporation_ground))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :evaporation_canopy))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :transpiration))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :sublimation))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :gross_primary_production))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :surface_runoff))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :temperature))

# Initialize the coupled simulation
@info "Initializing coupled simulation"
sim_coupled = @time Speedy.initialize!(primitive_wet_coupled; time = initial_date)

land_state = sim_coupled.variables.prognostic.land.terrarium
Terrarium.checkfinite!(land_state.prognostic)

period = Year(2)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled; period, output = true)
