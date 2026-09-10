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

# Build the SpeedyWeather ring + spectral grid
speedy_arch = RingGrids.Architectures.architecture(arch)
spectral_grid = Speedy.SpectralGrid(truncation = 64, architecture = speedy_arch)
ring_grid = spectral_grid.grid

# Here we declare some plotting helpers that ensure all Fields/arrays are transferred to CPU before attempting to plot:
plot_land_field(field, z_idx = 1; kwargs...) = heatmap(RingGrids.Field(CPU(), interior(field)[:, :, z_idx], land_grid); kwargs...)
plot_speedy_field(field; kwargs...) = heatmap(on_architecture(CPU(), field); kwargs...)


# To make the simulation a bit more interesting, we will use spatially periodic initial and boundary conditions.
# The climatology will be determined by latitude with a maximum of 20 °C at the equator and minimum of -20°C at
# the poles.
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

# Build the Terrarium land model on the matching ring grid with appropriate land-sea mask
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

# Prepare the input sources: first the SoilGrids data, then the LAI data for prescribed vegetation.
initial_date = DateTime(2024)
soilgrids_inputs = InputSources(SoilGrids2(), land_grid)
lai_highveg_fts = FieldTimeSeries(cat(lai_highveg_fields..., dims = 2), land_grid, 0.0:1day:365day)
lai_inputs = InputSource(lai_highveg_fts, name = :leaf_area_index, reftime = Speedy.DEFAULT_DATE)
inputs = InputSources(lai_inputs, soilgrids_inputs.sources...) # combine input sources

# Here we set our initial conditions for the soil
initializers = (
    temperature = initial_soil_temperature(land_grid),
    saturation_water_ice = 1.0, # fully saturated soil everywhere
)

# Wrap the Terrarium model as a SpeedyWeather land component
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
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :evaporation_ground))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :evaporation_canopy))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :transpiration))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :sublimation))
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :gross_primary_production))

# Initialize the coupled simulation
@info "Initializing coupled simulation"
sim_coupled = @time Speedy.initialize!(primitive_wet_coupled; time = initial_date)

# The Terrarium state lives inside SpeedyWeather's Variables tree.
land_state = sim_coupled.variables.prognostic.land.terrarium
Terrarium.checkfinite!(land_state.prognostic)

# Let's quickly check that the soil temperature was correctly initialized.
plot_land_field(land_state.temperature, Nz - 1)

# Now the sand fraction in the topmost horizon:
plot_land_field(land_state.horizon1.sand_fraction)

# and the Leaf Area Index:
plot_land_field(land_state.leaf_area_index)

# Looks good! Let's run it!
period = Year(1)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled; period, output = true)

# Land variables (use the SpeedyWeather-owned Terrarium state)
Tsoil_fig = plot_land_field(land_state.temperature, Nz - 1, title = "Soil temperature (10 cm depth)", size = (800, 400))
liq_fig = plot_land_field(land_state.liquid_water_fraction, Nz, title = "Soil surface liquid water fraction")
Tsurf_fig = plot_land_field(land_state.skin_temperature, title = "Skin temperature", size = (800, 400))
sat_fig = plot_land_field(land_state.saturation_water_ice, title = "Soil saturation water + ice", size = (800, 400))
Hs_fig = plot_land_field(land_state.sensible_heat_flux, title = "Sensible heat flux", size = (800, 400))
Hl_fig = plot_land_field(land_state.latent_heat_flux, title = "Latent heat flux", size = (800, 400))
E_fig = plot_land_field(land_state.evaporation_ground, title = "Evaporation (bare ground)", size = (800, 400))
E_sub = plot_land_field(land_state.sublimation, title = "Sublimation (bare ground)", size = (800, 400))
E_can_fig = plot_land_field(land_state.evaporation_canopy, title = "Evaporation (bare ground)", size = (800, 400))
Tr_fig = plot_land_field(land_state.transpiration, title = "Transpiration", size = (800, 400))
An_fig = plot_land_field(land_state.net_assimilation, title = "Net assimilation", size = (800, 400))
g_stm_fig = plot_land_field(land_state.canopy_water_conductance, title = "Canopy water conductance", size = (800, 400))
swe_fig = plot_land_field(land_state.snow_water_equivalent, title = "Snow water equivalent", colorrange = (0, 0.2), size = (800, 400))
scf_fig = plot_land_field(land_state.snow_cover_fraction, title = "Snow cover fraction", size = (800, 400))
snowT_fig = plot_land_field(land_state.snow_temperature, title = "Snow temperature", size = (800, 400))

# Atmosphere variables
Tair_fig = plot_speedy_field(sim_coupled.variables.parameterizations.surface_air_temperature .- 273.15, title = "Air temperature", size = (800, 400))
prcp_fig = plot_speedy_field(sim_coupled.variables.parameterizations.rain_rate, title = "Rainfall", size = (800, 400))
humid_fig = plot_speedy_field(sim_coupled.variables.grid.humidity, title = "Humidity", size = (800, 400))
snow_fig = plot_speedy_field(sim_coupled.variables.parameterizations.snow_rate, title = "Snowfall", size = (800, 400))
snow_ls_fig = plot_speedy_field(sim_coupled.variables.parameterizations.snow_large_scale, title = "Snowfall", size = (800, 400))
pres_fig = plot_speedy_field(exp.(sim_coupled.variables.grid.pressure), title = "Surface pressure", size = (800, 400))
srad_fig = plot_speedy_field(sim_coupled.variables.parameterizations.surface_shortwave_down, title = "Surface shortwave down", size = (800, 400))
albd_fig = plot_speedy_field(sim_coupled.variables.parameterizations.albedo, title = "Albedo", size = (800, 400))

# Pick a point somewhere in the mid-latitudes and plot the vertical soil layers
T = on_architecture(CPU(), interior(land_state.temperature)[2000, 1, :])
sat = on_architecture(CPU(), interior(land_state.saturation_water_ice)[2000, 1, :])
f = on_architecture(CPU(), interior(land_state.liquid_water_fraction)[2000, 1, :])
zs = on_architecture(CPU(), znodes(land_state.temperature))
Makie.scatterlines(T, zs)
Makie.scatterlines(sat, zs)
Makie.scatterlines(f, zs)
