using Terrarium

using CUDA
using Dates
using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.SoilGrids
using Oceananigans.Units: day
using Oceananigans.OutputReaders: Cyclical
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

# Interpolate a raster given on the `native_grid` of an asset onto the model ring grid; any dimensions after
# the two horizontal ones (e.g. time) become layers of the field and are interpolated in a single batched call
function interpolate_to_ring_grid(raster, asset)
    # convert explicitly: materializing a lazily loaded raster can give an `Array{Any}`
    data = Float32.(Array(raster))
    data = reshape(data, size(data, 1) * size(data, 2), :)
    src_field = RingGrids.Field(data, Terrarium.native_grid(asset))
    return RingGrids.interpolate(ring_grid, on_architecture(arch, src_field))
end

# ERA5-Land provides the leaf area index separately for high and low vegetation, each per unit area covered by
# that vegetation type, so the grid-cell mean leaf area index is cvh * lai_hv + cvl * lai_lv with the high and
# low vegetation cover fractions cvh and cvl
@info "Loading vegetation cover fractions"
invariants_asset = ERA5LandInvariants()
cover_highveg = interpolate_to_ring_grid(Terrarium.load_asset(invariants_asset, "cvh"; NF = Float32, fill_value = 0.0f0), invariants_asset)
cover_lowveg = interpolate_to_ring_grid(Terrarium.load_asset(invariants_asset, "cvl"; NF = Float32, fill_value = 0.0f0), invariants_asset)

lai_asset = ERA5LandLeafAreaIndex(Terrarium.N72)
lai_highveg = Terrarium.load_asset(lai_asset, "lai_hv"; NF = Float32, fill_value = 0.0f0)
lai_lowveg = Terrarium.load_asset(lai_asset, "lai_lv"; NF = Float32, fill_value = 0.0f0)
@info "Loading LAI data"
# all days of the year as layers: (grid point, day of year)
lai_highveg_daily = interpolate_to_ring_grid(lai_highveg, lai_asset)
lai_lowveg_daily = interpolate_to_ring_grid(lai_lowveg, lai_asset)
lai_daily = RingGrids.Field(cover_highveg.data .* lai_highveg_daily.data .+ cover_lowveg.data .* lai_lowveg_daily.data, ring_grid)

Nz = 30
land_sea_mask = Speedy.EarthLandSeaMask(spectral_grid)
Speedy.load_mask!(land_sea_mask)
land_grid = ColumnRingGrid(arch, Float32, ExponentialSpacing(; N = Nz), land_sea_mask.land_fraction .> 0)
porosity = SoilPorositySURFEX(eltype(land_grid))
strat = SoilGridsStratigraphy(eltype(land_grid); porosity)
soil = SoilEnergyWaterCarbon(eltype(land_grid); strat)
vegetation = PrescribedVegetation(eltype(land_grid))
surface_energy_balance = SurfaceEnergyBalance(eltype(land_grid); albedo = DiagnosticAlbedo(eltype(land_grid)))
# limit bare-ground evaporation by the moisture of the top soil layer relative to field capacity
evapotranspiration = PALADYNCanopyEvapotranspiration(eltype(land_grid); ground_resistance = SoilMoistureResistanceFactor(eltype(land_grid)))
surface_hydrology = SurfaceHydrology(eltype(land_grid); evapotranspiration)

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

# Initial soil moisture above the water table is given by the mean annual volumetric water content (m³/m³) of the
# root-zone layer of SpeedyWeather's monthly soil moisture climatology, which is converted to saturation using the
# porosity of each soil layer; the soil below the water table (5 m depth by default) is fully saturated
swl_climatology = RingGrids.get_asset(
    "data/boundary_conditions/soil_moisture.nc";
    from_assets = true,
    name = "swl2",
    ArrayType = RingGrids.FullGaussianField,
    FileFormat = NCDataset,
    output_grid = on_architecture(CPU(), ring_grid),
)
mean_vwc = RingGrids.Field(on_architecture(CPU(), ring_grid))
mean_vwc.data .= vec(mean(swl_climatology.data, dims = 2))

# Fall back to the global median water content for coastal columns where the climatology is undefined
missing_vwc = .!isfinite.(mean_vwc.data)
mean_vwc.data[missing_vwc] .= median(filter(isfinite, mean_vwc.data))

vwc_source = InputSource(land_grid, mean_vwc; name = :vadose_zone_saturation, domain = Terrarium.Ground())
hydrology_initializer = SaturationWaterTable(Float32; vadose_zone_saturation = VolumetricWaterContent(0.25f0))
initializer = SoilInitializer(eltype(land_grid); energy = energy_initializer, hydrology = hydrology_initializer)
terrarium_model = Terrarium.LandModel(land_grid; vegetation, soil, surface_energy_balance, surface_hydrology, initializer)

# start before the Northern Hemisphere snow season (and towards the end of the Southern Hemisphere one) to limit
# the spin-up of the seasonal snow cover, which starts from zero outside the ice sheets
initial_date = DateTime(2024, 10, 1)
soilgrids_inputs = InputSources(SoilGrids2(grid_type = SoilGrids.Clenshaw10km), land_grid)
# the LAI is a day-of-year climatology: repeat it cyclically, with times counted from January 1st of the start year
lai_fts = FieldTimeSeries(lai_daily, land_grid, 0.0:1day:365day; time_indexing = Cyclical())
lai_inputs = InputSource(lai_fts, name = :leaf_area_index, reftime = DateTime(year(initial_date)); domain = Terrarium.Canopy())
inputs = InputSources(T₀_source, vwc_source, lai_inputs, soilgrids_inputs.sources...) # combine input sources

# The ice sheets (Antarctica and Greenland) are represented as a deep, permanent snowpack. SpeedyWeather's
# snow-free surface albedo climatology assigns a constant albedo of 0.48 to land ice, well above that of any
# ice-free land (the Sahara reaches about 0.38), which gives the ice sheet mask.
albedo_climatology = RingGrids.get_asset(
    "data/boundary_conditions/albedo.nc";
    from_assets = true,
    name = "alb",
    ArrayType = RingGrids.FullGaussianField,
    FileFormat = NCDataset,
    output_grid = on_architecture(CPU(), ring_grid),
)
ice_sheet_snow_water_equivalent = 0.5f0 # (m)
initial_snow_water_equivalent = RingGrids.Field(on_architecture(CPU(), ring_grid))
is_ice_sheet = (on_architecture(CPU(), land_sea_mask.land_fraction).data .> 0.5) .& (albedo_climatology.data .>= 0.45f0)
initial_snow_water_equivalent.data .= ice_sheet_snow_water_equivalent .* is_ice_sheet
initializers = (snow_water_equivalent = Field(initial_snow_water_equivalent, land_grid),)

land = Speedy.LandModel(
    spectral_grid,
    terrarium_model;
    inputs,
    initializers,
    # must evenly divide the atmospheric time step (450 s at T64, see `time_stepping` below), since the
    # land is advanced by a whole number of land steps per atmospheric step
    Δt = Second(450)
)

# Build the coupled PrimitiveWetModel
surface_heat_flux = Speedy.SurfaceHeatFlux(land.spectral_grid, land = Speedy.PrescribedLandHeatFlux())
surface_humidity_flux = Speedy.SurfaceHumidityFlux(land.spectral_grid, land = Speedy.PrescribedLandHumidityFlux())
albedo = Speedy.OceanLandAlbedo(land.spectral_grid, land = Speedy.PrescribedAlbedo(land.spectral_grid))
# Pragmatic retuning of SpeedyWeather's default one-band radiation, whose clouds reflect too much sunlight
# (planetary albedo ≈ 0.43) so that the default atmosphere drifts far too cold: less reflective clouds
# (defaults 0.6 and 0.5), more shortwave absorption by water vapor (default 75), and a larger longwave optical
# depth (defaults 6 at the equator and 1.5 at the poles). This is not a calibration, only enough to give a
# reasonable near-surface climate for demonstrating the coupling.
clouds = Speedy.DiagnosticClouds(land.spectral_grid, cloud_albedo = 0.4, stratocumulus_albedo = 0.3)
shortwave_transmissivity = Speedy.BackgroundShortwaveTransmissivity(land.spectral_grid, absorptivity_water_vapor = 150)
longwave_transmissivity = Speedy.FriersonLongwaveTransmissivity(land.spectral_grid, τ₀_equator = 8, τ₀_pole = 2)
radiation = Speedy.Radiation(
    land.spectral_grid;
    shortwave = Speedy.OneBandShortwave(land.spectral_grid; clouds, transmissivity = shortwave_transmissivity),
    longwave = Speedy.OneBandLongwave(land.spectral_grid; transmissivity = longwave_transmissivity),
)
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveDryModel; interval = Hour(4), nlayers_soil = land.geometry.nlayers, path = output_dir)
time_stepping = Speedy.Leapfrog(land.spectral_grid, Δt_at_T32 = Minute(15))
primitive_wet_coupled = Speedy.PrimitiveWetModel(
    land.spectral_grid;
    land,
    surface_heat_flux,
    surface_humidity_flux,
    albedo,
    radiation,
    land_sea_mask,
    time_stepping,
    output,
)
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.SoilTemperature())
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.SnowDepth())
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Precipitation())
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.SurfaceSensibleHeatFlux())
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.SurfaceHumidityFlux())
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Albedo())
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :evaporation_ground))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :evaporation_canopy))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :transpiration))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :sublimation))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :gross_primary_production))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :surface_runoff))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :skin_temperature))

# Initialize the coupled simulation
@info "Initializing coupled simulation"
sim_coupled = @time Speedy.initialize!(primitive_wet_coupled; time = initial_date)

land_state = sim_coupled.variables.prognostic.land.terrarium
Terrarium.checkfinite!(land_state.prognostic)

period = Year(2)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled; period, output = true)
