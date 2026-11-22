using Terrarium

using ArgParse
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

# Command line options, e.g. `julia --project=docs/joss docs/joss/run_terrarium_speedy_simulation.jl --period 30d`
settings = ArgParseSettings(description = "Coupled Terrarium–SpeedyWeather simulation for the JOSS paper")
@add_arg_table! settings begin
    "--period"
    help = "length of the simulation: a number followed by y (years), d (days), or h (hours)"
    default = "5y"
    "--start-date"
    help = "start date of the simulation (yyyy-mm-dd)"
    default = "2024-10-01"
    "--timestep"
    help = "land time step (s); must evenly divide the atmospheric time step (default: the atmospheric time step)"
    arg_type = Int
    "--truncation"
    help = "spectral truncation of the atmosphere, e.g. 31, 42, 64, or 85"
    arg_type = Int
    default = 64
    "--architecture"
    help = "auto (GPU if CUDA is functional, otherwise CPU), cpu, or gpu"
    default = "auto"
    range_tester = in(("auto", "cpu", "gpu"))
end
args = parse_args(settings)

function parse_period(period::AbstractString)
    m = match(r"^\s*(\d+)\s*([ydh])\s*$", period)
    isnothing(m) && error("invalid period $period: expected a number followed by y, d, or h, e.g. 5y")
    n = parse(Int, m.captures[1])
    return m.captures[2] == "y" ? Year(n) : m.captures[2] == "d" ? Day(n) : Hour(n)
end
period = parse_period(args["period"])
initial_date = DateTime(args["start-date"])

# Make output directory for simulation results
output_dir = joinpath(@__DIR__, "outputs")
mkpath(output_dir)

# Choose the architecture
arch = if args["architecture"] == "gpu"
    CUDA.functional() || error("--architecture gpu requested, but CUDA is not functional")
    GPU()
elseif args["architecture"] == "cpu"
    CPU()
else
    CUDA.functional() ? GPU() : CPU()
end
speedy_arch = RingGrids.Architectures.architecture(arch)
spectral_grid = Speedy.SpectralGrid(truncation = args["truncation"], architecture = speedy_arch)
ring_grid = spectral_grid.grid
@info "Setting up simulation on grid $(ring_grid)"

# The atmospheric time step scales with the resolution (15 min at T32) and is adjusted by SpeedyWeather to divide
# the output interval; the land is advanced by a whole number of land steps per atmospheric step, so the land time
# step must divide it evenly
Δt_at_T32 = Minute(15)
output_interval = Hour(4)
atmosphere_Δt = Speedy.get_Δt_millisec(Δt_at_T32, args["truncation"], Speedy.DEFAULT_RADIUS, true, output_interval)
land_Δt = isnothing(args["timestep"]) ? atmosphere_Δt : Millisecond(Second(args["timestep"]))
Dates.value(atmosphere_Δt) % Dates.value(land_Δt) == 0 ||
    error("the land time step of $(land_Δt) does not evenly divide the atmospheric time step of $atmosphere_Δt")
@info "Time steps: atmosphere $(atmosphere_Δt), land $(land_Δt); simulating $period from $initial_date"

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
# Brooks–Corey retention curve with the previous default parameters (λ = 0.2, air-entry suction 1 cm): with the newer
# sandy loam default (air-entry suction 21.8 cm), the very dry cells of the soil moisture climatology reach matric
# potentials that make the first implicit steps of the Richards equation diverge
hydraulic_properties = SoilHydraulicsSURFEX(eltype(land_grid); swrc = BrooksCorey(λ = 0.2, ψₛ = 0.01u"m"))
hydrology = SoilHydrology(eltype(land_grid), RichardsEq(); hydraulic_properties)
soil = SoilEnergyWaterCarbon(eltype(land_grid); strat, hydrology)
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

# the default start date (October 1st) is before the Northern Hemisphere snow season (and towards the end of the
# Southern Hemisphere one) to limit the spin-up of the seasonal snow cover, which starts from zero outside the ice sheets
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
    Δt = land_Δt
)

# Build the coupled PrimitiveWetModel
surface_heat_flux = Speedy.SurfaceHeatFlux(land.spectral_grid, land = Speedy.PrescribedLandHeatFlux())
surface_humidity_flux = Speedy.SurfaceHumidityFlux(land.spectral_grid, land = Speedy.PrescribedLandHumidityFlux())
albedo = Speedy.OceanLandAlbedo(land.spectral_grid, land = Speedy.PrescribedAlbedo(land.spectral_grid))
# Longwave absorption coefficients 1.3 times the SpeedyWeather defaults (1.45 and 750): with the defaults, the
# coupled simulation drifts too cold and the Northern Hemisphere midlatitudes stay snow-covered. This is a
# pragmatic adjustment, not a calibration.
longwave_transmissivity = Speedy.ByrneOGormanLongwaveTransmissivity(land.spectral_grid; dry_absorption = 1.885, water_vapor_absorption = 975)
radiation = Speedy.Radiation(land.spectral_grid; longwave = Speedy.OneBandLongwave(land.spectral_grid; transmissivity = longwave_transmissivity))
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveDryModel; interval = output_interval, nlayers_soil = land.geometry.nlayers, path = output_dir)
time_stepping = Speedy.Leapfrog(land.spectral_grid; Δt_at_T32)
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
# surface radiation and ground heat flux of the land energy balance, R_net = SW↓ - SW↑ + LW↓ - LW↑ = H + LE + G
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :surface_shortwave_down))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :surface_shortwave_up))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :surface_longwave_down))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :surface_longwave_up))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :surface_net_radiation))
Speedy.add!(primitive_wet_coupled.output, Speedy.OutputVariables.Terrarium(terrarium_model, :ground_heat_flux))

# Initialize the coupled simulation
@info "Initializing coupled simulation"
sim_coupled = @time Speedy.initialize!(primitive_wet_coupled; time = initial_date)

land_state = sim_coupled.variables.prognostic.land.terrarium.state
Terrarium.checkfinite!(land_state.prognostic)

@info "Running simulation for $period"
@time Speedy.run!(sim_coupled; period, output = true)
