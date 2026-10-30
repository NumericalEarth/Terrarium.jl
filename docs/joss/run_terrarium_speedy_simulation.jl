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
@info "Loading LAI data"
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
terrarium_model = Terrarium.LandModel(land_grid; vegetation, soil, surface_energy_balance, initializer)

initial_date = DateTime(2024)
soilgrids_inputs = InputSources(SoilGrids2(), land_grid)
lai_highveg_fts = FieldTimeSeries(cat(lai_highveg_fields..., dims = 2), land_grid, 0.0:1day:365day)
lai_inputs = InputSource(lai_highveg_fts, name = :leaf_area_index, reftime = Speedy.DEFAULT_DATE; domain = Terrarium.Canopy())
inputs = InputSources(T₀_source, vwc_source, lai_inputs, soilgrids_inputs.sources...) # combine input sources

land = Speedy.LandModel(
    spectral_grid,
    terrarium_model;
    inputs,
    # must evenly divide the atmospheric time step (450 s at T64, see `time_stepping` below), since the
    # land is advanced by a whole number of land steps per atmospheric step
    Δt = Second(450)
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
Speedy.add!(primitive_wet_coupled.output, Speedy.TerrariumOutput(terrarium_model, :skin_temperature))

# Initialize the coupled simulation
@info "Initializing coupled simulation"
sim_coupled = @time Speedy.initialize!(primitive_wet_coupled; time = initial_date)

land_state = sim_coupled.variables.prognostic.land.terrarium
Terrarium.checkfinite!(land_state.prognostic)

period = Year(2)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled; period, output = true)
