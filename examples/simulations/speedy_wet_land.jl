using Terrarium

using CUDA
using Dates
using Rasters, NCDatasets
using Statistics

using CairoMakie, GeoMakie

import RingGrids
import SpeedyWeather as Speedy

# Choose architecture based on available hardware
arch = CUDA.functional() ? GPU() : CPU()

"""
Naive implementation of a SpeedyWeather "wet" land model based on Terrarium.
"""
struct TerrariumWetLand{
        NF,
        LG <: Speedy.LandGeometry,
        TM <: Terrarium.ModelIntegrator{NF},
    } <: Speedy.AbstractWetLand
    "Speedy spectral grid"
    spectral_grid::Speedy.SpectralGrid

    "Speedy land model geometry"
    geometry::LG

    "Initialized Terrarium model integrator"
    integrator::TM

    function TerrariumWetLand(integrator::Terrarium.ModelIntegrator{NF, Arch, Grid}; spectral_grid_kwargs...) where {NF, Arch, Grid <: ColumnRingGrid}
        spectral_grid = Speedy.SpectralGrid(integrator.model.grid.rings; NF, spectral_grid_kwargs...)
        land_grid = integrator.grid
        Δz = on_architecture(CPU(), land_grid.z.Δᵃᵃᶜ)
        geometry = Speedy.LandGeometry(1, Δz[end])
        return new{eltype(integrator), typeof(geometry), typeof(integrator)}(spectral_grid, geometry, integrator)
    end
end

Speedy.variables(land::TerrariumWetLand) = (
    Speedy.PrognosticVariable(name = :soil_temperature, dims = Speedy.Grid3D(), namespace = :land),
    Speedy.PrognosticVariable(name = :soil_moisture, dims = Speedy.Grid3D(), namespace = :land),
)

function Speedy.initialize!(
        ::Any,
        progn::Speedy.PrognosticVariables,
        diagn::Speedy.DiagnosticVariables,
        land::TerrariumWetLand{NF},
        model::Speedy.PrimitiveEquation,
    ) where {NF}
    Terrarium.initialize!(land.integrator)
    Tsoil = interior(land.integrator.state.temperature)[:, 1, end] .+ 273.15
    sat = interior(land.integrator.state.saturation_water_ice)[:, 1, end]
    progn.land.soil_temperature .= Tsoil
    progn.land.soil_moisture .= sat
    return nothing
end

function Speedy.timestep!(
        progn::Speedy.PrognosticVariables,
        diagn::Speedy.DiagnosticVariables,
        land::TerrariumWetLand{NF},
        model::Speedy.PrimitiveEquation,
    ) where {NF}
    speedy_timestep!(progn, diagn, land)
    return nothing
end

function speedy_timestep!(
        progn::Speedy.PrognosticVariables,
        diagn::Speedy.DiagnosticVariables,
        land::TerrariumWetLand{NF},
    ) where {NF}
    # land constants
    consts = land.integrator.model.constants
    # get speedy state variables
    Tair = @view diagn.grid.temp_grid[:, end]
    humid = @view diagn.grid.humid_grid[:, end]
    pres = diagn.grid.pres_grid
    wind = diagn.physics.surface_wind_speed
    rain = diagn.physics.rain_rate
    snow = diagn.physics.snow_rate
    SwIn = diagn.physics.surface_shortwave_down
    LwIn = diagn.physics.surface_longwave_down
    # update Terrarium input fields
    state = land.integrator.state
    set!(state.inputs.air_temperature, Tair) # first set directly to avoid allocating new fields
    set!(state.inputs.air_temperature, state.inputs.air_temperature - 273.15) # then convert to celsius
    set!(state.inputs.air_pressure, pres) # similar with pressure
    set!(state.inputs.air_pressure, exp(state.inputs.air_pressure)) # take exp to get pressure in Pa
    set!(state.inputs.specific_humidity, humid)
    set!(state.inputs.rainfall, rain)
    set!(state.inputs.snowfall, snow)
    set!(state.inputs.windspeed, wind)
    set!(state.inputs.surface_shortwave_down, SwIn)
    set!(state.inputs.surface_longwave_down, LwIn)
    # run land forward over speedy timestep interval;
    # we use a smaller actual timestep to ensure stability
    Terrarium.run!(land.integrator, period = progn.clock.Δt, Δt = 300.0)
    # Update speedy variables
    progn.land.soil_temperature .= state.skin_temperature .+ NF(273.15)
    progn.land.soil_moisture .= interior(state.saturation_water_ice)[:, 1, end]
    progn.land.sensible_heat_flux .= state.sensible_heat_flux
    progn.land.surface_humidity_flux .= state.latent_heat_flux ./ consts.Llg
    diagn.physics.surface_longwave_up .= state.surface_longwave_up
    diagn.physics.surface_shortwave_up .= state.surface_shortwave_up
    return nothing
end

# quick test of default Speedy PrimitiveWetModel
ring_grid = RingGrids.FullGaussianGrid(24)
spectral_grid = Speedy.SpectralGrid(ring_grid)
primitive_wet = Speedy.PrimitiveWetModel(spectral_grid)
sim = Speedy.initialize!(primitive_wet)
Speedy.run!(sim, period = Day(1))

Nz = 30
Δz_min = 0.05
grid = ColumnRingGrid(CPU(), Float32, ExponentialSpacing(; N = Nz, Δz_min), ring_grid)
# Initial conditions
soil_initializer = SoilInitializer(eltype(grid))
soil = SoilEnergyWaterCarbon(eltype(grid), hydrology = SoilHydrology(eltype(grid)))
# Land model with "prescribed" atmosphere (from the perspective of the land model at least...)
# vegetation = PrescribedVegetationCarbon(eltype(grid))
model = LandModel(grid; initializer = soil_initializer, vegetation = nothing, soil)
initializers = (;)
integrator = initialize(model, ForwardEuler(eltype(grid)); initializers)
# check if land model works standalone (with default atmospheric state)
timestep!(integrator, 60.0) # one step
run!(integrator, period = Hour(1), Δt = 300.0) # one hour
Terrarium.initialize!(integrator) # reinitialize before setting up atmosphere

# Initialize Terrarium-Speedy land model
land = TerrariumWetLand(integrator)
# Set up coupled model
land_sea_mask = Speedy.RockyPlanetMask(land.spectral_grid)
surface_heat_flux = Speedy.SurfaceHeatFlux(land.spectral_grid, land = Speedy.PrescribedLandHeatFlux())
surface_humidity_flux = Speedy.SurfaceHumidityFlux(land.spectral_grid, land = Speedy.PrescribedLandHumidityFlux())
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveDryModel, land.geometry, path = "outputs/")
time_stepping = Speedy.Leapfrog(land.spectral_grid, Δt_at_T31 = Minute(15))
primitive_wet_coupled = Speedy.PrimitiveWetModel(
    land.spectral_grid;
    land,
    surface_heat_flux,
    surface_humidity_flux,
    land_sea_mask,
    time_stepping,
    output
)
# add soil temperature as output variable for Speedy simulation
Speedy.add!(primitive_wet_coupled.output, Speedy.SoilTemperatureOutput())
# initialize coupled simulation
sim_coupled = Speedy.initialize!(primitive_wet_coupled)
# run it
period = Day(1)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled, period = period)
Terrarium.checkfinite!(integrator.state.prognostic)

# Land variables
Tsoil_fig = heatmap(RingGrids.Field(interior(integrator.state.temperature)[:, 1, end - 2], grid), title = "", size = (800, 400))
Tsurf_fig = heatmap(RingGrids.Field(interior(integrator.state.skin_temperature)[:, 1], grid), title = "", size = (800, 400))
Hs_fig = heatmap(RingGrids.Field(interior(integrator.state.sensible_heat_flux)[:, 1], grid), title = "", size = (800, 400))
Hl_fig = heatmap(RingGrids.Field(interior(integrator.state.latent_heat_flux)[:, 1], grid), title = "", size = (800, 400))
E_fig = heatmap(RingGrids.Field(interior(integrator.state.evaporation_ground)[:, 1], grid), title = "", size = (800, 400))
sat_fig = heatmap(RingGrids.Field(interior(integrator.state.saturation_water_ice)[:, 1, end], grid), title = "", size = (800, 400))
# Atmosphere variables
Tair_fig = heatmap(sim_coupled.diagnostic_variables.grid.temp_grid[:, 8] .- 273.15, title = "Air temperature", size = (800, 400))
pres_fig = heatmap(exp.(sim_coupled.diagnostic_variables.grid.pres_grid), title = "Surface pressure", size = (800, 400))
srad_fig = heatmap(sim.diagnostic_variables.physics.surface_shortwave_down, title = "Surface shortwave down", size = (800, 400))

# pick a point somewhere in the mid-lattitudes
T = interior(integrator.state.temperature)[2000, 1, :]
sat = interior(integrator.state.saturation_water_ice)[2000, 1, :]
f = interior(integrator.state.liquid_water_fraction)[2000, 1, :]
zs = znodes(integrator.state.temperature)

# Plot temperature and liquid fraction profiles in upper 15 layers
Makie.scatterlines(T, zs)
Makie.scatterlines(sat, zs)
Makie.scatterlines(f, zs)
