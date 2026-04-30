using Terrarium

using CUDA
using Dates
using Rasters, NCDatasets
using Statistics

using CairoMakie, GeoMakie

import RingGrids
import SpeedyWeather as Speedy

# Choose architecture based on available hardware
terrarium_arch = CUDA.functional() ? GPU() : CPU()
speedy_arch = CUDA.functional() ? Speedy.GPU() : Speedy.CPU()

"""
Naive implementation of a SpeedyWeather "wet" land model based on Terrarium.
Operates with two separate ring grids: a lower-resolution grid for Speedy and a
higher-resolution grid for Terrarium, with `RingGrids.interpolate!` used to couple them.
"""
struct TerrariumWetLand{
        NF,
        LG <: Speedy.LandGeometry,
        TM <: Terrarium.ModelIntegrator{NF},
        IU <: RingGrids.AbstractInterpolator,
        ID <: RingGrids.AbstractInterpolator,
        FT <: RingGrids.AbstractField,
        FS <: RingGrids.AbstractField,
    } <: Speedy.AbstractWetLand
    "Speedy spectral grid (low resolution)"
    spectral_grid::Speedy.SpectralGrid

    "Speedy land model geometry"
    geometry::LG

    "Initialized Terrarium model integrator (high resolution)"
    integrator::TM

    "Pre-computed interpolator: Speedy ring grid (low-res) → Terrarium ring grid (high-res)"
    interp_up::IU

    "Pre-computed interpolator: Terrarium ring grid (high-res) → Speedy ring grid (low-res)"
    interp_down::ID

    "Reusable 2D buffer Field on Terrarium ring grid (for upscaling Speedy → Terrarium)"
    buf_terrarium::FT

    "Reusable 2D buffer Field on Speedy ring grid (for downscaling Terrarium → Speedy)"
    buf_speedy::FS

    function TerrariumWetLand(
            integrator::Terrarium.ModelIntegrator{NF, Arch, Grid},
            speedy_ring_grid::RingGrids.AbstractGrid,
            terrarium_ring_grid::RingGrids.AbstractGrid;
            spectral_grid_kwargs...
        ) where {NF, Arch, Grid <: ColumnRingGrid}
        spectral_grid = Speedy.SpectralGrid(speedy_ring_grid; NF, spectral_grid_kwargs...)
        interp_up = RingGrids.interpolator(terrarium_ring_grid, speedy_ring_grid)
        interp_down = RingGrids.interpolator(speedy_ring_grid, terrarium_ring_grid)
        buf_terrarium = zeros(NF, terrarium_ring_grid)
        buf_speedy = zeros(NF, speedy_ring_grid)
        land_grid = integrator.grid
        Δz = on_architecture(CPU(), land_grid.z.Δᵃᵃᶜ)
        geometry = Speedy.LandGeometry(1, Δz[end])
        return new{NF, typeof(geometry), typeof(integrator), typeof(interp_up), typeof(interp_down), typeof(buf_terrarium), typeof(buf_speedy)}(
            spectral_grid, geometry, integrator, interp_up, interp_down, buf_terrarium, buf_speedy
        )
    end
end

Speedy.variables(land::TerrariumWetLand) = (
    Speedy.PrognosticVariable(name = :soil_temperature, dims = Speedy.Grid3D(), namespace = :land),
    Speedy.PrognosticVariable(name = :soil_moisture, dims = Speedy.Grid3D(), namespace = :land),
)

function Speedy.initialize!(
        vars::Speedy.Variables,
        land::TerrariumWetLand{NF},
        model::Speedy.PrimitiveEquation,
    ) where {NF}
    Terrarium.initialize!(land.integrator)
    state = land.integrator.state
    terrarium_ring_grid = land.buf_terrarium.grid
    # Downscale Terrarium initial state to Speedy resolution
    RingGrids.interpolate!(land.buf_speedy, RingGrids.Field(interior(state.temperature)[:, 1, end], terrarium_ring_grid), land.interp_down)
    vars.prognostic.land.soil_temperature .= land.buf_speedy .+ NF(273.15)
    RingGrids.interpolate!(land.buf_speedy, RingGrids.Field(interior(state.saturation_water_ice)[:, 1, end], terrarium_ring_grid), land.interp_down)
    vars.prognostic.land.soil_moisture .= land.buf_speedy
    return nothing
end

function Speedy.timestep!(
        vars::Speedy.Variables,
        land::TerrariumWetLand{NF},
        model::Speedy.PrimitiveEquation,
    ) where {NF}
    speedy_timestep!(vars, land)
    return nothing
end

function speedy_timestep!(
        vars::Speedy.Variables,
        land::TerrariumWetLand{NF},
    ) where {NF}
    # land constants
    consts = land.integrator.model.constants
    state = land.integrator.state
    terrarium_ring_grid = land.buf_terrarium.grid

    terrarium_grid = land.integrator.model.grid
    # Upscale Speedy fields to Terrarium resolution and update inputs
    RingGrids.interpolate!(land.buf_terrarium, vars.grid.temperature[:, end], land.interp_up)
    land.buf_terrarium.data .-= NF(273.15)  ## convert K → °C before set!
    set!(state.inputs.air_temperature, land.buf_terrarium)
    RingGrids.interpolate!(land.buf_terrarium, vars.grid.pressure, land.interp_up)
    land.buf_terrarium.data .= exp.(land.buf_terrarium.data)  ## convert log(Pa) → Pa before set!
    set!(state.inputs.air_pressure, land.buf_terrarium)
    RingGrids.interpolate!(land.buf_terrarium, vars.grid.humidity[:, end], land.interp_up)
    set!(state.inputs.specific_humidity, land.buf_terrarium)
    RingGrids.interpolate!(land.buf_terrarium, vars.parameterizations.rain_rate, land.interp_up)
    set!(state.inputs.rainfall, land.buf_terrarium)
    RingGrids.interpolate!(land.buf_terrarium, vars.parameterizations.snow_rate, land.interp_up)
    set!(state.inputs.snowfall, land.buf_terrarium)
    RingGrids.interpolate!(land.buf_terrarium, vars.parameterizations.surface_wind_speed, land.interp_up)
    set!(state.inputs.windspeed, land.buf_terrarium)
    RingGrids.interpolate!(land.buf_terrarium, vars.parameterizations.surface_shortwave_down, land.interp_up)
    set!(state.inputs.surface_shortwave_down, land.buf_terrarium)
    RingGrids.interpolate!(land.buf_terrarium, vars.parameterizations.surface_longwave_down, land.interp_up)
    set!(state.inputs.surface_longwave_down, land.buf_terrarium)
    # run land forward over speedy timestep interval;
    # we use a smaller actual timestep to ensure stability
    Terrarium.run!(land.integrator, period = vars.prognostic.clock.Δt, Δt = 300.0)
    # Downscale Terrarium output fields to Speedy resolution
    RingGrids.interpolate!(land.buf_speedy, RingGrids.Field(interior(state.skin_temperature)[:, 1, 1], terrarium_ring_grid), land.interp_down)
    vars.prognostic.land.soil_temperature .= land.buf_speedy .+ NF(273.15)
    RingGrids.interpolate!(land.buf_speedy, RingGrids.Field(interior(state.saturation_water_ice)[:, 1, end], terrarium_ring_grid), land.interp_down)
    vars.prognostic.land.soil_moisture .= land.buf_speedy
    RingGrids.interpolate!(land.buf_speedy, RingGrids.Field(interior(state.sensible_heat_flux)[:, 1, 1], terrarium_ring_grid), land.interp_down)
    vars.prognostic.land.sensible_heat_flux .= land.buf_speedy
    RingGrids.interpolate!(land.buf_speedy, RingGrids.Field(interior(state.latent_heat_flux)[:, 1, 1], terrarium_ring_grid), land.interp_down)
    vars.prognostic.land.surface_humidity_flux .= land.buf_speedy ./ consts.Llg
    RingGrids.interpolate!(land.buf_speedy, RingGrids.Field(interior(state.surface_longwave_up)[:, 1, 1], terrarium_ring_grid), land.interp_down)
    vars.parameterizations.land.surface_longwave_up .= land.buf_speedy
    RingGrids.interpolate!(land.buf_speedy, RingGrids.Field(interior(state.surface_shortwave_up)[:, 1, 1], terrarium_ring_grid), land.interp_down)
    vars.parameterizations.land.surface_shortwave_up .= land.buf_speedy
    return nothing
end

# quick test of default Speedy PrimitiveWetModel on GPU
speedy_ring_grid = RingGrids.FullGaussianGrid(24, speedy_arch)
speedy_ring_grid_cpu = on_architecture(CPU(), speedy_ring_grid)
spectral_grid = Speedy.SpectralGrid(speedy_ring_grid)
orography = Speedy.EarthOrography(spectral_grid, smoothing = false)
primitive_wet = Speedy.PrimitiveWetModel(spectral_grid; orography)
sim = Speedy.initialize!(primitive_wet)
Speedy.run!(sim, period = Day(1))

# Higher-resolution ring grid on Speedy's architecture for use in coupling buffers/interpolators
terrarium_ring_grid = RingGrids.FullGaussianGrid(48, speedy_arch)
terrarium_ring_grid_cpu = on_architecture(CPU(), terrarium_ring_grid)

Nz = 30
Δz_min = 0.05
grid = ColumnRingGrid(terrarium_arch, Float32, ExponentialSpacing(; N = Nz, Δz_min), terrarium_ring_grid_cpu)
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

# Initialize Terrarium-Speedy land model using separate low-res (Speedy) and high-res (Terrarium) ring grids
land = TerrariumWetLand(integrator, speedy_ring_grid, terrarium_ring_grid)
# Set up coupled model
land_sea_mask = Speedy.RockyPlanetMask(land.spectral_grid)
surface_heat_flux = Speedy.SurfaceHeatFlux(land.spectral_grid, land = Speedy.PrescribedLandHeatFlux())
surface_humidity_flux = Speedy.SurfaceHumidityFlux(land.spectral_grid, land = Speedy.PrescribedLandHumidityFlux())
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveWetModel, path = "outputs/")
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
period = Hour(1)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled; period)
Terrarium.checkfinite!(integrator.state.prognostic)

# Plotting: interpolate HEALPix fields to FullGaussianGrid on CPU (HEALPix not supported for heatmap)
land_plot_grid = RingGrids.FullGaussianGrid(48)  ## matches terrarium ring grid resolution
atm_plot_grid = RingGrids.FullGaussianGrid(24)  ## matches speedy ring grid resolution
land_interp_plot = RingGrids.interpolator(land_plot_grid, terrarium_ring_grid_cpu)
atm_interp_plot = RingGrids.interpolator(atm_plot_grid, speedy_ring_grid_cpu)

## Wrap a (GPU or CPU) array slice as a RingGrids.Field and interpolate to a FullGaussianField on CPU
function to_plotting_grid(data::AbstractArray, src_grid::RingGrids.AbstractGrid, plot_grid::RingGrids.AbstractGrid, interp)
    src = RingGrids.Field(Array(data), src_grid)
    out = zeros(eltype(src), plot_grid)
    RingGrids.interpolate!(out, src, interp)
    return out
end

# Land variables
Tsoil_fig = heatmap(to_plotting_grid(interior(integrator.state.temperature)[:, 1, end - 2], terrarium_ring_grid_cpu, land_plot_grid, land_interp_plot), title = "Soil temperature (°C)", size = (800, 400))
Tsurf_fig = heatmap(to_plotting_grid(interior(integrator.state.skin_temperature)[:, 1, 1], terrarium_ring_grid_cpu, land_plot_grid, land_interp_plot), title = "Skin temperature (°C)", size = (800, 400))
Hs_fig = heatmap(to_plotting_grid(interior(integrator.state.sensible_heat_flux)[:, 1, 1], terrarium_ring_grid_cpu, land_plot_grid, land_interp_plot), title = "Sensible heat flux (W/m²)", size = (800, 400))
Hl_fig = heatmap(to_plotting_grid(interior(integrator.state.latent_heat_flux)[:, 1, 1], terrarium_ring_grid_cpu, land_plot_grid, land_interp_plot), title = "Latent heat flux (W/m²)", size = (800, 400))
E_fig = heatmap(to_plotting_grid(interior(integrator.state.evaporation_ground)[:, 1, 1], terrarium_ring_grid_cpu, land_plot_grid, land_interp_plot), title = "Evaporation (m/s)", size = (800, 400))
sat_fig = heatmap(to_plotting_grid(interior(integrator.state.saturation_water_ice)[:, 1, end], terrarium_ring_grid_cpu, land_plot_grid, land_interp_plot), title = "Surface saturation", size = (800, 400))

# Atmosphere variables (Speedy v0.19 API)
Tair_fig = heatmap(to_plotting_grid(Array(sim_coupled.variables.grid.temperature[:, 8].data) .- 273.15f0, RingGrids.HEALPixGrid(24), atm_plot_grid, atm_interp_plot), title = "Air temperature (°C)", size = (800, 400))
pres_fig = heatmap(to_plotting_grid(exp.(Array(sim_coupled.variables.grid.pressure.data)), RingGrids.HEALPixGrid(24), atm_plot_grid, atm_interp_plot), title = "Surface pressure (Pa)", size = (800, 400))
srad_fig = heatmap(to_plotting_grid(Array(sim.variables.parameterizations.surface_shortwave_down.data), RingGrids.HEALPixGrid(24), atm_plot_grid, atm_interp_plot), title = "Surface shortwave down (W/m²)", size = (800, 400))

# Pick a point somewhere in the mid-latitudes and plot vertical profiles
T = Array(interior(integrator.state.temperature)[2000, 1, :])
sat = Array(interior(integrator.state.saturation_water_ice)[2000, 1, :])
f = Array(interior(integrator.state.liquid_water_fraction)[2000, 1, :])
zs = znodes(integrator.state.temperature)

# Plot temperature, saturation, and liquid fraction vertical profiles
Makie.scatterlines(T, zs)
Makie.scatterlines(sat, zs)
Makie.scatterlines(f, zs)
