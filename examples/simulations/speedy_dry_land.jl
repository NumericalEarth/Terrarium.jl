using Terrarium

using CUDA
using Dates
using Rasters, NCDatasets
using Statistics

using CairoMakie, GeoMakie

import RingGrids
import SpeedyWeather as Speedy

# Temporary workaround for SpeedyWeather.jl#919
Speedy.SpeedyTransforms.FFTW.set_num_threads(1)

"""
Naive implementation of a SpeedyWeather "dry" land model based on Terrarium.
"""
struct TerrariumDryLand{
        NF,
        LG <: Speedy.LandGeometry,
        TM <: Terrarium.ModelIntegrator{NF},
    } <: Speedy.AbstractDryLand
    "Speedy spectral grid"
    spectral_grid::Speedy.SpectralGrid

    "Speedy land model geometry"
    geometry::LG

    "Initialized Terrarium model integrator"
    integrator::TM

    function TerrariumDryLand(integrator::Terrarium.ModelIntegrator{NF, Arch, Grid}; spectral_grid_kwargs...) where {NF, Arch, Grid <: ColumnRingGrid}
        spectral_grid = Speedy.SpectralGrid(integrator.model.grid.rings; NF, spectral_grid_kwargs...)
        land_grid = integrator.grid
        Δz = on_architecture(CPU(), land_grid.z.Δᵃᵃᶜ)
        geometry = Speedy.LandGeometry(1, Δz[end])
        return new{eltype(integrator), typeof(geometry), typeof(integrator)}(spectral_grid, geometry, integrator)
    end
end

Speedy.variables(land::TerrariumDryLand) = (
    Speedy.PrognosticVariable(name = :soil_temperature, dims = Speedy.Grid3D(), namespace = :land),
)

function Speedy.initialize!(
        ::Any,
        progn::Speedy.PrognosticVariables,
        diagn::Speedy.DiagnosticVariables,
        land::TerrariumDryLand{NF},
        model::Speedy.PrimitiveEquation,
    ) where {NF}
    Tsoil = interior(land.integrator.state.temperature)[:, 1, end] .+ 273.15
    progn.land.soil_temperature .= Tsoil
    Terrarium.initialize!(land.integrator)
    return nothing
end

function Speedy.timestep!(
        progn::Speedy.PrognosticVariables,
        diagn::Speedy.DiagnosticVariables,
        land::TerrariumDryLand{NF},
        model::Speedy.PrimitiveEquation,
    ) where {NF}
    # get speedy state variables
    Tair = @view diagn.grid.temp_grid[:, end]
    # terrarium state
    state = land.integrator.state
    set!(state.inputs.air_temperature, Tair) # first set directly to avoid allocating new fields
    set!(state.inputs.air_temperature, state.inputs.air_temperature - 273.15) # then convert to celsius
    # run land forward over speedy timestep interval;
    # we use a smaller actual timestep to ensure stability
    Terrarium.run!(land.integrator, period = progn.clock.Δt, Δt = 300.0)
    # Get soil temperatures
    Tsoil = state.temperature
    # Get surface temperature (last z-layer in Oceananigans grids)
    Nx, Nz = Tsoil.grid.Nx, Tsoil.grid.Nz
    Tsurf = @view Tsoil[1:Nx, 1, Nz:Nz]
    # Update speedy soil/skin temperature
    progn.land.soil_temperature .= Tsurf .+ NF(273.15)
    return nothing
end

ring_grid = RingGrids.FullGaussianGrid(24)
Nz = 30
Δz_min = 0.05 # currently the coupling is only stable with a large surface layer
grid = ColumnRingGrid(CPU(), Float32, ExponentialSpacing(; N = Nz, Δz_min), ring_grid)
# Initial conditions
soil_initializer = SoilInitializer(eltype(grid))
# Soil model with prescribed surface temperature BC
model = SoilModel(grid, initializer = soil_initializer)
air_temperature = Field(grid, XY())
Tair_input = InputSource(grid, air_temperature; name = :air_temperature)
bcs = PrescribedSurfaceTemperature(:air_temperature)
integrator = initialize(model, ForwardEuler(eltype(grid)), Tair_input, boundary_conditions = bcs)

# Initialize Terrarium-Speedy land model
land = TerrariumDryLand(integrator)
# Set up coupled model
land_sea_mask = Speedy.RockyPlanetMask(land.spectral_grid)
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveDryModel, land.geometry, path = "outputs/")
time_stepping = Speedy.Leapfrog(land.spectral_grid, Δt_at_T31 = Minute(15))
primitive_dry_coupled = Speedy.PrimitiveDryModel(land.spectral_grid; land, land_sea_mask, time_stepping, output)
# add soil temperature as output variable for Speedy simulation
Speedy.add!(primitive_dry_coupled.output, Speedy.SoilTemperatureOutput())
# initialize coupled simulation
sim_coupled = Speedy.initialize!(primitive_dry_coupled)
# run it
Speedy.run!(sim_coupled, period = Day(2), output = true)

# Soil temperature in the 5th layer (~0.54 m)
Tsoil_fig = heatmap(RingGrids.Field(interior(integrator.state.temperature)[:, 1, end - 4], grid), title = "", size = (800, 400))
# Atmosphere variables
Tair_fig = heatmap(sim_coupled.diagnostic_variables.grid.temp_grid[:, 8] .- 273.15, title = "Air temperature", size = (800, 400))
pres_fig = heatmap(exp.(sim_coupled.diagnostic_variables.grid.pres_grid), title = "Surface pressure", size = (800, 400))
srad_fig = heatmap(exp.(sim_coupled.diagnostic_variables.physics.surface_shortwave_down), title = "Surface shortwave down", size = (800, 400))

# animate surface air and soil temperatures
Speedy.animate(sim_coupled, variable = "temp", coastlines = false, level = spectral_grid.nlayers, output_file = "plots/speedy_terrarium_dry_air_temperature.mp4")
# Speedy.animate(sim_coupled, variable = "st", coastlines = false, level = 1, output_file = "plots/speedy_terrarium_dry_soil_temperature.mp4") # TODO: this is broken now for some reason?

# pick a point somewhere in the mid-lattitudes
T = interior(integrator.state.temperature)[2000, 1, :]
f = interior(integrator.state.liquid_water_fraction)[2000, 1, :]
zs = znodes(integrator.state.temperature)
# Plot temperature and liquid fraction profiles in upper 15 layers
Makie.scatterlines(T[(end - 15):end], zs[(end - 15):end])
Makie.scatterlines(f, zs)
