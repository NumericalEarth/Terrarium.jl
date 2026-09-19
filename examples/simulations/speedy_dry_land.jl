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

# Build the SpeedyWeather ring + spectral grid
ring_grid = RingGrids.FullGaussianGrid(24)
spectral_grid = Speedy.SpectralGrid(ring_grid)

# Build the Terrarium soil model on the matching ring grid
Nz = 30
Δz_min = 0.05  # large surface layer keeps the coupling stable
grid = ColumnRingGrid(Terrarium.CPU(), Float32, ExponentialSpacing(; N = Nz, Δz_min), ring_grid)
soil_initializer = SoilInitializer(eltype(grid))

# Soil model with a prescribed surface-temperature boundary condition driven
# by SpeedyWeather's near-surface air temperature.
soil_model = SoilModel(grid; initializer = soil_initializer)
air_temperature_field = Field(grid, Terrarium.Ground(XY()))
Tair_input = InputSource(grid, air_temperature_field; name = :air_temperature)
bcs = PrescribedSurfaceTemperature(:air_temperature)

# Wrap the Terrarium model as a SpeedyWeather land component
land = SpeedyWeather.LandModel(
    spectral_grid, soil_model;
    timestepper = ForwardEuler(eltype(grid)),
    boundary_conditions = bcs,
    input_variables = Terrarium.variables(Tair_input),
    Δt = 300.0,
)

# Build the coupled PrimitiveDryModel
land_sea_mask = Speedy.RockyPlanetMask(land.spectral_grid)
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveDryModel; nlayers_soil = land.geometry.nlayers, path = "outputs/")
time_stepping = Speedy.Leapfrog(land.spectral_grid, Δt_at_T31 = Minute(15))
primitive_dry_coupled = Speedy.PrimitiveDryModel(
    land.spectral_grid;
    land,
    land_sea_mask,
    time_stepping,
    output,
)
Speedy.add!(primitive_dry_coupled.output, Speedy.SoilTemperatureOutput())

# Run the coupled simulation
sim_coupled = Speedy.initialize!(primitive_dry_coupled)
period = Day(2)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled, period = period, output = true)

# The Terrarium state lives inside SpeedyWeather's Variables tree
land_state = sim_coupled.variables.prognostic.land.terrarium

# Land variables
# Soil temperature in the 5th layer from top (~0.54 m)
Tsoil_fig = heatmap(
    RingGrids.Field(interior(land_state.temperature)[:, 1, end - 4], grid),
    title = "", size = (800, 400),
)

# Atmosphere variables
Tair_fig = heatmap(
    sim_coupled.variables.grid.temperature[:, 8] .- 273.15,
    title = "Air temperature", size = (800, 400),
)
pres_fig = heatmap(
    exp.(sim_coupled.variables.grid.pressure),
    title = "Surface pressure", size = (800, 400),
)
srad_fig = heatmap(
    sim_coupled.variables.parameterizations.surface_shortwave_down,
    title = "Surface shortwave down", size = (800, 400),
)

# Soil temperature and liquid fraction profiles at one mid-latitude point
T = interior(land_state.temperature)[2000, 1, :]
f = interior(land_state.liquid_water_fraction)[2000, 1, :]
zs = znodes(land_state.temperature)
Makie.scatterlines(T[(end - 15):end], zs[(end - 15):end])
Makie.scatterlines(f, zs)
