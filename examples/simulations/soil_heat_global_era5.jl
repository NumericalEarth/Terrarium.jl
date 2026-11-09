using Terrarium

using CUDA
using Rasters, NCDatasets

using CairoMakie, GeoMakie

import Pkg.Artifacts: @artifact_str
import RingGrids
import SpeedyWeather

# run on GPU if available
arch = CUDA.functional() ? GPU() : CPU()

ring_grid = RingGrids.FullGaussianGrid(72)
lon, lat = RingGrids.get_londlatds(ring_grid)

# Load land-sea mask at native ~0.1° resolution
land_sea_frac_native = RingGrids.Field(arch, ERA5LandInvariants(), "lsm"; NF)
land_sea_frac_N72 = RingGrids.interpolate(ring_grid, land_sea_frac_native)
heatmap(land_sea_frac_N72)

# Load ERA-5 2 meter air temperature at ~1° resolution
Tair_raster = Raster(Terrarium.get_asset(ERA5LandForcings()), name = "t2m")
Tsurf_0 = convert.(Float32, replace_missing(Tair_raster, NaN)) .- 273.15f0

# Set up grids
land_mask = land_sea_frac_N72 .> 0.5 # select only grid points with > 50% land
Nz = 30 # number of soil layers
grid = ColumnRingGrid(arch, ExponentialSpacing(N = Nz), land_mask)

# Construct input sources
Tair_forcing = InputSource(grid, rebuild(Tair_raster, name = :Tair))
Tsurf_0 = Tair_raster[Ti(1)][findall(land_mask)]

soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), NoFlow()))
model = SoilModel(grid; soil, timestepper = ForwardEuler(eltype(grid)))
boundary_conditions = PrescribedSurfaceTemperature(:Tair)
# Initial conditions
initializers = (
    # steady-ish state initial condition for temperature
    temperature = (x, z) -> Tsurf_0[Int(round(x)) + 1] - 0.02 * z,
    # dry soil
    saturation_water_ice = 1.0,
)
inputs = InputSources(Tair_forcing)
integrator = initialize(model; inputs, initializers, boundary_conditions)
@time timestep!(integrator)
@time run!(integrator, period = Day(10), dt = 120.0)

# plot heatmap of soil temperature at the surface
heatmap(RingGrids.Field(interior(integrator.state.temperature, :, :, Nz), grid))
