# currently only works in Julia 1.10 with Enzyme
using Terrarium

using CUDA
using Dates
using Rasters, NCDatasets
using Statistics

using CairoMakie, GeoMakie
using Enzyme, Checkpointing

import RingGrids
import SpeedyWeather

# run on GPU if available
arch = CUDA.functional() ? Terrarium.GPU() : Terrarium.CPU()

# Load land-sea mask at ~1° resolution
land_sea_frac = convert.(Float32, dropdims(Raster("inputs/era5-land_land_sea_mask_N72.nc"), dims=Ti))
land_sea_frac_field = RingGrids.FullGaussianField(Matrix(land_sea_frac), input_as=Matrix)
heatmap(land_sea_frac_field)

# Load ERA-5 2 meter air temperature at ~1° resolution
Tair_raster = Raster("inputs/external/era5-land/2m_temperature/era5_land_2m_temperature_2023_N72.nc")
Tsurf_0 = convert.(Float32, replace_missing(Tair_raster, NaN)) .- 273.15f0
# heatmap(Tair_raster[:,:,1])

# Set up grids
land_mask = land_sea_frac_field .> 0.5 # select only grid points with > 50% land

# for now let's do it actually without the mask to keep the example simple
grid = ColumnRingGrid(arch, Float64, ExponentialSpacing(N=30), land_mask.grid) #, land_mask.grid, land_mask)
lon, lat = RingGrids.get_londlatds(grid.rings)

# Construct input sources
Tair_forcing = InputSource(grid, rebuild(Tair_raster, name=:Tair))
Tsurf_0 = Tair_raster[Ti(1)][findall(land_mask)]

# Initial conditions
initializer = FieldInitializers(
	# steady-ish state initial condition for temperature
	temperature = (x,z) -> -1 - 0.01*z,
	# fully saturated soil pores
	saturation_water_ice = 1.0,
)
model = SoilModel(grid; initializer)
# constant surface temperature of 1°C
boundary_conditions = PrescribedSurfaceTemperature(:Tair)
integrator = initialize(model, ForwardEuler(), Tair_forcing; boundary_conditions)

# spin up a little
@time run!(integrator, period=Day(5), Δt=900.0)

# Enzyme prep 
scheme = Revolve(1)
dintegrator = make_zero(integrator)
N_t = 200 

# this uses checkpointing 
autodiff(set_runtime_activity(Enzyme.Reverse), run!, Const, Duplicated(integrator, dintegrator), Const(scheme), Const(N_t))
