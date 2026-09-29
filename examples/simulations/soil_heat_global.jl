# # [Soil heat conduction at global scale](@id soil_heat_global)
# Here we extend the single column soil heat conduction [example](@ref soil_heat_column)
# to do global scale simulations, accelerated by GPU (if available).

using Terrarium

using CUDA
using Dates
using Rasters, NCDatasets
using NumericalEarth.DataWrangling
using NumericalEarth.SoilGrids
using Statistics

using CairoMakie, GeoMakie

import RingGrids

import DisplayAs #hide

input_dir = "inputs" #hide
@info "Current working directory: $(pwd())" #hide

# First we check if GPU is available and choose the architecture correspondingly.
arch = CUDA.functional() ? GPU() : CPU()
@info "Setting up simulation on $arch" #hide

# Next, we load a land-sea mask at ~1° resolution. The mask is a full Gaussian grid, as defined by the
# [FullGaussianGrid](@extref RingGrids.FullGaussianGrid) from RingGrids.jl. Irrespective of the architecture
# used for simulation, the land-sea mask is kept on the CPU for easy scalar indexing, which is by default not
# allowed for GPU arrays (see [here](https://cuda.juliagpu.org/stable/usage/workflow/#UsageWorkflowScalar)).
NF = Float32
land_sea_frac_native = RingGrids.Field(arch, ERA5LandInvariants(), "lsm"; NF)
ring_grid = on_architecture(arch, RingGrids.FullGaussianGrid(72))
land_sea_frac_N72 = RingGrids.interpolate(ring_grid, land_sea_frac_native)
fig = heatmap(land_sea_frac_N72)
DisplayAs.PNG(fig) #hide

# Then we set up a masked [`ColumnRingGrid`](@ref), selecting only grid points
# with >50% land cover:
land_mask = land_sea_frac_N72 .> 0.5
land_mask_cpu = on_architecture(CPU(), land_mask)
grid = ColumnRingGrid(arch, NF, ExponentialSpacing(N = 30), land_mask.grid, land_mask)
grid_lon, grid_lat = RingGrids.get_lonlats(grid.rings) # in radians, on CPU

# Remember from the documentation section on [grids](@ref Grids), that the `x`-axis of the Oceananigans [`RectilinearGrid`](@extref Oceananigans.Grids.RectilinearGrid)
# corresponds to a single index following the ring order (for more details, see the [corresponding section in the
# RingGrids.jl documentation](https://speedyweather.github.io/SpeedyWeatherDocumentation/stable/ringgrids/#Indexing-Fields)).

# Now we create our [`SoilModel`](@ref), this time without a model initializer:
model = SoilModel(grid)

# To make the simulation a bit more interesting, we will use spatially periodic initial and boundary conditions.
# The climatology will be determined by latitude with a maximum of 20 °C at the equator and minimum of -20°C at
# the poles.
mean_annual_temperature(lat) = 20 - abs(40 * sin(lat)) # maximum at equator

fig = heatmap(RingGrids.Field(mean_annual_temperature.(grid_lat), grid.rings))
DisplayAs.PNG(fig) #hide

# The initial temperature profiles will be linear temperature profiles similar to what we would get from
# [`QuasiThermalSteadyState`](@ref) but with a hardcoded geothermal gradient of 0.05 K/m. We don't use
# the `QuasiThermalSteadyState` initializer here because it does not (yet) support spatially variable parameters.
function initial_soil_temperature(x, z)
    latᵢ = lat_masked[round(Int, x)]
    T₀ = mean_annual_temperature(latᵢ)
    T = T₀ - 0.05 * z
    return T
end

# We will impose a periodic temperature boundary condition at the surface to represent the daily cycle.
# We can specify it directly as a continuous function thanks to the power of `Oceananigans` `Field`s.
# However, we will need to use an enclosing function here to i) copy the vector of latitudes onto the
# device specified by `arch`, and ii) ensure that the compiler is able to infer the correct type of the
# coordinate values in the boundary condition function `periodic_bc`, which returns a temperature value
# based on the coordinate `x` of the `RectiLinearGrid` and the time `t` (s).
function get_temperature_bc(lon::AbstractVector, lat::AbstractVector, amplitude = 10.0)
    ## make sure coordinate arrays are on the same device
    lon_device = on_architecture(arch, NF.(lon))
    lat_device = on_architecture(arch, NF.(lat))
    ## function matching the expected signature for boundary conditions on a column-based grid
    function periodic_bc(x::NF, t::NF) where {NF}
        ## x coordinate is just the grid cell index
        lonₓ = lon_device[round(Int, x)]
        latₓ = lat_device[round(Int, x)]
        ## use climatology at latₓ as the mean of BC
        T₀ = mean_annual_temperature(latₓ)
        seconds_per_day = NF(24 * 3600)
        ## shift BC by longitude in radians to (roughly) mimic the global daily cycle
        T = T₀ + NF(amplitude) * sin(2π * t / seconds_per_day - lonₓ)
        return T
    end
    return periodic_bc
end


lon_masked = grid_lon[land_mask_cpu]
lat_masked = grid_lat[land_mask_cpu] # mask out non-land points
bc = PrescribedSurfaceTemperature(:T_ub, get_temperature_bc(lon_masked, lat_masked))
inits = (temperature = initial_soil_temperature,)

# We are finally ready to initialize our model with the above initial and boundary conditions:
integrator = initialize(model, boundary_conditions = bc, initializers = inits)

# Let's already plot the initial surface temperature state to see what it looks like:
T_surface_initial = RingGrids.Field(arch, interior(integrator.state.ground_temperature), grid)
fig = heatmap(T_surface_initial[:, 1, 1], title = "Temperature of uppermost soil layer", colorrange = (-20, 20))
DisplayAs.PNG(fig) #hide

# We will do a quick check, advancing the simulation by one timestep with Δt = 15 minutes (900 seconds)
timestep!(integrator)
@time timestep!(integrator)

# ...then run the simulation for 12 hours to see the temperature change!
@time run!(integrator, period = Hour(12), Δt = 600.0)

# Now we can plot the resulting soil (surface) temperature map using RingGrids + GeoMakie:
T_surface = RingGrids.Field(arch, interior(integrator.state.ground_temperature), grid)
fig = heatmap(T_surface[:, 1, 1], title = "Temperature of uppermost soil layer", colorrange = (-20, 20))
DisplayAs.PNG(fig) #hide

# We can also wrap the `integrator` in an Oceananigans `Simulation` which can be used to add
# callbacks and save outputs.
using Oceananigans.OutputWriters: JLD2Writer, AveragedTimeInterval
using Oceananigans.Units: days

sim = Simulation(integrator; Δt = 600.0, stop_time = 2days)
run!(sim)
