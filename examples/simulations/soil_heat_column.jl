# # [Soil heat conduction in a 1D vertical column](@id soil_heat_column)
# ## Part I: Nonlinear heat conduction with phase change
# This example shows how to set up a simple model of nonlinear heat conduction
# in a single vertical soil column (similar to the example shown in the README).
# The [`SoilThermodynamics`](@ref) process in Terrarium solves the nonlinear form
# of the heat equation with phase change. This allows for the simulation of
# freeze/thaw dynamics in both seasonally and perennially frozen soils.

using Terrarium
## For plotting
import CairoMakie as Makie
import DisplayAs

# We start by creating a single column grid with 10 exponentially spaced vertical soil layers:
grid = ColumnGrid(CPU(), Float32, ExponentialSpacing(N = 10))

# Next we specify an initializer suitable for [`SoilModel`](@ref). We will use
# a quasi-steady-state initialization for soil temperature (linear with depth)
# and a fully saturated state for soil water/ice content.
initializer = SoilInitializer(
    eltype(grid),
    energy = QuasiThermalSteadyState(eltype(grid), T₀ = -1.0),
    hydrology = ConstantSaturation(eltype(grid), sat = 1.0)
)

# We're now ready to create our model.
# We also choose the time stepper to be [`ForwardEuler`](@ref) for this example.
model = SoilModel(grid; timestepper = ForwardEuler(eltype(grid)), initializer = initializer)

# Boundary conditions are imposed directly on the corresponding `Field`s during
# initialization. We here set a constant surface temperature of 1°C, while the lower
# boundary is left with its default zero flux condition. Note that the name assigned
# to the boundary condition `:T_ub` is arbitrary; you can set it to whatever you want.
boundary_conditions = PrescribedSurfaceTemperature(:T_ub, 1.0)

# Then, we initialize the integrator:
integrator = initialize(model; boundary_conditions)

# We can now try taking a single one timestep and `@time` it; note that the first evaluation
# will be slower due to compilation.
timestep!(integrator)
@time timestep!(integrator)

# We an also run the simulation forward for a set period of time:
@time run!(integrator, period = Day(3))

# Now let's extract the relevant state variables for inspection. The function
# [`interior`](@extref Oceananigans.Fields.interior) comes from Oceananigans
# and is used to extract the interior values of the spatial field on grid cell centers,
# excluding the halo regions used for representing boundary conditions.
T = interior(integrator.state.temperature)[1, 1, :]
f = interior(integrator.state.liquid_water_fraction)[1, 1, :]

# Finally, we plot the temperature and liquid fraction profiles. With
# [`znodes`](@extref Oceananigans.Grids.znodes), the positions of the interior
# nodes in the $z$-direction can be extracted.
zs = znodes(integrator.state.temperature)
let fig = Makie.Figure()
    ax1 = Makie.Axis(fig[1, 1], ylabel = "Depth / m", xlabel = "Temperature / °C")
    ax2 = Makie.Axis(fig[1, 2], xlabel = "Liquid fraction")

    Makie.scatterlines!(ax1, T, zs)
    Makie.scatterlines!(ax2, f, zs)
    DisplayAs.PNG(fig)
end

# ## Part II: Validation against analytical solution
#
# Here we set up an idealized simulation for which the numerical solution
# can be verified against the analytical solution of the heat equation,
#
#     ∂T/∂t = α ∂²T/∂z²
#
# with the following sinusoidal surface boundary condition,
#
#     T(0,t) = T₀ + A sin(2πt/P)
#
# In this case, the analytical solution is
#
#     T(z,t) = T₀ + A exp(-z√(π/(αP))) sin(2πt/P − z√(π/(αP)))
#
# To make the analytical solution valid, we will need to configure the model
# to represent a homogeneous solid material without pore space or water/ice
# such that the dynamics match that of the linear heat equation above.

# ### Physical parameters
# We first define the thermal properties of a pure mineral soil and the parameters
# of the sinusoidal surface forcing.

T₀ = 2.0               # mean surface temperature
A = 1.0                # forcing amplitude
P = 24 * 3600          # forcing period (seconds)
k = 2.0                # thermal conductivity
c = 1.0e6              # volumetric heat capacity
α = k / c              # thermal diffusivity
w = 2 * pi / P         # angular frequency of surface forcing
d = sqrt(2 * α / w)    # thermal damping depth

# ### Analytical solution

# The exact solution for sinusoidal surface forcing decays
# exponentially with depth and develops a phase lag proportional to z/d,
# where d is the thermal damping depth.

function heat_conduction_solution(T₀, A, P, α)
    w = 2 * pi / P              # angular frequency of surface forcing
    d = sqrt(2 * α / w)         # thermal damping depth
    T(z, t) = T₀ +
        A *
        exp(-z / d) *
        sin(w * t - (z / d))
    return T
end

## T_sol(z, t) is a function of depth and time
T_sol = heat_conduction_solution(T₀, A, P, α);

# For evaluation against an analytical solution, we will use a much denser grid with 50 vertical
# soil layers and double precision:

grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_min = 0.02, N = 50))

# First, we need to configure the solid material properties for linear heat diffusion.
# We will set soil carbon density and porosity to zero. This corresponds to an idealized
# solid material with homogeneous thermal properties and no pore space (e.g. a slab of rock).

biogeochem = ConstantSoilCarbonDensity(eltype(grid), ρ_soc = 0.0);
soil_porosity = ConstantSoilPorosity(eltype(grid), mineral_porosity = 0.0);
## a pure-clay (non-quartz) texture makes the bulk solid conductivity equal the `mineral`
## endpoint `k`, giving the homogeneous slab assumed by the analytical solution
strat = HomogeneousSoilStratigraphy(eltype(grid); texture = SoilTexture(eltype(grid), :clay), porosity = soil_porosity);
thermal_properties = SoilThermalProperties(
    eltype(grid);
    conductivities = SoilThermalConductivities(eltype(grid), mineral = k),
    heat_capacities = SoilHeatCapacities(eltype(grid), mineral = c),
)

# Here we set up the "soil" processes according to our configuration.

energy = SoilThermodynamics(eltype(grid); thermal_properties);
soil = SoilEnergyWaterCarbon(eltype(grid); energy, strat, biogeochem);
model = SoilModel(grid; soil);

# Here we prescribe a sinusoidal temperature at the surface, again leaving the bottom boundary condition
# set to its default (zero flux). The initial condition is set to the analytical solution at `t = 0`.

upper_bc(z, t) = T₀ + A * sin(2π * t / P);
bcs = PrescribedSurfaceTemperature(:T_ub, upper_bc);
initializers = (temperature = (x, z) -> T_sol(-z, 0.0),);
integrator = initialize(model; initializers, boundary_conditions = bcs);

# We integrate forward for two full forcing periods using an Oceananigans
# Simulation, saving the temperature profile to a JLD2 file at every time step.
using Oceananigans
using JLD2

output_dir = mkpath("outputs")
output_file = joinpath(output_dir, "soil_heat_output.jld2")

Δt = 60.0
simulation = Simulation(integrator; Δt = Δt, stop_time = 2P)
simulation.output_writers[:temperature] = Oceananigans.OutputWriters.JLD2Writer(
    integrator,
    (; temperature = integrator.state.temperature);
    filename = output_file,
    schedule = TimeInterval(Δt),
    including = [:grid],
    overwrite_existing = true
)

run!(simulation)

# We can then read the model output as a `FieldTimeSeries` and compute the analytical solution for comparison:

Ts = FieldTimeSeries(output_file, "temperature")
Makie.lines(Ts[1, 1, 50, :])
z = znodes(Ts)
T_numeric = Ts[1, 1, :, :]  # shape: (100, 2881)
T_target = T_sol.(reshape(-z, :, 1), reshape(Ts.times, 1, :))  # shape: (100, 2881)
relative_error = abs.((T_numeric .- T_target) ./ T_target)  # relative error

max_error = maximum(relative_error)

println("Maximum relative error = ", max_error)

@assert max_error < 0.01 "Max relative error exceeded threshold: $max_error"

# ### Plot comparison at final timestep

# The numerical and analytical profiles should overlay eachother
# within the top few damping depths.

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1], xlabel = "Temperature", ylabel = "Depth")
    Makie.ylims!(ax, -10d, 0.0)
    Makie.lines!(ax, T_numeric[:, end], z, label = "Numerical")
    Makie.lines!(ax, T_target[:, end], z, linestyle = :dash, label = "Analytical")
    Makie.axislegend(ax, position = :lb)
    DisplayAs.PNG(fig)
end
