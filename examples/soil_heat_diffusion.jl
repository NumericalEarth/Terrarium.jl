# # Soil heat diffusion with periodic surface forcing
#
# This example demonstrates how to simulate 1D soil heat conduction
# in a single vertical column using Terrarium.jl.
#
# We verify the numerical solution against the analytical solution of the heat equation:
#
#     ∂T/∂t = α ∂²T/∂z²
#
# with the following sinusoidal surface boundary condition:
#
#     T(0,t) = T₀ + A sin(2πt/P)
#
# where the analytical solution is:
#
#     T(z,t) = T₀ + A exp(-z√(π/(αP))) sin(2πt/P − z√(π/(αP)))
#
# This example validates:
#
# - exponential damping with depth
# - phase lag with depth
# - agreement between analytic and numerical solution

using Terrarium
using CairoMakie
using JLD2

# ## Physical parameters
# We define the thermal properties of a pure mineral soil and the parameters of the sinusoidal surface forcing.

T₀ = 2.0;               # mean surface temperature
A = 1.0 ;               # forcing amplitude
P = 24 * 3600  ;       # forcing period (seconds)
k = 2.0   ;             # thermal conductivity
c = 1.0e6  ;            # volumetric heat capacity
α = k / c  ;            # thermal diffusivity
w = 2*pi/P   ;           # angular frequency of surface forcing
d = sqrt(2*α/w)  ;       # thermal damping depth

# ## Analytical solution

# The exact solution for sinusoidal surface forcing decays 
# exponentially with depth and develops a phase lag proportional to z/d, 
# where d is the thermal damping depth.

function heat_conduction_solution(T₀, A, P, α)
    w = 2*pi/P              # angular frequency of surface forcing
    d = sqrt(2*α/w)         # thermal damping depth
    T(z, t) = T₀ +
              A *
              exp(-z / d) *
              sin(w * t - (z/d))
    return T
end

T_sol = heat_conduction_solution(T₀, A, P, α);

# ## Model setup

# We build a 1D column grid with exponential vertical spacing, refining
# near the surface where temperature gradients are steepest. To isolate
# pure heat conduction, we zero out soil carbon and porosity so only the
# mineral thermal properties are active.

grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_min = 0.05, Δz_max = 100.0, N = 100,));

# Configure soil properties for pure heat diffusion:

# set carbon content to zero

biogeochem = ConstantSoilCarbonDensity(ρ_soc = 0.0);

# set porosity to zero

soil_porosity = ConstantSoilPorosity(mineral_porosity = 0.0);
strat = HomogeneousStratigraphy(Float64; porosity = soil_porosity);

# set thermal properties

thermal_properties = SoilThermalProperties(
    eltype(grid);
    conductivities = SoilThermalConductivities(mineral = k),
    heat_capacities = SoilHeatCapacities(mineral = c),)

energy = SoilEnergyBalance(eltype(grid); thermal_properties);
soil = SoilEnergyWaterCarbon(eltype(grid); energy, strat, biogeochem);
model = SoilModel(grid; soil);

# ## Boundary and initial conditions

# The upper boundary follows the prescribed sinusoidal forcing. The initial
# condition is set to the analytical solution at t=0.

upper_bc(z, t) = T₀ + A * sin(2π * t / P);
bcs = PrescribedSurfaceTemperature(:Tsurf, upper_bc);

# Temp IC
initializers = (temperature = (x, z) -> T_sol(-z, 0.0),);
integrator = initialize(model, ForwardEuler(); initializers, boundary_conditions=bcs);



# ## Run simulation
#
# We integrate forward for two full forcing periods using an Oceananigans
# Simulation, saving the temperature profile to a JLD2 file at every time step.
Δt = 60.0;

simulation = Simulation(integrator; Δt = Δt, stop_time = 2P)

simulation.output_writers[:temperature] = Oceananigans.OutputWriters.JLD2Writer(
    integrator,
    (; temperature = integrator.state.temperature);
    filename = "soil_heat_output.jld2",
    schedule = TimeInterval(Δt),
    overwrite_existing = true
)

run!(simulation)

# ## Read output back from file

times, Ts_buffer = jldopen("soil_heat_output.jld2") do file
    iters = string.(0:2880)
    t = [file["timeseries/t/$i"] for i in iters]
    T = [file["timeseries/temperature/$i"] for i in iters]
    return t, T
end
# ## Extract numerical solution

# We reshape both the numerical and analytical solutions onto the same
# (time × depth) grid and compute the pointwise relative error.

z = znodes(integrator.state.temperature);

T_numeric = reduce(hcat, Ts_buffer)[1, 1, 4:103, :];  # shape: (100, 2881)

T_target = T_sol.(reshape(-z, :, 1), reshape(times, 1, :));  # shape: (100, 2881)

relative_error = abs.((T_numeric .- T_target) ./ T_target);

max_error = maximum(relative_error);

println("Maximum relative error = ", max_error)

@warn max_error < 0.1

# ## Plot comparison at final timestep

# The numerical and analytical profiles should overlay eachother
# within the top few damping depths.

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Temperature", ylabel = "Depth",);

empty!(ax);
ylims!(ax, -5d, 0.0);
lines!(ax, T_numeric[:, end], z, label = "Numerical");
lines!(ax, T_target[:, end], z, linestyle = :dash, label = "Analytical");
axislegend(ax);
fig