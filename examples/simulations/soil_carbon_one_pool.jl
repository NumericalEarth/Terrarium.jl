using Terrarium

grid = ColumnGrid(CPU(), Float64, UniformSpacing(N = 1))
biogeochem = OnePoolSoilCarbon(eltype(grid))
soil = SoilEnergyWaterCarbon(eltype(grid); biogeochem) # coupled soil processes
model = SoilModel(grid; soil) # soil model
display(variables(model))
initializers = (density_soc = 10.0, temperature = 5.0, saturation_water_ice = 1.0)
integrator = initialize(model, ForwardEuler(eltype(grid)); initializers)
timestep!(integrator)

dyear = 60.0 * 60.0 * 24.0 * 365.0
t_F = 0:dyear:(10 * dyear)
F = FieldTimeSeries(grid, XY(), t_F)
F.data .= 0.3 / yr_to_sec .+ 0.001 .* randn(size(F));

T = FieldTimeSeries(grid, XY(), t_F)
T.data .= 10.0 .+ 2.0 .* randn(size(T));

input = InputSource(; F, T)

model = OnePoolSoilCarbon(grid; initializer)

initializers = (C = 10.0,)
integrator = initialize(model, ForwardEuler(Δt = dyear), input; initializers)

out = run!(integrator, period = Day(365 * 10))

out.state.C

sim = Simulation(integrator; stop_time = 10 * dyear, Δt = dyear)
run!(sim)

using Oceananigans: TimeInterval, JLD2Writer
using Oceananigans.Units: seconds

# Reset the integrator to its initial state
Terrarium.initialize!(integrator)

output_file = tempname()
sim.output_writers[:snapshots] = JLD2Writer(
    integrator,
    (C = integrator.state.C,);
    filename = output_file,
    overwrite_existing = true,
    schedule = TimeInterval(10seconds)
)

run!(sim)


fts = FieldTimeSeries(output_file, "C")

C = fts[end]

using Plots
plot(1:length(fts), [fts[i][1, 1, 1] for i in 1:length(fts)])
