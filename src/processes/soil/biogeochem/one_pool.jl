using Terrarium

yr_to_sec = 60.0*60.0*24.0*365.0

Base.@kwdef struct Respiration{NF} <: Terrarium.AbstractProcess
    "Reference decomposition rate [1/yr]"
    k_ref::NF = 0.1/yr_to_sec

    "Temperature sensitivity of decomposition rate (Q10)"
    Q10::NF = 2.0

    "Reference temperature for k_ref [°C]"
    T_ref::NF = 10.0
end

Base.@kwdef struct OnePoolSoilCarbon{NF, Grid <: Terrarium.AbstractLandGrid{NF}, Dyn, Init} <: Terrarium.AbstractModel{NF, Grid}
    "Spatial grid on which state variables are discretized"
    grid::Grid

    "Decomposition dynamics / biogeochemistry"
    dynamics::Dyn = Respiration()

    "Model initializer"
    initializer::Init = Terrarium.DefaultInitializer()
end

Terrarium.variables(::OnePoolSoilCarbon) = (
    Terrarium.prognostic(:C, Terrarium.XY()),
    Terrarium.auxiliary(:Respiration, Terrarium.XY()),
    Terrarium.input(:F, Terrarium.XY()),
    Terrarium.input(:T, Terrarium.XY()),
)

function Terrarium.compute_auxiliary!(state, model::OnePoolSoilCarbon)
    compute_auxiliary!(state, model, model.dynamics)
end

function Terrarium.compute_tendencies!(state, model::OnePoolSoilCarbon)
    compute_tendencies!(state, model, model.dynamics)
end

function Terrarium.compute_auxiliary!(
    state,
    model::OnePoolSoilCarbon,
    dynamics::Respiration
)
# set auxiliary variable for offset c
return state.auxiliary.Respiration .= dynamics.k_ref * dynamics.Q10^((state.inputs.T - dynamics.T_ref)/10.0)*state.prognostic.C
end

function Terrarium.compute_tendencies!(
    state,
    model::OnePoolSoilCarbon,
    dynamics::Respiration
)
# define the dynamics; we'll use some special characters to make the equation nicer to look at :)
return let C = state.prognostic.C,
        dC = state.tendencies.C,
        Respiration = state.auxiliary.Respiration,
        F = state.inputs.F
    # Write into tendency variable dC/dt
    dC .= F - Respiration
end
end

initializer = FieldInitializers(C = 10.0)

grid = ColumnGrid(CPU(), Float64, UniformSpacing(N=1))

dyear = 60.0*60.0*24.0*365.0
t_F = 0:dyear:10*dyear
F = FieldTimeSeries(grid, XY(), t_F)
F.data .= 0.3/yr_to_sec .+ 0.001 .* randn(size(F));

T = FieldTimeSeries(grid, XY(), t_F)
T.data .= 10.0 .+ 2.0 .* randn(size(T));

input = InputSource(; F, T)

model = OnePoolSoilCarbon(grid; initializer)

function compute_tendencies!(
    state,
    model::OnePoolSoilCarbon,
    dynamics::Respiration
)
# define the dynamics; we'll use some special characters to make the equation nicer to look at :)
return let C = state.prognostic.C,
        dC = state.tendencies.C,
        Respiration = state.auxiliary.Respiration,
        F = state.inputs.F
    # Write into tendency variable dC/dt
    dC .= F - Respiration
end
end

integrator = initialize(model, ForwardEuler(Δt=dyear), input)

out = run!(integrator, period=Day(365*10))

out.state.C

sim = Simulation(integrator; stop_time = 10*dyear, Δt = dyear)
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