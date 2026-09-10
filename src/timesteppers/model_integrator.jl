"""
    $TYPEDEF

Represents a "integrator" for a simulation of a given `model`. `ModelIntegrator` consists of a
`clock`, a `model`, and an initialized `StateVariables` data structure, as well as any relevant
`inputs` provided by a corresponding `InputProvider`. 
The `ModelIntegrator` implements the `Oceananigans.AbstractModel` interface and can thus be
treated as a "model" in `Oceananigans` `Simulation`s and output reading/writing utilities.
"""
struct ModelIntegrator{
        NF,
        Arch <: AbstractArchitecture,
        Grid <: AbstractLandGrid{NF},
        TimeStepper <: AbstractTimeStepper{NF},
        Model <: AbstractModel{NF, Grid},
        StateVars <: AbstractStateVariables,
        ClockType <: Clock,
        Inits <: NamedTuple,
        Inputs <: InputSources,
    } <: Oceananigans.AbstractModel{TimeStepper, Arch}
    "The clock holding all information about the current timestep/iteration of a simulation"
    clock::ClockType

    "Underlying model evaluated by this integrator"
    model::Model

    "Input sources"
    inputs::Inputs

    "Collection of all state variables defined on the simulation `model`"
    state::StateVars

    "Optional named tuple of user-specified field initializers"
    initializers::Inits
end

# Outer constructor so that `ModelIntegrator` can be constructed with a timestepper type
# so that it can correctly subtype Oceananigans.AbstractModel
function ModelIntegrator(
        clock::Clock,
        model::AbstractModel{NF},
        inputs::InputSources,
        state::AbstractStateVariables,
        initializers::NamedTuple,
    ) where {NF}
    grid = get_grid(model)
    timestepper = get_timestepper(model)
    return ModelIntegrator{
        NF, typeof(architecture(grid)), typeof(grid), typeof(timestepper),
        typeof(model), typeof(state), typeof(clock), typeof(initializers), typeof(inputs),
    }(clock, model, inputs, state, initializers)
end

# Oceananigans model interface

Base.time(integrator::ModelIntegrator) = integrator.clock.time

Base.eltype(::ModelIntegrator{NF}) where {NF} = NF

function Base.getproperty(integrator::ModelIntegrator, name::Symbol)
    # Temporary hack to make Oceananigans output writers play nicely with ModelIntegrator
    # TODO: Raise an issue in Oceananigans for better long-term solution
    if name == :grid
        model = getfield(integrator, :model)
        return get_field_grid(get_grid(model))
    else
        return getfield(integrator, name)
    end
end

Oceananigans.initialize!(integrator::ModelIntegrator) = initialize!(integrator)

Oceananigans.Solvers.iteration(integrator::ModelIntegrator) = integrator.clock.iteration

Oceananigans.Architectures.architecture(integrator::ModelIntegrator) = architecture(get_grid(integrator.model))

Oceananigans.TimeSteppers.update_state!(integrator::ModelIntegrator; compute_tendencies = true) = update_state!(integrator.state, integrator.model, integrator.inputs; compute_tendencies)

# for now, just forward Oceananigans.time_step! to timestep!
# consider renaming later...
Oceananigans.TimeSteppers.time_step!(integrator::ModelIntegrator, Δt; kwargs...) = timestep!(integrator, Δt)

Oceananigans.Simulations.timestepper(integrator::ModelIntegrator) = get_timestepper(integrator.model)

function Oceananigans.Simulations.conjure_time_step_wizard!(
        simulation::Simulation{<:ModelIntegrator},
        schedule = IterationInterval(1);
        diffusive_cfl = 0.5,
        min_change = 0.2,
        show_progress = true,
        kwargs...
    )
    wizard = TimeStepWizard(; diffusive_cfl, min_change, kwargs...)
    simulation.callbacks[:time_step_wizard] = Callback(wizard, schedule)
    if show_progress
        simulation.callbacks[:progress] = ProgressCallback(simulation)
    end
    return nothing
end

"""
    $SIGNATURES

Run the simulation for `steps` or a given time `period` with timestep size `Δt` (in seconds or Dates.Period).
"""
function Oceananigans.Simulations.run!(
        integrator::ModelIntegrator{NF};
        steps::Union{Int, Nothing} = nothing,
        period::Union{Period, Nothing} = nothing,
        Δt = default_dt(timestepper(integrator)),
        checkpointing = false,
        show_progress = false
    ) where {NF}
    Δt = convert_dt(NF, Δt)
    if !isnothing(period) && convert_dt(NF, period) < Δt
        @warn "Reducing Δt from $Δt to match length of given period: $period"
        Δt = min(Δt, convert_dt(NF, period))
    end
    steps = get_steps(steps, period, Δt)
    # Run for the specified number of time steps
    run_timesteps!(integrator, Δt, steps, checkpointing; show_progress)
    # Update auxiliary variables for final timestep
    compute_auxiliary!(integrator.state, integrator.model)
    return integrator
end

"""
    run_timesteps!(integrator, Δt, steps, checkpointing = false)

Advance `integrator` for the given number of `steps` of size `Δt`.

The generic (host) implementation is a plain loop and ignores `checkpointing`. `ReactantState`
integrators override this method in `TerrariumReactantExt`, compiling the loop into a single
traced program in which `checkpointing` selects the reverse-mode-AD checkpointing scheme
(`false`, or a scheme such as `Reactant.Periodic(n)`).
"""
function run_timesteps!(integrator::ModelIntegrator, Δt, steps, checkpointing = false; show_progress = false)
    if show_progress
        @showprogress "Integrating $(nameof(typeof(integrator.model))) for $steps steps" for _ in 1:steps
            timestep!(integrator, Δt, finalize = false)
        end
    else
        for _ in 1:steps
            timestep!(integrator, Δt, finalize = false)
        end
    end
    compute_auxiliary!(integrator.state, integrator.model)
    return nothing
end

# Terrarium method interfaces

"""
    $TYPEDEF

Resets the simulation `clock` and calls `initialize!(state, model)` on the underlying model which
should reset all state variables to their values as defiend by the model initializer.
"""
function initialize!(integrator::ModelIntegrator)
    # reset state variables and clock
    reset!(integrator.state)
    reset!(integrator.clock)
    # set inputs based on updated clock/state
    initialize!(integrator.state, get_grid(integrator.model), integrator.inputs)
    # evaluate user-specified field initializers
    initialize!(integrator.state, integrator.initializers)
    # evaluate model initializer
    initialize!(integrator.state, integrator.model)
    return integrator
end

current_time(integrator::ModelIntegrator) = integrator.clock.time

get_fields(integrator::ModelIntegrator, queries...) = get_fields(integrator.state, queries...)

"""
    $TYPEDSIGNATURES

Advance the model forward by one timestep with optional timestep size `Δt`. If `finalize = true`,
`compute_auxiliary!` is called after the time step in order to update the values of auxiliary/diagnostic
variables.
"""
timestep!(integrator::ModelIntegrator; finalize = true) = timestep!(integrator, default_dt(get_timestepper(integrator.model)); finalize)
function timestep!(integrator::ModelIntegrator{NF}, Δt; finalize = true) where {NF}
    timestep!(integrator, get_timestepper(integrator.model), convert_dt(NF, Δt))
    if finalize
        compute_auxiliary!(integrator.state, integrator.model)
    end
    return nothing
end

# Define Terrarium timestep! for Simulation
timestep!(sim::Simulation) = Oceananigans.TimeSteppers.time_step!(sim)

"""
    timestep!(integrator::ModelIntegrator, timestepper::AbstractTimeStepper, Δt)

Advance the model forward by one timestep of size `Δt` using a single `timestepper`, which integrates
*all* prognostic variables. Dispatches on the timestepper's [`timestepping`](@ref) trait
([`Explicit`](@ref)/[`Implicit`](@ref)) to `timestep!(integrator, timestepper, ::Timestepping, Δt)`.
"""
timestep!(integrator::ModelIntegrator, timestepper::AbstractTimeStepper, Δt) =
    timestep!(integrator, timestepper, timestepping(timestepper), Δt)

"""
    timestep!(integrator::ModelIntegrator, timestepper::AbstractTimeStepper, ::Timestepping, Δt)

Trait-dispatched single-timestepper step: forward the prognostic variable names to the scheme's
`timestep!(integrator, timestepper, Δt, names)` method and advance the clock once for the whole step. 
"""
function timestep!(integrator::ModelIntegrator, timestepper::AbstractTimeStepper, ::Timestepping, Δt)
    # a single time stepper integrates all prognostic variables
    names = prognostic_names(integrator.state)
    timestep!(integrator, timestepper, Δt, names)
    # advance the clock once for the entire step
    tick!(integrator.state.clock, Δt)
    return nothing
end

"""
    default_dt(integrator::ModelIntegrator)

Return the default timestep size for the given `integrator`, taken from its model's `timestepper`.
"""
default_dt(integrator::ModelIntegrator) = default_dt(get_timestepper(integrator.model))

"""
    $TYPEDSIGNATURES

Return the default `Clock` used by [`initialize`](@ref) for the given `model`. The generic method
returns a plain host clock starting at time zero; architecture extensions may specialize on the
model's grid to return an architecture-specific clock (e.g. `TerrariumReactantExt` returns a
traced `ConcreteRNumber`-backed clock so that time advances inside the compiled step).
"""
default_clock(model::AbstractModel{NF}) where {NF} = Clock(time = zero(NF))

"""
    $TYPEDSIGNATURES

Creates and initializes a `ModelIntegrator` for the given `model` with input variables populated by
the given `inputs` and optionally `params` . `InputSource`s can be specified via the `inputs` keyword argument.
This method allocates all necessary `Field`s for the state variables and subsequently calls
`initialize!(::ModelIntegrator)`.

Note that this method is **not type stable** and thus should not be called from Enzyme `autodiff`. To reinitialize
the model for an existing `state`, use `initialize!(state, model)`.

See the docstring for [`initialize(::AbstractModel)`](@ref) for further details.
"""
function initialize(
        model::AbstractModel{NF},
        params = nothing;
        clock::Clock = default_clock(model),
        inputs::InputSource = InputSources(NF),
        boundary_conditions = (;),
        initializers = (;),
        fields = (;)
    ) where {NF}
    inputs = InputSources(inputs)
    input_vars = variables(inputs)
    updated_model = isnothing(params) ? model : ParameterEditing.reconstruct(model, params)
    state = StateVariables(updated_model; clock, boundary_conditions, fields, input_variables = input_vars)
    integrator = ModelIntegrator(clock, updated_model, inputs, state, initializers)
    initialize!(integrator)
    return integrator
end

"""
    $SIGNATURES

Reconstruct the given `integrator` using the same underlying model populated with the given `params`.
The `clock` and `inputs` can also optionally be updated via their respective keyword arguments.
"""
function initialize(
        integrator::ModelIntegrator,
        params;
        clock::Clock = integrator.clock,
        inputs::InputSource = integrator.inputs
    )
    inputs = InputSources(inputs)
    model = ParameterEditing.reconstruct(integrator.model, params)
    integrator = ModelIntegrator(clock, model, inputs, integrator.state, integrator.initializers)
    initialize!(integrator)
    return integrator
end

get_steps(steps::Nothing, period::Period, Δt::Real) = div(convert_dt(Second, period).value, Δt)
get_steps(steps::Int, period::Nothing, Δt::Real) = steps
get_steps(steps::Nothing, period::Nothing, Δt::Real) = throw(ArgumentError("either `steps` or `period` must be specified"))
get_steps(steps::Int, period::Period, Δt::Real) = throw(ArgumentError("both `steps` and `period` cannot be specified"))

function Base.show(io::IO, integrator::ModelIntegrator)
    modelstr = summary(integrator.model)
    statestr = summary(integrator.state)
    tsstr = get_timestepper(integrator.model)
    println(io, "Integrator of $modelstr with timestepper $tsstr")
    println(io, "├── Current time: $(current_time(integrator))")
    println(io, "├── $statestr")
    # TODO: add more information?
    return nothing
end
