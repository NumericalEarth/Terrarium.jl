"""
    $TYPEDEF

First-order implicit (backward) Euler timestepper for use as the implicit sub-stepper of an
[`IMEX`](@ref) timestepper, e.g. `IMEX(ForwardEuler(NF), BackwardEuler(NF))`. For each prognostic
variable routed to it (see [`timestepping`](@ref)), the implicit step solves

    uⁿ⁺¹ = u⁺ + Δt G_imp(uⁿ⁺¹),    u⁺ = uⁿ + Δt G_exp(uⁿ),

where `G_exp` and `G_imp` are the explicitly and implicitly treated tendency terms. The system is solved
by `solver` (an [`AbstractImplicitSolver`](@ref)); the linearized operator of all routed variables is
provided by the model (see [`implicit_operator`](@ref)). `BackwardEuler` has no time step size of its
own: the [`IMEX`](@ref) takes `Δt` from its explicit partner (see [`default_dt`](@ref)).

Properties:
$(TYPEDFIELDS)
"""
struct BackwardEuler{NF, S <: AbstractImplicitSolver} <: AbstractTimeStepper{NF}
    "Solver for the implicit system"
    solver::S
end

BackwardEuler{NF}(solver::AbstractImplicitSolver) where {NF} = BackwardEuler{NF, typeof(solver)}(solver)

"""
    $TYPEDSIGNATURES

Construct a [`BackwardEuler`](@ref) timestepper with numeric type `NF` and implicit `solver`.
"""
function BackwardEuler(::Type{NF}; solver::AbstractImplicitSolver = Picard(NF)) where {NF}
    return BackwardEuler{NF}(solver)
end

timestepping(::BackwardEuler) = Implicit()

function default_dt(::BackwardEuler)
    throw(
        ArgumentError(
            "`BackwardEuler` has no default time step: as the implicit sub-stepper of an `IMEX`, it takes `Δt` from the explicit partner, e.g. `IMEX(ForwardEuler(NF; Δt), BackwardEuler(NF))`"
        )
    )
end

"""
    $TYPEDEF

Cache for the [`BackwardEuler`](@ref) timestepper. `previous_state` and `workspaces` are
`NamedTuple`s keyed by the names of the prognostic variables routed to the implicit sub-stepper:
`previous_state` holds `uⁿ` of each variable (stored by [`update_cache!`](@ref)), and `workspaces`
holds the [`ImplicitSolverWorkspace`](@ref) of each variable (right-hand side, explicit stage, and
non-final [`Iteration`](@ref) label). The `target` of the label is the closure variable (as given by
[`varname`](@ref) of the closure) if the variable has a closure, and the variable itself otherwise.
`operator` holds the [`AbstractLinearOperator`](@ref) of all routed variables, as returned by
[`implicit_operator`](@ref)`(model, state, names)` (e.g. a [`SplitOperator`](@ref) of one
[`TridiagonalOperator`](@ref) per variable).

Properties:
$(TYPEDFIELDS)
"""
struct BackwardEulerCache{NF, US, Operator <: AbstractLinearOperator, Workspaces <: NamedTuple} <: AbstractTimeStepperCache{NF}
    "Copy of the state `uⁿ` at the beginning of the step for each implicitly routed prognostic variable, stored by [`update_cache!`](@ref)"
    previous_state::US

    "Linearized implicit operator of all implicitly routed prognostic variables"
    operator::Operator

    "[`ImplicitSolverWorkspace`](@ref) of each implicitly routed prognostic variable"
    workspaces::Workspaces
end

function BackwardEulerCache{NF}(previous_state, operator, workspaces) where {NF}
    return BackwardEulerCache{NF, typeof(previous_state), typeof(operator), typeof(workspaces)}(previous_state, operator, workspaces)
end

"""
    $TYPEDSIGNATURES

Allocate the [`BackwardEulerCache`](@ref) for the prognostic variables in `progvars`, which under an
[`AbstractIMEX`](@ref) are exactly those routed to the implicit sub-stepper. Per-variable fields and
[`ImplicitSolverWorkspace`](@ref)s are allocated with `similar` from the prognostic fields of `state`,
the operator with [`implicit_operator`](@ref)`(model, state, names)`, and the target of each variable
with [`resolve_implicit_target`](@ref).

Throws an `ArgumentError` if `BackwardEuler` is not the implicit sub-stepper of the `model`'s
[`AbstractIMEX`](@ref) timestepper, if any variable in `progvars` has a non-scalar element type (the
implicit solvers are scalar only), if no implicit operator supports the configuration of the variables,
or if a target cannot be resolved.
"""
function initialize(timestepper::BackwardEuler{NF}, state::StateVariables, progvars, model) where {NF}
    model isa AbstractModel || throw(ArgumentError("`BackwardEuler` requires the owning model to allocate its cache"))
    get_timestepper(model) isa AbstractIMEX || throw(
        ArgumentError(
            "`BackwardEuler` can only be used as the implicit sub-stepper of an `IMEX` timestepper, e.g. `IMEX(ForwardEuler(NF), BackwardEuler(NF))`"
        )
    )
    check_scalar_eltypes(progvars)
    names = Tuple(keys(progvars))
    operator = implicit_operator(model, state, names)
    prognostic = getfield(state, :prognostic)
    previous_state = NamedTuple{names}(map(name -> similar(prognostic[name]), names))
    workspaces = NamedTuple{names}(
        map(names) do name
            iteration = Iteration(name; target = resolve_implicit_target(progvars[name], state))
            ImplicitSolverWorkspace(similar(prognostic[name]), similar(prognostic[name]), iteration)
        end
    )
    return BackwardEulerCache{NF}(previous_state, operator, workspaces)
end

"""
    $TYPEDSIGNATURES

Return the name of the field of `state` that the linear system of the implicitly integrated variable
`var` is solved for (its target): [`varname`](@ref) of its closure if it has one, and
[`implicit_variable`](@ref) of `var` otherwise (the variable itself by default). Implicit operators are thus
linearized in the closure variable, e.g. `temperature` for `internal_energy` and `pressure_head` for
`saturation_water_ice`. Called on the host when the
[`BackwardEuler`](@ref) cache is allocated; throws an `ArgumentError` if the closure does not implement
`varname` or `state` has no field of the resolved name. Not exported.
"""
function resolve_implicit_target(var::PrognosticVariable, state)
    name = implicit_variable(var)
    if hasclosure(var)
        applicable(varname, var.closure) || throw(
            ArgumentError(
                "the closure $(nameof(typeof(var.closure))) of variable `$(varname(var))` does not implement `varname`, which is required to determine the target of its implicit operator"
            )
        )
        name = varname(var.closure)
    end
    hasproperty(state, name) || throw(
        ArgumentError("the target `$name` of the implicit operator of variable `$(varname(var))` is not a field of the state")
    )
    return name
end

# Host-side check that all implicitly routed variables have a scalar element type.
function check_scalar_eltypes(progvars)
    for (name, var) in pairs(progvars)
        T = Base.eltype(var)
        isnothing(T) || T <: Number || throw(
            ArgumentError("prognostic variable `$name` has non-scalar element type $T and cannot be integrated implicitly")
        )
    end
    return nothing
end

"""
    $TYPEDSIGNATURES

Store `uⁿ` of each variable in `names` in `previous_state` of the [`BackwardEulerCache`](@ref). Called
by the [`AbstractIMEX`](@ref) driver before the explicit stage overwrites the fields with `u⁺`.
"""
function update_cache!(cache::BackwardEulerCache, integrator::ModelIntegrator, Δt, names::Tuple)
    prognostic = getfield(integrator.state, :prognostic)
    fastiterate(names) do name
        copyto!(cache.previous_state[name], prognostic[name])
    end
    return nothing
end

"""
    $TYPEDSIGNATURES

Apply the implicit Euler step to the prognostic variables in `names`. On entry, each prognostic field
holds `u⁺ = uⁿ + Δt G_exp` and `previous_state` of the [`BackwardEulerCache`](@ref) holds `uⁿ` (see
[`update_cache!`](@ref)). The step stores `Δt G_exp = u⁺ - uⁿ` of every variable in the `explicit_stage`
of its [`ImplicitSolverWorkspace`](@ref), resets every prognostic field to the initial iterate `uⁿ`, and
then solves all variables with the solver-level `implicit_step!` (see [`Picard`](@ref)) using the
`operator` and `workspaces` of the cache. Since all variables are reset before any is solved, processes
see the other implicit variables at `uⁿ` (or at `uⁿ⁺¹` once solved), never at `u⁺`. On exit, the
prognostic fields hold `uⁿ⁺¹` and the closure-derived auxiliary variables are consistent with them.
"""
function implicit_step!(integrator::ModelIntegrator, timestepper::BackwardEuler, Δt, names::Tuple)
    (; state, model) = integrator
    cache = get_cache(state.timestepper_cache, timestepper)
    Δt = convert_dt(eltype(get_grid(model)), Δt)
    prognostic = getfield(state, :prognostic)
    # store Δt G_exp = u⁺ - uⁿ and reset to the initial iterate u⁰ = uⁿ
    fastiterate(names) do name
        explicit_stage = cache.workspaces[name].explicit_stage
        copyto!(explicit_stage, prognostic[name])
        add_scaled!(explicit_stage, cache.previous_state[name], -one(eltype(explicit_stage)))
        copyto!(prognostic[name], cache.previous_state[name])
    end
    implicit_step!(timestepper.solver, cache.operator, cache.workspaces, state, model, Δt)
    return nothing
end
