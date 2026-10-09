"""
    $TYPEDEF

First-order implicit (backward) Euler timestepper for use as the implicit sub-stepper of an
[`IMEX`](@ref) timestepper, e.g. `IMEX(ForwardEuler(NF), BackwardEuler(NF))`. For each prognostic
variable routed to it (see [`timestepping`](@ref)), the implicit step solves

    uⁿ⁺¹ = u* + Δt G_imp(uⁿ⁺¹),    u* = uⁿ + Δt G_exp(uⁿ),

where `G_exp` and `G_imp` are the explicitly and implicitly treated tendency terms. The method used
to solve this system is selected by `solver` (an [`AbstractImplicitSolver`](@ref)), and the
linearized operator of all routed variables is provided by the model (see
[`implicit_operator`](@ref)), which assembles it from the operators of the processes that own the
variables. `BackwardEuler` has no time step size of its own: the [`IMEX`](@ref) takes `Δt` from its explicit partner (see
[`default_dt`](@ref)).

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

Construct an [`BackwardEuler`](@ref) timestepper with numeric type `NF` and implicit `solver`.
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

Cache for the [`BackwardEuler`](@ref) timestepper. The fields `rhs`, `previous_state`,
`explicit_state`, and `iterations` are `NamedTuple`s keyed by the names of the prognostic variables
routed to the implicit sub-stepper. Besides the fields that any backward Euler step needs,
`operator` holds the single [`AbstractLinearOperator`](@ref) of all routed variables, as returned
by the block query [`implicit_operator`](@ref)`(model, state, names)` (e.g. a
[`SplitOperator`](@ref) of one [`TridiagonalOperator`](@ref) per variable), and `iterations` holds, for
each variable, its non-final [`Iteration`](@ref), whose `target` is the field that its linear system is
solved for: the variable of its closure relation (as given by [`varname`](@ref) of the closure) if it
has one, and the prognostic variable itself otherwise. Storing the names as type parameters lets the
solver retrieve the target field type-stably.

The step `implicit_step!(integrator, timestepper::BackwardEuler, Δt, names)` obtains this cache with
`cache = get_cache(integrator.state.timestepper_cache, timestepper)` and passes `operator`, `rhs`,
`explicit_state`, and `iterations` to the solver-level `implicit_step!` of the
[`AbstractImplicitSolver`](@ref).

Properties:
$(TYPEDFIELDS)
"""
struct BackwardEulerCache{NF, RHS, US, Operator <: AbstractLinearOperator, Iterations <: NamedTuple} <: AbstractTimeStepperCache{NF}
    "Right-hand side field of the implicit system for each implicitly routed prognostic variable"
    rhs::RHS

    "Copy of the state `uⁿ` at the beginning of the step for each implicitly routed prognostic variable, stored by [`update_cache!`](@ref)"
    previous_state::US

    "Copy of the explicit stage `u* = uⁿ + Δt G_exp` for each implicitly routed prognostic variable"
    explicit_state::US

    "Linearized implicit operator of all implicitly routed prognostic variables"
    operator::Operator

    "Non-final [`Iteration`](@ref) of each implicitly routed prognostic variable, which names the field its linear system is solved for"
    iterations::Iterations
end

function BackwardEulerCache{NF}(rhs, previous_state, explicit_state, operator, iterations) where {NF}
    return BackwardEulerCache{NF, typeof(rhs), typeof(explicit_state), typeof(operator), typeof(iterations)}(
        rhs, previous_state, explicit_state, operator, iterations
    )
end

function Adapt.adapt_structure(to, cache::BackwardEulerCache{NF}) where {NF}
    return BackwardEulerCache{NF}(
        Adapt.adapt_structure(to, cache.rhs),
        Adapt.adapt_structure(to, cache.previous_state),
        Adapt.adapt_structure(to, cache.explicit_state),
        Adapt.adapt_structure(to, cache.operator),
        cache.iterations,
    )
end

"""
    $TYPEDSIGNATURES

Allocate the [`BackwardEulerCache`](@ref) for the prognostic variables in `progvars`, which under an
[`AbstractIMEX`](@ref) are exactly the variables routed to the implicit sub-stepper. The per-variable
fields are allocated with `similar` from the corresponding prognostic fields of `state`, the
operator of all variables with the block query [`implicit_operator`](@ref)`(model, state, names)`,
which also validates the configuration of the variables, and the name of the target of each
variable is resolved with [`resolve_implicit_target`](@ref).

Throws an `ArgumentError` if `BackwardEuler` is not the implicit sub-stepper of the `model`'s
[`AbstractIMEX`](@ref) timestepper, if any variable in `progvars` has a non-scalar element type
(the implicit solvers are scalar only), if no implicit operator is defined for (or supports the
configuration of) any variable in `progvars`, or if the target of a variable cannot be resolved.
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
    iterations = NamedTuple{names}(map(name -> Iteration(name; target = resolve_implicit_target(progvars[name], state)), names))
    prognostic = getfield(state, :prognostic)
    rhs = NamedTuple{names}(map(name -> similar(prognostic[name]), names))
    previous_state = NamedTuple{names}(map(name -> similar(prognostic[name]), names))
    explicit_state = NamedTuple{names}(map(name -> similar(prognostic[name]), names))
    return BackwardEulerCache{NF}(rhs, previous_state, explicit_state, operator, iterations)
end

"""
    $TYPEDSIGNATURES

Return the name of the field of `state` that the linear system of the implicitly integrated
prognostic variable `var` is solved for (its target): the name of the variable returned by its
closure relation, as given by [`varname`](@ref) of the closure, if `var` has a closure, and the name
of `var` itself otherwise. This fixes the convention that implicit operators are linearized in the
closure variable, e.g. the `temperature` for `internal_energy` and the `pressure_head` for
`saturation_water_ice`. Called on the host by [`BackwardEuler`](@ref) when its cache is allocated;
throws an `ArgumentError` if the closure does not implement `varname` or `state` has no field of the
resolved name. Not exported.
"""
function resolve_implicit_target(var::PrognosticVariable, state)
    name = varname(var)
    if hasclosure(var)
        applicable(varname, var.closure) || throw(
            ArgumentError(
                "the closure $(nameof(typeof(var.closure))) of variable `$name` does not implement `varname`, which is required to determine the target of its implicit operator"
            )
        )
        name = varname(var.closure)
    end
    hasproperty(state, name) || throw(
        ArgumentError("the target `$name` of the implicit operator of variable `$(varname(var))` is not a field of the state")
    )
    return name
end

# Host-side check that every variable routed to an implicit solver has a scalar element type.
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

Store `uⁿ` of each prognostic variable in `names` in the `previous_state` field of the
[`BackwardEulerCache`](@ref). Called by the [`AbstractIMEX`](@ref) driver before the explicit stage
overwrites the prognostic fields with `u*`.
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

Apply the implicit Euler step to the prognostic variables in `names`. On entry, each prognostic
field holds `u* = uⁿ + Δt G_exp`, and the `previous_state` field of the [`BackwardEulerCache`](@ref)
holds `uⁿ` (see [`update_cache!`](@ref)). The step first stores `u*` of every variable in the
`explicit_state` of the cache and sets every prognostic field to the initial iterate `uⁿ`, and then
solves the implicit system of all variables at once with the solver-level
`implicit_step!(timestepper.solver, operator, state, model, rhs, explicit_state, Δt, iterations)`,
with the `operator`, `rhs`, `explicit_state`, and `iterations` of the cache (see [`Picard`](@ref)).
Since all variables are reset before any is solved, the processes see the other implicit variables
at `uⁿ` (or, after they have been solved, at `uⁿ⁺¹`), never at the explicit stage `u*`. On exit, the
prognostic fields hold `uⁿ⁺¹`, and the auxiliary variables derived from them by the closures are
consistent with them.
"""
function implicit_step!(integrator::ModelIntegrator, timestepper::BackwardEuler, Δt, names::Tuple)
    (; state, model) = integrator
    cache = get_cache(state.timestepper_cache, timestepper)
    Δt = convert_dt(eltype(get_grid(model)), Δt)
    prognostic = getfield(state, :prognostic)
    # store u* and set the initial iterate u⁰ = uⁿ of every implicit variable
    fastiterate(names) do name
        copyto!(cache.explicit_state[name], prognostic[name])
        copyto!(prognostic[name], cache.previous_state[name])
    end
    implicit_step!(timestepper.solver, cache.operator, state, model, cache.rhs, cache.explicit_state, Δt, cache.iterations)
    return nothing
end
