# Implicit solver interface: the solver type, the solver-level `implicit_step!`, the implicit operator query, and
# the process hooks that solvers call.

"""
    $TYPEDEF

Base type for solvers of the (nonlinear) system of an implicit time step, held by an implicit
timestepper such as [`BackwardEuler`](@ref) (e.g. [`Picard`](@ref)). Solvers are generic over the
[`AbstractLinearOperator`](@ref) and implement the solver-level [`implicit_step!`](@ref).
"""
abstract type AbstractImplicitSolver end

"""
    Iteration{name, target, final}(number)

Identifies one iteration of an implicit solver for the prognostic variable `name`, whose linear system is
solved for the field `target` (the closure variable, e.g. `temperature` for `internal_energy`; `target`
defaults to `name` for variables without a closure); `final` is `true` in the solver's last iteration.
All three are type parameters, so hooks dispatch on them at compile time. The field `number` is the
1-based index of the iteration within the solver; it is stored for debugging purposes only.

```jldoctest
julia> iteration = Terrarium.Iteration(:internal_energy; target = :temperature)
Terrarium.Iteration{:internal_energy, :temperature, false}(1)
```
"""
struct Iteration{name, target, final}
    "Current iteration index"
    number::Int
end

# Host side; the keywords are not compile-time constants.
function Iteration(name::Symbol; target::Symbol = name, final::Bool = false, number::Int = 1)
    return Iteration{name, target, final}(number)
end

function Iteration(iter::Iteration{oldname, oldtarget, isfinal}; target::Symbol = oldname, final::Bool = isfinal, number::Int = iter.number) where {oldname, oldtarget, isfinal}
    return Iteration{name, target, final}(number)
end

# Name of the prognostic variable solved for in `iteration`.
@inline varname(::Iteration{name}) where {name} = name

# Name of the field that the linear system of `iteration` is solved for.
@inline targetname(::Iteration{name, target}) where {name, target} = target

# Whether `iteration` is the last iteration of the solver.
@inline isfinal(::Iteration{name, target, final}) where {name, target, final} = final

"""
    implicit_operator(model::AbstractModel, state, names::Tuple)

Return a newly allocated [`AbstractLinearOperator`](@ref) for the implicitly routed prognostic
variables `names` of `model` (e.g. a [`SplitOperator`](@ref) of per-variable operators in solve
order). Called once on the host at `initialize`; methods validate the configuration and throw an
`ArgumentError` for unsupported ones. The fallback throws an `ArgumentError`. [`SoilModel`](@ref)
and [`LandModel`](@ref) forward to `implicit_operator(soil::AbstractSoil, state, grid, names)`.
"""
function implicit_operator(model::AbstractModel, state, names::Tuple{Vararg{Symbol}})
    throw(
        ArgumentError(
            "no implicit operator is defined for the variables $names of $(nameof(typeof(model))); define `implicit_operator(model, state, names)` and the implicit process hooks (`linearize!`, `implicit_update!`, `implicit_closure!`) for them, or route the variables explicitly"
        )
    )
end

# Process hooks, called by the solver for one variable and dispatched on its `Iteration`; models forward them
# to the process that owns the variable.

"""
    linearize!(operator::AbstractLinearOperator, rhs, state, model::AbstractModel, explicit_stage, Δt, ::Iteration{name})

Process hook of the implicit solver. Fill the lagged coefficients of `operator` and the right-hand
side `rhs` of the linear system for variable `name` at the current iterate in `state`, given its
explicit stage `explicit_stage` (`u* = uⁿ + Δt G_exp`). The fallback throws an `ArgumentError`.
"""
function linearize!(operator::AbstractLinearOperator, rhs, state, model::AbstractModel, explicit_stage, Δt, ::Iteration{name}) where {name}
    throw(
        ArgumentError(
            "no implicit linearization (`linearize!`) is defined for variable `$name` of $(nameof(typeof(model))) with operator $(nameof(typeof(operator)))"
        )
    )
end

"""
    implicit_update!(state, model::AbstractModel, operator::AbstractLinearOperator, explicit_stage, Δt, ::Iteration{name})

Process hook of the implicit solver. Update the prognostic variable `name` in `state` from the
solved target and `explicit_stage`, conservatively, i.e. from the fluxes implied by `operator`.
"""
function implicit_update! end

"""
    implicit_closure!(state, model::AbstractModel, ::Iteration{name, target, final})

Process hook of the implicit solver. Re-evaluate the closure of variable `name` (and the auxiliary
variables the next linearization needs) after [`implicit_update!`](@ref). Side effects that must
happen once per step (e.g. routing excess water to runoff) should be restricted to `final = true`.
"""
function implicit_closure! end

"""
    implicit_step!(solver::AbstractImplicitSolver, operator::AbstractLinearOperator, state, model::AbstractModel, rhs::NamedTuple, explicit_state::NamedTuple, Δt, iterations::NamedTuple)
    implicit_step!(solver::AbstractImplicitSolver, operator::AbstractLinearOperator, state, model::AbstractModel, rhs, explicit_stage, Δt, iteration::Iteration)

Solve the implicit system of `model` with `solver` and `operator` (from
[`implicit_operator`](@ref)), using the hooks [`linearize!`](@ref), [`solve!`](@ref),
[`implicit_update!`](@ref), and [`implicit_closure!`](@ref).

The first form solves all routed variables; `rhs`, `explicit_state`, and `iterations` are keyed by
variable name, with `iterations` holding the non-final [`Iteration`](@ref) of each variable. The
second form solves a single variable, e.g. one component of a [`SplitOperator`](@ref). On entry, the
prognostic fields hold the initial iterate; on exit, they hold the solution and their closure
variables are consistent with it. The fallbacks throw an `ArgumentError`.
"""
function implicit_step!(
        solver::AbstractImplicitSolver,
        operator::AbstractLinearOperator,
        state,
        model::AbstractModel,
        rhs::NamedTuple,
        explicit_state::NamedTuple,
        Δt,
        iterations::NamedTuple
    )
    throw(
        ArgumentError(
            "solver $(nameof(typeof(solver))) does not support the implicit operator $(nameof(typeof(operator))) of variables $(keys(iterations))"
        )
    )
end

function implicit_step!(
        solver::AbstractImplicitSolver,
        operator::AbstractLinearOperator,
        state,
        model::AbstractModel,
        rhs,
        explicit_stage,
        Δt,
        ::Iteration{name}
    ) where {name}
    throw(
        ArgumentError(
            "solver $(nameof(typeof(solver))) does not support the implicit operator $(nameof(typeof(operator))) of variable `$name`"
        )
    )
end
