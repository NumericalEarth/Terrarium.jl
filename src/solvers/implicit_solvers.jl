# Implicit solver interface: solver type, solver-level `implicit_step!`, operator query, and process hooks.

"""
    $TYPEDEF

Base type for solvers of the (nonlinear) system of an implicit time step (e.g. [`Picard`](@ref)),
held by an implicit timestepper such as [`BackwardEuler`](@ref). Solvers are generic over the
[`AbstractLinearOperator`](@ref) and implement [`implicit_step!`](@ref).
"""
abstract type AbstractImplicitSolver end

"""
    Iteration{name, target, final}(number)

Identifies one iteration of an implicit solver for the prognostic variable `name`, whose linear system is
solved for the field `target` (the closure variable, e.g. `temperature` for `internal_energy`; defaults to
`name`). `final` is `true` in the solver's last iteration. All three are type parameters, so hooks dispatch
on them. `number` is the 1-based iteration index, kept for debugging only.

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

function Iteration(iter::Iteration{name, oldtarget, isfinal}; target::Symbol = oldtarget, final::Bool = isfinal, number::Int = iter.number) where {name, oldtarget, isfinal}
    return Iteration{name, target, final}(number)
end

# Name of the prognostic variable solved for in `iteration`.
@inline varname(::Iteration{name}) where {name} = name

# Name of the field that the linear system of `iteration` is solved for.
@inline targetname(::Iteration{name, target}) where {name, target} = target

# Whether `iteration` is the last iteration of the solver.
@inline isfinal(::Iteration{name, target, final}) where {name, target, final} = final

"""
    $TYPEDEF

Per-variable working data of an implicit step: the right-hand side `rhs` of the linear system, the
`explicit_stage` (the explicit increment `Δt G_exp` on entry to [`implicit_step!`](@ref), from which the
solver forms `u⁺ = u⁰ + Δt G_exp` at the initial iterate `u⁰`), and the non-final [`Iteration`](@ref)
label `iteration` of the variable. Allocated by the implicit timestepper (see
[`BackwardEulerCache`](@ref)). Not exported.

Properties:
$(TYPEDFIELDS)
"""
struct ImplicitSolverWorkspace{I <: Iteration, R, E}
    "Right-hand side of the linear system"
    rhs::R

    "Explicit increment `Δt G_exp` on entry to the step, and the explicit stage `u⁺` once the solver has formed it"
    explicit_stage::E

    "Non-final iteration label of the variable, naming the variable and the target field of its linear system"
    iteration::I
end

function Adapt.adapt_structure(to, workspace::ImplicitSolverWorkspace)
    return ImplicitSolverWorkspace(
        Adapt.adapt_structure(to, workspace.rhs),
        Adapt.adapt_structure(to, workspace.explicit_stage),
        workspace.iteration,
    )
end

"""
    implicit_variable(var::AbstractVariable)

Name of the field that the linear system of the implicitly integrated variable `var` is solved for if `var`
has no closure: the variable itself by default. Processes may linearize in a derived field instead, e.g. the
snow meltwater drainage of `snow_water_equivalent` in `snow_liquid_fraction`. Not exported.
"""
implicit_variable(var::AbstractVariable) = varname(var)

"""
    implicit_operator(model::AbstractModel, state, names::Tuple)

Return a newly allocated [`AbstractLinearOperator`](@ref) for the implicitly routed prognostic
variables `names` of `model` (e.g. a [`SplitOperator`](@ref) of per-variable operators in solve
order). Called once on the host at `initialize`; methods throw an `ArgumentError` for unsupported
configurations, as does the fallback. [`SoilModel`](@ref) and [`LandModel`](@ref) forward to
`implicit_operator(soil::AbstractSoil, state, grid, names)`.
"""
function implicit_operator(model::AbstractModel, state, names::Tuple{Vararg{Symbol}})
    throw(
        ArgumentError(
            "no implicit operator is defined for the variables $names of $(nameof(typeof(model))); define `implicit_operator(model, state, names)` and the implicit process hooks (`linearize!`, `implicit_update!`, `implicit_closure!`) for them, or route the variables explicitly"
        )
    )
end

# Process hooks called by the solver per variable, dispatched on `Iteration`; models forward them to the owning process.

"""
    linearize!(operator::AbstractLinearOperator, rhs, state, model::AbstractModel, explicit_stage, Δt, ::Iteration{name})

Process hook of the implicit solver. Fill the lagged coefficients of `operator` and the right-hand
side `rhs` for variable `name` at the current iterate in `state`, given the explicit stage
`explicit_stage` (`u⁺ = uⁿ + Δt G_exp`). The fallback throws an `ArgumentError`.
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

Process hook of the implicit solver. Update the prognostic variable `name` in `state` conservatively
from the solved target and `explicit_stage`, using the fluxes implied by `operator`.
"""
function implicit_update! end

"""
    implicit_closure!(state, model::AbstractModel, ::Iteration{name, target, final})

Process hook of the implicit solver. Re-evaluate the closure of variable `name` (and the auxiliary
variables the next linearization needs) after [`implicit_update!`](@ref). Once-per-step side effects
(e.g. routing excess water to runoff) must be restricted to `final = true`.
"""
function implicit_closure! end

"""
    implicit_step!(solver::AbstractImplicitSolver, operator::AbstractLinearOperator, workspaces::NamedTuple, state, model::AbstractModel, Δt)
    implicit_step!(solver::AbstractImplicitSolver, operator::AbstractLinearOperator, workspace::ImplicitSolverWorkspace, state, model::AbstractModel, Δt)

Solve the implicit system of `model` with `solver` and `operator` (from
[`implicit_operator`](@ref)), using the hooks [`linearize!`](@ref), [`solve!`](@ref),
[`implicit_update!`](@ref), and [`implicit_closure!`](@ref).

The first form solves all routed variables (`workspaces` is a `NamedTuple` of
[`ImplicitSolverWorkspace`](@ref)s keyed by variable name); the second solves the single variable of
`workspace`, e.g. one component of a [`SplitOperator`](@ref). On entry, the prognostic fields hold
the initial iterate and each `explicit_stage` holds `Δt G_exp`; on exit, the prognostic fields hold
the solution and their closure variables are consistent with it. The fallbacks throw an `ArgumentError`.
"""
function implicit_step!(
        solver::AbstractImplicitSolver,
        operator::AbstractLinearOperator,
        workspaces::NamedTuple,
        state,
        model::AbstractModel,
        Δt
    )
    throw(
        ArgumentError(
            "solver $(nameof(typeof(solver))) does not support the implicit operator $(nameof(typeof(operator))) of variables $(keys(workspaces))"
        )
    )
end

function implicit_step!(
        solver::AbstractImplicitSolver,
        operator::AbstractLinearOperator,
        workspace::ImplicitSolverWorkspace,
        state,
        model::AbstractModel,
        Δt
    )
    throw(
        ArgumentError(
            "solver $(nameof(typeof(solver))) does not support the implicit operator $(nameof(typeof(operator))) of variable `$(varname(workspace.iteration))`"
        )
    )
end
