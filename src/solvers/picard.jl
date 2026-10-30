# Picard (lagged-coefficient) solver of implicit time steps: `implicit_step!` for a `SplitOperator` (operator
# splitting) and for a single variable (fixed number of Picard iterations over the process hooks).

"""
    $TYPEDEF

Picard (lagged-coefficient) solver for the implicit system of a timestepper such as
[`BackwardEuler`](@ref). Each iteration linearizes the implicit operator of a prognostic variable at the
latest iterate ([`linearize!`](@ref)), solves the linear system for the closure variable (e.g.
temperature) with [`solve!`](@ref), which dispatches on the [`AbstractLinearOperator`](@ref) (e.g. a
batched tridiagonal solve for a [`TridiagonalOperator`](@ref)), updates the conserved variable
conservatively ([`implicit_update!`](@ref)), and re-evaluates the closure
([`implicit_closure!`](@ref)). The solver is generic over the operator; the process-specific parts are
the hooks, which the model forwards to the process that owns the variable.

A single iteration (`iterations = 1`) is the linearly implicit scheme used by most land surface models.
The default of two iterations substantially reduces the lagged-coefficient error near freezing fronts and
is the minimum recommended for the modified Picard iteration of the Richards equation
[CeliaEtAl1990](@cite).

Properties:
$(TYPEDFIELDS)

# References

* [CeliaEtAl1990](@cite) Celia et al., Water Resources Research 26, 1483–1496 (1990)
"""
struct Picard <: AbstractImplicitSolver
    "Number of Picard iterations (linearizations and linear solves) per implicit step"
    iterations::Int
end

"""
    $TYPEDSIGNATURES

Construct a [`Picard`](@ref) solver. `NF` is accepted for consistency with other timestepping
constructors; the solver holds no floating-point parameters. `iterations` must be at least one.

```jldoctest
julia> Picard(Float32).iterations
2

julia> Picard(Float32; iterations = 3).iterations
3
```
"""
function Picard(::Type{NF}; iterations::Int = 2) where {NF}
    iterations >= 1 || throw(ArgumentError("`iterations` must be at least one, got $iterations"))
    return Picard(iterations)
end

"""
    $TYPEDSIGNATURES

Solve the implicit system of a [`SplitOperator`](@ref) with the [`Picard`](@ref) `solver` by operator
splitting: the single-variable method below is applied to each component of `operator` in order, with
the [`ImplicitSolverWorkspace`](@ref) of that variable from `workspaces`. Each variable is solved at the
latest state of the variables solved before it and the initial iterate of those after it, and forms its
explicit stage from its current field, as in a sequential split.
"""
function implicit_step!(solver::Picard, operator::SplitOperator, workspaces::NamedTuple, state, model::AbstractModel, Δt)
    fastiterate(split_variable_names(operator)) do name
        implicit_step!(solver, operator.operators[name], workspaces[name], state, model, Δt)
    end
    return nothing
end

"""
    $TYPEDSIGNATURES

Solve the implicit system of the variable of `workspace` with the [`Picard`](@ref) `solver` and the
variable's linear `operator`. On entry, the prognostic field in `state` holds the initial iterate `u⁰`
(`uⁿ` for [`BackwardEuler`](@ref), plus any change `δ` made by the closures of variables solved before
it) and `workspace.explicit_stage` holds `Δt G_exp`. The step forms `u⁺ = u⁰ + Δt G_exp` in
`workspace.explicit_stage` (keeping `δ`), then for each of the `solver.iterations` iterations calls
[`linearize!`](@ref) (lagged coefficients of `operator` and `workspace.rhs`), [`solve!`](@ref) (for the
`target` field of the [`Iteration`](@ref) label `workspace.iteration`), [`implicit_update!`](@ref)
(conservative update from `u⁺`), and [`implicit_closure!`](@ref), passing the label with its 1-based
`number` and `final = true` in the last iteration only. On exit, the prognostic field holds `uⁿ⁺¹` and
the closure variables are consistent with it.
"""
function implicit_step!(solver::Picard, operator::AbstractLinearOperator, workspace::ImplicitSolverWorkspace, state, model::AbstractModel, Δt)
    (; rhs, explicit_stage, iteration) = workspace
    # explicit stage u⁺ = u⁰ + Δt G_exp
    add_scaled!(explicit_stage, getproperty(state, varname(iteration)), one(eltype(explicit_stage)))
    # all but the last iteration
    for number in 1:(solver.iterations - 1)
        picard_iteration!(state, model, operator, rhs, explicit_stage, Δt, Iteration(iteration; number))
    end
    picard_iteration!(state, model, operator, rhs, explicit_stage, Δt, Iteration(iteration; final = true, number = solver.iterations))
    return nothing
end

# Single Picard iteration: the four process hooks, solving for the target of `iteration`.
function picard_iteration!(state, model, operator, rhs, explicit_stage, Δt, iteration::Iteration)
    linearize!(operator, rhs, state, model, explicit_stage, Δt, iteration)
    solve!(getproperty(state, targetname(iteration)), operator, rhs, Δt)
    implicit_update!(state, model, operator, explicit_stage, Δt, iteration)
    implicit_closure!(state, model, iteration)
    return nothing
end
