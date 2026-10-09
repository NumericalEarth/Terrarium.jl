# Picard (lagged-coefficient) solver of implicit time steps
#
# Implements the solver-level `implicit_step!` (see `implicit_solvers.jl`) for a `SplitOperator`, by operator
# splitting, and for the operator of a single variable as a fixed number of Picard iterations, each composed of
# the process hooks `linearize!`, `solve!`, `implicit_update!`, and `implicit_closure!`.

"""
    $TYPEDEF

Picard (lagged-coefficient) solver for the implicit system of an implicit timestepper such as
[`BackwardEuler`](@ref). Each Picard iteration linearizes the implicit operator of a prognostic
variable at the latest iterate ([`linearize!`](@ref)), solves the resulting linear system for the
closure variable (e.g. temperature) with [`solve!`](@ref), which dispatches on the
[`AbstractLinearOperator`](@ref) (e.g. a batched tridiagonal solve for a [`TridiagonalOperator`](@ref)),
updates the conserved prognostic variable conservatively ([`implicit_update!`](@ref)), and re-evaluates
the closure ([`implicit_closure!`](@ref)). The solver is generic over the operator: the
process-specific parts are the hooks, which the model forwards to the process that owns the variable.

A single iteration (`iterations = 1`) corresponds to the linearly implicit scheme used by most land
surface models. The default is two iterations, which substantially reduces the error of the lagged
coefficients near freezing fronts and is the minimum recommended for the modified Picard iteration of
the Richards equation [CeliaEtAl1990](@cite).

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

Construct a [`Picard`](@ref) solver with numeric type `NF` (accepted for consistency with the other
timestepping constructors; the solver itself holds no floating-point parameters). `iterations` must
be at least one and defaults to two.

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

Solve the implicit system of the prognostic variables of a [`SplitOperator`](@ref) with the
[`Picard`](@ref) `solver` by operator splitting: the Picard iteration of a single variable (see the
method below) is applied to each component of `operator` in turn, in the order of the components,
with the `rhs` and `explicit_state` fields and the [`Iteration`](@ref) in `iterations` of that variable. Each
variable is solved at the latest state of the variables solved before it, and at the initial
iterate of those solved after it.
"""
function implicit_step!(
        solver::Picard,
        operator::SplitOperator,
        state,
        model::AbstractModel,
        rhs::NamedTuple,
        explicit_state::NamedTuple,
        Δt,
        iterations::NamedTuple
    )
    fastiterate(split_variable_names(operator)) do name
        implicit_step!(solver, operator.operators[name], state, model, rhs[name], explicit_state[name], Δt, iterations[name])
    end
    return nothing
end

"""
    $TYPEDSIGNATURES

Solve the implicit system of the prognostic variable of `iteration` with the [`Picard`](@ref) `solver` and the
variable's linear `operator`. On entry, the prognostic field in `state` holds the initial iterate
(`uⁿ` for [`BackwardEuler`](@ref)) and `explicit_stage` holds the explicit stage `u* = uⁿ + Δt G_exp`.
For each of the `solver.iterations` iterations, this calls the process hooks [`linearize!`](@ref)
(lagged coefficients of `operator` and the right-hand side `rhs`), [`solve!`](@ref) (for the field
`target` of the [`Iteration`](@ref), e.g. the closure variable), [`implicit_update!`](@ref) (conservative
update of the prognostic variable from `u*`), and [`implicit_closure!`](@ref), passing `iteration` with
its 1-based `number` and, in the last iteration only, `final = true`. On exit, the prognostic field holds the last iterate
(`uⁿ⁺¹`), and the closure variables are consistent with it.
"""
function implicit_step!(
        solver::Picard,
        operator::AbstractLinearOperator,
        state,
        model::AbstractModel,
        rhs,
        explicit_stage,
        Δt,
        iteration::Iteration
    )
    # all but the last iteration
    for number in 1:(solver.iterations - 1)
        picard_iteration!(state, model, operator, rhs, explicit_stage, Δt, Iteration(iteration; number))
    end
    picard_iteration!(state, model, operator, rhs, explicit_stage, Δt, Iteration(iteration; final = true, number = solver.iterations))
    return nothing
end

# Single Picard iteration composed of the four process hooks; the linear system is solved for the target of `iteration`.
function picard_iteration!(state, model, operator, rhs, explicit_stage, Δt, iteration::Iteration)
    linearize!(operator, rhs, state, model, explicit_stage, Δt, iteration)
    solve!(getproperty(state, targetname(iteration)), operator, rhs, Δt)
    implicit_update!(state, model, operator, explicit_stage, Δt, iteration)
    implicit_closure!(state, model, iteration)
    return nothing
end
