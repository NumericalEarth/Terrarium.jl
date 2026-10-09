"""
    $TYPEDEF

Base type for the linearized operator of a model's implicitly integrated prognostic variables.
An operator owns its working state (coefficient fields and linear solver), is filled by
[`linearize!`](@ref), and is solved by [`solve!`](@ref) in each iteration of the implicit solver.
Operators of several variables either compose per-variable operators (e.g. [`SplitOperator`](@ref))
or couple them. They are allocated by [`implicit_operator`](@ref) at `initialize` and stored in the
implicit timestepper's cache (e.g. [`BackwardEulerCache`](@ref)).
"""
abstract type AbstractLinearOperator end

# Operator splitting

"""
    $TYPEDEF

Composite [`AbstractLinearOperator`](@ref) for operator splitting: `operators` holds per-variable
operators keyed by prognostic variable name, in the order in which they are solved. The model sets
this order in [`implicit_operator`](@ref); e.g. [`SoilModel`](@ref) and [`LandModel`](@ref) solve
`internal_energy` before `saturation_water_ice`. Solvers apply their single-variable iteration to
each component in turn (see [`Picard`](@ref)). Not exported.

```jldoctest
julia> grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 4));

julia> operator = Terrarium.SplitOperator((b = TridiagonalOperator(ground_domain(grid)), a = TridiagonalOperator(ground_domain(grid))));

julia> keys(operator.operators)
(:b, :a)
```

Properties:
$(TYPEDFIELDS)
"""
struct SplitOperator{Operators <: NamedTuple} <: AbstractLinearOperator
    "Per-variable operators, keyed by prognostic variable name, in the order in which they are applied"
    operators::Operators
end

Adapt.adapt_structure(to, operator::SplitOperator) = SplitOperator(Adapt.adapt_structure(to, operator.operators))

# Names of the prognostic variables of a `SplitOperator`, in the order of application, as a compile-time constant.
@inline split_variable_names(::SplitOperator{<:NamedTuple{names}}) where {names} = names

# Batched vertical tridiagonal operator

"""
    $TYPEDEF

Marker passed as the `lower_diagonal` of the `BatchedTridiagonalSolver` of a [`TridiagonalOperator`](@ref);
its values come from `Oceananigans.Solvers.get_coefficient`.
"""
struct ImplicitDiffusionLowerDiagonal end

"""
    $TYPEDEF

Marker passed as the `diagonal` of the `BatchedTridiagonalSolver` of a [`TridiagonalOperator`](@ref);
its values come from `Oceananigans.Solvers.get_coefficient`.
"""
struct ImplicitDiffusionDiagonal end

"""
    $TYPEDEF

Marker passed as the `upper_diagonal` of the `BatchedTridiagonalSolver` of a [`TridiagonalOperator`](@ref);
its values come from `Oceananigans.Solvers.get_coefficient`.
"""
struct ImplicitDiffusionUpperDiagonal end

"""
    $TYPEDEF

Batched vertical tridiagonal [`AbstractLinearOperator`](@ref) for implicit vertical diffusion, with
one system per column. Row `k` of the system for the closure variable `ϕ` reads

    C_k ϕ_k - Δt [∂z(κ ∂z ϕ)]_k - Δt L_k ϕ_k = rhs_k,

with the lagged capacity `C` in `diagonal_coefficient` (e.g. the apparent heat capacity, or `∂θ/∂ψ`
for Richards flow), the lagged face diffusivity `κ` in `face_diffusivity` (thermal or hydraulic
conductivity), and the optional linear coefficient `L` ([`compute_implicit_linear_coefficient`](@ref)).
Boundary-face fluxes are masked, so the operator is flux-free and boundary fluxes enter through the
right-hand side.

Properties:
$(TYPEDFIELDS)
"""
struct TridiagonalOperator{Solver, DiagonalCoefficient, FaceDiffusivity} <: AbstractLinearOperator
    "Batched tridiagonal solver with `ImplicitDiffusion*Diagonal` marker coefficients"
    tridiagonal_solver::Solver

    "Lagged capacity `C` (Center, Center, Center)"
    diagonal_coefficient::DiagonalCoefficient

    "Lagged face diffusivity `κ` (Center, Center, Face)"
    face_diffusivity::FaceDiffusivity
end

"""
    $TYPEDSIGNATURES

Allocate a [`TridiagonalOperator`](@ref) on `grid`, typically `ground_domain(get_grid(model))`.

```jldoctest
julia> grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 4));

julia> operator = TridiagonalOperator(ground_domain(grid));

julia> size(operator.diagonal_coefficient)
(1, 1, 4)
```
"""
function TridiagonalOperator(grid::AbstractGrid)
    diagonal_coefficient = Field{Center, Center, Center}(grid)
    face_diffusivity = Field{Center, Center, Face}(grid)
    tridiagonal_solver = BatchedTridiagonalSolver(
        grid;
        lower_diagonal = ImplicitDiffusionLowerDiagonal(),
        diagonal = ImplicitDiffusionDiagonal(),
        upper_diagonal = ImplicitDiffusionUpperDiagonal(),
    )
    return TridiagonalOperator(tridiagonal_solver, diagonal_coefficient, face_diffusivity)
end

function Adapt.adapt_structure(to, operator::TridiagonalOperator)
    return TridiagonalOperator(
        adapt_tridiagonal_solver(to, operator.tridiagonal_solver),
        Adapt.adapt_structure(to, operator.diagonal_coefficient),
        Adapt.adapt_structure(to, operator.face_diffusivity),
    )
end

# Oceananigans defines no `Adapt` rule for `BatchedTridiagonalSolver`, so adapt its array-valued members here.
function adapt_tridiagonal_solver(to, solver::BatchedTridiagonalSolver)
    return BatchedTridiagonalSolver(
        solver.a,
        solver.b,
        solver.c,
        Adapt.adapt_structure(to, solver.t),
        Adapt.adapt_structure(to, solver.grid),
        Adapt.adapt_structure(to, solver.parameters),
        solver.tridiagonal_direction,
    )
end

# Tridiagonal coefficients, following Oceananigans' `ivd_*_diagonal` construction

"""
    $TYPEDSIGNATURES

Upper diagonal of row `k`, `upper_k = -Δt κ_{k+½} / (Δz^c_k Δz^f_{k+½})`, zero at the top (`k = Nz`).
"""
@propagate_inbounds function compute_implicit_diffusion_upper_diagonal(i, j, k, grid, face_diffusivity, Δt)
    Nz = size(grid, 3)
    κ = face_diffusivity[i, j, k + 1]
    upper = -Δt * κ / (Δzᵃᵃᶜ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k + 1, grid))
    return ifelse(k == Nz, zero(upper), upper)
end

"""
    $TYPEDSIGNATURES

Lower diagonal at index `k′`, which multiplies `ϕ_{k′}` in row `k = k′ + 1` (the
`BatchedTridiagonalSolver` convention): `lower_{k′} = -Δt κ_{k-½} / (Δz^c_k Δz^f_{k-½})`, zero at the
bottom (`k = 1`).
"""
@propagate_inbounds function compute_implicit_diffusion_lower_diagonal(i, j, k′, grid, face_diffusivity, Δt)
    k = k′ + 1
    κ = face_diffusivity[i, j, k]
    lower = -Δt * κ / (Δzᵃᵃᶜ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid))
    return ifelse(k == 1, zero(lower), lower)
end

"""
    compute_implicit_linear_coefficient(i, j, k, grid, args...)

Coefficient `L_k` (1/s) of an implicit linear term `L ϕ` in row `k`, e.g. a linear sink or a
linearized boundary flux; zero by default. `args` are the trailing arguments of [`solve!`](@ref) after
`Δt`, for dispatch. A nonzero `L` must also enter the conservative update as `Δt L ϕ`.
"""
@inline compute_implicit_linear_coefficient(i, j, k, grid, args...) = zero(grid)

"""
    $TYPEDSIGNATURES

Main diagonal of row `k`, `diag_k = C_k - lower_{k-1} - upper_k - Δt L_k`, with `L_k` from
[`compute_implicit_linear_coefficient`](@ref) evaluated with the trailing `args`.
"""
@propagate_inbounds function compute_implicit_diffusion_diagonal(i, j, k, grid, face_diffusivity, diagonal_coefficient, Δt, args...)
    C = diagonal_coefficient[i, j, k]
    upper = compute_implicit_diffusion_upper_diagonal(i, j, k, grid, face_diffusivity, Δt)
    lower = compute_implicit_diffusion_lower_diagonal(i, j, k - 1, grid, face_diffusivity, Δt)
    L = compute_implicit_linear_coefficient(i, j, k, grid, args...)
    return C - lower - upper - Δt * L
end

"""
    $TYPEDSIGNATURES

Divergence `[∂z(κ ∂z ϕ)]_k` through the interior faces of cell `k` only, consistent with the masked
coefficients, used for the conservative update after each solve. The stencil mirrors the explicit
[`diffusive_heat_flux`](@ref), with the lagged `face_diffusivity` in place of the interpolated
conductivity.
"""
@propagate_inbounds function compute_implicit_diffusion_divergence(i, j, k, grid, face_diffusivity, ϕ)
    Nz = size(grid, 3)
    # diffusive fluxes κ ∂z ϕ through the lower (k) and upper (k + 1) faces of cell k
    F_lower = face_diffusivity[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, ϕ)
    F_upper = face_diffusivity[i, j, k + 1] * ∂zᵃᵃᶠ(i, j, k + 1, grid, ϕ)
    # omit the boundary faces
    F_bottom = ifelse(k == 1, zero(F_lower), F_lower)
    F_top = ifelse(k == Nz, zero(F_upper), F_upper)
    return (F_top - F_bottom) / Δzᵃᵃᶜ(i, j, k, grid)
end

@inline Oceananigans.Solvers.get_coefficient(i, j, k, grid, ::ImplicitDiffusionLowerDiagonal, p, ::ZDirection, face_diffusivity, diagonal_coefficient, Δt, args...) =
    compute_implicit_diffusion_lower_diagonal(i, j, k, grid, face_diffusivity, Δt)

@inline Oceananigans.Solvers.get_coefficient(i, j, k, grid, ::ImplicitDiffusionDiagonal, p, ::ZDirection, face_diffusivity, diagonal_coefficient, Δt, args...) =
    compute_implicit_diffusion_diagonal(i, j, k, grid, face_diffusivity, diagonal_coefficient, Δt, args...)

@inline Oceananigans.Solvers.get_coefficient(i, j, k, grid, ::ImplicitDiffusionUpperDiagonal, p, ::ZDirection, face_diffusivity, diagonal_coefficient, Δt, args...) =
    compute_implicit_diffusion_upper_diagonal(i, j, k, grid, face_diffusivity, Δt)

"""
    solve!(target, operator::TridiagonalOperator, rhs, Δt, args...)

Solve the system of `operator` for `target` with right-hand side `rhs` (one Thomas sweep per column).
The trailing `args` are passed on to [`compute_implicit_linear_coefficient`](@ref).
"""
function solve!(target, operator::TridiagonalOperator, rhs, Δt, args...)
    (; tridiagonal_solver, face_diffusivity, diagonal_coefficient) = operator
    Oceananigans.Solvers.solve!(target, tridiagonal_solver, rhs, face_diffusivity, diagonal_coefficient, Δt, args...)
    return nothing
end
