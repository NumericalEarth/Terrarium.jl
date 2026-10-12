"""
    $TYPEDEF

Base type for the linearized operator of a model's implicitly integrated prognostic variables.
An operator owns its working state (coefficient fields and linear solver), is filled by
[`linearize!`](@ref), and is solved by [`solve!`](@ref) in each solver iteration. Operators of several
variables either compose per-variable operators (e.g. [`SplitOperator`](@ref)) or couple them. They are
allocated by [`implicit_operator`](@ref) at `initialize` and stored in the timestepper's cache (e.g.
[`BackwardEulerCache`](@ref)).
"""
abstract type AbstractLinearOperator end

# Operator splitting

"""
    $TYPEDEF

Composite [`AbstractLinearOperator`](@ref) for operator splitting: `operators` holds per-variable
operators keyed by prognostic variable name, in solve order. The model sets this order in
[`implicit_operator`](@ref); e.g. [`SoilModel`](@ref) and [`LandModel`](@ref) solve
`saturation_water_ice` before `internal_energy`. Solvers apply their single-variable iteration to each
component in turn (see [`Picard`](@ref)). Not exported.

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

# Variable names of a `SplitOperator` in solve order, as a compile-time constant.
@inline split_variable_names(::SplitOperator{<:NamedTuple{names}}) where {names} = names

# Pointwise diagonal operator

"""
    $TYPEDEF

Diagonal [`AbstractLinearOperator`](@ref) for implicit terms without spatial coupling, e.g. the meltwater
outflow of a single-layer snowpack. Each row of the system reads `D ϕ = rhs`, with the lagged diagonal `D`
in `diagonal`, so that [`solve!`](@ref) is a pointwise division. The `diagonal` field is allocated at the
location of the field the system is solved for.

```jldoctest
julia> grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 4));

julia> operator = DiagonalOperator(Field{Center, Center, Nothing}(ground_domain(grid)));

julia> size(operator.diagonal)
(1, 1, 1)
```

Properties:
$(TYPEDFIELDS)
"""
struct DiagonalOperator{Diagonal} <: AbstractLinearOperator
    "Lagged diagonal `D`, at the location of the field it is solved for"
    diagonal::Diagonal
end

Adapt.adapt_structure(to, operator::DiagonalOperator) = DiagonalOperator(Adapt.adapt_structure(to, operator.diagonal))

"""
    solve!(target, operator::DiagonalOperator, rhs, Δt, args...)

Solve the system of `operator` for `target` with right-hand side `rhs`, `target = rhs / D` pointwise.
"""
function solve!(target, operator::DiagonalOperator, rhs, Δt, args...)
    launch!(target.grid, XYZ, solve_diagonal_kernel!, target, operator.diagonal, rhs)
    return nothing
end

function solve!(target::Field{<:Any, <:Any, Nothing}, operator::DiagonalOperator, rhs, Δt, args...)
    launch!(target.grid, XY, solve_diagonal_xy_kernel!, target, operator.diagonal, rhs)
    return nothing
end

@kernel inbounds = true function solve_diagonal_kernel!(target, grid, diagonal, rhs)
    i, j, k = @index(Global, NTuple)
    target[i, j, k] = rhs[i, j, k] / diagonal[i, j, k]
end

@kernel inbounds = true function solve_diagonal_xy_kernel!(target, grid, diagonal, rhs)
    i, j = @index(Global, NTuple)
    target[i, j, end] = rhs[i, j, end] / diagonal[i, j, end]
end

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

Batched vertical tridiagonal [`AbstractLinearOperator`](@ref) for implicit vertical diffusion and,
optionally, first-order upwind advection, with one system per column. Row `k` of the system for the
closure variable `ϕ` reads

    C_k ϕ_k - Δt [∂z(κ ∂z ϕ)]_k + Δt [∂z(u ϕ_up)]_k - Δt L_k ϕ_k = rhs_k,

with the lagged capacity `C` in `diagonal_coefficient` (e.g. the apparent heat capacity, or `∂θ/∂ψ`
for Richards flow), the lagged face conductivity `κ` in `face_conductivity` (thermal or hydraulic
conductivity), the optional advective face velocity `u` in `face_velocity`, and the optional linear
coefficient `L` ([`compute_implicit_linear_coefficient`](@ref)). `u` is in units of capacity times
velocity, so that `u ϕ_up` is the advective flux; e.g. `u = C_w q` (W/m²/K) for the heat carried by
the water flux `q` (m/s), with `C_w` the volumetric heat capacity of water. `ϕ_up` is the value in the
upwind cell of the face, which keeps the system an M-matrix for any `u`. `face_velocity` is `nothing`
for operators without advection, and the advective terms then vanish by dispatch
([`compute_implicit_face_velocity`](@ref)). Diffusive fluxes and advective inflow through the two
boundary faces of the column are masked and enter through the right-hand side; advective outflow
through the boundary faces (upwind value of the boundary cell itself) is part of the operator, so a
boundary cell cannot lose more than its content.

Properties:
$(TYPEDFIELDS)
"""
struct TridiagonalOperator{Solver, DiagonalCoefficient, FaceDiffusivity, FaceVelocity} <: AbstractLinearOperator
    "Batched tridiagonal solver with `ImplicitDiffusion*Diagonal` marker coefficients"
    tridiagonal_solver::Solver

    "Lagged capacity `C` (Center, Center, Center)"
    diagonal_coefficient::DiagonalCoefficient

    "Lagged face conductivity `κ` (Center, Center, Face)"
    face_conductivity::FaceDiffusivity

    "Lagged advective face velocity `u` (Center, Center, Face), or `nothing` without advection"
    face_velocity::FaceVelocity
end

"""
    $TYPEDSIGNATURES

Allocate a [`TridiagonalOperator`](@ref) on `grid`, typically `ground_domain(get_grid(model))`. With
`advection = true`, also allocate a `face_velocity` field; otherwise it is `nothing`.

```jldoctest
julia> grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 4));

julia> operator = TridiagonalOperator(ground_domain(grid));

julia> size(operator.diagonal_coefficient)
(1, 1, 4)

julia> isnothing(operator.face_velocity)
true

julia> size(TridiagonalOperator(ground_domain(grid); advection = true).face_velocity)
(1, 1, 5)
```
"""
function TridiagonalOperator(grid::AbstractGrid; advection::Bool = false)
    diagonal_coefficient = Field{Center, Center, Center}(grid)
    face_conductivity = Field{Center, Center, Face}(grid)
    face_velocity = advection ? Field{Center, Center, Face}(grid) : nothing
    tridiagonal_solver = BatchedTridiagonalSolver(
        grid;
        lower_diagonal = ImplicitDiffusionLowerDiagonal(),
        diagonal = ImplicitDiffusionDiagonal(),
        upper_diagonal = ImplicitDiffusionUpperDiagonal(),
    )
    return TridiagonalOperator(tridiagonal_solver, diagonal_coefficient, face_conductivity, face_velocity)
end

function Adapt.adapt_structure(to, operator::TridiagonalOperator)
    return TridiagonalOperator(
        adapt_tridiagonal_solver(to, operator.tridiagonal_solver),
        Adapt.adapt_structure(to, operator.diagonal_coefficient),
        Adapt.adapt_structure(to, operator.face_conductivity),
        Adapt.adapt_structure(to, operator.face_velocity),
    )
end

# Oceananigans defines no `Adapt` rule for `BatchedTridiagonalSolver`.
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
@propagate_inbounds function compute_implicit_diffusion_upper_diagonal(i, j, k, grid, face_conductivity, Δt)
    Nz = size(grid, 3)
    κ = face_conductivity[i, j, k + 1]
    upper = -Δt * κ / (Δzᵃᵃᶜ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k + 1, grid))
    return ifelse(k == Nz, zero(upper), upper)
end

"""
    $TYPEDSIGNATURES

Lower diagonal at index `k′`, which multiplies `ϕ_{k′}` in row `k = k′ + 1` (the
`BatchedTridiagonalSolver` convention): `lower_{k′} = -Δt κ_{k-½} / (Δz^c_k Δz^f_{k-½})`, zero at the
bottom (`k = 1`).
"""
@propagate_inbounds function compute_implicit_diffusion_lower_diagonal(i, j, k′, grid, face_conductivity, Δt)
    k = k′ + 1
    κ = face_conductivity[i, j, k]
    lower = -Δt * κ / (Δzᵃᵃᶜ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid))
    return ifelse(k == 1, zero(lower), lower)
end

"""
    compute_implicit_face_velocity(i, j, k, grid, face_velocity)

Advective face velocity `u` at face `i, j, k` of a [`TridiagonalOperator`](@ref): `face_velocity`, or
`zero(grid)` if it is `nothing` (pure diffusion).
"""
@propagate_inbounds compute_implicit_face_velocity(i, j, k, grid, face_velocity) = face_velocity[i, j, k]

@inline compute_implicit_face_velocity(i, j, k, grid, ::Nothing) = zero(grid)

"""
    $TYPEDSIGNATURES

Upwind advection part of the upper diagonal of row `k`, `Δt min(u_{k+½}, 0) / Δz^c_k`: a downward
velocity through the upper face carries `ϕ_{k+1}` into cell `k`. Zero at the top (`k = Nz`).
"""
@propagate_inbounds function compute_implicit_advection_upper_diagonal(i, j, k, grid, face_velocity, Δt)
    Nz = size(grid, 3)
    u = compute_implicit_face_velocity(i, j, k + 1, grid, face_velocity)
    upper = Δt * min(u, zero(u)) / Δzᵃᵃᶜ(i, j, k, grid)
    return ifelse(k == Nz, zero(upper), upper)
end

"""
    $TYPEDSIGNATURES

Upwind advection part of the lower diagonal at index `k′` (row `k = k′ + 1`),
`-Δt max(u_{k-½}, 0) / Δz^c_k`: an upward velocity through the lower face carries `ϕ_{k-1}` into
cell `k`. Zero at the bottom (`k = 1`).
"""
@propagate_inbounds function compute_implicit_advection_lower_diagonal(i, j, k′, grid, face_velocity, Δt)
    k = k′ + 1
    u = compute_implicit_face_velocity(i, j, k, grid, face_velocity)
    lower = -Δt * max(u, zero(u)) / Δzᵃᵃᶜ(i, j, k, grid)
    return ifelse(k == 1, zero(lower), lower)
end

"""
    $TYPEDSIGNATURES

Upwind advection part of the main diagonal of row `k`,
`Δt (max(u_{k+½}, 0) - min(u_{k-½}, 0)) / Δz^c_k`, i.e. the outflow of `ϕ_k` through both faces,
including the boundary faces of the column. It is nonnegative, which with the nonpositive advective
off-diagonals keeps the system an M-matrix.
"""
@propagate_inbounds function compute_implicit_advection_diagonal(i, j, k, grid, face_velocity, Δt)
    u_upper = compute_implicit_face_velocity(i, j, k + 1, grid, face_velocity)
    u_lower = compute_implicit_face_velocity(i, j, k, grid, face_velocity)
    outflow = max(u_upper, zero(u_upper)) - min(u_lower, zero(u_lower))
    return Δt * outflow / Δzᵃᵃᶜ(i, j, k, grid)
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

Main diagonal of row `k`, `diag_k = C_k - lower_{k-1} - upper_k + a_k - Δt L_k`, with the diffusive
off-diagonals `lower` and `upper`, the upwind advection part `a_k`
([`compute_implicit_advection_diagonal`](@ref)), and `L_k` from
[`compute_implicit_linear_coefficient`](@ref) evaluated with the trailing `args`.
"""
@propagate_inbounds function compute_implicit_diffusion_diagonal(i, j, k, grid, face_conductivity, face_velocity, diagonal_coefficient, Δt, args...)
    C = diagonal_coefficient[i, j, k]
    upper = compute_implicit_diffusion_upper_diagonal(i, j, k, grid, face_conductivity, Δt)
    lower = compute_implicit_diffusion_lower_diagonal(i, j, k - 1, grid, face_conductivity, Δt)
    advection = compute_implicit_advection_diagonal(i, j, k, grid, face_velocity, Δt)
    L = compute_implicit_linear_coefficient(i, j, k, grid, args...)
    return C - lower - upper + advection - Δt * L
end

"""
    $TYPEDSIGNATURES

Diffusive tendency `[∂z(κ ∂z ϕ)]_k`, the rate of change of the conserved quantity from the diffusive
fluxes through the interior faces of cell `k` only (consistent with the masked coefficients), used for
the conservative update after each solve. Mirrors the explicit [`diffusive_heat_flux`](@ref) with the
lagged `face_conductivity`.
"""
@propagate_inbounds function compute_implicit_diffusion_tendency(i, j, k, grid, face_conductivity, ϕ)
    Nz = size(grid, 3)
    # diffusive fluxes κ ∂z ϕ through the lower (k) and upper (k + 1) faces of cell k
    F_lower = face_conductivity[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, ϕ)
    F_upper = face_conductivity[i, j, k + 1] * ∂zᵃᵃᶠ(i, j, k + 1, grid, ϕ)
    # omit the boundary faces
    F_bottom = ifelse(k == 1, zero(F_lower), F_lower)
    F_top = ifelse(k == Nz, zero(F_upper), F_upper)
    return (F_top - F_bottom) / Δzᵃᵃᶜ(i, j, k, grid)
end

"""
    $TYPEDSIGNATURES

First-order upwind advective flux `u ϕ_up` through face `i, j, k`, with `ϕ_up` taken from the cell
below the face (`k - 1`) for `u > 0` and above it (`k`) otherwise, as in the coefficients of the
[`TridiagonalOperator`](@ref).
"""
@propagate_inbounds function compute_implicit_advective_flux(i, j, k, grid, face_velocity, ϕ)
    u = compute_implicit_face_velocity(i, j, k, grid, face_velocity)
    return max(u, zero(u)) * ϕ[i, j, k - 1] + min(u, zero(u)) * ϕ[i, j, k]
end

"""
    $TYPEDSIGNATURES

Advective tendency `-[∂z(u ϕ_up)]_k`, the rate of change of the conserved quantity from the upwind
fluxes ([`compute_implicit_advective_flux`](@ref)) through the faces of cell `k`, used for the
conservative update after each solve. Through the boundary faces of the column only the outflow
`u ϕ_k` is included; the inflow enters through the right-hand side. Zero
without advection (`face_velocity = nothing`).
"""
@propagate_inbounds function compute_implicit_advection_tendency(i, j, k, grid, face_velocity, ϕ)
    Nz = size(grid, 3)
    F_lower = compute_implicit_advective_flux(i, j, k, grid, face_velocity, ϕ)
    F_upper = compute_implicit_advective_flux(i, j, k + 1, grid, face_velocity, ϕ)
    # only the outflow through the boundary faces, downward at the bottom and upward at the top
    u_lower = compute_implicit_face_velocity(i, j, k, grid, face_velocity)
    u_upper = compute_implicit_face_velocity(i, j, k + 1, grid, face_velocity)
    ϕ_k = ϕ[i, j, k]
    F_bottom = ifelse(k == 1, min(u_lower, zero(u_lower)) * ϕ_k, F_lower)
    F_top = ifelse(k == Nz, max(u_upper, zero(u_upper)) * ϕ_k, F_upper)
    return (F_bottom - F_top) / Δzᵃᵃᶜ(i, j, k, grid)
end

@inline compute_implicit_advection_tendency(i, j, k, grid, ::Nothing, ϕ) = zero(grid)

@inline Oceananigans.Solvers.get_coefficient(i, j, k, grid, ::ImplicitDiffusionLowerDiagonal, p, ::ZDirection, face_conductivity, face_velocity, diagonal_coefficient, Δt, args...) =
    compute_implicit_diffusion_lower_diagonal(i, j, k, grid, face_conductivity, Δt) +
    compute_implicit_advection_lower_diagonal(i, j, k, grid, face_velocity, Δt)

@inline Oceananigans.Solvers.get_coefficient(i, j, k, grid, ::ImplicitDiffusionDiagonal, p, ::ZDirection, face_conductivity, face_velocity, diagonal_coefficient, Δt, args...) =
    compute_implicit_diffusion_diagonal(i, j, k, grid, face_conductivity, face_velocity, diagonal_coefficient, Δt, args...)

@inline Oceananigans.Solvers.get_coefficient(i, j, k, grid, ::ImplicitDiffusionUpperDiagonal, p, ::ZDirection, face_conductivity, face_velocity, diagonal_coefficient, Δt, args...) =
    compute_implicit_diffusion_upper_diagonal(i, j, k, grid, face_conductivity, Δt) +
    compute_implicit_advection_upper_diagonal(i, j, k, grid, face_velocity, Δt)

"""
    solve!(target, operator::TridiagonalOperator, rhs, Δt, args...)

Solve the system of `operator` for `target` with right-hand side `rhs` (one Thomas sweep per column).
The trailing `args` are passed on to [`compute_implicit_linear_coefficient`](@ref).
"""
function solve!(target, operator::TridiagonalOperator, rhs, Δt, args...)
    (; tridiagonal_solver, face_conductivity, face_velocity, diagonal_coefficient) = operator
    Oceananigans.Solvers.solve!(target, tridiagonal_solver, rhs, face_conductivity, face_velocity, diagonal_coefficient, Δt, args...)
    return nothing
end
