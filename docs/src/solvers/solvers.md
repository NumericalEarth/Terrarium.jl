# Nonlinear solvers

```@meta
CurrentModule = Terrarium
```

## Overview

Some land processes, such as the implicit formulation of the [surface energy balance](@ref surface_energy_balance_docs), require solving pointwise nonlinear algebraic equations. To keep the numerics of such routines modular, device-agnostic, and differentiable, Terrarium provides a barebones interface for solving scalar optimization problems consisting of an *objective function* and a set of interchangeable *solvers*.

The general problem is to find values of a target field that drives a residual `F(x)` to zero,
```math
F(x) = 0
```
where `F` is evaluated by mutating the relevant output fields and reading back the residual. The unknown `x` is stored in the target field, so the same machinery works for any prognostic or auxiliary quantity solved implicitly.

## Objective functions

An [`ObjectiveFunction`](@ref) couples a residual kernel function to the name of the `target`
field it solves for. The residual kernel has the standard kernel-function call signature
`F(out, indices..., grid, fields, args...)` and is expected to write any intermediate fluxes into `out` and return the scalar residual. An optional derivative function may be supplied when analytic derivatives are available; if omitted, a simple finite-difference approximation is used.

```@docs; canonical = false
ObjectiveFunction
```

## Root solvers

[`RootSolver`](@ref) wraps the root-finding methods provided by
[RootSolvers.jl](https://github.com/CliMA/RootSolvers.jl) (e.g. Newton's method). When the [`ObjectiveFunction`](@ref) carries no analytic derivative, a finite-difference derivative is formed automatically so that derivative-based methods such as Newton's method can still be used.
This is the default solver for the implicit skin temperature, configured by [`default_skin_temperature_solver`](@ref).

```@docs; canonical = false
RootSolver
```

## Fixed-point solvers

[`FixedPointSolver`](@ref) implements a simple fixed-point iteration $x_{n+1} = g(x_n)$. Since the [`ObjectiveFunction`](@ref)
is formulated as a residual `r(x)`, the iterative scheme uses `g(x) = x - \alpha r(x)` for the fixed-point update. Updates may be
under-relaxed by setting $\alpha < 1$ via [`RelaxationFactor`](@ref).

```@docs; canonical = false
FixedPointSolver
RelaxationFactor
relaxed_update
```

## Solvers for implicit time stepping

The nonlinear solvers above act pointwise across the underlying `grid`. An implicit time step requires additionally solving a coupled system along each column (see [Implicit-explicit time stepping](@ref)). As an example, a backward Euler step of the system state $u$,
```math
u^{n+1} = u^+ + \Delta t\, G_\text{imp}\left(u^{n+1}\right)\,, \qquad u^+ = u^n + \Delta t\, G_\text{exp}\left(u^n\right)\,,
```
forms a nonlinear system in ``u^{n+1}``, since the coefficients of the implicit tendency ``G_\text{imp}`` (e.g. the heat capacity and conductivity of vertical heat conduction) depend on the solution, and its discretization couples each cell to its vertical neighbors. Its solution is split into two parts: implicit solvers, which subtype [`AbstractImplicitSolver`](@ref) and are components of an implicit timestepper such as [`BackwardEuler`](@ref), define how the nonlinear system is iterated, while linear operators subtyping [`AbstractLinearOperator`](@ref) define the numerical scheme for *linearizing* and solving the system of equations at each iteration. A solver is generic over the operator, and an operator is independent of the solver.

In the code, the unit of the implicit solve is the set of all variables of a model that are routed to the implicit timestepper: the model provides a single operator for them, which is either a composite of per-variable operators, such as the [`SplitOperator`](@ref) below, or an operator that couples the variables. A solver implements the solver-level methods `implicit_step!(solver, operator, workspaces::NamedTuple, state, model, Δt)` for all routed variables and `implicit_step!(solver, operator, workspace::ImplicitSolverWorkspace, state, model, Δt)` for a single one (see [`implicit_step!`](@ref)), which contain only its iteration; the bookkeeping of the time step, such as storing the explicit increment and resetting the prognostic fields to the initial iterate, belongs to the implicit timestepper.

The working data buffers for each (implicit) prognostic variable are bundled in an [`ImplicitSolverWorkspace`](@ref): the right-hand side `rhs`, the `explicit_stage` (which holds the explicit increment `Δt G_exp` on entry and the explicit stage `u⁺` once the solver has formed it from the variable's current field), and the [`Iteration`](@ref) label `iteration`, which names the variable and the field that its linear system is solved for, as resolved by the implicit timestepper from the variable's closure (see [`BackwardEuler`](@ref)). The workspaces are passed as a `NamedTuple` keyed by variable name, so solvers depend on no timestepper type.

```@docs; canonical = false
ImplicitSolverWorkspace
```

```@docs; canonical = false
AbstractImplicitSolver
AbstractLinearOperator
```

### Picard solver

The backward Euler step of a nonlinear equation is a nonlinear system in ``u^{n+1}``, written as ``F(u^{n+1}) = 0``, since the coefficients of the operator (e.g. the heat capacity or the hydraulic conductivity) depend on the solution. A [`Picard`](@ref) scheme solves this root-finding problem via fixed-point iteration. With the coefficients evaluated (*lagged*) at the current iterate ``u^{n+1,m}``, the system is linear, and its solution is the next iterate,
```math
u^{n+1,m+1} = g\left(u^{n+1,m}\right)\,,
```
where ``g(v)`` denotes the solution of the linear system with coefficients evaluated at ``v``. A fixed point ``u = g(u)`` is a solution of ``F(u) = 0``, and the iteration ``u^{n+1,m+1} = g(u^{n+1,m})`` starting from ``u^{n+1,0} = u^n`` is the Picard (fixed-point) iteration [CeliaEtAl1990](@cite). With a single iteration, it is the linearly implicit scheme common in land surface models [bonanClimateChangeTEMSoilMoisture2019](@cite).

[`Picard`](@ref) performs a fixed number of iterations configurable by the user (default: `Picard(NF; iterations = 2)`). For the operator of a single variable, its solver-level `implicit_step!` runs a fixed number of iterations, each of which calls the process hooks [`linearize!`](@ref), [`solve!`](@ref), [`implicit_update!`](@ref), and [`implicit_closure!`](@ref) of the variable (see [Process hooks](@ref imex-process-hooks)). For a [`SplitOperator`](@ref), it runs these iterations for each component in turn. Since each variable forms its explicit stage as its current field plus the explicit increment when it is solved, each component after the first is stepped from the state left by the components solved before it, as in a sequential split: changes that their closures make to its prognostic field (e.g. the heat moved with the water redistributed by the soil hydrology closure) are kept.

```@docs; canonical = false
Picard
```

### Operator splitting

A [`SplitOperator`](@ref) holds the operators of several variables in a `NamedTuple`, in the order in which they are solved. An example of this is the soil component of [`SoilModel`](@ref) and [`LandModel`](@ref) which returns a `SplitOperator` from [`implicit_operator`](@ref)`(model, state, names)` that always solves `saturation_water_ice` before `internal_energy`, so that the heat step sees the saturation at the end of the step and advects heat with the water flux of the step, while the Richards step sees the temperature and liquid water fraction at its beginning (see [Coupling of water and energy](@ref imex-water-energy-coupling)). Operator splitting is therefore a choice of the model rather than a property of the timestepper; a model that solves its implicit variables together would return a coupled operator from the same query.

```@docs; canonical = false
SplitOperator
```

### Tridiagonal operator

[`TridiagonalOperator`](@ref) is the linear operator of vertical diffusion and, optionally, first-order upwind advection. Its systems are solved with Oceananigans' [`BatchedTridiagonalSolver`](@extref Oceananigans.Solvers.BatchedTridiagonalSolver), which solves one system per column and runs on CPU and GPU. The operator owns the batched solver and the lagged coefficient fields, `diagonal_coefficient` at cell centers and `face_conductivity` at cell faces, which are filled by [`linearize!`](@ref). The coefficients of the solver are looked up through `Oceananigans.Solvers.get_coefficient`, for which Terrarium defines methods on three marker structs, [`ImplicitDiffusionLowerDiagonal`](@ref), [`ImplicitDiffusionDiagonal`](@ref), and [`ImplicitDiffusionUpperDiagonal`](@ref). These coefficients follow from a finite-volume discretization of the diffusion equation
```math
C \frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\left(\kappa \frac{\partial T}{\partial z}\right)
```
with a cell-centered capacity ``C`` and a conductivity ``\kappa`` that may vary with depth (hence ``\partial_z(\kappa\, \partial_z T)`` rather than ``\kappa\, \partial_z^2 T``).
Integrating over cell ``k``, of thickness ``\Delta z_k`` and bounded by the faces ``k - 1/2`` (below) and ``k + 1/2`` (above), and approximating the gradient at each face by the difference of the adjacent cell values over the distance ``\Delta z_{k \pm 1/2}`` between their centers, gives
```math
C_k \frac{\mathrm{d} T_k}{\mathrm{d} t} = \frac{1}{\Delta z_k}\left[ \kappa_{k+1/2}\, \frac{T_{k+1} - T_k}{\Delta z_{k+1/2}} - \kappa_{k-1/2}\, \frac{T_k - T_{k-1}}{\Delta z_{k-1/2}} \right]\,.
```
A backward Euler step evaluates the right-hand side at ``T^{n+1}`` and replaces the time derivative by ``(T_k^{n+1} - T_k^n) / \Delta t``.
Multiplying by ``\Delta t`` and collecting the unknowns gives one row per cell,
```math
a_k\, T_{k-1}^{n+1} + b_k\, T_k^{n+1} + c_k\, T_{k+1}^{n+1} = d_k\,, \qquad
a_k = -\frac{\Delta t\, \kappa_{k-1/2}}{\Delta z_k\, \Delta z_{k-1/2}}\,, \quad
c_k = -\frac{\Delta t\, \kappa_{k+1/2}}{\Delta z_k\, \Delta z_{k+1/2}}\,, \quad
b_k = C_k - a_k - c_k\,,
```
with ``d_k = C_k T_k^n`` for this equation.
Each row couples a cell only to its two neighbors, so the system of a column is tridiagonal.
There is no flux through the column boundaries in the operator, i.e. ``a_1 = 0`` at the bottom and ``c_{N_z} = 0`` at the top; boundary fluxes are treated explicitly and enter through ``d_k``.
In the implicit steps of Terrarium, ``C`` and ``\kappa`` are lagged coefficients and ``d_k`` is the right-hand side assembled by [`linearize!`](@ref), e.g. ``C_\text{app} T^m - U^m + U^+`` for the heat step (see [Implicit vertical heat conduction](@ref)).
Since ``C_k > 0`` and ``\kappa \geq 0``, the off-diagonal coefficients ``a_k`` and ``c_k`` are nonpositive, and the main diagonal ``b_k`` is positive and dominates each row (the row sum is ``C_k``).
A matrix with these two properties is an [M-matrix](https://en.wikipedia.org/wiki/M-matrix): it is invertible and its inverse has no negative entries.
As a consequence, the solution obeys a discrete maximum principle: it creates no new extrema and no spurious oscillations, for any time step.

If the operator has a `face_velocity` ``u`` (allocated with `TridiagonalOperator(grid; advection = true)`), the equation also contains the divergence of the advective flux ``u\, T``, discretized with first-order upwinding: the flux through face ``k + 1/2`` is ``F_{k+1/2} = u^+_{k+1/2}\, T_k + u^-_{k+1/2}\, T_{k+1}``, with ``u^+ = \max(u, 0)`` and ``u^- = \min(u, 0)``, i.e. it carries the value of the cell below the face for an upward velocity and of the cell above it otherwise.
The term ``\Delta t\, (F_{k+1/2} - F_{k-1/2}) / \Delta z_k`` of row ``k`` adds the outflow of the cell to the main diagonal and its inflow, with a negative sign, to the off-diagonals:
```math
b_k \mathrel{+}= \frac{\Delta t}{\Delta z_k}\left(u^+_{k+1/2} - u^-_{k-1/2}\right)\,, \qquad
a_k \mathrel{+}= -\frac{\Delta t}{\Delta z_k}\, u^+_{k-1/2}\,, \qquad
c_k \mathrel{+}= \frac{\Delta t}{\Delta z_k}\, u^-_{k+1/2}\,.
```
The velocity is given in units of the capacity times a velocity, e.g. ``u = C_\text{w} j_\text{w}`` for the heat carried by the vertical soil water flux ``j_\text{w}``.
The off-diagonals remain nonpositive, but the rows are no longer dominant in general.
The columns are, once each row is multiplied by its ``\Delta z_k``: every interior flux leaves one cell and enters its neighbor, so the ``\Delta z``-weighted column sums are ``C_k \Delta z_k > 0`` for any ``u``, or larger for a boundary cell with outflow through the boundary face.
The system is therefore still an M-matrix, and the solution obeys the same discrete maximum principle, without a CFL limit on the advection.
Through the boundary faces, the outflow, whose upwind value is that of the boundary cell itself, is part of the operator, while the inflow must enter through the right-hand side.
Without advection, `face_velocity` is `nothing` and the advective terms vanish by dispatch.

An operator may in addition treat a linear term ``L_k T_k`` implicitly (e.g. a linear sink or a linearized boundary flux), which adds ``-\Delta t\, L_k`` to the main diagonal. Its coefficient is given by [`compute_implicit_linear_coefficient`](@ref), which an operator specializes via the trailing arguments it passes to [`solve!`](@ref); by default it is zero.

[`DiagonalOperator`](@ref) is the operator of implicit terms without spatial coupling, such as the meltwater drainage of a single-layer snowpack: its system is diagonal, so [`solve!`](@ref) is a pointwise division by the lagged `diagonal`.

```@docs; canonical = false
DiagonalOperator
solve!(target, operator::DiagonalOperator, rhs, Δt, args...)
TridiagonalOperator
solve!(target, operator::TridiagonalOperator, rhs, Δt, args...)
ImplicitDiffusionLowerDiagonal
ImplicitDiffusionDiagonal
ImplicitDiffusionUpperDiagonal
compute_implicit_diffusion_diagonal
compute_implicit_diffusion_upper_diagonal
compute_implicit_diffusion_lower_diagonal
compute_implicit_face_velocity
compute_implicit_advection_diagonal
compute_implicit_advection_upper_diagonal
compute_implicit_advection_lower_diagonal
compute_implicit_linear_coefficient
compute_implicit_diffusion_tendency
compute_implicit_advective_flux
compute_implicit_advection_tendency
```

Process-specific regularization of the lagged coefficients is a property of the process rather than of the solver or the operator; for example, the regularized `FreeWater` apparent heat capacity of [`compute_implicit_capacity!`](@ref).

The operator of the implicitly routed variables is obtained from the model with the block query [`implicit_operator`](@ref)`(model, state, names)` when the implicit timestepper's cache is allocated at initialization. Every model with implicitly routed variables defines this query; its methods also validate the configuration on the host. For a model without an operator, the fallback throws an `ArgumentError`, so that a missing operator is reported at initialization rather than at the first time step:

```@docs; canonical = false
implicit_operator
```

## Interface

```@docs; canonical = false
solve!
```

## [References](@id "solvers.refs")

```@bibliography
Pages = ["solvers.md"]
Canonical = false
```
