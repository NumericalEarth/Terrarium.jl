# Nonlinear solvers

```@meta
CurrentModule = Terrarium
```

## Overview

Some land processes such as the implicit [skin temperature](@ref "Skin temperature"), require solving pointwise nonlinear algebraic equations. To keep the numerics of such routines modular, device-agnostic, and differentiable, Terrarium provides a barebones interface for solving scalar optimization problems consisting of an *objective function* and a set of interchangeable *solvers*.

The general problem is to find values of a target field that drives a residual `F(x)` to zero,
```math
F(x) = 0
```
where `F` is evaluated by mutating the relevant output fields and reading back the residual. The unknown `x` is stored in the target field, so the same machinery works for any prognostic or
auxiliary quantity solved implicitly.

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

## Implicit solvers

The nonlinear solvers above act pointwise. Implicit time stepping (see [Implicit-explicit time stepping](@ref)) instead requires solving a coupled system along each column. This is split into two parts: implicit solvers, which subtype [`AbstractImplicitSolver`](@ref) and are held by an implicit timestepper such as [`BackwardEuler`](@ref), define the (nonlinear) iteration, and linear operators, which subtype [`AbstractLinearOperator`](@ref), define the form of the linearized system and how it is solved. A solver is generic over the operator, and an operator is independent of the solver. The unit of the implicit solve is the set of all variables of a model that are routed to the implicit timestepper: the model provides a single operator for them, which is either a composite of per-variable operators, such as the [`SplitOperator`](@ref) below, or an operator that couples the variables. A solver implements the solver-level methods `implicit_step!(solver, operator, workspaces::NamedTuple, state, model, Δt)` for all routed variables and `implicit_step!(solver, operator, workspace::ImplicitSolverWorkspace, state, model, Δt)` for a single one (see [`implicit_step!`](@ref)), which contain only its iteration; the bookkeeping of the time step, such as storing the explicit increment and resetting the prognostic fields to the initial iterate, belongs to the implicit timestepper. The per-variable working data are bundled in an [`ImplicitSolverWorkspace`](@ref): the right-hand side `rhs`, the `explicit_stage` (which holds the explicit increment `Δt G_exp` on entry and the explicit stage `u⁺` once the solver has formed it from the variable's current field), and the [`Iteration`](@ref) label `iteration`, which names the variable and the field that its linear system is solved for, as resolved by the implicit timestepper from the variable's closure (see [`BackwardEuler`](@ref)). The workspaces are passed as a `NamedTuple` keyed by variable name, so solvers depend on no timestepper type.

```@docs; canonical = false
ImplicitSolverWorkspace
```

```@docs; canonical = false
AbstractImplicitSolver
AbstractLinearOperator
```

### Picard solver

[`Picard`](@ref) solves the implicit system by a sequence of linear solves with lagged coefficients (Picard iteration [CeliaEtAl1990](@cite)); with a single iteration, it is the linearly implicit scheme [KnollKeyes2004](@cite) common in land surface models. The default is two iterations (`Picard(NF; iterations = 2)`). For the operator of a single variable, its solver-level `implicit_step!` runs a fixed number of iterations, each of which calls the process hooks [`linearize!`](@ref), [`solve!`](@ref), [`implicit_update!`](@ref), and [`implicit_closure!`](@ref) of the variable (see [Enabling implicit timestepping](@ref)). For a [`SplitOperator`](@ref), it runs these iterations for each component in turn. Since each variable forms its explicit stage as its current field plus the explicit increment when it is solved, each component after the first is stepped from the state left by the components solved before it, as in a sequential split: changes that their closures make to its prognostic field (e.g. the heat moved with the water redistributed by the soil hydrology closure) are kept.

```@docs; canonical = false
Picard
```

### Operator splitting

A [`SplitOperator`](@ref) holds the operators of several variables in a `NamedTuple`, in the order in which they are solved. [`SoilModel`](@ref) and [`LandModel`](@ref) return a `SplitOperator` from the block query [`implicit_operator`](@ref)`(model, state, names)` that always solves `saturation_water_ice` before `internal_energy`, so that the heat step sees the saturation at the end of the step and advects heat with the water flux of the step, while the Richards step sees the temperature and liquid water fraction at its beginning (see [Coupling of water and energy](@ref imex-water-energy-coupling)). Operator splitting is therefore a choice of the model rather than a property of the timestepper; a model that solves its implicit variables together would return a coupled operator from the same query. `SplitOperator` is not exported.

```@docs; canonical = false
SplitOperator
```

### Tridiagonal operator

[`TridiagonalOperator`](@ref) is the linear operator of vertical diffusion and, optionally, first-order upwind advection. Its systems are solved with Oceananigans' `BatchedTridiagonalSolver`, which solves one system per column and runs on CPU and GPU. The operator owns the batched solver and the lagged coefficient fields, `diagonal_coefficient` at cell centers and `face_conductivity` at cell faces, which are filled by [`linearize!`](@ref). The coefficients of the solver are looked up through `Oceananigans.Solvers.get_coefficient`, for which Terrarium defines methods on three marker structs, [`ImplicitDiffusionLowerDiagonal`](@ref), [`ImplicitDiffusionDiagonal`](@ref), and [`ImplicitDiffusionUpperDiagonal`](@ref). For a diffusion operator with face conductivity ``\kappa`` and cell-centered capacity ``C``, these evaluate to the coefficients of
```math
C_k\, T_k - \frac{\Delta t}{\Delta z_k}\left[ \frac{\kappa_{k+1/2}}{\Delta z_{k+1/2}} (T_{k+1} - T_k) - \frac{\kappa_{k-1/2}}{\Delta z_{k-1/2}} (T_k - T_{k-1}) \right]\,,
```
with no flux through the column boundaries (boundary fluxes are treated explicitly).

If the operator has a `face_velocity` ``u`` (allocated with `TridiagonalOperator(grid; advection = true)`), the row also contains the divergence ``\Delta t\, (F_{k+1/2} - F_{k-1/2}) / \Delta z_k`` of the upwind advective flux ``F = u\, T_\text{up}``, where ``T_\text{up}`` is the value of the cell below the face for ``u > 0`` and of the cell above it otherwise. The velocity is given in units of the capacity times a velocity, e.g. ``u = C_\text{w} q`` for the heat carried by the soil water flux ``q``. The advective terms add the outflow of each cell to the main diagonal and its inflow to the off-diagonals with a negative sign, so the system remains an M-matrix for any ``u`` and the solution obeys a discrete maximum principle. Through the boundary faces, the outflow, whose upwind value is that of the boundary cell itself, is part of the operator, while the inflow must enter through the right-hand side. Without advection, `face_velocity` is `nothing` and the advective terms vanish by dispatch.

An operator may in addition treat a linear term ``L_k T_k`` implicitly (e.g. a linear sink or a linearized boundary flux), which adds ``-\Delta t\, L_k`` to the main diagonal. Its coefficient is given by [`compute_implicit_linear_coefficient`](@ref), which an operator specializes via the trailing arguments it passes to [`solve!`](@ref); by default it is zero.

```@docs; canonical = false
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
