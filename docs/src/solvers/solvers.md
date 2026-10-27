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

## Interface

```@docs; canonical = false
solve!
```
