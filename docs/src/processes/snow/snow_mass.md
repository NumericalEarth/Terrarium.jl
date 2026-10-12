# Snow mass balance

```@meta
CurrentModule = Terrarium
```

```@setup snowmass
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

The snow water equivalent ``W_\text{snow}`` (m) evolves according to a continuous mass balance,
```math
\begin{equation}
\frac{\partial W_\text{snow}}{\partial t} = P_s + R_\text{on snow} - M_r - E_\text{subl},
\end{equation}
```
where ``P_s`` is snowfall, ``R_\text{on snow} = f_\text{snow}\,R`` is the rainfall intercepted by the snow-covered fraction, ``M_r`` is the meltwater outflow draining from the snowpack base, and ``E_\text{subl}`` is the surface sublimation rate. In the coupled land model the bare-ground fraction ``(1 - f_\text{snow})`` of the rainfall reaches the soil directly and the meltwater outflow is added to it, forming the water input to the [Surface runoff](@ref) scheme.

## Meltwater outflow

Liquid water in excess of the capillary retention drains from the snowpack with a Darcy-type cubic
conductivity ([tarbotonSpatiallyDistributedEnergy1994](@cite), eqns 23–24, in excess-saturation form).

```@docs; canonical = false
compute_meltwater_outflow

compute_meltwater_outflow_derivative

snow_meltwater_flux

compute_meltwater_flux!
```

## Implicit meltwater drainage

With the default saturated conductivity of 20 m/h, a fully melted pack drains its liquid water within seconds, so
an explicit step of useful length drains more water than the pack holds as liquid. Since meltwater leaves at the
zero-enthalpy reference of the closure (liquid water at 0 °C), the drained ice then leaves its latent deficit in the
remaining pack, which cools far below the air temperature. Under an [`IMEX`](@ref) timestepper, `snow_water_equivalent`
is therefore integrated implicitly (see [Implicit-explicit time stepping](@ref)), and the drainage is excluded from its
explicit tendency.

Drainage changes neither the snow energy ``\bar{U}`` nor the ice content ``W_i = (1 - \theta_\text{liq}) W = -\bar{U} / (\rho_w L_f)``.
With ``W^+`` and ``\bar{U}^+`` from the explicit stage (snowfall, rain, and sublimation), the backward Euler step of the
drainage is a scalar equation per column, written in the liquid fraction ``\theta`` via ``W = W_i / (1 - \theta)``:
```math
G(\theta) = \frac{W_i}{1 - \theta} + \Delta t\, M_r(\theta) - W^+ = 0.
```
``G`` is increasing and convex, so each [`Picard`](@ref) iteration takes a Newton step in ``\theta``, which approaches the
root from above: no iterate drains more than the backward Euler solution, at any step and any number of iterations.
The system of each iteration is solved by a [`DiagonalOperator`](@ref) for `snow_liquid_fraction`. A fully liquid pack
(``W_i = 0``) drains at the constant rate ``M_r(1)`` until it is empty, so its step takes the exact solution
``\max(W^+ - \Delta t\, M_r(1), 0)``.

The drainage of each step is stored in the auxiliary variable `meltwater_flux`, which the surface hydrology of a
[`LandModel`](@ref) reads as the meltwater input to the soil in the following step. Snow and soil water are thus
conserved, with the meltwater input lagged by one step. Under explicit integration, `meltwater_flux` is the outflow at
the beginning of the step.

```@docs; canonical = false
linearize!(operator::DiagonalOperator, rhs, state, grid, snow::SingleLayerSnow, constants::PhysicalConstants, explicit_water, Δt, ::Iteration{:snow_water_equivalent})

implicit_update!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants, operator::DiagonalOperator, explicit_water, Δt, ::Iteration{:snow_water_equivalent})

implicit_closure!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants, ::Iteration{:snow_water_equivalent})

compute_snow_ice_content

compute_implicit_meltwater_coefficients!

compute_implicit_snow_water_update!
```

## Tendencies

```@docs; canonical = false
compute_snow_water_tendency

compute_snow_tendencies!
```

## [References](@id "snowmass.refs")

```@bibliography
Pages = ["snow_mass.md"]
Canonical = false
```
