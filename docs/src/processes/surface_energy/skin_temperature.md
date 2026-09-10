# Skin temperature and ground heat flux

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

The **skin temperature** $T_s$ is the effective radiative temperature of the land surface. It represents the temperature at which the surface emits longwave radiation and is the temperature that directly controls evaporation and sensible heat flux in the surface energy balance.

Skin temperature differs from the air temperature at height $z$ by the effects of turbulent mixing and surface properties. It is also not necessarily the same as the subsurface ground temperature $T_g$ (the temperature of the uppermost soil or snow layer) due to the thermal resistance between the surface and the ground.

The skin temperature $T_s$ is determined by balancing the energy arriving at the skin from below (the ground heat flux $G$) with the radiative and turbulent losses above,
```math
R_{\text{net}}(T_s) + H_s(T_s) + H_l(T_s) = G(T_s, T_g)
```
where $R_{\text{net}}$ is the net radiation budget, $H_s$ is the sensible heat flux, and $H_l$ is the latent heat flux from sublimation and evapotranspiration.

## Implicit skin temperature

```@docs; canonical = false
ImplicitSkinTemperature
```

Skin temperature can be determined instantaneously at each time step by finding the roots of the above nonlinear energy balance equation at each grid point. Given a trial skin temperature $T_s$, the **demanded flux** required to close the energy balance is
```math
G^\star = R_{\text{net}}(T_s) + H_s(T_s) + H_l(T_s)
```
This demanded flux is equated to the area-weighted sum of two *unblended* conductive targets: the flux $G$ between the skin and the snow-free ground surface (uppermost soil layer) at temperature $T_g$, across the half-cell of thickness $\Delta z_1 / 2$,
```math
G(T_s, T_g) = \frac{2 \kappa_s}{\Delta z_1} (T_g - T_s)
```
and, over the snow-covered fraction $f_{\text{snow}}$, the flux $S$ between the skin and the top of the snowpack at temperature $T_{\text{snow}}$, across its (floored) depth $d_{\text{snow}}$,
```math
S(T_s, T_{\text{snow}}) = \frac{2 \kappa_{\text{snow}}}{d_{\text{snow}}} (T_{\text{snow}} - T_s)
```
Keeping $G$ and $S$ unblended (rather than averaging the ground and snow conduction targets into a single cell-wide target) prevents the flux delivered to a thin or patchy snowpack from being diluted by the bare-ground share. Setting the demanded flux equal to the area-weighted conductive fluxes and solving for $T_s$ (both conductive terms are affine in $T_s$, so this is an exact linear solve) yields
```math
T_s^\star = \frac{(1 - f_{\text{snow}}) \frac{2\kappa_s}{\Delta z_1} T_g + f_{\text{snow}} \frac{2\kappa_{\text{snow}}}{d_{\text{snow}}} T_{\text{snow}} - G^\star}{(1 - f_{\text{snow}}) \frac{2\kappa_s}{\Delta z_1} + f_{\text{snow}} \frac{2\kappa_{\text{snow}}}{d_{\text{snow}}}}
```
which reduces to $T_s^\star = T_g - G^\star \Delta z_1 / (2\kappa_s)$ without snow ($f_{\text{snow}} = 0$). The residual driving the nonlinear solve is the difference between the current and implied skin temperatures,
```math
r(T_s) = T_s - T_s^\star
```
The root $r(T_s) = 0$ is the skin temperature at which the conductive fluxes balance the radiative and turbulent fluxes, i.e. the solution of the surface energy balance. Note that the *stored* `ground_heat_flux` field is not $G^\star$ but the explicit conductive flux $G(T_s, T_g)$ evaluated at the converged $T_s$ — the two coincide only at convergence (see [`compute_ground_heat_flux`](@ref)).

The solve is performed by [`solve_skin_temperature!`](@ref), which wraps [`compute_skin_temperature_residual!`](@ref) in an [`ObjectiveFunction`](@ref) targeting the `skin_temperature` field and hands it to the configured solver. The default solver, [`default_skin_temperature_solver`](@ref), is a Newton-Raphson ([`RootSolver`](@ref) provided by [RootSolvers.jl](https://github.com/CliMA/RootSolvers.jl)) with a fixed iteration budget of $n = 5$ to balance accuracy with GPU efficiency. The prognostic `skin_temperature` is seeded with the current `ground_temperature` by [`initialize!`](@ref) so that the iteration starts from a physically sensible guess close to the root. After the solve converges, the surface energy fluxes are recomputed from the final skin temperature.

## Prescribed skin temperature

```@docs; canonical = false
PrescribedSkinTemperature
```

When the skin temperature is prescribed, it is supplied directly as the `skin_temperature` input field (for example by an external coupler) and no nonlinear solve is performed: [`solve_skin_temperature!`](@ref) is a no-op for [`PrescribedSkinTemperature`](@ref). The fused surface-energy-balance kernel then evaluates the radiative and turbulent fluxes from the prescribed skin temperature and closes the ground heat flux as the residual
```math
G = R_{\text{net}} + H_s + H_l
```
This configuration can be used to defer the surface energy balance to an external solver, often in tandem with [`PrescribedTurbulentFluxes`](@ref) when the turbulent fluxes are also supplied as inputs (see [Surface energy balance](@ref surface_energy_balance_docs)).

## Process interface

```@docs; canonical = false
initialize!(state, grid, ::ImplicitSkinTemperature, args...)
compute_auxiliary!(state, grid, skinT::ImplicitSkinTemperature, seb::AbstractSurfaceEnergyBalance, args...)
```

## Methods

```@docs; canonical = false
compute_ground_heat_flux_demand(::AbstractSkinTemperature, R_net, H_s, H_l)
compute_ground_heat_flux!(state, grid, skinT::AbstractSkinTemperature, seb::AbstractSurfaceEnergyBalance)
default_skin_temperature_solver
```

## Kernel functions

```@docs; canonical = false
compute_ground_heat_flux_demand(i, j, grid, fields, skinT::AbstractSkinTemperature, ::AbstractSurfaceEnergyBalance)
compute_ground_heat_flux(i, j, grid, fields, skinT::PrescribedSkinTemperature, seb::AbstractSurfaceEnergyBalance)
compute_ground_heat_flux(i, j, grid, fields, skinT::ImplicitSkinTemperature, ::AbstractSurfaceEnergyBalance)
compute_ground_heat_flux!(out, i, j, grid, fields, skinT::AbstractSkinTemperature, seb::AbstractSurfaceEnergyBalance)
compute_skin_temperature(i, j, grid, fields, skinT::ImplicitSkinTemperature{NF}, args...) where {NF}
compute_skin_temperature(i, j, grid, fields, skinT::ImplicitSkinTemperature{NF}, constants::PhysicalConstants, snow::AbstractSnow) where {NF}
compute_skin_temperature_residual!
solve_skin_temperature!
```
