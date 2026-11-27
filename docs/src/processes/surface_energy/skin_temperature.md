# Skin temperature

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

The **skin temperature** $T_s$ is the effective radiative temperature of the land surface. It represents the temperature at which the surface emits longwave radiation and is the temperature that directly controls evaporation and sensible heat flux in the surface energy balance.

Skin temperature differs from the air temperature at height $z$ by the effects of turbulent mixing and surface properties. It is also not necessarily the same as the subsurface ground temperature $T_g$ (the temperature of the uppermost soil or snow layer) due to the thermal resistance between the surface and the ground.

The skin temperature $T_s$ is determined by balancing the energy arriving at the skin from below (the conductive heat flux $G$) with the radiative and turbulent losses above,
```math
R_{\text{net}}(T_s) + H(T_s) + LE(T_s) = G(T_s)
```
where $R_{\text{net}}$ is the net radiation budget, $H$ is the sensible heat flux, and $LE$ is the latent heat flux from evapotranspiration or sublimation (all positive upward).
On a snow-free surface (superscript $sf$), $G = G^{sf}$ conducts into the uppermost ground layer and $LE$ is the evapotranspiration latent flux; on a snow-covered surface (superscript $sn$), $G = G^{sn}$ conducts into the top of the snowpack and $LE$ is the sublimation latent flux.

## Implicit skin temperature

```@docs; canonical = false
ImplicitSkinTemperature
```

Skin temperature can be determined instantaneously at each time step by finding the roots of the above nonlinear energy balance equation at each grid point. Given a trial skin temperature $T_s$, the **demanded flux** required to close the energy balance is
```math
G_0 = R_{\text{net}}(T_s) + H(T_s) + LE(T_s)
```
Both conductive fluxes are affine in $T_s$, so each can be inverted exactly for the skin temperature that would deliver the demanded flux. On the snow-free surface, the flux between the skin and the uppermost ground layer at temperature $T_g$, across the half-cell of thickness $\Delta z_g / 2$, is
```math
G^{sf}(T_s) = \frac{2 \kappa_s}{\Delta z_g} (T_g - T_s), \qquad T_s^{sf\star} = T_g - \frac{G_0\, \Delta z_g}{2 \kappa_s}
```
and on the snow-covered surface, the flux between the skin and the top of the snowpack at temperature $T_{\text{snow}}$, across its (floored) depth $d_{\text{snow}}$, is
```math
G^{sn}(T_s) = \frac{2 \kappa_{\text{snow}}}{d_{\text{snow}}} (T_{\text{snow}} - T_s), \qquad T_s^{sn\star} = T_{\text{snow}} - \frac{G_0\, d_{\text{snow}}}{2 \kappa_{\text{snow}}}
```
These two inverses are computed by the methods of [`compute_skin_temperature`](@ref) for the [`SnowFree`](@ref) and [`SnowCovered`](@ref) markers. Without snow, $T_s^\star = T_s^{sf\star}$.

!!! note "Transitional single solve with snow"
    With a snow component, the surface energy balance is currently solved for a single grid-cell skin temperature, for which the two conduction relations are combined by area weighting with the snow-covered fraction $f_{\text{snow}}$, $(1 - f_{\text{snow}})\, G^{sf}(T_s) + f_{\text{snow}}\, G^{sn}(T_s) = G_0$. Since both fluxes are affine in $T_s$, the solution is the conductance-weighted mean of the two tile inverses,
    ```math
    T_s^\star = \frac{(1 - f_{\text{snow}}) \frac{2\kappa_s}{\Delta z_g}\, T_s^{sf\star} + f_{\text{snow}} \frac{2\kappa_{\text{snow}}}{d_{\text{snow}}}\, T_s^{sn\star}}{(1 - f_{\text{snow}}) \frac{2\kappa_s}{\Delta z_g} + f_{\text{snow}} \frac{2\kappa_{\text{snow}}}{d_{\text{snow}}}}
    ```
    which the grid-cell method of [`compute_skin_temperature`](@ref) forms from the two tile inverses. Keeping $G^{sf}$ and $G^{sn}$ unblended (rather than averaging the ground and snow conduction targets into a single cell-wide target) prevents the flux delivered to a thin or patchy snowpack from being diluted by the bare-ground share. The snow-free and snow-covered tiles are to be solved separately, each with its own skin temperature.

The residual driving the nonlinear solve is the difference between the current and implied skin temperatures,
```math
r(T_s) = T_s - T_s^\star
```
The root $r(T_s) = 0$ is the skin temperature at which the conductive fluxes balance the radiative and turbulent fluxes, i.e. the solution of the surface energy balance. Note that the *stored* `ground_heat_flux` field is not $G_0$ but the explicit conductive flux $G^{sf}(T_s)$ evaluated at the converged $T_s$; the two coincide only at convergence. Both the demand $G_0$ and the stored flux are owned by the [ground heat flux](@ref ground_heat_flux_docs) sub-process.

The solve is performed by [`solve_skin_temperature!`](@ref), which wraps [`compute_skin_temperature_residual!`](@ref) in an [`ObjectiveFunction`](@ref) targeting the `skin_temperature` field and hands it to the configured solver. The default solver, [`default_skin_temperature_solver`](@ref), is a Newton-Raphson ([`RootSolver`](@ref) provided by [RootSolvers.jl](https://github.com/CliMA/RootSolvers.jl)) with a fixed iteration budget of $n = 5$ to balance accuracy with GPU efficiency. The auxiliary `skin_temperature` is seeded with the current `ground_temperature` by [`initialize!`](@ref) so that the iteration starts from a physically sensible guess close to the root; since the field persists across time steps, every later solve starts from the previous solution. After the solve converges, the surface energy fluxes are recomputed from the final skin temperature.

## Prescribed skin temperature

```@docs; canonical = false
PrescribedSkinTemperature
```

When the skin temperature is prescribed, it is supplied directly as the `skin_temperature` input field (for example by an external coupler) and no nonlinear solve is performed: [`solve_skin_temperature!`](@ref) is a no-op for [`PrescribedSkinTemperature`](@ref). The fused surface-energy-balance kernel then evaluates the radiative and turbulent fluxes from the prescribed skin temperature. Paired with [`DiagnosedGroundHeatFlux`](@ref), the ground heat flux is closed as the residual
```math
G = R_{\text{net}} + H + LE
```
This configuration can be used to defer the surface energy balance to an external solver, often in tandem with [`PrescribedTurbulentFluxes`](@ref) when the turbulent fluxes are also supplied as inputs (see [Surface energy balance](@ref surface_energy_balance_docs)).

## Process interface

```@docs; canonical = false
initialize!(state, grid, ::ImplicitSkinTemperature, args...)
```

## Methods

```@docs; canonical = false
skin_temperature(i, j, grid, fields, ::AbstractSkinTemperature)
ground_thermal_interface
snow_thermal_interface
default_skin_temperature_solver
```

## Kernel functions

```@docs; canonical = false
compute_skin_temperature(i, j, grid, fields, skinT::ImplicitSkinTemperature{NF}, seb::AbstractSurfaceEnergyBalance, constants::PhysicalConstants, snow::AbstractSnow) where {NF}
compute_skin_temperature(i, j, grid, fields, skinT::ImplicitSkinTemperature, seb::AbstractSurfaceEnergyBalance, constants::PhysicalConstants, ::Nothing)
compute_skin_temperature(i, j, grid, fields, ::SnowFree, skinT::ImplicitSkinTemperature{NF}, seb::AbstractSurfaceEnergyBalance, constants::PhysicalConstants, snow::Optional{AbstractSnow}) where {NF}
compute_skin_temperature(i, j, grid, fields, ::SnowCovered, skinT::ImplicitSkinTemperature{NF}, seb::AbstractSurfaceEnergyBalance, constants::PhysicalConstants, snow::AbstractSnow) where {NF}
compute_skin_temperature_residual!
solve_skin_temperature!
```
