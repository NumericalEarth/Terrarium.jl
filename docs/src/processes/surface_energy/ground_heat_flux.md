# [Ground heat flux](@id ground_heat_flux_docs)

```@meta
CurrentModule = Terrarium
```

```@setup ghf
using Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

The **ground heat flux** $G$ (W/m²) is the energy flux across the top of the ground (soil) column. It is the term through which the surface energy balance drives the subsurface: without snow, the `ground_heat_flux` field *is* the Neumann boundary condition on the soil's internal energy (see [`SoilHeatFlux`](@ref)), and with snow it feeds the blended conductive flux across the snow base.

Following the standard convention of Terrarium and Oceananigans, all surface energy fluxes are defined **positive upward**. A positive $G$ therefore removes energy from the ground column.

The quantity that the rest of the surface energy balance *demands* of the ground is the residual of the remaining three flux terms,
```math
G^\star = R_{\text{net}} + H_s + H_l
```
where $R_{\text{net}}$ is the net radiation budget, $H_s$ is the sensible heat flux, and $H_l$ is the latent heat flux. How that demand is realized, and whether it is imposed at all, is the responsibility of the [`AbstractGroundHeatFlux`](@ref) sub-process.

```@docs; canonical = false
AbstractGroundHeatFlux
```

## Implementations

### Diagnosed ground heat flux

```@docs; canonical = false
DiagnosedGroundHeatFlux
```

[`DiagnosedGroundHeatFlux`](@ref) closes the surface energy balance internally. What is stored depends on the accompanying skin temperature scheme:

- With [`PrescribedSkinTemperature`](@ref), there is no separate conduction target, so the stored flux *is* the demand, $G = G^\star = R_{\text{net}} + H_s + H_l$.
- With [`ImplicitSkinTemperature`](@ref), the stored flux is the explicit bare-ground conductive flux evaluated at the current skin temperature, $G = 2\kappa_s (T_g - T_s) / \Delta z_1$. This coincides with the demand only at convergence of the skin temperature solve (see [Skin temperature](@ref "Skin temperature")).

```@example ghf
variables(DiagnosedGroundHeatFlux(Float32))
```

### Prescribed ground heat flux

```@docs; canonical = false
PrescribedGroundHeatFlux
```

[`PrescribedGroundHeatFlux`](@ref) declares `ground_heat_flux` as an *input* variable and computes nothing. Whatever is written into the field, typically by an external coupler that owns the atmosphere-land interface, survives untouched through `compute_auxiliary!` and reaches the ground boundary condition. This is the configuration that makes [`PrescribedSurfaceEnergyBalance`](@ref) possible; see [Surface energy balance](@ref surface_energy_balance_docs) for the supported combinations.

```@example ghf
variables(PrescribedGroundHeatFlux(Float32))
```

!!! warning "Over-determination"
    Nothing enforces $G = R_{\text{net}} + H_s + H_l$ when the ground heat flux is prescribed alongside prescribed radiative and turbulent fluxes. It is the caller's responsibility to supply a mutually consistent set; Terrarium will happily integrate an inconsistent one.

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, ghf::AbstractGroundHeatFlux, seb::AbstractSurfaceEnergyBalance, args...)
```

## Methods

```@docs; canonical = false
ground_heat_flux(i, j, grid, fields, ::AbstractGroundHeatFlux)
compute_ground_heat_flux_demand(::DiagnosedGroundHeatFlux, R_net, H_s, H_l)
compute_ground_heat_flux!(state, grid, ghf::AbstractGroundHeatFlux, skinT::AbstractSkinTemperature, seb::AbstractSurfaceEnergyBalance)
compute_ground_heat_flux!(state, grid, ::PrescribedGroundHeatFlux, args...)
```

## Kernel functions

```@docs; canonical = false
compute_ground_heat_flux_demand(i, j, grid, fields, ghf::DiagnosedGroundHeatFlux)
compute_ground_heat_flux_demand(i, j, grid, fields, ::PrescribedGroundHeatFlux)
compute_ground_heat_flux(i, j, grid, fields, ghf::DiagnosedGroundHeatFlux, ::PrescribedSkinTemperature, ::AbstractSurfaceEnergyBalance)
compute_ground_heat_flux(i, j, grid, fields, ::DiagnosedGroundHeatFlux, skinT::ImplicitSkinTemperature, ::AbstractSurfaceEnergyBalance)
compute_ground_heat_flux!(out, i, j, grid, fields, ghf::DiagnosedGroundHeatFlux, skinT::AbstractSkinTemperature, seb::AbstractSurfaceEnergyBalance)
compute_ground_heat_flux!(out, i, j, grid, fields, ::PrescribedGroundHeatFlux, ::AbstractSkinTemperature, ::AbstractSurfaceEnergyBalance)
```
