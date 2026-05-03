# Skin temperature and ground heat flux

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

The **skin temperature** (sometimes also called *surface temperature*) $T_s$ is the effective radiative temperature of the land surface. It represents the temperature at which the surface emits longwave radiation and is the temperature that directly controls evaporation and sensible heat flux in the surface energy balance.

Skin temperature differs from the air temperature at height $z$ by the effects of turbulent mixing and surface properties. It is not necessarily the same as subsurface soil temperature, $T_g$ (the temperature of the top soil layer), due to the thermal resistance between surface and soil.

## Implicit skin temperature

```@docs; canonical = false
ImplicitSkinTemperature
```

The implicit approach requires solving a nonlinear equation at each grid point and time step. In order to avoid iteration, the current approach implementation of [`ImplicitSkinTemperature`](@ref) in Terrarium approximately solves for the skin temperature using a fixed point iteration where the ground heat flux is computed as the residual energy flux given the current prognostic state of the `skin_temperature` $T_s$,
```math
G^\star = R_{\text{net}}(T_s) + H_s(T_s) + H_l(T_s)
```
and the new skin temperature $T_s^\star$ is determined by setting this heat flux equal to gradient between the skin and the ground surface temperature (uppermost soil layer) $T_g$,
```math
T_s^\star = T_g - G^\star * \frac{\Delta z}{2 \kappa_s}
```
This procedure can be iterated until convergence; however, the current implementation simply performs one iteration to shift the skin temperature towards its root. This incurs some numerical approximation error in the calculation of the surface energy flux which still needs to be quantified. More accurate and stable schemes will be implemented in the future.

## Prescribed skin temperature

```@docs; canonical = false
PrescribedSkinTemperature
```

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, skinT::ImplicitSkinTemperature, seb::AbstractSurfaceEnergyBalance, args...)
```

## Methods

```@docs; canonical = false
update_skin_temperature!
```

```@docs; canonical = false
compute_skin_temperature
```

```@docs; canonical = false
compute_ground_heat_flux!
```

```@docs; canonical = false
compute_ground_heat_flux
```
