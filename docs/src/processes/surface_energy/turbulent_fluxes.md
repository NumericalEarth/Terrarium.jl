# Turbulent fluxes

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Turbulent motion in the atmosphere transports heat away from the surface. Two primary mechanisms are involved: the **sensible heat flux** due to the temperature gradient between the atmosphere and land surface, and the **latent heat flux** (from evaporation, transpiration, and sublimation).

The surface energy budget partitioning strongly depends on the strength of these fluxes:
- Strong winds and atmospheric instability → large turbulent fluxes
- Calm conditions and stable boundary layer → weak turbulent fluxes

The coupling between the turbulent fluxes and atmospheric conditions are primarily captured through aerodynamic resistance (or equivalently conductance) terms that approximate the instantaneous resistance of the land surface to energy losses due to turbulent effects.

## Implementations

```@docs; canonical = false
PrescribedTurbulentFluxes
```

```@docs; canonical = false
DiagnosedTurbulentFluxes
```

### Sensible heat flux

Sensible heat is transported by the mean wind and turbulent eddies using bulk aerodynamic theory:

```math
\begin{equation}
H_s = c_a \rho_a \frac{\Delta T}{r_a}
\end{equation}
```

where $c_a$ is the specific heat capacity of moist air (J/kg/K) and $\rho_a$ is the moist air density (kg/m³), both computed dynamically based on the local atmosphere state using `Thermodynamics.jl`. $\Delta T = T_s - T_a$ is the temperature difference (K) (positive if surface warmer than air), and $r_a$ is the aerodynamic resistance (s/m). $H_s$ is **positive when surface is warmer than air** (heat flows upward), and **negative when surface is cooler** (heat flows downward).


### Latent heat flux

Evaporation and sublimation remove heat from the surface through the latent heat pathway. It is driven by the specific humidity difference $\Delta q$ (kg/kg), equivalent with the vapor pressure difference $\Delta e$ (Pa), between the surface and atmosphere.  Under the typical assumption that the land surface is saturated (see e.g. [zhouPhysicalBasisEp2024](@cite)), the vapor pressure difference is computed as:

```math
\begin{equation}
\Delta e = e_s - e_a = e_{\text{sat}}(T_s) - e_a(q_a, p)
\end{equation}
```

with $T_s$ the skin temperature. The specific humidity difference $\Delta q$ is 

```math
\begin{equation}
\Delta q = q_s - q_a = q_{\text{sat}}(T_s) - q_a
\end{equation}
```
with $q_{\text{sat}}$ the saturation specific humidity at the surface temperature and $q_a$ the specific humidity of the atmosphere. 

The latent heat flux is then computed as:

```math
\begin{equation}
H_l = L \rho_a \frac{\Delta q}{r_a}
\end{equation}
```

where $L$ is the latent heat of vaporization or sublimation (J/kg). $H_l$ is **always non-negative** (≥ 0) and represents energy lost due to evaporation, transpiration, or sublimation. Currently, condensation (dew formation) is neglected so $\Delta q \geq 0$ and negative latent heat fluxes cannot occur.

The latent heat flux is directly tied to:
- **Vegetation**: Transpiration through stomata (see [Photosynthesis](@ref))
- **Soil moisture**: Availability of water for evaporation (see [Soil hydrology](@ref))
- **Surface roughness**: Vegetation height affects aerodynamic properties

## Diagnosed turbulent fluxes

```@docs; canonical = false
DiagnosedTurbulentFluxes
```

## Prescribed turbulent fluxes

```@docs; canonical = false
PrescribedTurbulentFluxes
```

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, tur::DiagnosedTurbulentFluxes, seb::AbstractSurfaceEnergyBalance, constants::PhysicalConstants, atmos::AbstractAtmosphere, args...)
```

## Methods

```@docs; canonical = false
compute_sensible_heat_flux
```

```@docs; canonical = false
compute_latent_heat_flux
```

```@docs; canonical = false
compute_vapor_pressure_difference(i, j, grid, fields, atmos::AbstractAtmosphere, c::PhysicalConstants, Ts)
```

```@docs; canonical = false
compute_specific_humidity_difference(i, j, grid, fields, atmos::AbstractAtmosphere, c::PhysicalConstants, Ts)
```

## References

```@bibliography
Pages = ["turbulent_fluxes.md"]
Canonical = false
```