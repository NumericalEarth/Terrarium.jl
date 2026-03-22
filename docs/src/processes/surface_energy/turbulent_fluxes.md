# Turbulent fluxes

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Turbulent heat transport fundamentals

Turbulent motion in the atmosphere transports heat away from the surface. Two primary mechanisms are involved: the **sensible heat flux** due to the temperature gradient between the atmosphere and land surface, and the **latent heat flux** (from evaporation, transpiration, and sublimation). All turbulent fluxes are **positive upward** (away from the surface, toward the atmosphere).

The surface energy budget partitioning strongly depends on the strength of these fluxes:
- Strong winds and atmospheric instability → large turbulent fluxes, cooler surface
- Calm conditions and stable layers → weak turbulent fluxes, warmer surface

### Sensible heat flux

Sensible heat is transported by the mean wind and turbulent eddies using bulk aerodynamic theory:

```math
\begin{equation}
H_s = c_a \rho_a \frac{\Delta T}{r_a}
\end{equation}
```

where:
- $c_a$ is the specific heat capacity of air [J/kg/K]
- $\rho_a$ is the air density [kg/m³]
- $\Delta T = T_s - T_a$ is the temperature difference (positive if surface warmer than air) [K]
- $r_a$ is the aerodynamic resistance [s/m]
- $H_s$ is **positive when surface is warmer than air** (heat flows upward), and **negative when surface is cooler** (heat flows downward)


### Latent heat flux

Evaporation and sublimation remove heat from the surface through the latent heat pathway:

```math
\begin{equation}
H_l = L \rho_a \frac{\Delta q}{r_a}
\end{equation}
```

where:
- $L$ is the latent heat of vaporization or sublimation [J/kg]
- $\rho_a$ is the air density [kg/m³]
- $\Delta q = q_{\text{sat}}(T_s) - q_a$ is the specific humidity gradient [kg/kg] derived from the vapor pressure deficit which is always ≥ 0 by definition
- $r_a$ is the aerodynamic resistance [s/m]
- $H_l$ is **always non-negative** (≥ 0) and represents energy lost due to evaporation, transpiration, or sublimation. Currently condensation (dew formation) is neglected so negative latent heat fluxes cannot occur.

### Coupling to vegetation and soil

Latent heat flux is directly tied to:
- **Vegetation**: Transpiration through stomata (see [Photosynthesis and Gas Exchange](../vegetation/photosynthesis.md))
- **Soil moisture**: Availability of water for evaporation
- **Surface roughness**: Vegetation height affects aerodynamic properties

## Abstract types

```@docs; canonical = false
AbstractTurbulentFluxes
```

## Concrete types

```@docs; canonical = false
PrescribedTurbulentFluxes
```

```@docs; canonical = false
DiagnosedTurbulentFluxes
```

## Methods

### Sensible heat flux

```@docs; canonical = false
compute_sensible_heat_flux
```

### Latent heat flux

```@docs; canonical = false
compute_latent_heat_flux
```

## Kernel functions

