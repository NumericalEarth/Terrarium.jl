# Evapotranspiration

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Evapotranspiration fundamentals

Evapotranspiration (ET) is the combined process of water evaporation from soil and open water surfaces, evaporation of intercepted water from vegetation, and transpiration through plant stomata. These processes remove water from the surface, driving latent heat flux and competing with sensible heat and ground heat fluxes in the surface energy balance.

All evapotranspiration pathways are driven by the **vapor pressure deficit** or **specific humidity deficit** $\Delta q = q_{\text{sat}}(T_s) - q_a$, where:
- $q_{\text{sat}}(T_s)$ is the saturation specific humidity at surface temperature $T_s$ [kg/kg]
- $q_a$ is the atmospheric specific humidity at reference height [kg/kg]

Each pathway is also modulated by **aerodynamic resistance** $r_a$ (between surface and atmosphere) and/or **stomatal resistance** $r_s$ (in transpiration).

### Canopy evaporation

Evaporation of water intercepted by the canopy depends on the saturation state of the canopy (fraction of leaves wet):

```math
\begin{equation}
E_{\text{can}} = f_{\text{can}} \frac{\Delta q}{r_a}
\end{equation}
```

where:
- $f_{\text{can}}$ is the canopy saturation fraction (0 = dry, 1 = saturated) [dimensionless]
- $\Delta q$ is the vapor pressure deficit [kg/kg]
- $r_a$ is the aerodynamic resistance [s/m]

When $f_{\text{can}} = 0$ (completely dry canopy), $E_{\text{can}} = 0$. When $f_{\text{can}} = 1$ (wet canopy), evaporation proceeds at the potential rate.

### Ground evaporation

Evaporation from exposed soil or under-canopy surfaces is limited by soil water availability:

```math
\begin{equation}
E_{\text{ground}} = \beta \frac{\Delta q}{r_a + r_e}
\end{equation}
```

where:
- $\beta$ is the ground evaporation resistance factor (0 to 1) [dimensionless]
- $r_a$ is the aerodynamic resistance [s/m]
- $r_e$ is the aerodynamic resistance between ground and canopy [s/m]

The resistance factor $\beta$ is computed from soil moisture in the upper layer: $\beta = 1$ when soil is wet (at field capacity) and $\beta \to 0$ as soil dries.

### Transpiration

Plant transpiration occurs through stomata and is controlled by stomatal conductance:

```math
\begin{equation}
T_{\text{canopy}} = \frac{\Delta q}{r_a + r_s}
\end{equation}
```

where:
- $r_s = 1 / g_w$ is the stomatal resistance [s/m]
- $g_w$ is the stomatal conductance (computed from photosynthesis; see [Photosynthesis and Gas Exchange](../vegetation/photosynthesis.md)) [m/s]
- $r_a$ is the aerodynamic resistance [s/m]

High stomatal conductance (when photosynthetically active) leads to low stomatal resistance and high transpiration. This creates a strong coupling between carbon uptake (photosynthesis) and water loss (transpiration).

### Total evapotranspiration

The PALADYN implementation combines all three pathways:

```math
\begin{equation}
\text{ET} = E_{\text{can}} + E_{\text{ground}} + T_{\text{canopy}}
\end{equation}
```

The latent heat flux in the energy balance (see [Turbulent Fluxes](../surface_energy/turbulent_fluxes.md)) is then:

```math
\begin{equation}
H_l = L \rho_a \text{ET}
\end{equation}
```

where $L$ is the latent heat of vaporization and $\rho_a$ is air density.

## Abstract types

```@docs; canonical = false
AbstractEvapotranspiration
```

```@docs; canonical = false
AbstractGroundEvaporationResistanceFactor
```

## Concrete types

### Evapotranspiration schemes

```@docs; canonical = false
PALADYNCanopyEvapotranspiration
```

```@docs; canonical = false
GroundEvaporation
```

### Ground resistance parameterizations

```@docs; canonical = false
ConstantEvaporationResistanceFactor
```

```@docs; canonical = false
SoilMoistureResistanceFactor
```

## Methods

```@docs; canonical = false
surface_humidity_flux
```

## Kernel functions

### Canopy evapotranspiration

```@docs; canonical = false
compute_transpiration
```

```@docs; canonical = false
compute_evaporation_ground
```

```@docs; canonical = false
compute_evaporation_canopy
```

### Ground evaporation

```@docs; canonical = false
ground_evaporation_resistance_factor
```
