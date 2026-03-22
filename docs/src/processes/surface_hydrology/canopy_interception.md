# Canopy interception

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Canopy water storage and interception

When precipitation falls on a forested canopy, a fraction is intercepted and stored as liquid water on leaves and stems. This intercepted water either evaporates directly back to the atmosphere or drips to the ground. The amount intercepted depends on vegetation density and structural properties (leaf area index and stem area index).

### Interception fraction

Following PALADYN (Willeit 2016), the fraction of precipitation intercepted is:

```math
\begin{equation}
I_{\text{can}} = \alpha_{\text{int}} P (1 - e^{-k_{\text{ext}}(L + S)})
\end{equation}
```

where:
- $\alpha_{\text{int}}$ is the interception factor (typically 0.2) [dimensionless]
- $P$ is incident precipitation [m/s]
- $k_{\text{ext}}$ is the radiation extinction coefficient (typically 0.5) [dimensionless]
- $L$ is the leaf area index (LAI) [m²/m²]
- $S$ is the stem area index (SAI) [m²/m²]

The term $(1 - e^{-k_{\text{ext}}(L+S)})$ represents the vegetation cover fraction and increases from 0 (bare ground) toward 1 (dense forest).

### Canopy water storage evolution

The canopy water budget is:

```math
\begin{equation}
\frac{\partial w_{\text{can}}}{\partial t} = I_{\text{can}} - E_{\text{can}} - R_{\text{can}}
\end{equation}
```

where:
- $w_{\text{can}}$ is liquid water stored on the canopy [kg/m²]
- $I_{\text{can}}$ is the interception rate [m/s]
- $E_{\text{can}}$ is evaporation of intercepted water [m/s]
- $R_{\text{can}}$ is the removal (drip) rate [m/s]

The canopy storage capacity is proportional to vegetation area:

```math
\begin{equation}
w_{\text{can, max}} = w_0 (L + S)
\end{equation}
```

where $w_0$ is the specific water capacity per unit leaf/stem area (typically 0.2 kg/m²).

### Canopy saturation and drip

The canopy water removal (drip to ground) follows a simple drainage model:

```math
\begin{equation}
R_{\text{can}} = \frac{w_{\text{can}}}{\rho_w \tau_w}
\end{equation}
```

where:
- $\rho_w$ is the density of water [kg/m³]
- $\tau_w$ is the canopy water removal timescale (typically 1 day = 86400 s) [s]

The saturation fraction of the canopy is:

```math
\begin{equation}
f_{\text{can}} = \frac{w_{\text{can}}}{w_{\text{can, max}}}
\end{equation}
```

This dimensionless saturation fraction ranges from 0 (dry) to 1 (saturated) and controls the efficiency of canopy evaporation.

### Precipitation reaching the ground

After accounting for interception and drip, the precipitation reaching the ground is:

```math
\begin{equation}
P_{\text{ground}} = P - I_{\text{can}} + R_{\text{can}} - E_{\text{can}}
\end{equation}
```

This represents the water available for soil infiltration and surface runoff.

## Abstract types

```@docs; canonical = false
AbstractCanopyInterception
```

## Concrete types

```@docs; canonical = false
PALADYNCanopyInterception
```

## Methods

```@docs; canonical = false
canopy_water
```

```@docs; canonical = false
saturation_canopy_water
```

```@docs; canonical = false
ground_precipitation
```

## Kernel functions

```@docs; canonical = false
compute_canopy_interception
```

```@docs; canonical = false
compute_canopy_saturation_fraction
```

```@docs; canonical = false
compute_canopy_water_removal
```

```@docs; canonical = false
compute_w_can_tendency
```

```@docs; canonical = false
compute_precip_ground
```
