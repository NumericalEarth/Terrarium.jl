# Radiative fluxes

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Radiative energy components

The net radiation at the surface is the sum of all radiative fluxes, with fluxes defined as **positive upward** (away from surface):

```math
\begin{equation}
R_{\text{net}} = S_{\uparrow} - S_{\downarrow} + L_{\uparrow} - L_{\downarrow}
\end{equation}
```

where:
- $S_{\uparrow} = \alpha S_{\downarrow}$ is upwelling (reflected) shortwave radiation [W/m²]
- $S_{\downarrow}$ is downwelling (incident) shortwave radiation [W/m²]
- $L_{\uparrow} = \epsilon \sigma T_0^4 + (1-\epsilon) L_{\downarrow}$ is upwelling longwave radiation from the surface and reflected downwelling longwave [W/m²]
- $L_{\downarrow}$ is downwelling (incident) longwave radiation [W/m²]
- $\epsilon$ is the surface emissivity (see [Albedo and Emissivity](albedo.md))
- $\alpha$ is the surface albedo (see [Albedo and Emissivity](albedo.md))
- $\sigma$ is the Stefan-Boltzmann constant
- $T_0$ is the skin temperature

This can be equivalently written in absorbed form as:

```math
\begin{equation}
R_{\text{net}} = -(1 - \alpha) S_{\downarrow} + \epsilon \sigma T_0^4 - \epsilon L_{\downarrow}
\end{equation}
```

where the first term represents absorbed solar radiation (negative because downwelling radiation contributes negatively when using the upward-positive convention).

### Shortwave radiation

Shortwave radiation is determined by the solar angle, atmospheric transmittance, and cloud cover—all of which are typically prescribed from forcing data or a coupled atmosphere model. The absorbed shortwave depends on surface albedo $(1 - \alpha)$, which may vary with surface type (soil, vegetation, snow) and solar zenith angle.

### Longwave radiation

The surface emits longwave radiation according to the Stefan-Boltzmann law, with the emission temperature determined by the skin temperature $T_0$. The downwelling longwave is controlled by atmospheric water vapor, cloud cover, and air temperature.

## Abstract types

```@docs; canonical = false
AbstractRadiativeFluxes
```

## Concrete types

```@docs; canonical = false
PrescribedRadiativeFluxes
```

```@docs; canonical = false
DiagnosedRadiativeFluxes
```

## Methods

```@docs; canonical = false
compute_shortwave_up
```

```@docs; canonical = false
compute_longwave_up
```

```@docs; canonical = false
compute_surface_net_radiation
```

## Kernel functions

```@docs; canonical = false
compute_surface_upwelling_radiation
```

```@docs; canonical = false
compute_radiative_fluxes!
```
