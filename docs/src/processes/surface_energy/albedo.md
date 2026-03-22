# Albedo and emissivity

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Surface albedo

Albedo is the fraction of incident shortwave radiation reflected back to space:

```math
\begin{equation}
\alpha = \frac{S_{\uparrow}}{S_{\downarrow}}
\end{equation}
```

where $S_{\uparrow}$ is the reflected (upwelling) shortwave and $S_{\downarrow}$ is the incident (downwelling) shortwave. Albedo ranges from 0 (perfectly absorbing) to 1 (perfectly reflecting).

The absorbed shortwave radiation is (in the upward-positive convention):

```math
\begin{equation}
\text{Absorbed SW} = -(1 - \alpha) S_{\downarrow}
\end{equation}
```

The negative sign indicates that downwelling radiation (negative in upward-positive convention) is absorbed and contributes to heating the surface.

### Albedo variability

Albedo depends on multiple factors:
- **Surface type**: Snow (0.7–0.9, highly reflective), vegetation (0.1–0.3), bare soil (0.2–0.4), water (0.05–0.15)
- **Soil moisture**: Darker, wetter soils have lower albedo
- **Vegetation density**: Denser vegetation lowers albedo
- **Solar zenith angle**: Lower sun angles increase albedo due to geometric effects
- **Snow age**: Fresh snow is bright; aging snow darkens with impurities

### Surface emissivity

Emissivity is the efficiency with which the surface emits thermal (longwave) radiation:

```math
\begin{equation}
\epsilon = \frac{L_{\uparrow}}{\sigma T_0^4}
\end{equation}
```

where $L_{\uparrow}$ is the total upwelling longwave radiation (including both surface emission and reflection of incident longwave), and $\sigma T_0^4$ is the blackbody emission at skin temperature $T_0$.

Most natural surfaces are nearly black in the longwave (emissivity 0.9–0.99), but dry sand and snow may have slightly lower values. In the energy balance, lower emissivity reduces cooling via longwave emission and also increases reflection of incident longwave radiation.

The total upwelling longwave includes both surface emission and reflection:
$$L_{\uparrow} = \epsilon \sigma T_0^4 + (1-\epsilon) L_{\downarrow}$$

## Abstract types

```@docs; canonical = false
AbstractAlbedo
```

## Concrete types

```@docs; canonical = false
ConstantAlbedo
```

```@docs; canonical = false
PrescribedAlbedo
```

## Methods

Albedo and emissivity are accessed through kernel functions at grid points:

```@docs; canonical = false
albedo
```

```@docs; canonical = false
emissivity
```

## Related processes

The albedo and thermal properties of the surface are influenced by:
- **Vegetation**: Leaf area, vegetation type (see [Vegetation Phenology](../vegetation/vegetation_phenology.md))
- **Soil moisture**: Saturation state (see [Soil Hydrology](../soil/soil_hydrology.md))
- **Snow cover**: Accumulation and melt (not yet implemented)
