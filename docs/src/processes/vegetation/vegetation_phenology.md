# Vegetation phenology

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

!!! warning
    This process is currently not fully implemented.

## Theory

### Phenological processes

Phenology describes the seasonal timing of plant development stages: bud break (leaf-out), peak green vegetation cover, and leaf senescence (leaf-fall). These processes are driven by climate and vary significantly among plant functional types (PFTs).

Deciduous plants undergo rapid changes in leaf biomass with distinct leafless periods, while evergreen plants maintain relatively constant foliage year-round. The phenology factor $\phi$ (0 to 1) represents the fraction of maximum foliage present:
```math
\begin{equation}
\phi = f_{\text{deciduous}} \cdot \phi_{\text{season}} + (1 - f_{\text{deciduous}})
\end{equation}
```

where $f_{\text{deciduous}}$ (0 to 1) is a smooth transition parameter between deciduous and evergreen behavior, and $\phi_{\text{season}}$ is the seasonal phenology factor.

### Leaf area index modulation

The instantaneous leaf area index (LAI) is derived from the balanced LAI by applying the phenology factor:
```math
\begin{equation}
\text{LAI} = (f_{\text{deciduous}} \cdot \phi_{\text{season}} + (1 - f_{\text{deciduous}})) \cdot \text{LAI}_b
\end{equation}
```

This allows the effective canopy to vary seasonally while maintaining consistent carbon pools in the background.

## Abstract types

```@docs; canonical = false
AbstractPhenology
```

## Concrete types

```@docs; canonical = false
PALADYNPhenology
```

## Methods

```@docs; canonical = false
compute_f_deciduous
```

```@docs; canonical = false
compute_phenology_factor
```

```@docs; canonical = false
compute_LAI
```

## Kernel functions

```@docs; canonical = false
compute_phenology
```

```@docs; canonical = false
compute_phenology!
```
