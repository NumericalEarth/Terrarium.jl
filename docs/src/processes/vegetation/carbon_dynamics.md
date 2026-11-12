# Vegetation carbon dynamics

```@meta
CurrentModule = Terrarium
```

```@setup vegcarbon
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Gross primary productivity ($\text{GPP}$) is the total carbon uptake by plants through photosynthesis. A fraction of this carbon is lost to the atmosphere through autotrophic respiration. The remainder is net primary productivity ($\text{NPP}$), which is then partitioned through carbon allocation among the different plant compartments, such as leaves, stems, and roots (often referred to as carbon pools). At the same time, carbon is removed from these pools through turnover processes such as litterfall, mortality, and disturbances.

```@docs; canonical = false
AbstractVegetationCarbonDynamics
```

```@example vegcarbon
subtypes(Terrarium.AbstractVegetationCarbonDynamics)
```

## PALADYN carbon dynamics

```@docs; canonical = false
PALADYNCarbonDynamics
```

```@example vegcarbon
variables(PALADYNCarbonDynamics(Float32))
```

This implementation is based on PALADYN [willeitPALADYNV10Comprehensive2016](@cite), in which the temporal evolution of the total vegetation carbon pool $C_{\text{veg}}$ for each PFT is described by an ordinary differential equation, where $C_{\text{veg}}$ is the single prognostic variable. The partitioning of $C_{\text{veg}}$ into the different carbon pools (leaf, stem, and root) is handled afterward through fixed allometric relationships (not yet implemented).

```math
\begin{equation}
\frac{d C_{\text{veg}}}{dt}
= \left(1-\lambda_{\text{NPP}}\right)\,\text{NPP}
- \Lambda_{\text{loc}}
\end{equation}
```

where $\lambda_{\text{NPP}}$ is a dimensionless factor ranging from 0 to 1 that determines how NPP is partitioned between the spatial expansion of a PFT and growth of that PFT over its existing vegetated area, and $\Lambda_{\text{loc}}$ is the local litterfall rate, computed as follows for evergreen PFTs:

```math
\begin{equation}
\Lambda_{\text{loc}} = \left(\frac{\gamma_L}{\text{SLA}} + \frac{\gamma_R}{\text{SLA}} + \gamma_S\,a_{wl}\right) \text{LAI}_b
\end{equation}
```

where $\gamma_L$, $\gamma_R$, and $\gamma_S$ are the leaf, root, and stem turnover rates respectively.

$\lambda_{\text{NPP}}$ is computed from the balanced leaf area index $\text{LAI}_b$ relative to the PFT-specific threshold values $\text{LAI}_{\text{min}}$ and $\text{LAI}_{\text{max}}$. Low values of $\text{LAI}_b$ favor allocation of NPP to local growth, while high values favor allocation to spatial expansion. Accordingly, $\lambda_{\text{NPP}}$ is set to 0 below the lower threshold, to 1 above the upper threshold, and varies linearly between the two thresholds.

```math
\begin{equation}
\lambda_{\text{NPP}} = \begin{cases}
0 & \text{if } \text{LAI}_b < \text{LAI}_{\min} \\
\frac{\text{LAI}_b - \text{LAI}_{\min}}{\text{LAI}_{\max} - \text{LAI}_{\min}} & \text{if } \text{LAI}_{\min} \leq \text{LAI}_b \leq \text{LAI}_{\max} \\
1 & \text{if } \text{LAI}_b > \text{LAI}_{\max}
\end{cases}
\end{equation}
```

where $\text{LAI}_b$ is the balanced leaf area index that would be reached if the plant were in full leaf. The actual LAI is computed by the phenology model (see [Phenology](@ref)).

The balanced leaf area index $\text{LAI}_b$ is related to the total vegetation carbon pool $C_{\text{veg}}$ through the following equation derived from allometric relationships:

```math
\begin{equation}
C_{\text{veg}} = 2 \frac{\text{LAI}_b}{\text{SLA}} + a_{wl}\,\text{LAI}_b^{b_{wl}}
\end{equation}
```

where $\text{SLA}$ is the specific leaf area, and $a_{wl}$ and $b_{wl}$ are PFT-dependent allometric coefficients.

In PALADYN, $b_{wl}=1$, which simplifies this relationship to

```math
\begin{equation}
C_{\text{veg}} = \left(\frac{2}{\text{SLA}} + a_{wl}\right) \text{LAI}_b
\end{equation}
```

Therefore, $\text{LAI}_b$ can be diagnosed from the total vegetation carbon pool $C_{\text{veg}}$ as follows

```math
\begin{equation}
\text{LAI}_b = \frac{C_{\text{veg}}}{\frac{2}{\text{SLA}} + a_{wl}}
\end{equation}
```


## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, vegcarbon_dynamics::PALADYNCarbonDynamics, traits::PlantTraits, args...)
```

```@docs; canonical = false
compute_tendencies!(state, grid, vegcarbon_dynamics::PALADYNCarbonDynamics, traits::PlantTraits, args...)
```

## Methods

```@docs; canonical = false
compute_balanced_leaf_area_index
```

```@docs; canonical = false
compute_leaf_carbon
```

```@docs; canonical = false
compute_root_carbon
```

```@docs; canonical = false
compute_stem_carbon
```

```@docs; canonical = false
compute_λ_NPP
```

```@docs; canonical = false
compute_Λ_loc
```

```@docs; canonical = false
compute_C_veg_tend
```

## Kernel functions

```@docs; canonical = false
compute_veg_carbon_tendency
```

```@docs; canonical = false
compute_veg_carbon_auxiliary!
```

```@docs; canonical = false
compute_veg_carbon_tendencies!
```

## [References](@id "carbon_dynamics.refs")

```@bibliography
Pages = ["carbon_dynamics.md"]
Canonical = false
```
