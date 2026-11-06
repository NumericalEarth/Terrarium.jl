# Autotrophic respiration

```@meta
CurrentModule = Terrarium
```


```@setup vegresp
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Gross primary production ($\text{GPP}$) provides the carbon input to the system. A fraction of this carbon is released back to the atmosphere through autotrophic respiration $R_a$, which is represented as the sum of two terms: maintenance respiration $R_m$ (the carbon cost of maintaining existing plant tissues) and growth respiration $R_g$ (the carbon cost of producing new plant tissues).


```math
\begin{equation}
R_a = R_m + R_g
\end{equation}
```

The remaining carbon is the net primary production ($\text{NPP}$), which is then allocated to the different plant components.

```math
\begin{equation}
\text{NPP} = \text{GPP} - R_a
\end{equation}
```

```@docs; canonical = false
AbstractAutotrophicRespiration
```

```@example vegresp
subtypes(Terrarium.AbstractAutotrophicRespiration)
```

## PALADYN autotrophic respiration model

```@docs; canonical = false
PALADYNAutotrophicRespiration
```

```@example vegresp
variables(PALADYNAutotrophicRespiration(Float32))
```

This implementation follows the autotrophic respiration scheme of PALADYN [willeitPALADYNV10Comprehensive2016](@cite).

### Maintenance respiration

Maintenance respiration is computed as the sum of leaf, stem, and root respiration

```math
\begin{equation}
R_m = R_{\text{leaf}} + R_{\text{stem}} + R_{\text{root}}
\end{equation}
```

where $R_{\text{leaf}}$ is the dark respiration computed by the photosynthesis scheme, and $R_{\text{stem}}$ and $R_{\text{root}}$ are each computed from the corresponding tissue carbon pool and their assigned C:N ratios,

```math
\begin{equation}
R_{\text{stem}} = r_{10} \, f(T_{\text{air}}) \, \frac{C_{\text{stem}}}{a_{ws} \, \text{CN}_{\text{sapwood}}}, \qquad
R_{\text{root}} = r_{10} \, f(T_{\text{soil}}) \, \phi \, \frac{C_{\text{root}}}{\text{CN}_{\text{root}}}
\end{equation}
```

where $r_{10}$ (`resp10`) is the tissue respiration rate at 10 °C per unit nitrogen (kgC/kgN/day, converted to per second), $a_{ws}$ is the ratio of total to respiring stem carbon, $\phi$ is the phenology factor, and $f(T)$ is the temperature factor.
All fluxes are in kgC/m²/s.

The temperature factor is a modified Arrhenius equation [lloydTemperatureDependenceSoil1994](@cite),

```math
\begin{equation}
f(T) = \exp\left(308.56 \left(\frac{1}{56.02} - \frac{1}{T + 46.02}\right)\right),
\end{equation}
```

where $T$ is the air or soil temperature in °C; $f(10\,°\text{C}) = 1$.

The vegetation carbon $C_{\text{veg}}$ is a single prognostic pool.
The tissue pools are diagnosed from it with the allometry of [willeitPALADYNV10Comprehensive2016](@cite) (see [`compute_leaf_carbon`](@ref), [`compute_root_carbon`](@ref), and [`compute_stem_carbon`](@ref)), with the balanced leaf area index $\text{LAI}_b = C_{\text{veg}} / (2/\text{SLA} + a_{wl})$,

```math
\begin{equation}
C_{\text{leaf}} = C_{\text{root}} = \frac{\text{LAI}_b}{\text{SLA}}, \qquad
C_{\text{stem}} = a_{wl} \, \text{LAI}_b
\end{equation}
```

so that $C_{\text{leaf}} + C_{\text{root}} + C_{\text{stem}} = C_{\text{veg}}$, and stem and root respiration are proportional to $C_{\text{veg}}$.

### Growth respiration

Growth respiration is taken as a fixed fraction of the carbon available after maintenance respiration

```math
\begin{equation}
R_g = 0.25 \cdot (\text{GPP} - R_m)
\end{equation}
```

## Methods

```@docs; canonical = false
compute_f_temp
```

```@docs; canonical = false
compute_resp10
```

```@docs; canonical = false
compute_Rm
```

```@docs; canonical = false
compute_Rg
```

```@docs; canonical = false
compute_Ra
```

```@docs; canonical = false
compute_NPP
```

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, autoresp::PALADYNAutotrophicRespiration, vegcarbon::AbstractVegetationCarbonDynamics, phenology::AbstractPhenology, traits::PlantTraits, atmos::AbstractAtmosphere)
```

## Kernel functions

```@docs; canonical = false
compute_autotrophic_respiration
```

```@docs; canonical = false
compute_autotrophic_respiration!
```

## [References](@id "autotrophic_respiration.refs")

```@bibliography
Pages = ["autotrophic_respiration.md"]
Canonical = false
```

