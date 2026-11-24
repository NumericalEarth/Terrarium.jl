# Vegetation dynamics 

```@meta
CurrentModule = Terrarium
```

```@setup vegdynamics
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Vegetation dynamics describes the temporal evolution of the fractional coverage of plant functional types (PFTs) within a grid cell. In dynamic vegetation models, PFTs within a grid cell compete for space and resources, and their expansion is driven by establishment and constrained by mortality and disturbance processes such as fire.

```@docs; canonical = false
AbstractVegetationDynamics
```

```@example vegdynamics
subtypes(Terrarium.AbstractVegetationDynamics)
```

## PALADYN vegetation dynamics

```@docs; canonical = false
PALADYNVegetationDynamics
```

```@example vegdynamics
variables(PALADYNVegetationDynamics(Float32))
```

This implementation follows the Lotka–Volterra approach of PALADYN [willeitPALADYNV10Comprehensive2016](@cite), in which the vegetation area fraction $\nu_i$ for each PFT $i$ evolves according to:

```math
\begin{equation}
\frac{d\nu_i}{dt} = \lambda_{\text{NPP}} \frac{\text{NPP}}{C_{\text{veg,i}}} \cdot \nu_i^*  (1 - \sum_j c_{ij}\nu_j) - \gamma_v \nu_i^*
\end{equation}
```

where $\lambda_{\text{NPP}}$ is the $\text{NPP}$-partitioning factor computed by the vegetation carbon dynamics process (see [Vegetation carbon dynamics](@ref)), $\text{NPP}$ is the net primary production, $C_{\text{veg}}$ is the total vegetation carbon pool, $\nu^* = \max(\nu, \nu_{\text{seed}})$ where $\nu_{\text{seed}}$ is a small seeding fraction to ensure that a PFT is always seeded and $\gamma_v$ is the disturbance rate.

The first term on the right-hand side represents the expansion of PFT $i$ driven by the fraction of NPP allocated to spreading. 

The second term is a Lotka–Volterra competition term that limits expansion through competition with other PFTs for space, with $c_{ij}$ describing the competitive effect of PFT $j$ on PFT $i$. Since only one PFT is considered per grid cell in the current implementation, this competition term is ignored for now.

The last term represents the loss of vegetation cover due to disturbance, for example from fire or other mortality related processes. In PALADYN, the disturbance coefficient $\gamma_v$ mainly represents fire-induced disturbance through a simple parameterization based on topsoil moisture and aboveground biomass, together with a minimum constant disturbance rate used to represent disturbances other than fire. In the current implementation, fire is ignored, and $\gamma_v = \gamma_{\min}$. 


## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, veg_dynamics::PALADYNVegetationDynamics, args...)
```

```@docs; canonical = false
compute_tendencies!(state, grid, veg_dynamics::PALADYNVegetationDynamics, vegcarbon_dynamics::PALADYNCarbonDynamics, traits::AbstractPlantTraits, args...)
```

## Methods

```@docs; canonical = false
compute_γv
```

```@docs; canonical = false
compute_ν_star
```

## Kernel functions

```@docs; canonical = false
compute_ν_tendency
```

```@docs; canonical = false
compute_ν_tendencies!
```

## [References](@id "vegetation_dynamics.refs")

```@bibliography
Pages = ["vegetation_dynamics.md"]
Canonical = false
```
