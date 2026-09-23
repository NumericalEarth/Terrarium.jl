# Stomatal conductance

```@meta
CurrentModule = Terrarium
```

```@setup vegstomcond
using Terrarium
using InteractiveUtils
```


!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Stomata regulate gas exchange between the leaf and atmosphere, directly controlling the trade-off between carbon uptake during photosynthesis and water loss through transpiration. Similar to photosynthesis, stomatal conductance, $g_w$, depends on multiple environmental factors, including light availability, CO₂ concentration, and temperature. 

```@docs; canonical = false
AbstractStomatalConductance
```

```@example vegstomcond
subtypes(Terrarium.AbstractStomatalConductance)
```

### The Medlyn stomatal conductance model

```@docs; canonical = false
MedlynStomatalConductance
```

```@example vegstomcond
variables(MedlynStomatalConductance(Float32))
```

This implementation uses the optimal stomatal conductance model of Medlyn (2011) [medlynReconcilingOptimalEmpirical2011](@cite), in the corrected form of the 2012 corrigendum [medlynCorrigendumReconcilingOptimal2012](@cite) which supplies the factor 1.6, with the canopy scaling of $g_0$ and the PFT-specific parameter values taken from PALADYN [willeitPALADYNV10Comprehensive2016](@cite). It derives stomatal conductance from water-use efficiency optimization as follows

```math
\begin{equation}
g_w = g_0 + 1.6 \frac{A_n}{c_a}  \left(1 + \frac{g_1}{\sqrt{\text{VPD}}}\right) 
\end{equation}
```

where $g_0$ is the minimum stomatal conductance,  $g_1$ is a PFT-specific slope parameter, $\text{VPD}$ is the vapor pressure deficit, $A_n$ is the net photosynthesis and $c_a$ is the atmospheric CO₂ concentration. The factor 1.6 is the ratio of the diffusivities of water vapor and CO₂ in air.

Following PALADYN, $\text{VPD}$ is expressed in kPa, so $g_1$ carries units of $\sqrt{\text{kPa}}$ and the tabulated values of [linOptimalStomatalBehaviour2015](@cite) apply directly. Terrarium computes the vapor pressure deficit in Pa and converts before evaluating $g_1/\sqrt{\text{VPD}}$.

The variables $g_w$ and $A_n$ are also related by the diffusion equation

```math
\begin{equation}
g_w = g_0 + 1.6 \frac{A_n}{c_a - c_i} 
\end{equation}
```

where $c_i$ is the intercellular CO2 concentration.

Eliminating $A_n$ between the two gives the ratio of intercellular to atmospheric CO₂ concentration $\lambda_c$, in which the factor 1.6 cancels:

```math
\begin{equation}
\lambda_c = 1 - \frac{1}{1 + \frac{g_1}{\sqrt{\text{VPD}}}}
\end{equation}
```

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, stomcond::MedlynStomatalConductance, traits::PlantTraits, constants::PhysicalConstants, atmos::AbstractAtmosphere, args...)
```

## Methods

```@docs; canonical = false
compute_stomatal_conductance
```

```@docs; canonical = false
compute_λc
```

## Kernel functions

```@docs; canonical = false
compute_stomatal_conductance!
```

```@docs; canonical = false
compute_stomatal_conductance
```

## [References](@id "stomatal_conductance.refs")

```@bibliography
Pages = ["stomatal_conductance.md"]
Canonical = false
```
