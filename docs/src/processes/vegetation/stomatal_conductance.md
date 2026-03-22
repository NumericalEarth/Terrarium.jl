# Stomatal conductance

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

Stomata are microscopic pores on plant leaves that regulate gas exchange: they allow CO₂ uptake for photosynthesis while controlling water loss through transpiration. Stomatal conductance ($g_w$) quantifies the ease with which water vapor diffuses from inside the leaf to the atmosphere. Higher conductance means greater water vapor flux and higher transpiration rates.

Terrarium implements the optimal stomatal conductance model of Medlyn et al. (2011) adapted from PALADYN (Willeit 2016), which derives stomatal conductance from the ecological principle that plants optimize the trade-off between photosynthetic carbon gain and water loss.

### The Medlyn optimal stomatal conductance model

The Medlyn et al. (2011) model describes the relationship between stomatal conductance and net photosynthetic assimilation as:

```math
\begin{equation}
g_w = g_{\min} (1 - e^{-k_{\text{ext}} \cdot \text{LAI}}) \cdot \beta + \left(1 + \frac{g_1}{\sqrt{\text{VPD}}}\right) \frac{A_n}{C_a} \times 10^6
\end{equation}
```

where:
- $g_{\min}$ is the minimum stomatal conductance (residual conductance when no photosynthesis occurs)
- $k_{\text{ext}}$ is the light extinction coefficient
- $\text{LAI}$ is the leaf area index
- $\beta$ is the soil moisture limitation factor (0 to 1)
- $g_1$ is the stomatal sensitivity parameter (species/PFT-specific, typically 1–4)
- $\text{VPD}$ is the vapor pressure deficit in Pa
- $A_n$ is the net photosynthetic assimilation rate [gC/m²/s]
- $C_a$ is the atmospheric CO₂ concentration [ppm]

The model captures the empirical observation that stomata:
1. Close when soil water is limiting ($\beta < 1$)
2. Open in response to high photosynthetic demand ($A_n$ increases)
3. Close when atmospheric water demand is high (VPD increases)

### Leaf-level CO₂ concentration

The ratio of leaf-internal to air CO₂ concentration ($\lambda_c$) is derived from the optimal stomatal conductance theory:

```math
\begin{equation}
\lambda_c = 1 - \frac{1.6}{1 + \frac{g_1}{\sqrt{\text{VPD} \times 10^{-3}}}}
\end{equation}
```

where VPD is in Pa. This ratio determines the driving force for CO₂ diffusion into the leaf and is essential for the photosynthesis calculation.

### Coupling to photosynthesis

Stomatal conductance and photosynthesis form a tight feedback loop:
- Photosynthesis produces net assimilation ($A_n$) → higher $A_n$ increases stomatal conductance
- Stomatal conductance determines the leaf-internal CO₂ concentration ($\lambda_c$) → this drives photosynthesis
- In reality, this coupling is iterative; here we compute both simultaneously as part of the vegetation process

## Abstract types

```@docs; canonical = false
AbstractStomatalConductance
```

## Concrete types

```@docs; canonical = false
MedlynStomatalConductance
```

## Methods

### Stomatal conductance helper functions

```@docs; canonical = false
compute_gw_can
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
