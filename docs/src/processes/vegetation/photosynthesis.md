# Photosynthesis

```@meta
CurrentModule = Terrarium
```

```@setup vegphoto
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Photosynthesis is the process by which plants use solar radiation to convert carbon dioxide into chemical energy (stored in carbohydrates) and occurs in two stages: light-dependent reactions and light independent reactions (also known as dark reactions or the Calvin cycle). In the first stage, light energy is used to produce two compounds (ATP and NADPH) which are then used during the second stage to produce carbohydrates from carbon dioxide.

The net carbon assimilation per leaf area is the difference between carbon uptake and carbon losses, for example through mitochondrial respiration (also known as dark respiration). Net photosynthesis, $A_n$, is usually modeled as the difference between gross photosynthesis, $A_g$, and dark respiration, $R_d$

```math
\begin{equation}
A_n = A_g - R_d.
\end{equation}
```

Gross photosynthesis is limited by two processes: RuBisCO activity and light availability. A simple approach is to model $A_g$ as the minimum of these two rates: the RuBisCO-limited rate $J_C$ and the light-limited rate $J_E$:

```math
\begin{equation}
A_g = \min(J_C, J_E)\,.
\end{equation}
```

While $A_g$ remains continuous, this formulation produces an abrupt transition between the two limiting factors. Under some conditions, however, photosynthesis can be co-limited by both light availability and RuBisCO activity. This co-limitation allows for a smoother transition and can be represented by taking the smaller root of the following quadratic equation:


```math
\begin{equation}
\theta_r A_g^2 - (J_C + J_E)A_g + J_C J_E = 0\,,
\end{equation}
```

where $\theta_r$ is an empirical shape parameter between 0 and 1 used to control the transition between the two limiting factors. 

```@docs; canonical = false
AbstractPhotosynthesis
```

```@example vegphoto
subtypes(Terrarium.AbstractPhotosynthesis)
```

## Light use efficiency (LUE) model

```@docs; canonical = false
LUEPhotosynthesis
```

```@example vegphoto
variables(LUEPhotosynthesis(Float32))
```

This implementation uses the light-use efficiency model of [haxeltineBIOME3EquilibriumTerrestrial1996](@citet), adapted from PALADYN [willeitPALADYNV10Comprehensive2016](@cite), where the original equations were derived assuming a daily time step. Unlike PALADYN, however, all photosynthetic rates here are computed as instantaneous rates (e.g. mol/m²/s or kgC/m²/s) to ensure compatibility with generic timestepping schemes.

### Light and RuBisCO-limited photosynthesis rates

The light-limited photosynthesis rate is
```math
\begin{equation}
J_E = c_1 \cdot \text{APAR}\,,
\end{equation}
```

with 

```math
\begin{equation}
c_1 = \alpha_{C3} \cdot f_{\text{temp}} \cdot C_{\text{mass}} \cdot \frac{p_i - \Gamma^*}{p_i + 2\Gamma^*}\,,
\end{equation}
```

where $\alpha_{C3}$ is the intrinsic quantum efficiency of CO₂ uptake in C3 plants, $C_{mass}$ is the atomic mass of carbon (kgC/mol), $p_i$ is intercellular CO₂ partial pressure and $\Gamma^*$ is the CO₂ compensation point. 

The temperature stress factor $f_{\text{temp}}$ is defined as

```math
\begin{equation}
f_{\text{temp}}(T) = 
\begin{cases}
\sigma_{\text{low}}(T) \cdot \sigma_{\text{high}}(T) & \text{if } T_{\text{CO2,low}} < T < T_{\text{CO2,high}} \\
0 & \text{otherwise}
\end{cases}
\end{equation}
```

Assuming that half of the downwelling shortwave radiation ($\text{SW}_{\downarrow}$) is in the photosynthetically active wavelength range, photosynthetically active radiation ($\text{PAR}$) can be computed as

```math
\begin{equation}
\text{PAR} = 0.5 \cdot  $\text{SW}_{\downarrow}$ \cdot (1 - \alpha_\text{leaf}) \cdot c_q\,,
\end{equation}
```

where $\alpha_\text{leaf}$ is the leaf albedo in the PAR range and $c_q$ is the conversion factor from W/m² to mol/m²/s. 

Absorbed PAR ($\text{APAR}$) can then be computed as,

```math
\begin{equation}
\text{APAR} = \alpha_a \cdot \text{PAR} \cdot (1 - e^{-k_{\text{ext}} \cdot \text{LAI}})\,,
\end{equation}
```

where $\alpha_a$ accounts for reductions in PAR utilization in natural ecosystems, $k_{\text{ext}}$ is the light extinction coefficient, and $\text{LAI}$ is the leaf area index.



The RuBisCO-limited (enzyme-limited) photosynthesis rate is:
```math
\begin{equation}
J_C = c_2 \cdot V_c^{\max}\,,
\end{equation}
```

where $V_c^{\max}$ is the maximum carboxylation rate and

```math
\begin{equation}
c_2 = \frac{p_i - \Gamma^*}{p_i + K_c(1 + p_o/K_o)}\,,
\end{equation}
```

where $K_c$ and $K_o$ are the temperature-dependent Michaelis-Menten constants for CO₂ and O₂, respectively, and $p_O$ is the O₂ partial pressure.

### Leaf respiration

Leaf (dark) respiration is a fixed fraction $b$ (`leaf_respiration_fraction`, 0.015 for C3 plants) of the maximum carboxylation capacity [haxeltineBIOME3EquilibriumTerrestrial1996; Eq. (10)](@cite), limited by soil moisture through the factor $\beta$,

```math
\begin{equation}
R_d = b \cdot V_c^{\max} \cdot \beta\,.
\end{equation}
```


### Stomatal conductance coupling

Both light-limited and RuBisCO-limited photosynthesis rates $J_E$ and $J_C$ depend on the intercellular CO₂ partial pressure $p_i$.
The exchange of carbon during photosynthesis is regulated by stomata, which open and close to control gas exchanges between the leaf and the atmosphere, while balancing carbon uptake for photosynthesis against water loss through transpiration.

This coupling is captured through the intercellular-to-ambient CO₂ concentration ratio $\lambda_c$, which is derived in the stomatal conductance model (see [Stomatal conductance](@ref)) from the optimization-based and diffusion-based expressions of stomatal conductance. The intercellular CO₂ partial pressure can then be computed as

```math
\begin{equation}
p_i = \lambda_c \cdot p_a\,,
\end{equation}
```

where $p_a$ is the atmospheric CO₂ partial pressure.


## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, photo::LUEPhotosynthesis, stomcond::AbstractStomatalConductance, traits::PlantTraits, constants::PhysicalConstants, atmos::AbstractAtmosphere, args...)
```

## Methods

```@docs; canonical = false
compute_kinetic_parameters
```

```@docs; canonical = false
compute_Γ_star
```

```@docs; canonical = false
compute_PAR
```

```@docs; canonical = false
compute_APAR
```

```@docs; canonical = false
compute_pres_i
```

```@docs; canonical = false
compute_temperature_stress
```

```@docs; canonical = false
compute_assimilation_factors
```

```@docs; canonical = false
compute_Vc_max
```

```@docs; canonical = false
compute_JE_JC
```

```@docs; canonical = false
compute_Rd
```

```@docs; canonical = false
compute_Ag
```

```@docs; canonical = false
compute_respiration_assimilation
```

```@docs; canonical = false
compute_GPP
```

## Kernel functions

```@docs; canonical = false
compute_photosynthesis
```

```@docs; canonical = false
compute_photosynthesis!
```

## [References](@id "photosynthesis.refs")

```@bibliography
Pages = ["photosynthesis.md"]
Canonical = false
```
