# Photosynthesis and gas exchange

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

Photosynthesis is the process by which plants convert solar radiation into chemical energy (sugars). The rate of photosynthesis depends on photosynthetically active radiation (PAR), ambient CO₂ concentration, leaf temperature, and leaf water status. Terrarium currently implements the mechanistic approach of Haxeltine and Prentice (1996) adapted from PALADYN (Willeit 2016), which explicitly represents two limitations to photosynthesis: light availability and enzyme (RuBisCO) activity.

Unlike in PALADYN, however, all photosynthetic rates are computed as **instantaneous rates** (e.g. mol/m²/s or gC/m²/s) to ensure compatibility with generic timestepping schemes.

### Light and RuBisCO-limited photosynthesis (two-leaf model)

The Haxeltine and Prentice (1996) approach computes photosynthesis by considering two potentially limiting processes and smoothly interpolating between them.

First, absorbed photosynthetically active radiation (APAR) is computed accounting for canopy light extinction:
```math
\begin{equation}
\text{APAR} = \alpha_a \cdot \text{PAR} \cdot (1 - e^{-k_{\text{ext}} \cdot \text{LAI}})
\end{equation}
```

where $\alpha_a$ is the canopy light-use efficiency, PAR is incident photosynthetically active radiation, $k_{\text{ext}}$ is the light extinction coefficient, and LAI is leaf area index.

The light-limited photosynthesis rate is:
```math
\begin{equation}
J_E = c_1 \cdot \text{APAR}
\end{equation}
```

where $c_1$ depends on the intrinsic quantum efficiency $\alpha_{C3}$ and enzyme kinetics:
```math
\begin{equation}
c_1 = \alpha_{C3} \cdot f_{\text{temp}} \cdot C_{\text{mass}} \cdot \frac{p_i - \Gamma^*}{p_i + 2\Gamma^*}
\end{equation}
```

where $p_i$ is intercellular CO₂ partial pressure and $f_{\text{temp}}$ is the temperature stress factor. $\Gamma^*$ is the CO₂ compensation point which represents the intercellular CO₂ concentration
at which no CO₂ uptake occurs in the absence of mitochondrial respiration.

The RuBisCO-limited (enzyme-limited) photosynthesis rate is:
```math
\begin{equation}
J_C = c_2 \cdot V_c^{\max}
\end{equation}
```

where $V_c^{\max}$ is the maximum carboxylation rate and:
```math
\begin{equation}
c_2 = \frac{p_i - \Gamma^*}{p_i + K_c(1 + p_O/K_o)}
\end{equation}
```

with $K_c$ and $K_o$ being Michaelis-Menten constants for CO₂ and O₂, and $p_O$ the O₂ partial pressure.

### Smooth minimum between light and RuBisCO limitations

Rather than taking a simple minimum, the two rates are blended smoothly using:
```math
\begin{equation}
A_g = \frac{J_E + J_C - \sqrt{(J_E + J_C)^2 - 4\theta_r J_E J_C}}{2\theta_r} \cdot \beta
\end{equation}
```

where $\theta_r$ is a shape parameter (typically 0.7), and $\beta$ is the soil moisture limitation factor. This smooth minimum approach avoids unrealistic discontinuities at the transition between light and enzyme limitation.

The net photosynthesis is then:
```math
\begin{equation}
A_n = A_g - R_d
\end{equation}
```

where $R_d$ is the leaf dark respiration rate.

### Temperature stress on photosynthesis

Temperature has a critical impact on photosynthetic enzyme kinetics and overall physiological capacity. A temperature stress factor $f_{\text{temp}}$ modulates the light-limited rate, with performance decreasing outside the range $[T_{\text{CO2,low}}, T_{\text{CO2,high}}]$ (PFT-specific):
```math
\begin{equation}
f_{\text{temp}}(T) = 
\begin{cases}
\sigma_{\text{low}}(T) \cdot \sigma_{\text{high}}(T) & \text{if } T_{\text{CO2,low}} < T < T_{\text{CO2,high}} \\
0 & \text{otherwise}
\end{cases}
\end{equation}
```

where $\sigma_{\text{low}}$ and $\sigma_{\text{high}}$ are sigmoid-shaped functions controlling the lower and upper temperature boundaries. This representation captures the observation that photosynthesis is inhibited both by cold and heat stress.

### Stomatal conductance coupling

Stomatal conductance (gas exchange) is computed separately from photosynthesis using the Medlyn et al. (2011) optimal stomatal control theory (see [Stomatal conductance](@ref)). This gives the internal leaf CO₂ concentration ratio $\lambda_c$ that drives the intercellular CO₂ pressure: $p_i = \lambda_c \cdot p_a$, which is used in the photosynthesis calculation above.

The stomatal conductance model captures the empirical observation that stomata open more when CO₂ uptake is high and close when air is dry (high VPD), creating a tight coupling between photosynthesis, transpiration, and atmospheric water demand.

## Abstract types

```@docs; canonical = false
AbstractPhotosynthesis
```

## Concrete types

```@docs; canonical = false
LUEPhotosynthesis
```

## Methods

### Photosynthesis helper functions

These functions compute the intermediate quantities needed for the Haxeltine and Prentice (1996) photosynthesis scheme:

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
compute_photosynthesis
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
