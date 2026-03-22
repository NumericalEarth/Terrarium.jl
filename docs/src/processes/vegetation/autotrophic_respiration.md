# Plant respiration

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

Autotrophic respiration is the metabolic cost of maintaining and growing plant tissues. It consists of maintenance respiration (the energy cost of tissue upkeep) and growth respiration (the carbon cost of synthesizing new biomass from photosynthates). Maintenance respiration is temperature-dependent and occurs in all living plant tissues.

### Autotrophic respiration

The total autotrophic respiration can be approximated as a fraction of gross primary production (GPP):
```math
\begin{equation}
R_a = R_{\text{leaf}} + R_{\text{maint}} + R_{\text{growth}}
\end{equation}
```

where $R_{\text{leaf}}$ is leaf respiration at the leaf level, $R_{\text{maint}}$ is maintenance respiration across all tissues, and $R_{\text{growth}}$ is the cost of growth. The net primary production (NPP) is then:
```math
\begin{equation}
\text{NPP} = \text{GPP} - R_a
\end{equation}
```

### Temperature dependence

Respiration rates follow an exponential temperature dependence:
```math
\begin{equation}
f_{\text{temp}}(T) = \exp\left( 308.56 \left( \frac{1}{56.02} - \frac{1}{46.02 + T} \right) \right)
\end{equation}
```

where $T$ is temperature in degrees Celsius. This exponential form captures the enzyme kinetics underlying respiration across a physiologically relevant temperature range.

### Maintenance respiration

Maintenance respiration varies across plant tissues according to their tissue-specific respiration rates and carbon content. For stem respiration, the respiring biomass is estimated from total stem carbon using an allometric factor:
```math
\begin{equation}
R_{\text{maint}} = R_{10} \cdot f_{\text{temp}}(T) \cdot C_{\text{resp}} 
\end{equation}
```

where $R_{10}$ is the respiration rate at 10°C, and $C_{\text{resp}}$ is the respiring biomass derived from plant structural and sapwood carbon pools.

## Abstract types

```@docs; canonical = false
AbstractAutotrophicRespiration
```

## Concrete types

```@docs; canonical = false
PALADYNAutotrophicRespiration
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

## Kernel functions

```@docs; canonical = false
compute_autotrophic_respiration
```

```@docs; canonical = false
compute_autotrophic_respiration!
```
