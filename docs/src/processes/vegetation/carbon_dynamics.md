# Plant carbon dynamics and vegetation evolution

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Vegetation carbon pools

Plant biomass is partitioned among multiple carbon pools: leaves, stems, and roots. These pools have different turnover rates and play distinct roles in plant physiology and ecosystem function. The carbon dynamics system tracks the total vegetation carbon and derives the balanced leaf area index (LAI) assuming a fixed biomass allocation scheme.

### Leaf area index

The specific leaf area (SLA) and allometric relationships determine how vegetation carbon is distributed:
```math
\begin{equation}
\text{LAI}_b = \frac{C_{\text{leaf}}}{C_0} = \frac{1}{\text{SLA}} \left( C_{\text{veg}} - C_{\text{struct}} \right)
\end{equation}
```

where $C_{\text{veg}}$ is the total vegetation carbon, $C_{\text{struct}}$ is the structural carbon (stems and roots), and LAI$_b$ is the balanced leaf area index that would exist at equilibrium with the current carbon pool.

### Leaf turnover and litterfall

Leaf turn occurs at a constant annual rate $\gamma_L$ (year$^{-1}$), representing senescence and leaf drop. The litterfall rate integrating across the canopy is given by
```math
\begin{equation}
\Lambda = \gamma_L C_{\text{leaf}} = \gamma_L \text{SLA}^{-1} \left( C_{\text{veg}} - C_{\text{struct}} \right)
\end{equation}
```

Other carbon pools (roots and stems) turn over at their respective rates $\gamma_R$ and $\gamma_S$ (year$^{-1}$).

### Net primary productivity partitioning

Net primary production (NPP) is partitioned between two pathways: increasing vegetation carbon in existing vegetation and enabling the spread of new vegetation. The partitioning factor $\lambda_{\text{NPP}}$ depends on whether the current LAI is above the minimum LAI threshold:
```math
\begin{equation}
\lambda_{\text{NPP}} = \begin{cases}
0 & \text{if } \text{LAI}_b < \text{LAI}_{\min} \\
\frac{\text{LAI}_b - \text{LAI}_{\min}}{\text{LAI}_{\max} - \text{LAI}_{\min}} & \text{if } \text{LAI}_{\min} \leq \text{LAI}_b \leq \text{LAI}_{\max} \\
1 & \text{if } \text{LAI}_b > \text{LAI}_{\max}
\end{cases}
\end{equation}
```

This ensures that when vegetation is below its minimum viable LAI, all NPP goes to increasing carbon rather than being available for spreading new vegetation. The vegetation carbon tendency is then:

```math
\begin{equation}
\frac{dC_{\text{veg}}}{dt} = (1 - \lambda_{\text{NPP}}) \text{NPP} - \Lambda_{\text{loc}}
\end{equation}
```

The first term represents NPP allocated to carbon accumulation on already-vegetated land, while the second term represents carbon losses through litterfall and turnover of all pools.

## Abstract types

```@docs; canonical = false
AbstractVegetationCarbonDynamics
```

## Concrete types

```@docs; canonical = false
PALADYNCarbonDynamics
```

## Methods

```@docs; canonical = false
compute_LAI_b
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
