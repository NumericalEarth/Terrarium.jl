# Soil stratigraphy

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Soil composition and material properties

The subsurface soil column consists of multiple material constituents that determine its physical and chemical properties. These constituents include water and ice occupying the pore space, air filling unsaturated pores, and a solid matrix composed of mineral and organic material. To accurately represent many soil proccesses in land surface models, it is necessary to characterize both the spatial distribution of soil material properties (stratigraphy) and the material composition of each soil volume element.

The stratigraphy of a soil column defines its layering structure and spatial variation in texture and other properties. Soil texture refers to the relative proportions of sand, silt, and clay in the mineral soil component, which is a fundamental property affecting hydraulic conductivity, water retention, and thermal properties. These textural classes are typically classified according to soil classification systems such as USDA or FAO standards.

The total void space available in a soil volume, termed porosity, controls the maximum amount of water and air that can occupy the pore space. Porosity varies depending on soil type, bulk density, and the presence of organic material. Organic soil components typically have higher porosity than mineral soil, reflecting their loose, aggregated structure.

### Soil volume composition

An elementary volume $V$ of soil can be decomposed into its constituent phases,
```math
\begin{equation}
V = V_{\text{water}} + V_{\text{ice}} + V_{\text{air}} + V_{\text{solids}}
\end{equation}
```

The volumetric fractions of these constituents are related through saturation and liquid water content,
```math
\begin{equation}
\theta_{\text{por}} = \frac{V_{\text{pore}}}{V_{\text{total}}} = \frac{V_{\text{water}} + V_{\text{ice}} + V_{\text{air}}}{V_{\text{total}}}
\end{equation}
```

where saturation $S \in [0,1]$ represents the fraction of pore space occupied by liquid water and ice,
```math
\begin{equation}
S = \frac{V_{\text{water}} + V_{\text{ice}}}{V_{\text{pore}}}
\end{equation}
```

The liquid fraction $\ell \in [0,1]$ represents the fraction of water in the pore space that is unfrozen,
```math
\begin{equation}
\ell = \frac{V_{\text{water}}}{V_{\text{water}} + V_{\text{ice}}}
\end{equation}
```

The [solid matrix](@ref AbstractSoilMatrix) may contain both mineral constituents (sand, silt, clay fractions) and organic material (e.g., soil organic matter). The organic fraction influences both thermal and hydraulic properties due to its distinct density and structural characteristics.

## Abstract types

```@docs; canonical = false
AbstractStratigraphy
```

```@docs; canonical = false
AbstractSoilMatrix
```

```@docs; canonical = false
AbstractSoilPorosity
```

## Concrete types

```@docs; canonical = false
SoilTexture
```

```@docs; canonical = false
HomogeneousStratigraphy
```

```@docs; canonical = false
ConstantSoilPorosity
```

```@docs; canonical = false
SoilPorositySURFEX
```

```@docs; canonical = false
SoilVolume
```

```@docs; canonical = false
MineralOrganic
```

## Methods

```@docs; canonical = false
soil_texture
```

```@docs; canonical = false
soil_matrix
```

```@docs; canonical = false
soil_volume
```

```@docs; canonical = false
mineral_porosity
```

```@docs; canonical = false
organic_porosity
```

```@docs; canonical = false
volumetric_fractions
```

## Kernel functions

```@docs; canonical = false
organic_fraction
```

```@docs; canonical = false
porosity
```
