# Plant available water

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Soil water availability for plants

The plant available water (PAW) represents the fraction of soil water that is available for uptake by plant roots. This is often assumed to be the fraction of water between the wilting point (where plants can no longer extract water) and field capacity (approximately optimal water availability). This quantity is critical for constraining photosynthesis and transpiration.

The water availability coefficient for a given soil layer is defined as:
```math
\begin{equation}
W = \min\left(\frac{\theta_w - \theta_{wp}}{\theta_{fc} - \theta_{wp}}, 1\right)
\end{equation}
```

where $\theta_w$ is the current volumetric water content, $\theta_{wp}$ is the wilting point, and $\theta_{fc}$ is the field capacity matric potential. Both wilting point and field capacity are derived from soil texture properties and the water retention characteristics of the soil.

### Soil moisture limitation of photosynthesis

The impact of soil water availability on photosynthesis is aggregated across the rooting depth using the [root distribution](@ref "Root distribution"):
```math
\begin{equation}
\beta = \int_0^{z_{\text{max}}} W(z) \cdot r(z) \, dz
\end{equation}
```

where $r(z)$ is the normalized root fraction at depth $z$, and $z_{\text{max}}$ is the rooting depth. This soil moisture limiting factor $\beta$ (0 to 1) directly modulates vegetation photosynthetic capacity.

### Water retention and texture

The relationship between soil matric potential and water content depends on soil texture through the water retention curve (SWRC). Coarse soils (sandy) drain more readily than fine-textured soils (clay), affecting the range of plant-available water. Field capacity is typically defined as the water content at -33 kPa matric potential, while the wilting point corresponds to -1500 kPa.

## Abstract types

```@docs; canonical = false
AbstractPlantAvailableWater
```

## Concrete types

```@docs; canonical = false
FieldCapacityLimitedPAW
```

## Methods

```@docs; canonical = false
soil_moisture_limiting_factor
```

## Kernel functions

```@docs; canonical = false
compute_plant_available_water
```

```@docs; canonical = false
compute_plant_available_water!
```
