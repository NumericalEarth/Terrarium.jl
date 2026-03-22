# Soil hydrology

```@meta
CurrentModule = Terrarium
```

```@setup default
using Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Richardson-Richards equation for variably saturated flow

The vertical flow of water in porous media, such as soils, can be formulated as following the conservation law
```math
    \phi\frac{\partial\vartheta(\psi)}{\partial t} - \boldsymbol{\nabla} \cdot \textbf{j}_{\text{w}} - F_{\text{w}}(z,t) = 0,
```
where $\phi$ is the natural porosity (or saturated water content) of the soil volume and $F_{\text{w}}(z,t)$ (m/s) is an inhomogeneous source/sink (forcing) term.

Vertical fluxes in the soil column be represented by combining gravity-driven advection with Darcy's law
```math
\begin{equation}
\textbf{j}_{\text{w}} \cdot \mathbf{n} = -\kappa_{\text{w}}\frac{\partial \left(\psi + z\right)}{\partial z},
\end{equation}
```
where $\psi$ (m) is the matric potential. Substituting this equation into the aforementioned conservation law yields the widely known Richardson-Richards equation for variably saturated flow in porous media (Richards 1931).

## Abstract types

```@docs; canonical = false
AbstractSoilHydrology
```

```@docs; canonical = false
AbstractSoilWaterClosure
```

## Concrete types

```@docs; canonical = false
SoilHydrology
```

### [State variables](@id soilhydrology.vars)

```@example default
variables(SoilHydrology(Float32))
```

### [Process method dispatches](@id soilhydrology.dispatches)

```@docs; canonical = false
initialize!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)
initialize!(
    state, grid,
    hydrology::SoilHydrology{NF, RichardsEq},
    soil::AbstractSoil,
    constants::PhysicalConstants,
    args...
) where {NF}
```

```@docs; canonical = false
compute_auxiliary!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)
compute_auxiliary!(
    state, grid,
    hydrology::SoilHydrology{NF, RichardsEq},
    soil::AbstractSoil,
    args...
) where {NF}
```

```@docs; canonical = false
compute_tendencies!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)
compute_tendencies!(
    state, grid,
    hydrology::SoilHydrology{NF, RichardsEq},
    soil::AbstractSoil,
    constants::PhysicalConstants,
    evtr::Optional{AbstractEvapotranspiration} = nothing,
    runoff::Optional{AbstractSurfaceRunoff} = nothing,
    args...
) where {NF}
```

### Vertical flow

```@docs; canonical = false
NoFlow
RichardsEq
```

## Closures

```@docs; canonical = false
SoilSaturationPressureClosure
```

## Methods

```@docs; canonical = false
get_hydraulic_properties
```

```@docs; canonical = false
get_swrc
```

```@docs; canonical = false
get_closure
```

```@docs; canonical = false
compute_water_table!
```

```@docs; canonical = false
adjust_saturation_profile!
```

```@docs; canonical = false
compute_hydraulics!
```

## Kernel functions

```@docs; canonical = false
saturation_water_ice
```

```@docs; canonical = false
hydraulic_conductivity
```

```@docs; canonical = false
liquid_water_fraction
```

```@docs; canonical = false
water_table
```

```@docs; canonical = false
surface_excess_water
```
