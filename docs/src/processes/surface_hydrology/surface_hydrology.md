# Surface hydrology

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Overview

The surface hydrology module manages water exchange between the atmosphere and land, including three tightly coupled processes:

1. [Canopy interception](@ref): Temporary storage and removal of intercepted rainfall
2. [Evapotranspiration](@ref): Water vapor fluxes from ground, canopy, and plant stomata
3. [Surface runoff](@ref): Partitioning of ground-reaching rainfall into infiltration and streamflow

These processes couple the water and energy cycles: evapotranspiration removes energy from the surface through latent heat flux (competing with sensible and ground heat), while infiltration controls water availability in the soil.

For detailed physics and governing equations of each process, see the dedicated pages linked above.

## Abstract types

```@docs; canonical = false
AbstractSurfaceHydrology
```

## Concrete types

### Surface Hydrology

```@docs; canonical = false
SurfaceHydrology
```
