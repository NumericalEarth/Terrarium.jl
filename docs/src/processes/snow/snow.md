# [Snow](@id snow_process_docs)

```@meta
CurrentModule = Terrarium
```

```@setup snow
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Snow is represented in Terrarium as one or more vertical layers of an ice-water-air mixture overlaying the land surface with fractional subgrid coverage diagnosed from the mass balance (see [Snow cover](@ref)). The default scheme, [`SingleLayerSnow`](@ref), is a simple single-layer snowpack loosely based on the Utah Energy Balance model [tarbotonSpatiallyDistributedEnergy1994](@cite). The snowpack is treated as a single, integrated layer with constant density and prognostic state defined by the depth-integrated internal energy ``\bar{U}_\text{snow}`` (J/m²) and the snow water equivalent ``W_\text{snow}`` (m).  The mass balance dynamics are detailed on the [Snow mass balance](@ref) page and the energy balance and enthalpy closure on the [Snow energy balance](@ref) page.

### Coupling

In the [Land model](@ref "Land models") the snowpack couples to the surface energy balance and the soil:

- The albedo and emissivity are blended between snow and bare ground by the snow-covered fraction ``f_\text{snow}`` (see [Albedo and emissivity](@ref)).
- The skin-temperature conduction target is an ``f_\text{snow}``-weighted blend of the snow layer and the underlying ground (see [Skin temperature](@ref "Skin temperature")).
- The surface latent flux is partitioned by ``f_\text{snow}`` into ground/canopy evaporation (latent heat of vaporization) and snow sublimation (latent heat of sublimation).
- The snow→soil basal conductive flux is blended into the soil-top boundary condition, and the meltwater outflow is routed into the [Surface runoff](@ref) water balance.

!!! todo "Snow tiling"
    The current approach of blending snow-covered and snow-free surface fluxes is expected to underperform in many temperate regions where partial snow cover is common due to heterogeneous terrain or intermittent snowfall. This will be improved in future iterations through the use of tiling.

```@docs; canonical = false
SingleLayerSnow
```

## [Process interface](@id snow.dispatches)

[`SingleLayerSnow`](@ref) implements the standard `AbstractProcess` interface: `initialize!` sets the internal
energy from a prescribed temperature and SWE via the inverse closure; `compute_auxiliary!` diagnoses the
geometric and thermal properties; `compute_tendencies!` accumulates the mass and energy tendencies; and
the `closure!`/`invclosure!` pair maps between the prognostic energy and the diagnosed temperature and
liquid water fraction.

```@docs; canonical = false
initialize!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants, args...)

compute_auxiliary!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants, args...)

compute_tendencies!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants, atmos::AbstractAtmosphere, args...)

closure!(state, grid, closure::SnowEnergyTemperatureClosure, snow::SingleLayerSnow, constants::PhysicalConstants, args...)

invclosure!(state, grid, closure::SnowEnergyTemperatureClosure, snow::SingleLayerSnow, constants::PhysicalConstants, args...)
```

## [References](@id "snow.refs")

```@bibliography
Pages = ["snow.md"]
Canonical = false
```
