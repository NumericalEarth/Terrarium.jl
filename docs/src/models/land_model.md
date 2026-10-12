# Land models

```@meta
CurrentModule = Terrarium
```

```@setup landmodel
using Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

[`LandModel`](@ref) is a fully-coupled land surface and terrestrial ecosystem model that simulates the integrated dynamics of soil, surface hydrology, surface energy balance, vegetation, and atmospheric forcing. Conceptually, the model represents a single, homogeneous land surface tile within each `grid` cell with a predetermined set of processes.

```@example landmodel
arch = CPU()
grid = ColumnGrid(arch, Float32, ExponentialSpacing(N = 10)) # 10 soil layers
model = LandModel(grid) # Default configuration
integrator = initialize(model)
```

```@docs; canonical = false
LandModel
```

By default, `LandModel` is integrated with `IMEX(ForwardEuler(NF; Δt = 900), BackwardEuler(NF))`: soil heat conduction, Richards-equation water flow, and the snow meltwater drainage are implicit, while the snow heat conduction, vegetation, surface hydrology, and the surface energy balance are explicit, with a 15-minute time step, which is limited by the stability of the explicit snow heat conduction. See [Time stepping](@ref) for details.

```@example landmodel
variables(model)
```

## Components

[`LandModel`](@ref) integrates five major sub-processes: **atmosphere**, **soil**, **surface energy balance**, **surface hydrology**, and **vegetation**:

| Field | Type | Scope | Process page |
|-------|------|-------|---------------|
| `atmosphere` | [`AbstractAtmosphere`](@ref) | Meteorological input variables | [Atmosphere](@ref atmosphere_docs) |
| `soil` | [`AbstractSoil`](@ref) | Energy, water, carbon in soil | [Soil processes](@ref) |
| `surface_energy_balance` | [`AbstractSurfaceEnergyBalance`](@ref) | Radiative and turbulent energy fluxes | [Surface energy balance](@ref surface_energy_balance_docs) |
| `surface_hydrology` | [`AbstractSurfaceHydrology`](@ref) | Infiltration, evapotranspiration, interception | [Surface hydrology](@ref surface_hydrology_docs) |
| `vegetation` | `Optional{`[`AbstractVegetation`](@ref)`}` | Coupled vegetation carbon processes | [Vegetation](@ref vegetation_docs) |

Each component can be configured separately when constructing a `LandModel`. `vegetation` is optional and can be disabled by setting `vegetation = nothing` in the constructor; this corresponds to a bare ground land-atmosphere coupling. Each component of `LandModel` is summarized briefly below. See the linked pages for further details about each component process.

### Atmosphere

The `atmosphere` component provides (possibly time-varying) meteorological inputs. The default implementation is [`PrescribedAtmosphere`](@ref), which reads air temperature, humidity, wind, radiation, and precipitation from [`InputVariable`](@ref)s and provides them as boundary conditions to the surface energy balance, hydrology, and vegetation components. See [Atmospheric inputs](@ref atmosphere_docs) for further details on the atmosphere interface.

### Soil

The `soil` component represents the solid land surface extending from the topmost soil layer down to an arbitrary depth determined by the `grid`. The default configuration of `LandModel` uses [`SoilEnergyWaterCarbon`](@ref) which represents coupled energy, water, and carbon transport within the soil column. See [Soil processes](@ref) for detailed descriptions of energy, hydrology, and biogeochemistry implementations.

### Surface energy balance

The `surface_energy_balance` component computes the energy fluxes at the land-atmosphere interface, including solar radiation, thermal radiation, and turbulent heat fluxes. The default implementation is [`SurfaceEnergyBalance`](@ref). See [Surface energy balance](@ref surface_energy_balance_docs) for details.

### Surface hydrology

The `surface_hydrology` component manages water exchange between the atmosphere and the land surface, including canopy interception, evapotranspiration, and surface runoff partitioning. The default implementation is [`SurfaceHydrology`](@ref). See [Surface hydrology](@ref surface_hydrology_docs) for details.

### Vegetation

The `vegetation` component represents vegetation carbon cycling, including photosynthesis, stomatal conductance, respiration, phenology, and carbon dynamics. The default implementation is [`VegetationCarbonCycle`](@ref). See [Vegetation](@ref vegetation_docs) for details.

## Initializers

!!! todo "Initializers and boundary conditions"
    `LandModel` does not yet have its own dedicated boundary conditions and initializers, but it will soon! Stay tuned!
