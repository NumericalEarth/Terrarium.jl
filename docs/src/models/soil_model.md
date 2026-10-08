# Soil models

```@meta
CurrentModule = Terrarium
```

```@setup soilmodel
using Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

[`SoilModel`](@ref) is a model of soil physics that couples the relevant processes governing energy, water, and carbon in natural soils. It is the primary model type for simulating heat conduction, freeze-thaw processes (e.g. permafrost), and variably saturated water flow in the subsurface.

```@example soilmodel
arch = CPU()
grid = ColumnGrid(arch, Float32, ExponentialSpacing(N = 10)) # 10 soil layers
model = SoilModel(grid)
integrator = initialize(model)
```

```@docs; canonical = false
SoilModel
```

```@example soilmodel
variables(model)
```

## Components

| Field | Type | Scope | Process page |
|-------|------|-------|---------------|
| `strat` | [`AbstractStratigraphy`](@ref) | Vertical structure of soil | [Soil stratigraphy](@ref) |
| `energy` | [`AbstractSoilThermodynamics`](@ref) | Heat conduction and freeze-thaw of soil water | [Soil energy balance](@ref) |
| `hydrology` | [`AbstractSoilHydrology`](@ref) | Vertical flow of water between soil layers | [Soil hydrology](@ref) |
| `biogeochem` | [`AbstractSoilBiogeochemistry`](@ref) | Soil organic carbon and biogeochemical fluxes | Not yet added |

Each component is summarized briefly below. Follow the linked process pages for full theoretical background, available concrete types, state variables, and method signatures.

### Stratigraphy

The `strat` component parameterizes the vertical distribution of soil material properties (texture, porosity, organic content). It provides kernel functions used by the energy and hydrology sub-processes to look up spatially varying material properties at each grid cell. By default [`HomogeneousSoilStratigraphy`](@ref) is used, which specifies a single uniform material throughout all vertical profiles. See [Soil stratigraphy](@ref) for details.

### Energy balance

The `energy` component represents heat conduction in the soil column, including the latent heat of freeze-thaw phase change. The default implementation is [`SoilThermodynamics`](@ref), which evolves the volumetric internal energy $U$ (J m⁻³) as the prognostic variable and derives temperature via the [`SoilEnergyTemperatureClosure`](@ref). See [Soil energy balance](@ref) for details.

### Hydrology

The `hydrology` component governs the vertical movement of water in the soil column. The default implementation is [`SoilHydrology`](@ref), which evolves total saturation (liquid + ice) as the prognostic variable. Vertical flow is disabled by default ([`NoFlow`](@ref)); enabling the Richards equation requires selecting [`RichardsEq`](@ref) as the `vertical_flow` operator. See [Soil hydrology](@ref) for details.

### Biogeochemistry

The `biogeochem` component simulates the spatial distribution of soil organic carbon and associated biogeochemical fluxes. The default implementation is [`ConstantSoilCarbonDensity`](@ref), which prescribes a spatially homogeneous organic carbon density and does not define any prognostic variables.

## [Initializers](@id soil.init)

Terrarium provides a hierarchy of initializers for `SoilModel`. The top-level initializer is [`SoilInitializer`](@ref), which composes separate sub-initializers for energy, hydrology, and biogeochemistry. It is the recommended initializer for most use cases:

```@docs; canonical = false
SoilInitializer
```

Passing a `SoilInitializer` to `SoilModel` overrides the `DefaultInitializer`:

```@example soilmodel
initializer = SoilInitializer(Float32;
    energy    = QuasiThermalSteadyState(Float32; T₀ = 2.0f0),
    hydrology = SaturationWaterTable(Float32; water_table_depth = 3.0f0),
)
model = SoilModel(grid; initializer)
```

### Spatially varying initial values

The parameters of the energy and hydrology initializers are declared as `input` variables of the model, with the values given to the initializer as their defaults:

| Initializer | Input variables |
|:--|:--|
| [`ConstantSoilTemperature`](@ref) | `initial_surface_temperature` (°C) |
| [`QuasiThermalSteadyState`](@ref) | `initial_surface_temperature` (°C), `geothermal_heat_flux` (W/m²) |
| [`ConstantSaturation`](@ref) | `initial_saturation` |
| [`SaturationWaterTable`](@ref) | `vadose_zone_saturation`, `water_table_depth` (m) |

Each value may therefore be a number, a function of the horizontal node coordinates, an array, a `Field`, or an [`AbstractFieldInitializer`](@ref).
These defaults are re-applied at every initialization, so changes to their parameters take effect, and an [`InputSource`](@ref) with the same name and units always takes precedence over them.
Spatial data, such as a regridded climatology, should preferably be supplied this way rather than as a `Field` stored in the initializer: the initializer is part of the model, which should stay independent of any particular grid and free of state.
Field initializers such as [`LatitudinalClimatology`](@ref) expose their own parameters as model parameters.
Note that `geothermal_heat_flux` is the same variable read by the [`GeothermalHeatFlux`](@ref) bottom boundary condition, so the initial profile and the boundary condition stay consistent.

```@example soilmodel
column_grid = ColumnGrid(arch, Float32, ExponentialSpacing(N = 10), 3) # three columns
energy = QuasiThermalSteadyState(Float32; T₀ = x -> 2.0f0 * x, Qgeo = 0.05f0)
model = SoilModel(column_grid; initializer = SoilInitializer(Float32; energy))
integrator = initialize(model)
interior(integrator.state.initial_surface_temperature)
```

### Energy initializers

```@docs; canonical = false
ConstantSoilTemperature
```

```@docs; canonical = false
QuasiThermalSteadyState
```

```@docs; canonical = false
PiecewiseLinearInitialSoilTemperature
```

### Hydrology initializers

```@docs; canonical = false
SaturationWaterTable
```

```@docs; canonical = false
ConstantSaturation
```

### Kernel functions

```@docs; canonical = false
compute_quasi_steady_state_temperature
compute_constant_temperature
compute_water_table_saturation
compute_constant_saturation
```

### Fallback

```@docs; canonical = false
DefaultInitializer
```

## Boundary conditions

Terrarium provides a set of named boundary condition aliases for the most common `SoilModel` configurations. These return `NamedTuple`s in the format expected by the `boundary_conditions` keyword argument of [`initialize`](@ref).

### Energy boundary conditions

```@docs; canonical = false
SoilHeatFlux
```

```@docs; canonical = false
GeothermalHeatFlux
```

```@docs; canonical = false
PrescribedSurfaceTemperature
```

```@docs; canonical = false
PrescribedBottomTemperature
```

### Hydrology boundary conditions

```@docs; canonical = false
InfiltrationFlux
```

```@docs; canonical = false
ImpermeableBoundary
```

```@docs; canonical = false
FreeDrainage
```
