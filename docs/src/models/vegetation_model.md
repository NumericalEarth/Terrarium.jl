# Vegetation models

```@meta
CurrentModule = Terrarium
```

```@setup vegmodel
using Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

[`VegetationModel`](@ref) is a standalone model for vegetation processes that simulates the integrated dynamics of plant carbon cycling, including photosynthesis, respiration, phenology, and water stress for a single plant functional type (PFT). It is designed as a standalone model wrapper around a [`AbstractVegetation`](@ref) component coupled with an atmospheric forcing, providing a focused representation of terrestrial vegetation suitable for conceptual simulations of vegetation processes. Interactions with soil are neglected or idealized (e.g. infinite soil moisture availability) where appropriate.

```@example vegmodel
arch = CPU()
grid = ColumnGrid(arch, Float32, UniformSpacing(Δz = 0.1, N = 1)) # grid with one vertical layer
model = VegetationModel(grid) # Default configuration
integrator = initialize(model)
```

```@docs; canonical = false
VegetationModel
```

```@example vegmodel
variables(model)
```

## Components

[`VegetationModel`](@ref) represents a standalone model of vegetation carbon cycle processes coupled with an interface for the [atmosphere](@ref atmosphere_docs). It consists of a `grid`, a set of [`PhysicalConstants`](@ref), and the following component processes:

| Field | Type | Scope | Process page |
|-------|------|-------|---------------|
| `vegetation` | [`AbstractVegetation`](@ref) | Coupled vegetation carbon processes | [Vegetation](@ref vegetation_docs) |
| `atmosphere` | [`AbstractAtmosphere`](@ref) | Meteorological input variables | [Atmosphere](@ref atmosphere_docs) |

Both components are summarized briefly below. See the linked process pages for more details.

### Vegetation

The `vegetation` component should be a subtype of [`AbstractVegetation`](@ref) type that represents a functional representation of the vegetation carbon cycle. The default implementation is [`VegetationCarbonCycle`](@ref) (see also the relevant doc page on [vegetation processes](@ref vegetation_docs)), which couples photosynthesis, stomatal conductance, autotrophic respiration, phenology, and carbon and vegetation dynamics. See [Vegetation](@ref vegetation_docs) for detailed descriptions of photosynthesis, respiration, phenology, and carbon dynamics implementations.

### Atmosphere

The `atmosphere` component provides time-varying meteorological forcing. The default implementation is [`PrescribedAtmosphere`](@ref), which reads air temperature, humidity, wind, radiation, and CO₂ concentration from [`InputVariable`](@ref)s and provides them as boundary conditions to the vegetation and photosynthesis processes. See [Atmosphere](@ref atmosphere_docs) for details regarding the coupling interface with the atmosphere.

## Initializers

!!! todo "Initializers and boundary conditions"
    `VegetationModel` does not yet have its own dedicated boundary conditions and initializers, but it will soon! Stay tuned!
