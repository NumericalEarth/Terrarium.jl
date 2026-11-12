# Configuring models

```@meta
CurrentModule = Terrarium
```

```@setup configuring
using Terrarium
using InteractiveUtils
```

## Overview

Terrarium models are constructed by passing a `grid` and optional keyword arguments that customize component choices and parameters. This page shows how to build [`SoilModel`](@ref), [`LandModel`](@ref), and [`VegetationModel`](@ref) with various configurations, and how to inspect their state variables.

## Setting up a [`SoilModel`](@ref)

All models in Terrarium are constructed first and foremost from a `grid` representing the spatial discretization. As with many of our other examples, we will use here a simple [`ColumnGrid`](@ref) representing a single vertical column:

```@example configuring
num_columns = 1
grid = ColumnGrid(ExponentialSpacing(N = 10), num_columns)
```

!!! tip "Modular grids"
    Terrarium models are processes and designed to be largely invariant to the specific choice of grid. You can, for example, easily increase `num_columns` to add more (independent) columns or replace the above `grid` with a GPU-based grid with more soil layers `ColumnGrid(GPU(), ExponentialSpacing(N = 30))`. Alternatively, we could use a global [`ColumnRingGrid`](@ref) as shown in [this example](@ref soil_heat_global).

We will start here by constructing a simple [`SoilModel`](@ref) using its defaults:

```@example configuring
model = SoilModel(grid)
```

Calling `variables(model)` will give us a more detailed look at the state variables defined by this model:

```@example configuring
variables(model)
```

By default `SoilModel` uses a soil hydrology scheme in which water flows vertically following the Richards equation. Suppose we want to turn that off and treat the soil water/ice as constant in time instead. The (hopefully) easiest way to learn how to do this would be to go look at the documentation page for [soil hydrology](@ref "Soil hydrology"). But let's be lazy and try to figure it ourselves.

We can see above that `model` has a property `soil` (see also the page for [`SoilModel`](@ref "Soil models")). Let's inspect that:

```@example configuring
model.soil
```

We can see that `soil` is a coupled process type [`SoilEnergyWaterCarbon`](@ref) with processes `energy`, `hydrology`, and `biogeochem`. Looking at `model.soil.hydrology`:

```@example configuring
model.soil.hydrology
```

we can see that it has a property `vertical_flow`. As a general rule, most process and parameterization types in Terrarium declare abstract base types which can help us figure out what our choices are. We can use some built-in Julia tools for this.

```@example configuring
# First get the type of vertical_flow
vftype = typeof(model.soil.hydrology.vertical_flow)
# Then get the "super" type (i.e. the abstract type it inherits from)
suptype = supertype(vftype)
# Then list all of the subtypes of the abstract type
subtypes(suptype)
```

Aha! We see here a second implementation [`NoFlow`](@ref). This happens to correspond to the configuration option for [`SoilHydrology`](@ref) that keeps the soil water immobile. We can select it by changing `vertical_flow` when constructing the process and then building back up the `SoilModel` from there.

```@example configuring
hydrology = SoilHydrology(eltype(grid), vertical_flow = NoFlow())
soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
model = SoilModel(grid; soil)
variables(model)
```

We could also do something similar for the soil thermal properties:

```@example configuring
thermal_properties = SoilThermalProperties(eltype(grid))
energy = SoilThermodynamics(eltype(grid); thermal_properties)
hydrology = SoilHydrology(eltype(grid), NoFlow()) # also a valid constructor, same as above
biogeochem = ConstantSoilCarbonDensity(eltype(grid))
soil = SoilEnergyWaterCarbon(eltype(grid); energy, hydrology, biogeochem)
model = SoilModel(grid; soil)
```

!!! info "Number formats"
    Note the use of `eltype(grid)` when constructing process and parameterization types. This is important and mandatory! The number formats for all components must match, otherwise you will get errors during construction.

## Setting up a [`LandModel`](@ref)

[`LandModel`](@ref) couples soil, surface energy and water fluxes, and optionally vegetation. A key option here is whether or not the land surface has vegetation.

### No vegetation (bare soil)

For bare soil simulations, we can simply pass `vegetation=nothing`:

```@example configuring
model = LandModel(grid; vegetation = nothing)
variables(model)
```

The bare soil configuration uses simpler hydrology (no canopy interception or canopy evapotranspiration) and focuses on direct soil-atmosphere exchanges. You can still customize soil components the same way as with `SoilModel`:

```@example configuring
hydrology = SoilHydrology(eltype(grid), vertical_flow = RichardsEq())
soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
model = LandModel(grid; vegetation=nothing, soil)
```

### Customized vegetation processes

You can customize individual vegetation processes by passing them to [`PrognosticVegetation`](@ref):

```@example configuring
photosynthesis = LUEPhotosynthesis(eltype(grid))
stomatal_conductance = MedlynStomatalConductance(eltype(grid))
carbon_dynamics = PALADYNCarbonDynamics(eltype(grid))
vegetation = PrognosticVegetation(eltype(grid);
    photosynthesis,
    stomatal_conductance,
    carbon_dynamics
)
model = LandModel(grid; vegetation)
```

We could also customize surface energy balance processes, e.g. prescribed surface energy balance fluxes:

```@example configuring
radiative_fluxes = PrescribedRadiativeFluxes(eltype(grid))
turbulent_fluxes = PrescribedTurbulentFluxes(eltype(grid))
seb = SurfaceEnergyBalance(eltype(grid);
    radiative_fluxes,
    turbulent_fluxes
)
vegetation = PrognosticVegetation(eltype(grid))
model = LandModel(grid; vegetation, surface_energy_balance = seb)
```

## Changing parameters

Model parameters are stored as fields in their respective process implementations or the parameterization types therein. One simple way to change parameter values is to simply modify the values when constructing the relevant process types.

As an example, consider again the above example with `SoilModel`. Suppose we want to change the thermal conductivity of mineral and organic soil material. We could do as follows:

```@example configuring
conductivities = SoilThermalConductivities(eltype(grid); mineral = 3.0, organic = 0.8)
thermal_properties = SoilThermalProperties(eltype(grid); conductivities)
energy = SoilThermodynamics(eltype(grid); thermal_properties)
soil = SoilEnergyWaterCarbon(eltype(grid); energy)
model = SoilModel(grid; soil)
```

!!! todo "Improved parameter handling"
    We will soon make it easier to collect, select, and modify parameters *after* model construction using a parameter handling system based on [`ModelParameters`](https://github.com/rafaqz/ModelParameters.jl). Stay tuned!
