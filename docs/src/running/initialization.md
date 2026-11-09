# Initialization

```@meta
CurrentModule = Terrarium
```

## Overview

Model and process types in Terrarium are *stateless* and *immutable*; i.e. they only specify parameters and model configuration. To allocate state variables for a model or process, we use the [`StateVariables`](@ref) constructor:

```@docs; canonical = false
StateVariables(model::AbstractModel)
StateVariables(process::AbstractProcess{NF}, grid::AbstractLandGrid{NF}) where {NF}
```

which will create and return a [`StateVariables`](@ref) structure containing all of the initialized [`Field`](@ref)s corresponding to state variables defined by the model/process.

```@docs; canonical = false
StateVariables
```

By default, `Field`s will be initialized with zeros. Some state variable types, such as `input` variables, allow for the specification of alternative default values.

Of course, very few models can do anything useful or interesting starting from (literally) zero. Terrarium provides three complementary mechanisms through which initialization routines for model/process state variables can be defined:

- Direct initialization via the `initializers` keyword argument of [`initialize`](@ref)
- [`AbstractInitializer`](@ref) types specified in models encapsulate a sequence of initialization routines
- Model and process specific dispatches of `initialize!`

As a general rule, these initializers are invoked in the order that they are listed above. Implementations of `initialize!` should generally assume that the prognostic state has already been initialized by corresponding `Field` or model initializers.

## Direct initialization of `Field`s

The keyword argument must be a `NamedTuple` where the keys correspond to the name of the state variable and the values are either scalars, arrays matching the size of the model `grid`, or functions of the form `f(coords...)` where `coords` are the non-[`Flat`](@extref Oceananigans.Grids.Flat) dimensions of `grid`. For column-based grids, this is generally `f(x,z)` with `x` corresponding to a column index.

## Field initializers

Reusable, parameterized initializers of a single `Field` subtype [`AbstractFieldInitializer`](@ref).
They are accepted everywhere a `set!`-compatible value is: in the `initializers` keyword argument, as the default of an `input` variable, or as a parameter of a model initializer.
In the latter case, their `@param` fields become parameters of the model.

```@docs; canonical = false
AbstractFieldInitializer
LatitudinalClimatology
```

Field initializers that depend on geographic position can use `λnodes` and `φnodes`, which return longitudes and latitudes (degrees) on both an Oceananigans `LatitudeLongitudeGrid` and a [`ColumnRingGrid`](@ref).
On the latter, they are indexed by active column.

## Model initializers

These `Initializer` types can be supplied to subtypes of [`AbstractModel`](@ref) during construction. Models can/should typically define corresponding [`AbstractInitializer`](@ref) types that represent common initialization strategies appropriate for the processes included in that model; e.g. the [`SoilModel`](@ref) defines [`SoilInitializer`](@ref) with process-specific initialization types like [`QuasiThermalSteadyState`](@ref) and [`SaturationWaterTable`](@ref).

Model initializers may also implement [`variables`](@ref) to declare `input` variables for their parameters.
These are collected by `variables(model)` alongside the process variables, which allows initial values to vary in space and to be supplied by [`InputSource`](@ref)s.

## Built-in initialization routines

Model/process specific dispatches of [`initialize!`](@ref) can be defined for process initialization logic that should be run regardless of the choice of prognostic state variable initialization.
