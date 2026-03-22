# Baisc concepts

```@meta
CurrentModule = Terrarium
```

```@setup basics
using Terrarium
```

Terrarium should not be thought of as a single model but rather a *framework* or "toolkit" for building a wide range of different possible land models from a set of process-based building blocks. This page provides a brief overview of the core abstractions and concepts that form the basis of this framework. 

## Models

A “model” in Terrarium represents a collection of components that fully characterize the dynamics of a land simulation. All models must be defined as `struct`s that subtype [`AbstractModel`](@ref) and typically need consist of the following components:
- A `grid` that defines both a vertical and lateral discretization of the spatial domain,
- One or more [Processes](@ref) that define the state variables, parameters, and dynamics of the model,
- An `initializer` that defines a sequence of initialization routines (as well as any assocaited parameters) for the state variables declared by all of its components.

To see this in action, let's look again at a simple example of setting up a soil model for a single vertical column:

```@example soil_quickstart
using Terrarium

# Set up a SoilModel on a ColumnGrid with 10 exponentially spaced vertical soil layers that will run on the CPU with 32-bit precision
vert = ExponentialSpacing(Δz_min = 0.1, N = 10)
grid = ColumnGrid(CPU(), Float32, vert)
model = SoilModel(grid)
```

Here we first specify the `grid` which defines the spatial discretization (a single rectangular `column`), device architecture, and number format (`Float32`) to be used by all model components. The spacing of the $N = 10$ vertical layers is set to be exponentially increasing from a minimum thickness of 10 cm at the surface; note that alternative options for the vertical discretization include [`UniformSpacing`](@ref) and [`PrescribedSpacing`](@ref).

The `SoilModel` is then constructed directly from this `grid`. If the 

## Processes

Implementations of `AbstractProcess` represent physical processes characterized by:
- Zero or more state `variable`s that vary spatially across any given `grid`,
- Zero or more *parameters* that are spatially constant and defined somewhere within the process `struct`,
- One or more functions defining equations that compute quantities of interest from both the state variables and parameters defined by the process.

Concrete implementations of `AbstractProcess` are `struct` types that typically consist of zero or more parameters or parameter `struct`s (also sometimes referred to as *parameterizations*).

### Parameterizations

## State variables
State variables are symbolically defined for each process in their respective implementations of the `variables` method by returning instances of [`AbstractVariable`](@ref) which define (at minimum) its name, dimensionality, and physical units. State variables may be one of three types:
```@docs; canonical = false
PrognosticVariable
AuxiliaryVariable
InputVariable
```
which can also be instantiated using one of the convenience methods [`prognostic`](@ref), [`auxiliary`](@ref), or [`input`](@ref).

!!! info Variables vs. `Field`s
    Within the context of defining and handling model/process state, Terrarium distinguishes between *variables* and *Fields*. Variables are symbolic objects subtyping [`AbstractVariable`](@ref) that contain metadata about the state variable defined by the process or model type. These variables are realized as [`Fields`](@ref) defined over the model grid.

As a simple example, suppose we are implementing a new process `MyProcess` and we want to define the necessary state variables. We do this by defining a new dispatch of the `variables` method:
```julia
variables(::MyProcess) = (
    prognostic(:progvar, XYZ()),
    auxiliary(:auxvar, XYZ()),
    auxiliary(:bc, XY()),
    input(:input, XY())
)
```
This will result in a total of five state variables being allocated upon initialization: two auxiliary variables named `auxvar` and `bc` as well as one prognostic variable named `progvar` along with its corresponding tendency variable which is created automatically. The second argument to the variable metadata constructors `prognostic` and `auxiliary` is a subtype of `VarDims` which specifies on which spatial dimensions the state variable should be defined. `XYZ()` corresponds to a 3D `Field` which varies both laterally and with depth.

Currently, Terrarium only supports a single 3D grid representing variables defined in the soil domain, though this may change in the future in order to accommodate multi-layer snow and canopy processes. `XY()` corresponds to a 2D field which is discretized along the lateral dimension only. Note that Terrarium also currently supports only 1D (vertical) dynamics so all grid cells on the X and Y axes will be assumed independent. This is equivalent to what is typically called a *single column* model, or *column-based parameterization* in atmosphere and ocean modeling. However, building on Oceananigans means that we have a clear path to relax this assumption in the future!

## Initialization

A key abstraction in Terrarium is the [`initialize`](@ref) method:

```docs; canonical = false
initialize(model::AbstractModel)
initialize(process::AbstractProcess, grid::AbstractGrid)
```

Calling `initialize` on a model or process type returns a [`StateVariables`](@ref) structure containing all of the initialized [`Field`](@ref)s corresponding to state variables defined by the model/process. By default, `Field`s will be initialized with zeros. Some state variable types, such as `input` variables, allow for the specification of alternative default values.

Of course, very few models can do anything useful or interesting starting from (literally) zero. Terrarium provides three complementary mechanisms through which initialization routines for model/process state variables can be defined:

- Direct initialization via the `initializers` keyword argument of [`initialize`](@ref). The keyword argument must be a `NamedTuple` where the keys correspond to the name of the state variable and the values are either scalars, arrays matching the size of the model `grid`, or functions of the form `f(coords...)` where `coords` are the non-`Flat` dimensions of `grid`. For column-based grids, this is generally `f(x,z)` with `x` corresponding to a column index.
- [`AbstractInitializer`](@ref) types encapsulate a sequence of initialization routines. These `Initializer` types can be supplied to subtypes of `AbstractModel` during construction. Models can/should typically define corresponding `AbstractInitializer` types that represent common initilization strategies appropriate for the processes included in that model; e.g. the `SoilModel` defines [`SoilInitializer`](@ref) with process-specific initialization types like [`QuasiThermalSteadyState`](@ref) and [`SaturationWaterTable`](@ref).
- Model/process speciic dispatches of [`initialize!`](@ref) can be defined for process initialization logic that should be run regardless of the choice of prognostic state variable initialization.

As a general rule, these initializers are invoked in the order that they are listed above. Implementations of `initilaize!` should generally assume that the prognostic state has already been initilaized by corresponding `Field` or model initializers.

## Timestepping

Terrarium explicitly separates process computations in `compute_auxiliary!` and `compute_tendencies!` from the choice of time stepping scheme. As a general rule, only models can be configured for timestepping. A model can be initialized for timestepping via

```@doc; canonical = false
initialize(model::AbstractModel, timestepper::AbstractTimeStepper, inputs::InputSource...)
```

This will return a [`ModelIntegrator`](@ref):

```@doc; canonical = false
ModelIntegrator
```

A single time step of the `ModelIntegrator` roughly involves four steps:

1. Invoke [`update_inputs!`](@ref) to populate all input `Field`s from their respective `InputSource`s based on the current `clock` time,
2. Invoke [`compute_auxiliary!`](@ref) on all model components to derive auxiliary state variables from the current prognostic state,
3. Invoke [`compute_tendencies!`](@ref) on all model components to calculate tendencies for all prognostic variables,
4. Apply the tendencies computed in step (3) to update the prognostic state based on the selected timestepping scheme. Higher order explicit or implicit timesteppers may repeat steps 1-3 multiple times within a single time step.

As an example, let's consider the construction and initalization of a [`SoilModel`](@ref):

```@example basics
arch = CPU()
grid = ColumnGrid(arch, Float32, ExponentialSpacing(N=10))
model = SoilModel(grid)
integrator = initialize(model, ForwardEuler(Float32))
```

Here `integrator` corresponds to a `ModelIntegrator` configured for a [`ForwardEuler`](@ref) time stepping scheme. `ModelIntegrator` defines dispatches for [`timestep!`](@ref) and [`run!`](@ref) that allow for open-ended simulations starting from the initialized state. We can use [`timestep!`](@ref) to take a single time step,

```@example basics
timestep!(integrator)
```

and [`run!`](@ref) to advance multiple timesteps over a specified period,

```@example basics
run!(integrator, period=Hour(1))
```

Alternatively, the `integrator` can be passed into an Oceananigans [`Simulation`](https://clima.github.io/OceananigansDocumentation/stable/simulations/simulations_overview) like so,

```@example basics
sim = Simulation(integrator; stop_time = 3600.0, Δt = 60.0)
run!(sim)
```

where the `stop_time` here is the stopping time of the simulation (in seconds) and `Δt` is the timestep size.
