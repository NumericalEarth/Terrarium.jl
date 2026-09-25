# Core interfaces

!!! warning "🚧🚧 Under construction 🚧🚧"
    The software architecture of Terrarium.jl is still in a prototype stage and is the subject of ongoing discussion. These pages are a non-exhaustive summary of the current working concept and will be updated accordingly if and when the architecture changes.

```@meta
CurrentModule = Terrarium
```

## Modularity through multiple dispatch

Terrarium relies heavily on a core feature of Julia: *multiple dispatch*. Multiple dispatch is a programming pattern where methods (functions with a particular set of argument types) are dynamically invoked based on the (runtime) types of all their arguments. This can be contrasted to most object-oriented languages (e.g. Java, C/C++/C#, Python, etc.) where dispatch occurs based on only a single (implicit) argument, typically the type of the "object" (or `struct`) with which the method is associated.

Multiple dispatch allows us to implement model components based on specific combinations of types. As a concrete example, consider the following method signature from Terrarium's surface hydrology module that computes the aerodynamic resistance beneath the canopy:
```julia
aerodynamic_resistance(i, j, grid, fields, atmos::AbstractAtmosphere, evapotranspiration::PALADYNCanopyEvapotranspiration)
```
This method will be executed when the function `aerodynamic_resistance` is called with any implementation (subtype) of `AbstractAtmosphere` and the `PALADYNCanopyEvapotranspiration` evapotranspiration scheme. We could define further dispatches that have different implementations for specific subtypes of `AbstractAtmosphere` or alternative canopy evapotranspiration schemes which would then be invoked when the user configures a model with that choice of components.

In order to maximize code reuse and ease coupling of different components, Terrarium also defines *interfaces* for each model component (such as `AbstractAtmosphere`). These interfaces usually consist of a standard set of methods and behaviors that each component is expected to implement. As an example, `AbstractAtmosphere` defines a method of the form,
```julia
air_temperature(i, j, grid, fields, atmos::AbstractAtmosphere) = fields.air_temperature[i, j]
```
which defaults to assuming that a 2D input `Field` (see [Fields](@ref)) named `air_temperature` has been defined and is available as a property of `fields`. However, alternative implementations could derive this air temperature from other state variables or via some other function, without requiring any changes to the calling code. This kind of interface-based coupling is core to the software design of Terrarium. Method interfaces for individual process and parameterization types are summarized on their respective doc pages. Documentation and method dispatches can also be dynamically queried from the Julia REPL via the help function `?air_temperature` or with `methods(air_temperature)` and `methodswith`.

## Key abstractions

As discussed in [Basic concepts](@ref), Terrarium revolves around two key abstractions: **models** and **processes**. Models represent complete representations of a dynamical system defined over a particular spatial `grid` while processes represent individual components of that system. Both models and processes share a common method interface:

```@docs; canonical=false
variables
initialize!
compute_auxiliary!
compute_boundary_conditions!
compute_tendencies!
```
Remember from [Initialization](@ref) that `state` is a [StateVariables](@ref) structure formed during initialization of a model or process. 

Further details on these interfaces and their practical implementations are given in the following sections. 

## The `AbstractProcess` interface

In the example above, both `AbstractAtmosphere` and `PALADYNCanopyEvapotranspiration` are examples of *processes* that subtype the [`AbstractProcess`](@ref) type.

Implementations of `AbstractProcess` represent physical processes characterized by:
- Zero or more state `variable`s that vary spatially across any given `grid`,
- Zero or more *parameters* that are spatially constant and defined somewhere within the process `struct`,
- One or more functions defining equations that compute quantities of interest from both the state variables and parameters defined by the process.

Concrete implementations of `AbstractProcess` are `struct` types that typically consist of zero or more parameters or parameter `struct`s (also sometimes referred to as *parameterizations*).

Default (no-op) implementations of [`variables`](@ref) and [`initialize!`](@ref) are provided for convenience. However, to avoid ambiguity, `compute_auxiliary!` and `compute_tendencies!` **must be defined by each process** even if they are not needed.

The required additional `args` may vary for each type of `AbstractProcess`; typically they consist of either `AbstractProcess`es on which the process depends or universal parameter types like [`PhysicalConstants`](@ref). These argument types must be clearly documented and standardized for each abstract subtype of `AbstractProcess`. Changes to these interfaces, e.g. the addition of alternative call patterns with different `args`, should be made with great care and only when absolutely necessary. It is recommended for implementations to always include trailing `args...` to ensure forward compatibility.

For more details on the `state` structure and definition of `variables`, see the section on [State variables](@ref) below.

## The `AbstractModel` interface

The main difference between a "model" and a "process" in Terrarium lies in the specification of the `grid` and `initializer`. Processes should be generally be defined independently from any particular choice of initialization, boundary conditions, and `grid`. However, it is worth noting that processes can dispatch on specific types of `grid`, if necessary.

[`AbstractModel`](@ref) is parameterized by two type arguments:

```julia
abstract type AbstractModel{NF, Grid <: AbstractLandGrid{NF}} end
```

where `NF` is the numeric float type (e.g. `Float32`, `Float64`) and `Grid` is the concrete grid type. These type parameters propagate to all concrete subtypes, making the full model type checkable at compile time.

### Model type hierarchy

Terrarium defines a hierarchy of abstract model subtypes that correspond to the major components defined by most land surface models:

```
AbstractModel
└── AbstractSoilModel            # general soil models
└── AbstractSurfaceEnergyModel   # standalone land-atmosphere energy exchange
└── AbstractSnowModel            # standalone snow models
└── AbstractVegetationModel      # standalone vegetation models
└── AbstractHydrologyModel       # standalone (surface) hydrology models
└── AbstractLandModel            # fully coupled land models
```

New component models should subtype the most specific abstract type appropriate for the component being implemented.

### Required struct fields and accessor methods

By convention, concrete subtypes of `AbstractModel` are expected to store at least three standard fields that are accessed via the corresponding accessor methods:

| Field | Accessor | Description |
|-------|----------|-------------|
| `grid` | [`get_grid`](@ref) | The spatial grid defining the model domain |
| `initializer` | [`get_initializer`](@ref) | An `AbstractInitializer` defining the initial state |
| `constants` | [`get_constants`](@ref) | A [`PhysicalConstants`](@ref) struct |
| `timestepper` | [`get_timestepper`](@ref) | An [`AbstractTimestepper`](@ref) struct |

These accessors are used internally by timesteppers and simulation set-up code. If a model deviates from this convention, the relevant accessor methods outlined above must be overridden for that specific model type.

```@docs; canonical = false
get_grid
get_initializer
get_constants
get_timestepper
```

Note that subtypes of `AbstractModel` and `AbstractCoupledProcesses` also automatically inherit a default implementation of the `processes` method:
```@docs; canonical = false
processes(::Union{AbstractModel, AbstractCoupledProcesses})
```

### Standard constructor for model types

`AbstractModel` provides a universal convenience constructor that allows the `grid` to be passed as the first *positional* argument even when the concrete struct uses `@kwdef`:

```julia
SoilModel(grid; soil = SoilEnergyWaterCarbon(eltype(grid)), ...)
```

This is equivalent to `SoilModel(; grid, soil, ...)` and is the recommended calling convention for all model types.

The `grid` field of every model is an [`AbstractLandGrid`](@ref), but the positional constructor also accepts an ordinary spatial discretization (any Oceananigans `AbstractGrid`, including [`ColumnGrid`](@ref) and [`ColumnRingGrid`](@ref)) and converts it via [`create_land_grid`](@ref). The default conversion builds a ground-only [`LandGrid`](@ref); models which resolve a vertical snow or canopy profile should define their own constructor that passes the relevant process components to `create_land_grid`.

### Required method implementations

A concrete `AbstractModel` subtype must implement at minimum the same basic interface as `AbstractProcess`:

| Method | Purpose |
|--------|---------|
| `initialize!(state, model)` | Initialize all state variables from the `initializer` and any process-level initialization |
| `compute_auxiliary!(state, model)` | Compute all auxiliary (non-prognostic) variables from the current prognostic state |
| `compute_tendencies!(state, model)` | Compute tendencies of all prognostic variables |

Additionally, models with [closure relations](@ref "Closure relations") should implement:

| Method | Purpose |
|--------|---------|
| `closure!(state, model)` | Apply all forward closure relations |
| `invclosure!(state, model)` | Apply all inverse closure relations |

The typical pattern is to forward each of these model-level calls to the corresponding process-level call, passing the `grid` and `constants` explicitly.

Both models and processes may optionally implement [`compute_boundary_conditions!`](@ref), which is responsible for computing any necessary `BoundaryCondition`s for prognostic/auxiliary `Field`s defined therein. These boundary conditions may be either internal (e.g. soil/snow fluxes in a [`LandModel`](@ref)) or external (like [`SurfaceEnergyBalance`](@ref)) to the model domain. The default implementation provided for all `AbstractModel` types is a no-op.

### The `AbstractInitializer` interface

Standardized model initialization routines can be defined using the [`AbstractInitializer`](@ref) interface (see also [Initialization](@ref)). Each implementation of `AbstractModel` must allow for a user-defined `initializer` (the type can be constrained where appropriate). The simplest initializer is `DefaultInitializer`, which is a no-op that leaves all `Field`s at their default (zero) values. More complex models define their own composite initializers; for example, `SoilInitializer` composes separate initializers for the energy, hydrology, and biogeochemistry state variables. See [Soil models](@ref) for the full list of available initializer types.

```@docs; canonical = false
AbstractInitializer
```

## Closure relations

Some physical processes in Terrarium are described by conservation laws that take the form

$$\frac{\partial g(u)}{\partial t} = F(u, \ldots)$$

where $u$ is the *conserved* (prognostic) state variable and $g(u)$ is a constitutive relation mapping $u$ to the physical units matching the tendency $F$. A canonical example is the soil thermal energy balance, where the prognostic variable is the volumetric internal energy $U$ (J m⁻³), but the tendency $F$ — the divergence of the heat flux — is most naturally evaluated in terms of temperature $T$ (°C). The mapping $U \leftrightarrow T$ is therefore a *closure relation*.

[`AbstractClosureRelation`](@ref) is the base type for all such relations in Terrarium. A process that requires a closure stores its closure as a field and reports it via the `closures` method (which is auto-generated from the field types). The two primary interface methods are:

```@docs; canonical=false
closure!(state, grid, closure::AbstractClosureRelation, process::AbstractProcess, args...)
invclosure!(state, grid, closure::AbstractClosureRelation, process::AbstractProcess, args...)
```

The semantics of the forward vs. inverse closure can be summarized as:

| Method | Direction | Typical use |
|--------|-----------|-------------|
| `closure!` | conserved variable → closure variable | After each time step, compute the auxiliary variable for the closure (e.g. temperature) from the updated conserved quantity (e.g. internal energy) |
| `invclosure!` | closure variable → conserved variable | During initialization, convert user-supplied values (e.g. an initial temperature profile) into the corresponding conserved quantity (e.g. internal energy) |

Default implementations of both methods are no-ops (do nothing), so processes that do not require a closure relation do need to implement them.

Implementations of `AbstractModel` should define dispatches of `closure!` and `invclosure!` that automatically apply all closure relations defined by their processes:
```@docs; canonical = false
closure!(state, model::AbstractModel)
invclosure!(state, model::AbstractModel)
```
These methods provide a unified interface that can be used by timesteppers, callbacks, and user-defined analysis routines.

### Soil energy: temperature–enthalpy closure

The [`SoilThermodynamics`](@ref) process uses the [`SoilEnergyTemperatureClosure`](@ref) to relate volumetric internal energy $U$ (J/m³) to temperature $T$ (°C) and the liquid water fraction $l$ (-):

$$U(T) = T \cdot C(T) - L_{\text{sl}} \, \theta \, (1 - l(T))$$

where $C(T)$ is the temperature-dependent volumetric heat capacity (J / (K m³)), $L_{\text{sl}}$ is the volumetric latent heat of fusion (J / kg), and $\theta$ is the total water-ice content.

- `closure!(state, grid, ::SoilEnergyTemperatureClosure, energy, ground, constants)` — evaluates
  the forward mapping $U \mapsto T$, updating `state.temperature` and
  `state.liquid_water_fraction` (both auxiliary variables).
- `invclosure!(state, grid, ::SoilEnergyTemperatureClosure, energy, ground, constants)` — evaluates
  the inverse mapping $T \mapsto U$, updating `state.internal_energy` (the prognostic variable)
  and `state.liquid_water_fraction`. This direction is used during initialization when a
  temperature profile is given and must be converted to internal energy.

See the [Soil energy balance](@ref) doc page for further details and the full list of dispatch signatures.

### [Soil hydrology: saturation–pressure closure](@id core_interfaces_hydrology_closure)

The [`SoilHydrology`](@ref) process (Richards equation variant) uses the [`SoilSaturationPressureClosure`](@ref) to relate total water–ice saturation $\xi$ (-) to hydraulic head $\Psi$ (m) via the soil-water retention curve (SWRC):

$$\Psi = \Psi_m(\xi) + \Psi_z + \Psi_h$$

where $\Psi_m$ is the matric potential given by the SWRC, $\Psi_z$ is the elevation head, and $\Psi_h$ is the hydrostatic head contributed by free water above the water table.

- `closure!(state, grid, ::SoilSaturationPressureClosure, hydrology, soil)` — evaluates the
  forward mapping $\xi \mapsto \Psi$, updating the auxiliary `state.pressure_head`.
- `invclosure!(state, grid, ::SoilSaturationPressureClosure, hydrology, soil)` — evaluates the
  inverse mapping $\Psi \mapsto \xi$, updating the prognostic `state.saturation_water_ice`.

See the [Soil hydrology](@ref) doc page for further details and the full list of dispatch signatures.

### Implementing a new closure

To add a closure relation to a new process:

1. Define a concrete subtype of `AbstractClosureRelation` (or a process-specific abstract
   subtype, e.g. `AbstractEnergyClosure`)
2. Implement `variables(::MyClosureRelation)` which should, at minimum, define the relevant `auxiliary`
   variable for the closure (`temperature` and `pressure_head` for soil energy and hydrology respectively)
3. Implement `closure!` and `invclosure!` dispatching on the process type
4. Store an instance of the closure as a field of the process `struct`; the `closures`
   generated function will then pick it up automatically
