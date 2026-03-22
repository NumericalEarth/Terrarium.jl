# Core interfaces

!!! warning "🚧🚧 Under construction 🚧🚧"
    The software architecture of Terrarium.jl is still in an early prototype stage and is the subject of ongoing discussion. This page is a non-exhaustive summary of the current working concept and will be updated accordingly if and when the architecture changes.

```@meta
CurrentModule = Terrarium
```

## Modularity through multiple dispatch

Terrarium relies heavily on a core feature of Julia: *multiple dispatch*. Multiple dispatch is a programming pattern where methods (functions with a paritcular set of argument types) are dynamically invoked based on the (runtime) types of all their arguments. This can be constrasted to most object-oriented languages (e.g. Java, C/C++/C#, Python, etc.) where dispatch occurs based on only a single (implicit) argument, typically the type of the "object" (or `struct`) with which the method is associated.

Multiple dispatch allows us to implement model components based on specific combinations of types. As a concrete example, consider the following method signature from Terrarium's surface hydrology module that computes the aerodynamic resistance beneath the canopy:
```julia
aerodynamic_resistance(i, j, grid, fields, atmos::AbstractAtmosphere, evtr::PALADYNCanopyEvapotranspiration)
```
This method will be executed when the function `aerodynamic_resistance` is called with any implementation (subtype) of `AbstractAtmosphere` and the `PALADYNCanopyEvapotranspiration` evapotranpsiration scheme. We could define further dispatches that have different implementations for specific subtypes of `AbstractAtmosphere` or alternative canopy evapotranspiration schemes which would then be invoked when the user configures a model with that chocie of components.

In order to maximize code reuse and ease coupling of different components, Terrarium also defines *interfaces* for each model component (such as `AbstractAtmosphere`). These interfaces usually consist of a standard set of methods and behaviors that each component is expected to implement. As an example, `AbstractAtmosphere` defines a method of the form,
```julia
air_temperature(i, j, grid, fields, atmos::AbstractAtmosphere) = fields.air_temperature[i, j]
```
which defaults to assuming that a 2D input `Field` (see [Fields](@ref)) named `air_temperature` has been defined and is available as a property of `fields`. However, alternative implementations could derive this air temperature from other state variables or via some other function, without requiring any changes to the calling code. This kind of interface-based coupling is core to the software design of Terrarium.

## Core interfaces


```@docs; canonical=false
variables(proc::AbstractProcess)
initialize!
compute_auxiliary!
compute_tendencies!
```

### The `AbstractProcess` interface

In the example above, both `AbstractAtmosphere` and `PALADYNCanopyEvapotranspiration` are examples of "processes* that subtype the `AbstractProcess` type.

Implementations of `AbstractProcess` represent physical processes characterized by:
- Zero or more state `variable`s that vary spatially across any given `grid`,
- Zero or more *parameters* that are spatially constant and defined somewhere within the process `struct`,
- One or more functions defining equations that compute quantities of interest from both the state variables and parameters defined by the process.

Concrete implementations of `AbstractProcess` are `struct` types that typically consist of zero or more parameters or parameter `struct`s (also sometimes referred to as *parameterizations*).

Default (no-op) implementations of [`variables`](@ref) and [`initialize!`](@ref) are provided for convenience. However, to avoid ambiguity, `compute_auxiliary!` and `compute_tendencies!` **must be defined by each process** even if they are not needed.

The required additional `args` are may vary for each type of `AbstractProcess`; typically they consist of either other `AbstractProcess`es on which the process depends or universal parameter types like [`PhysicalConstants`](@ref). These argument types must be clearly documented and standardized for each abstract subtype of `AbstractProcess`. Changes to these interfaces, e.g. the addition of alternative call patterns with different `args`, should be made with great care and only when absolutely necessary. It is recommended for implementations to always include trailing `args...` to ensure forward compatibility.

For more details on the `state` structure and definition of `variables`, see the section on [State variables](@ref) below.

#### Coupled processes

Terrarium also defines a base type for *coupled processes*, i.e. processes that define a physical coupling of two or more sub-processes:
```@docs; canonical = false
AbstractCoupledProcesses
```

It is important to note that `AbstractCoupledProcesses` is itself a subtype of `AbstractProcess` and thus subtypes must also implement the same interface for `AbstractProcess`. The main reason to subtype `AbstractCoupledProcesses` instead of `AbstractProcess` is to define a coupling interface for logical groups of processes, e.g. [`SoilEnergyWaterCarbon`](@ref) or [`VegetationCarbon`](@ref). Additionally, concrete implementations may provide specialized computation kernels that *fuse* the respective kernel functions from each of the sub-processes for greater efficiency.

### The `AbstractModel` interface

The main difference between a "model" and a "process" in Terrarium lies in the specification of the `grid` and `initializer`. Processes should be generally be defined independently from any particular choice of initialization, boundary conditions, and `grid`. However, it is worth noting that processes can dispatch on specific types of `grid`, if necessary.

Note that subtypes of `AbstractModel` and `AbstractCoupledProcesses` also automatically inherit a default implementation of the `processes` method:
```@docs; canonical = false
processes(::Union{AbstractModel, AbstractCoupledProcesses})
```