# Basic concepts

```@meta
CurrentModule = Terrarium
```

Terrarium should not be thought of as a single model but rather a *framework* or "toolkit" for building a wide range of different possible land models from a set of process-based building blocks. This page provides a brief overview of the core abstractions and concepts that form the basis of this framework. 

## Models

A “model” in Terrarium represents a collection of components that fully characterize the dynamics of a land simulation. Models are standalone objects that satisfy a common interface for calculating the transient state of the system. See the [Core interfaces](@ref) for further discussion.

All models must be defined as `struct`s that subtype [`AbstractModel`](@ref) and typically consist of the following components:
- A `grid` that defines both a vertical and lateral discretization of the spatial domain,
- One or more [Processes](@ref basic_concepts_processes) that define the state variables, parameters, and dynamics of the model,
- An `initializer` that defines a sequence of initialization routines (as well as any associated parameters) for the state variables declared by all of its components.
- A `timestepper` that defines the scheme for advancing in time all prognostic variables defined by the model's processes.

To see this in action, let's look again at a simple example of setting up a soil model for a single vertical column:

```@example soil_quickstart
using Terrarium

# Set up a SoilModel on a ColumnGrid with 10 exponentially spaced vertical soil layers that will run on the CPU with 32-bit precision
vert = ExponentialSpacing(Δz_min = 0.1, N = 10)
grid = ColumnGrid(CPU(), Float32, vert)
model = SoilModel(grid)
```

Here we first specify the `grid` which defines the spatial discretization (a single rectangular `column`), device architecture, and number format (`Float32`) to be used by all model components. The spacing of the $N = 10$ vertical layers is set to be exponentially increasing from a minimum thickness of 10 cm at the surface; the vertical discretization can alternatively be given as [`UniformSpacing`](@ref) for layers of constant thickness, or directly as any vector or range of cell interfaces. The `SoilModel` is then constructed directly from this `grid` using its default process configurations. Individual processes and parameterizations can also be constructed and passed in as keyword arguments, e.g,

```@example soil_quickstart
hydrology = SoilHydrology(eltype(grid), RichardsEq())
soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
model = SoilModel(grid; soil)
```

`SoilEnergyWaterCarbon` is a *coupled* process of which `SoilHydrology` is one component. Note that the keyword argument syntax `; hydrology` is a Julia shorthand for `hydrology = hydrology`.

## [Processes](@id basic_concepts_processes)

Procceses are the building blocks of all Terrarium models. Implementations of `AbstractProcess` represent physical processes characterized by:
- Zero or more state `variable`s that vary spatially across any given `grid`,
- Zero or more *parameters* that are spatially constant and defined somewhere within the process `struct`,
- One or more functions defining equations that compute quantities of interest from both the state variables and parameters defined by the process.

Concrete implementations of `AbstractProcess` are `struct` types that typically consist of zero or more parameters or parameter `struct`s (also sometimes referred to as *parameterizations*). Spatially varying parameters should generally be defined as [`InputVariable`](@ref)s (see following section) or computed by [kernel functions](@ref "Kernel programming") based on spatially invariant parameters stored in parameterization `struct`s.

## State variables and `Field`s
State variables define both the prognostic and derived (auxiliary) state of the system, discretized over the model `grid`, at any given time step. Terrarium models and processes symbolically define required state variables by defining dispatches of the `variables` method. This method should return a tuple of [`AbstractVariable`](@ref) each of which specifies the variable name, spatial dimensions, and physical units. State variables may be one of three types: **prognostic**, **auxiliary**, or **input**.

Prognostic variables fully define the state of the system at any given time step. They have corresponding tendency (time derivative) variables automatically defined for them which are typically computed in `compute_tendencies!` and updated only by the timestepper (unless otherwise documented).

Auxiliary variables are derived in each time step from the current prognostic state in `compute_auxiliary!`. These variables represent intermediary computations or terms in the equations for the prognostic tendencies. Values of auxiliary variables from previous time steps should generally not be relied upon as they may change depending on the formulation of the time stepping scheme.

Input variables represent entry points for data or physical variables that are external to a particular process or model. They are typically either supplied by [`InputSource`](@ref)s or merged with matching prognostic/auxiliary variables during initialization.

!!! info "Variables vs. Fields"
    Terrarium distinguishes between *variables* and *Fields*. Variables are symbolic objects subtyping [`AbstractVariable`](@ref) that contain metadata about the state variable defined by the process or model type. These variables are realized as [`Fields`](@ref) defined over the model grid that contain the actual numerical data of the corresponding variable.

Currently, Terrarium only supports defining state variables on a single spatial grid where the vertical dimension corresponds to the subsurface (soil) domain. This may change in the future in order to accommodate multi-layer snow and canopy processes. Note that Terrarium also currently implements only 1D (vertical) dynamics so all grid cells on the X and Y axes can be assumed independent. This is equivalent to what is typically called a *single column* model, or *column-based parameterization* in atmosphere and ocean modeling. However, building on Oceananigans means that we have a clear path to relax this assumption in the future!
