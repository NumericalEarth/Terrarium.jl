# State variables

```@meta
CurrentModule = Terrarium
```

```@setup variables
using Terrarium
using Oceananigans
```

## Overview

Models and processes must define dispatches of the [`variables`](@ref) method that return a `Tuple` of variables subtyping `AbstractVariable`:

```@docs; canonical = false
PrognosticVariable
AuxiliaryVariable
InputVariable
```
Variables are typically constructed using one of the convenience functions [`prognostic`](@ref), [`auxiliary`](@ref), or [`input`](@ref).

These variable definitions are purely symbolic; they do not hold any data and cannot be used for computation. Constructing [`StateVariables`](@ref) from a model, process, or [`Variables`](@ref) container (see following sections) results in corresponding [Fields](@ref) being allocated for each variable. 

A default implementation of `variables` is provided for all [`AbstractModel`](@ref) and `AbstractCoupledProcesses` types that automatically collects variables from all [`AbstractProcess`](@ref) types defined therein:

```@docs; canonical = false
variables(obj::Union{AbstractCoupledProcesses, AbstractModel})
```

Most state variables will thus be defined by implementation of `AbstractProcess`. As an example, suppose we are implementing a new process `MyProcess` and we want to define the necessary state variables. We do this by defining a new dispatch of the `variables` method:
```julia
struct MyProcess{NF} <: Terrarium.AbstractProcess{NF} end

Terrarium.variables(::MyProcess) = (
    Terrarium.prognostic(:progvar, XYZ()),
    Terrarium.auxiliary(:auxvar, XYZ()),
    Terrarium.auxiliary(:bc, XY()),
    Terrarium.input(:input, XY())
)
```
This will result in a total of five state variables being allocated upon initialization: one input variable, two auxiliary variables named `auxvar` and `bc` and one prognostic variable named `progvar` along with its corresponding tendency variable which is created automatically. The second argument to the variable metadata constructors `prognostic` and `auxiliary` is a subtype of `VarDims` which specifies on which spatial dimensions the state variable should be defined. [`XYZ()`](@ref) corresponds to a 3D `Field` which varies both laterally and with depth. [`XY()`](@ref) corresponds to a 2D field which is discretized along the lateral X and Y dimensions only.

## Merging and promotion rules

Models rarely consist of only a single process, and many processes inevitably need access to the same physical variables. Terrarium handles this by automatically merging variables that are duplicated between model components. Duplicate variables must have the same dimensions and physical units in order to be merged, otherwise an error will be raised during initialization. Variables with identical names, dimensions, and units but different types (i.e. prognostic vs. auxiliary vs. input) are merged according to two simple promotion rules:

- `input` variables matching a corresponding `prognostic` or `auxiliary` variable are "promoted" to the corresponding prognostic/auxiliary type
- Clashing definitions of `prognostic` and `auxiliary` definitions are disallowed and result in an error

The reason for the latter rule is that variables which are treated as prognostic (integrated by the time stepper) cannot simultaneously be treated as auxiliary (i.e. derived) from the prognostic state.

These merging and promotion rules are defined and applied by the [`Variables`](@ref) container:

```@docs; canonical = false
Variables
```

`Variables` also provides helpful dispatches of `show` which concisely summarize the variables in the container after merging/promotion rules have been applied.

## Auxiliary variables with derived `Field`s

As briefly discussed in the doc section on [Fields](@ref), Oceananigans features a powerful system of defining `Field`s that are lazily computed as operators defined over the model `grid`. Terrarium takes advantage of this by permitting custom `Field` constructors in the definition of `auxiliary` variables:

```@docs; canonical = false
auxiliary(::Symbol, ::VarDims, ::Any, ::Any)
```

Here the `ctor` argument should be a function with signature `(process::ProcessType, grid, clock, fields)` that returns an Oceananigans operator-derived `Field`.

As a simple example, suppose we want to define an auxiliary variable `C` for the hypothetical process type `Pythagoras` which is derived from two other state variables (`Field`s) `length` and `width` via the relation $C = \sqrt{A^2 + B^2}$. This could be accomplished as follows:

```@example variables
struct Pythagoras{NF} <: Terrarium.AbstractProcess{NF} end

Terrarium.variables(pythag::Pythagoras) = (
    Terrarium.auxiliary(:hypotenuse, XY(), hypotenuse, pythag),
    Terrarium.input(:length, XY()),
    Terrarium.input(:width, XY())
)

function hypotenuse(grid, clock, fields, ::Pythagoras)
    hypotenuse = sqrt(fields.length^2 + fields.width^2)
    return hypotenuse
end

grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
state = StateVariables(Pythagoras{Float64}(), grid)
state.hypotenuse
```

The benefit of using this pattern is that we save memory overhead by avoiding the allocation of new `Field` memory for `hypotenuse`. Instead, the expression in the above function is evaluated each time `hypotenuse` is indexed:

```@example variables
set!(state.length, 2)
set!(state.width, 3)
state.hypotenuse[1,1,1]
```

!!! warning "Operators vs. Fields"
    Note that `hypotenuse` is of type [`UnaryOperation`](@extref Oceananigans.AbstractOperations.UnaryOperation) not [`Field`](@extref Oceananigans.Fields.Field). While operations typically behave like `Field`s and can be indexed normally in kernel functions, they are not identical and this can lead to errors if some downstream code assumes `hypotenuse` to be an actual `Field` rather than simply an array-like type. In such cases, one should instead return `Field(state.hypotenuse)` in the above constructor and then call `compute!(state.hypotenuse)` in each time step. See also the following example.

As another more concrete example, consider the [`soil_moisture_limiting_factor`](@ref) `Field` constructor for [`FieldCapacityLimitedPAW`](@ref):

```julia
function soil_moisture_limiting_factor(grid, clock, fields, ::FieldCapacityLimitedPAW)
    Δz = zspacings(ground_domain(grid), Center(), Center(), Center())
    β = Integral(fields.plant_available_water * fields.root_fraction / Δz, dims = 3)
    return Field(β)
end
```
The resulting factor `β` is an integral over the enclosed function of `plant_available_water` and `root_fraction`. Since [`Integral`](@extref Oceananigans.AbstractOperations.Integral) is a so-called "reduction" operator, it cannot be lazily computed and instead [`compute!`](@extref Oceananigans.Fields.compute!) must be called on the `Field` in each time step, e.g:

```julia
compute!(state.soil_moisture_limiting_factor)
```
