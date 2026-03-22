# Numerical core

```@meta
CurrentModule = Terrarium
```

Terrarium is based on the numerics and finite-volume method (FVM) operators provided by [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). All state variables are realized as Oceananigans [`Field`s](https://clima.github.io/OceananigansDocumentation/stable/fields/) defined over a particular choice of [grid](https://clima.github.io/OceananigansDocumentation/stable/grids/) which discretizes physical space into a finite number of *volumes* or "cells". This page gives a brief overview of the basic concepts behind the numerical building blocks of Oceanangians and Terrarium.

## Grids
Terrarium defines its own `AbstractLandGrid` types as wrappers around Oceananigans `AbstractGrid` types (note that these wrappers may be consolidated into full-fledged `AbstractGrid`s in the future). At the moment, Terrarium grids are based primarily on the Oceanangians `RectilinearGrid`, which represents a rectangular volume divided orthogonally into smaller control volumes along orthogonal X, Y, and Z axes. There are currently two `AbstractLandGrid` implementations:

- [`ColumnGrid`](@ref) which represents an unstructured collection of 1D vertical columns discretized along the `Z` axis with the `X` axis corresponding to a non-spatial index over each of those columns and the `Y` axis ignored (assigned a `Flat` grid "topology" in the underlying `RectilinearGrid`).
- [`ColumnRingGrid`](@ref) which is the same as `ColumnGrid` but with columns ordered according to a user-specified `RingGrid` from [RingGrids.jl](https://speedyweather.github.io/SpeedyWeatherDocumentation/stable/ringgrids/). Under this configuration, each dimension along the `X` axis corresponds to a point in the `RingGrid`, allowing for direct translation between Terrarium and RingGrids `Field`s. This is primarily motivated by our goal to couple Terrarium with [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl).

All Terrarium `AbstractLandGrid` implementations are required to implement the following methods:
- `architecture(grid)` (from Oceananigans) which returns the `Architecture` (e.g. `CPU` or `GPU`) on which the grid is defined,
- `get_field_grid(grid)` which returns the underlying `Oceananigans` grid.

The vertical `Z`-axis is currently limited to representing the subsurface (typically soil) domain, though we plan to expand this to include snow and canopy layers in the future.

!!! info Ordering of vertical layers
    Oceananigans follows a **positive-upwards** convention for the vertical axis. This also implies that the vertical layer at the first index of a 3D Terrarium field is actually the **bottom-most layer** in the ground/soil column; i.e. `interior(temperature)[1,1,1]` for a `Field` called `temperature` would correspond to the vertical layer at the bottom of the first grid cell (i.e. `X = 1`). To get the topmost layer, use instead `interior(temperature)[1,1,end]`. Note that here [`interior`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Fields.interior-Tuple{Field}) is a function from Oceananigans that retrieves a view of the `Field` excluding halo (boundary condition) cells.

For more information on how Oceananigans grids work, we recommend reading the [corresponding page](https://clima.github.io/OceananigansDocumentation/stable/grids/) in the Oceananigans documentation.

## Fields

`Field`s define `Array`s of data that align with a grid. In Terrarium, `Field`s are used exclusively to represent state and input variables that vary in space (and time). All `Field`s have a corresponding `location(field) = (LX, LY, LZ)` which describes how the `Field` aligns with the underlying grid. Each of `LX`, `LY`, and `LZ` can take on values of either `Center`, `Face`, or `nothing`, where `Center` refers to the spatial centerpoint of the cell (representing the spatial average over that volume in the typical finite-volume sense) and `Face` corresponds to values defined at the interfaces such as fluxes or conductivities. A `nothing` location refers to a reduced quantity averaged or integrated over the corresponding axis.

In addition to holding data, `Field`s also can also store one or more associated [boundary conditions](https://clima.github.io/OceananigansDocumentation/stable/fields/#Halo-regions-and-boundary-conditions) as well as "halo" regions used to define them. Boundary conditions can either be Dirichlet-type (`ValueBoundaryCondition`), Neumann-type (`GradientBoundaryCondition`), or fluxes (`FluxBoundaryCondition`). Boundary conditions can be defined using functions, scalars, `Array`s or other `Field`s.

`Field`s can also serve as the basis for building [expression trees](https://clima.github.io/OceananigansDocumentation/stable/operations/): e.g. writing `C = A * B + 1` where `A` and `B` are both `Field`s results in a derived `Field` `C` that will lazily compute the expression when accessed. Similarly, `Fields` can be reduced over a grid axis using operations like `Integral` and `Average`, though these must be explciitly computed via `compute!`.

For a more detailed overview of `Field`s, we recommend checking out the [corresponding page](https://clima.github.io/OceananigansDocumentation/stable/fields/) in the Oceananigans documentation.

## Parallel programming with KernelAbstractions.jl

Like Oceananigans and SpeedyWeather, the numerical core of Terrarium is based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) which allows for device-agnostic *parallel programming* via computational *kernels*. The purpose of this design is to allow for highly efficient, scalable, and cross-architecture (e.g. CPU/GPU/TPU) parallelization of all numerical computations. Kernels are defined using the `@kernel` macro from KernelAbstractions:

```julia
@kernel function square_kernel!(output, input, other_args...)
    i, j, k = @index(Global, NTuple)
    output[i, j, k] = input[i, j, k]^2
end
```
where `input` and `output` are here assumed to be 3D `Array`s or `Field`s. Note that kernels can also accept any arbitrary number of arguments of any plain data type (including immutable `struct`s) for which `isbits(arg)` is true.

Kernels represent self-contained programs that can be execute in parallel by large number of worker threads or processes. Intuitively, you can think of the kernel function as the body of a `for` loop that executes over all finite volumes defined on the grid. The [`@index`](https://juliagpu.github.io/KernelAbstractions.jl/stable/api/#KernelAbstractions.@index) macro extracts the index of the executing thread within the parallel kernel; `Global` here refers to the index across all workgroups (or "blocks" in CUDA language).

Terrarium and Oceananigans kernels are typically launched over a `grid` (or some subset thereof) via `launch!`,
```julia
launch!(architecture(grid), grid, :xyz, square_kernel!, output_field, input_field)
```
where `:xyz` indicates the dimensions of the grid over which the kernel should be executed. For 2D (e.g. surface) computations, this would instead be `:xy`. In Terrarium, we typically use a slightly condensed variant of `launch!` for convenience:

```julia
launch!(grid, XYZ, square_kernel!, output_field, input_field)
```
which automatically passes `grid` as the second argument to the kernel. Note that, in order to make use of this convenience dispatch of `launch!`, the above kernel would need to be modified to have the signature `square_kernel!(output, grid, input, other_args...)`. We generally find this convenient since it makes the kernel's dependence on the `grid` more explicit. However, it comes with a small amount of runtime overhead in cases where the `grid` is not needed. Note also the argument `XYZ` (or `XY` for 2D) corresponds to a `VarDims` type used to define Terrarium state variables; this syntax can be used interchangeably with the Oceananigans `Symbol` syntax.

!!! warning
    The kernel launching patterns used in Terrarium are still in an early stage of development and may change in the future.

## Numeric types

A common pattern that you may notice throughout the Terrarium codebase is the widespread use of the type argument `NF`. This stands for "number format" (in Oceananigans/ClimA it's usually called `FT` for "floating-point type") and is usually set to either double-precision `Float64` or single-precision `Float32`, though lower precisions (or even non-`Float` types) are also theoretically possible. The widespread use of these type arguments is necessary to ensure (i) flexibility in the number format for all model parameters and state variables, and (ii) consistency throughout any given configuration of the model. We typically want to avoid the situation where some variables have type `Float64` and some have `Float32`, for example, since this can reuslt in errors and severe degradation of performance when running on GPU. As such, we typically require all parameter and configuration `struct`s to be defined following the pattern:

```julia
@kwdef struct SomeParameters{NF}
    a::NF = 1.0
    b::NF = 2.0
end
```
with a constructor that accepts the numeric type as the first argument, e.g.
```julia
SomeParameters(::Type{NF}; kwargs...) where {NF} = SomeParameters{NF}(; kwargs...)
```
or
```julia
SomeParameters(::Type{NF}; a::NF = 1.0, b::NF = 2.0) where {NF} = SomeParameters(a, b)
```
In the latter case, the `@kwdef` macro on the `struct` definition may be omitted.
