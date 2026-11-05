# Implementing processes

```@meta
CurrentModule = Terrarium
```

This page is a step-by-step guide for implementing a new concrete process type in Terrarium. We assume familiarity with the [Basic concepts](@ref) and [Core interfaces](@ref) pages, both of which cover the `AbstractProcess` interface and variable system. For a concrete, self-contained example that follows the workflow below step by step, see the [Implementing a process: 1D linear heat diffusion](@ref linear_heat_conduction_example) example.

## Control flow

Every process in Terrarium is implemented across three levels of abstraction:

- I. **Top-level interface methods** for `AbstractProcess`, i.e. `variables`, `initialize!`, `compute_auxiliary!`, and `compute_tendencies!` (see doc section on the [`AbstractProcess` interface](@ref "The AbstractProcess interface"))
- II. **Kernel functions**; this includes both the [`@kernel`](@extref KernelAbstractions.@kernel) entry point and subsequently inlined functions with signatures of the form `compute_*(i,j[,k], grid, fields, ::Process, args...)`
- III. **Primitive methods** defined directly by the subtype of `AbstractProcess`. These are scalar-valued, grid-independent functions corresponding to individual terms or expressions in the mathematical formulation of the process physics.

The flow of execution is I → II  → III: top-level  interface methods are invoked by the enclosing `AbstractModel`, these methods then `launch!` their corresponding `@kernel`s which in turn call the inner kernel functions.

## Development workflow

We recommend to start implementing new processes using the following workflow:

1. Start by drafting an initial version of the process `struct` subtyping [`AbstractProcess`](@ref); it's good to start simple by directly putting all relevant parameters into this `struct`. You can later consider grouping these parameters into logical parameterization types.
2. Define `variables(::Process)` for the new process type. Identify which variables are `prognostic`, which are `auxiliary`, and which might be assumed to be `input`s, provided either by external forcing data or by another process.
3. Write down unit tests and initial implementations of the scalar-valued **process methods** (**level III**). These should correspond to the key equations or terms in the mathematical formulation of the physics. Note that these methods are invoked on **each elementary volume** and thus cannot be used to define equations involving spatial gradients or fluxes.
4. Implement the `compute_*(i, j[, k], grid, fields, ...)` kernel functions with the corresponding `@kernel` entry points `compute_auxiliary_kernel!` and `compute_tendencies_kernel!` (**level II**). Functions which might be independently useful outside of the process interface (e.g. [`adjust_saturation_profile!`](@ref) in `SoilHydrology`) can also define their own `@kernel` entry points.
5. Finally, implement the top-level process methods for `initialize!`, `compute_auxiliary!` and `compute_tendencies!` that `launch!` their respective `@kernel`s (**level I**).

As mentioned above, see the example [Implementing a process: 1D linear heat diffusion](@ref linear_heat_conduction_example) for a detailed walkthrough of this workflow.

## Enabling implicit timestepping

By default, all prognostic variables are integrated explicitly. A diffusion-like operator can instead be integrated implicitly under an [`IMEX`](@ref) timestepper (see [Implicit-explicit time stepping](@ref)). To support this in a process:

1. **Accept a `Timestepping` argument.** Give the process's `compute_tendencies!` a trailing positional argument `timestepping::Timestepping = Explicit()` and pass it through the kernel to the kernel functions, which should then dispatch on [`Explicit`](@ref) or [`Implicit`](@ref). The model obtains the resolved class of a variable with `timestepping(state, Val(:name))`. See `compute_tendencies!` of `SoilThermodynamics` for an example.
2. **Omit the implicitly treated terms under `Implicit()`.** The tendency under `Implicit()` must contain only the explicitly treated terms (e.g. sources and boundary fluxes). The omitted terms are instead applied via the corresponding implicit operator. The `Explicit()` implementation of `compute_tendencies!` should typically remain unchanged so that users also have the option to use explicit timesteppers.
3. **Declare prognostic variables that should be solved implicitly.** Define `timestepping(::AbstractVariable{:name}, ::YourModel, ::AbstractIMEX) = Implicit()` for the model that owns the variable.
4. **Define the implicit operator.** The implicit timestepper calls [`implicit_operator`](@ref)`(model, state, names)` once on the host with the names of all routed variables when its cache is allocated at initialization, and the model returns one [`AbstractLinearOperator`](@ref) for all of them. Every model with implicitly routed variables must define this method; the fallback throws an `ArgumentError` at initialization. To solve the variables one after the other, return a [`SplitOperator`](@ref) of newly allocated per-variable operators (e.g. a [`TridiagonalOperator`](@ref) for a vertical diffusion operator) in the order in which they should be solved; a model that couples its variables returns its own operator type instead.
5. **Implement the process hooks.** In each iteration, the implicit solver calls the hooks [`linearize!`](@ref), [`implicit_update!`](@ref), and [`implicit_closure!`](@ref) for the variable, which the model forwards to the process that owns it (the linear solve, [`solve!`](@ref), is generic over the operator). Implement them for the process; their signatures and what each must do are described in [Process hooks](@ref imex-process-hooks). The linear system is solved for the variable of the prognostic variable's closure relation, so a closure implements [`varname`](@ref) if its variable can be integrated implicitly. See `SoilThermodynamics` and `SoilHydrology` with [`RichardsEq`](@ref) for examples.

```@docs; canonical = false
implicit_operator
timestepping
linearize!
implicit_update!
implicit_closure!
```

