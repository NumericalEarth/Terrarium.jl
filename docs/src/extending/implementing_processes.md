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
3. Write down unit tests and initial implementations of the the scalar-valued **process methods** (**level III**). These should correspond to the key equations or terms in the mathematical formulation of the physics. Note that these methods are invoked on **each elementary volume** and thus cannot be used to define equations involving spatial gradients or fluxes.
4. Implement the `compute_*(i, j[, k], grid, fields, ...)` kernel functions with the corresponding `@kernel` entry points `compute_auxiliary_kernel!` and `compute_tendencies_kernel!` (**level II**). Functions which might be independently useful outside of the process interface (e.g. [`adjust_saturation_profile!`](@ref) in `SoilHydrology`) can also define their own `@kernel` entry points.
5. Finally, write implement the top-level process methods for `initialize!`, `compute_auxiliary!` and `compute_tendencies!` that `launch!` their respective `@kernel`s (**level I**).

As mentioned above, see the example [Implementing a process: 1D linear heat diffusion](@ref linear_heat_conduction_example) for a detailed walkthrough of this workflow.

## Enabling implicit timestepping

By default, all prognostic variables are integrated explicitly. A diffusion-like operator can instead be integrated implicitly under an [`IMEX`](@ref) timestepper (see [Implicit-explicit time stepping](@ref)). To support this in a process:

1. **Accept a `Timestepping` argument.** Give the process's `compute_tendencies!` a trailing positional argument `timestepping::Timestepping = Explicit()` and pass it through the kernel to the kernel functions, which dispatch on [`Explicit`](@ref) or [`Implicit`](@ref). The model obtains the resolved class of a variable with `timestepping(state, Val(:name))`. See `compute_tendencies!` of `SoilThermodynamics` for an example.
2. **Omit the implicit operator under `Implicit()`.** The tendency under `Implicit()` must contain only the explicitly treated terms (e.g. sources and boundary fluxes). The omitted terms are applied by the implicit sub-stepper. The `Explicit()` method must remain unchanged so that the process continues to work with explicit timesteppers.
3. **Route the variable.** Define `timestepping(::AbstractVariable{:name}, ::YourModel, ::AbstractIMEX) = Implicit()` for the model that owns the variable.
4. **Provide the implicit operator.** The implicit timestepper calls the block query [`implicit_operator`](@ref)`(model, state, names)` once on the host with the names of all routed variables when its cache is allocated at initialization, and the model returns one [`AbstractLinearOperator`](@ref) for all of them. Every model with implicitly routed variables must define this method; the fallback throws an `ArgumentError` at initialization. To solve the variables one after the other, return a [`SplitOperator`](@ref) of newly allocated per-variable operators (e.g. a [`TridiagonalOperator`](@ref) for a vertical diffusion operator) in the order in which they should be solved; a model that couples its variables returns its own operator type instead. For a model built on the soil, the query is forwarded as `implicit_operator(model, state, names)` → `implicit_operator(soil::AbstractSoil, state, grid, names)`, which orders the names (energy before water) and assembles the `SplitOperator` from `implicit_operator(soil::SoilEnergyWaterCarbon, state, grid, ::Val{name})` → `implicit_operator(process, state, grid)`. These process-level methods also validate the configuration on the host.
5. **Implement the process hooks.** The implicit solver (e.g. [`Picard`](@ref)) owns the iteration and is generic over the operator and the process. On entry to the implicit step, the prognostic field holds `u⁺ = uⁿ + Δt G_exp`; the implicit timestepper ([`BackwardEuler`](@ref)) stores the explicit increment `Δt G_exp = u⁺ - uⁿ` in the `explicit_stage` of the variable's [`ImplicitSolverWorkspace`](@ref) in the [`BackwardEulerCache`](@ref), resets the prognostic field to `uⁿ` (for all implicit variables before any of them is solved), and then calls the solver-level `implicit_step!(solver, operator, workspaces, state, model, Δt)`. When the variable is solved, the solver forms its explicit stage `u⁺` from its current field and the increment, so the hooks receive `u⁺` as before; it then calls, in each iteration, the hooks below with the model, which forwards them to the process that owns the variable in the same way as the query. The hooks dispatch on an [`Iteration`](@ref)`{name, target, final}`, which identifies the variable `name`, the field `target` that its linear system is solved for, and whether it is the solver's last iteration; its field `number` holds the 1-based index of the iteration for information only. Their model-level signatures are

    - [`linearize!`](@ref)`(operator, rhs, state, model, explicit_stage, Δt, ::Iteration{name})`: fill the lagged coefficients of `operator` and the right-hand side `rhs` at the current iterate, where `explicit_stage` is the field holding `u⁺`;
    - [`solve!`](@ref)`(target, operator, rhs, Δt)`: solve the linear system for the field `target`, which dispatches on the operator type and needs no process-specific method;
    - [`implicit_update!`](@ref)`(state, model, operator, explicit_stage, Δt, ::Iteration{name})`: update the prognostic variable from the solution, conservatively, i.e. from the fluxes implied by the operator;
    - [`implicit_closure!`](@ref)`(state, model, ::Iteration{name, target, final})`: re-evaluate the closure, where `final` is `true` only in the last iteration.

    The target is not a hook: it is the variable of the prognostic variable's closure relation, as given by [`varname`](@ref) of the closure, or, if it has no closure, its [`implicit_target`](@ref), which is the prognostic variable itself unless specialized (e.g. `snow_liquid_fraction` for `snow_water_equivalent`). A closure relation therefore implements `varname` if its variable can be integrated implicitly (see, e.g., the method for `SoilEnergyTemperatureClosure`), and the operator is linearized in that variable. If the operator depends on another implicitly integrated variable, the hooks can read that variable from its prognostic field: since the implicit timestepper resets every implicit variable to `uⁿ` before the first solve, it holds `uⁿ` if that variable is solved later and `uⁿ⁺¹` if it was solved earlier, never the explicit stage. On exit, the prognostic field must hold `uⁿ⁺¹`, with any closure variables derived from it consistent. The hooks receive the model as an argument rather than the timestepper cache, so the process does not depend on the solver or on the timestepper. See the implementations for `SoilThermodynamics` and `SoilHydrology` with [`RichardsEq`](@ref) for examples.

The implicit operator itself must still be a well-formed continuous-time operator.

```@docs; canonical = false
implicit_operator
Iteration
linearize!
implicit_update!
implicit_closure!
BackwardEulerCache
ImplicitSolverWorkspace
```

