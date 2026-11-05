# Time stepping

```@meta
CurrentModule = Terrarium
```

```@setup simulation
using Terrarium
```

## Overview

Terrarium explicitly separates process computations in `compute_auxiliary!` and `compute_tendencies!` from the choice of time stepping scheme. As a general rule, only models can be configured for timestepping. A model can be initialized for timestepping via

```@docs; canonical = false
initialize(model::AbstractModel, params::Union{Nothing, ComponentVector, ParameterTable})
```

This will return a [`ModelIntegrator`](@ref):

```@docs; canonical = false
ModelIntegrator
```

A single iteration of the `ModelIntegrator` roughly involves four steps:

1. Invoke [`update_inputs!`](@ref) to populate all input `Field`s from their respective `InputSource`s based on the current `clock` time,
2. Invoke [`compute_auxiliary!`](@ref) on all model components to derive auxiliary state variables from the current prognostic state,
3. Invoke [`compute_tendencies!`](@ref) on all model components to calculate tendencies for all prognostic variables,
4. Apply the tendencies computed in step (3) to update the prognostic state based on the selected timestepping scheme. Higher order explicit or implicit timesteppers may repeat steps 1-3 multiple times within a single time step.

As an example, let's consider the construction and initialization of a [`SoilModel`](@ref):

```@example simulation
arch = CPU()
grid = ColumnGrid(arch, Float32, ExponentialSpacing(N=10))
model = SoilModel(grid, timestepper=ForwardEuler(Float32))
integrator = initialize(model)
```

Here `integrator` corresponds to a `ModelIntegrator` configured for a [`ForwardEuler`](@ref) time stepping scheme that was set in the `SoilModel`. State variable `Field`s can be accessed via `integrator.state`:

```@example simulation
integrator.state.temperature   # current temperature Field
```

## Time stepping schemes

Each model carries a single `timestepper` (set via the `timestepper` keyword on the model constructor).
[`LandModel`](@ref), and [`SoilModel`](@ref) with Richards-equation water flow, default to the implicit-explicit `IMEX(ForwardEuler(NF; Δt), BackwardEuler(NF))`, with a time step of 15 minutes for `LandModel` and 1 hour for `SoilModel` (see [Implicit-explicit time stepping](@ref)); all other models, including `SoilModel` without water flow, default to `ForwardEuler(NF)`, whose default time step is 300 s.
Terrarium currently provides two explicit time steppers, [`ForwardEuler`](@ref) and [`Heun`](@ref):

```@docs; canonical = false
ForwardEuler
Heun
```

Both are constructed with a default timestep `Δt` in seconds. `Δt` can be overridden at run
time by specifying it when calling `run!` or `timestep!`.

## Implicit-explicit time stepping

Explicit schemes are cheap and simple, but for diffusive processes the stable time step shrinks quadratically with the grid spacing (see [Adaptive time stepping](@ref)). A thin surface layer or a fine vertical resolution can therefore force steps of seconds to minutes, even when the quantity of interest evolves over hours. Terrarium addresses this with an implicit-explicit (IMEX) time stepper, [`IMEX`](@ref), which integrates selected prognostic variables with an implicit scheme and all remaining variables with an explicit one.

For an implicitly integrated variable ``u``, the tendency is split into the terms treated explicitly, the *explicit tendency* ``G_\text{exp}``, and the terms treated implicitly, the *implicit tendency* ``G_\text{imp}`` (e.g. the vertical diffusion operator):
```math
\frac{\partial u}{\partial t} = G_\text{exp}(u) + G_\text{imp}(u)\,.
```
The explicit tendency holds everything else, such as sources, sinks, and fluxes through the boundaries of the column. For explicitly integrated variables, ``G_\text{imp}(u) = 0`` and thus the whole tendency is explicit.

```@docs; canonical = false
IMEX
```

### Structure of an IMEX step

A single IMEX step of size ``\Delta t`` consists of the following stages:

1. A single `update_state!` evaluates all inputs, auxiliary variables, and tendencies at ``t^n``. For implicitly routed variables, the tendency holds only the *explicitly treated* terms ``G_\text{exp}``; the implicit operator is omitted (see [Routing variables](@ref) below).
2. [`update_cache!`](@ref) on the implicit sub-stepper's cache stores what it needs from ``u^n`` (for [`BackwardEuler`](@ref), a copy of ``u^n`` of each implicitly routed variable) before the explicit stage overwrites it.
3. The explicit sub-stepper's update ([`explicit_step!`](@ref)) is applied to *all* prognostic variables, giving ``u^+ = u^n + \Delta t\, G_\text{exp}(u^n)``. For explicitly routed variables, this is the final result. For implicitly routed variables, it is the right-hand side of the implicit system.
4. For the implicitly routed variables, [`implicit_step!`](@ref) of the implicit sub-stepper solves
   ```math
   u^{n+1} = u^+ + \Delta t\, G_\text{imp}(u^{n+1})
   ```
   in place.
5. The model hook `timestep!(state, model, timestepper, Δt)` and `closure!` are applied once, and the clock ticks once.

`IMEX(ForwardEuler(NF; Δt), BackwardEuler(NF))` is the default timestepper of [`LandModel`](@ref) (`Δt = 900`) and of [`SoilModel`](@ref) with Richards-equation water flow (`Δt = 3600`). Under `IMEX`, the snow meltwater drainage of a `LandModel` is implicit as well (see [Implicit meltwater drainage](@ref)), while the snow heat conduction, vegetation, surface hydrology, and surface energy balance remain explicit and are integrated with an explicit integrator (by default `ForwardEuler`) at the same step; the shorter default time step of `LandModel` is due to the explicit snow heat conduction (which will later also be treated implicitly). Value boundary conditions on `temperature` or `pressure_head` (e.g. [`PrescribedSurfaceTemperature`](@ref)) currently require an explicit timestepper instead (see [Limitations](@ref imex-limitations)).

Because there is only one `update_state!` per step, an IMEX step costs one evaluation of the explicit right-hand side plus the implicit solve. The sub-steppers are given in the order `IMEX(explicit, implicit)`, which is checked when the integrator is initialized. The explicit sub-stepper must be a single-stage scheme; currently this means [`ForwardEuler`](@ref), whose `Δt` is also the default time step of the `IMEX`. The implicit sub-stepper is an [`AbstractTimeStepper`](@ref) whose behavior is further specialized by an [`AbstractImplicitSolver`](@ref):

```@docs; canonical = false
AbstractImplicitSolver
update_cache!
explicit_step!
implicit_step!
```

### Routing variables

Which variables are integrated implicitly is determined by the [`timestepping`](@ref) trait, evaluated once when the integrator is initialized as `timestepping(var, model, imex)`. The default for every variable is [`Explicit`](@ref). [`SoilModel`](@ref) and [`LandModel`](@ref) both specialize it so that `internal_energy` is [`Implicit`](@ref) under any [`AbstractIMEX`](@ref) timestepper, via `timestepping(::AbstractVariable{:internal_energy}, ::SoilModel, ::AbstractIMEX) = Implicit()`. The soil saturation `saturation_water_ice` is [`Implicit`](@ref) as well whenever the soil hydrology follows the Richards equation (`SoilHydrology` with [`RichardsEq`](@ref)), and [`Explicit`](@ref) otherwise, as given by `default_timestepping(hydrology)`. [`LandModel`](@ref) and [`SnowModel`](@ref) also route `snow_water_equivalent` implicitly, for the meltwater drainage (see [Implicit meltwater drainage](@ref)); the snow energy, vegetation, and surface hydrology variables always remain explicit.

To change the routing, define a more specific method for your own model type or timestepper type. For example, to keep soil energy explicit under a custom IMEX subtype `MyIMEX`, define `timestepping(::AbstractVariable{:internal_energy}, ::SoilModel, ::MyIMEX) = Explicit()`. The resolved class of each prognostic variable is stored in the type of the timestepper cache, so that routing is resolved at compile time and kernels remain type stable.

```@docs; canonical = false
timestepping
```

Processes see the resolved class as a trailing `Timestepping` argument of `compute_tendencies!`. Under `Implicit()`, the soil thermodynamics omit the diffusive and advective heat fluxes through interior cell faces from the tendency (see [Implicit vertical heat conduction](@ref)), and the Richards-equation hydrology omits the Darcy fluxes through interior cell faces (see [Implicit Richards equation](@ref soil-hydrology-implicit-richards)).

### Implicit Euler with a Picard solver

The only implicit scheme currently provided is first-order backward Euler, [`BackwardEuler`](@ref), with a [`Picard`](@ref) nonlinear solver. When the integrator is initialized, `BackwardEuler` queries the model for the linearized operator of all variables routed to it with [`implicit_operator`](@ref)`(model, state, names)` and stores this single operator in its cache. For [`SoilModel`](@ref) and [`LandModel`](@ref), it is a [`SplitOperator`](@ref) that holds a batched vertical [`TridiagonalOperator`](@ref) for each routed soil variable and solves `saturation_water_ice` before `internal_energy` (operator splitting; see [Solvers for implicit time stepping](@ref) and [Coupling of water and energy](@ref imex-water-energy-coupling)).

`BackwardEuler` also resolves, for each variable, the field its linear system is solved for: the variable of its closure relation (as given by [`varname`](@ref) of the closure), i.e. the `temperature` for `internal_energy` and the `pressure_head` for `saturation_water_ice`, and for a variable without closure its [`implicit_variable`](@ref), e.g. the `snow_liquid_fraction` for `snow_water_equivalent`. In each step, `BackwardEuler` stores the explicit increment ``u^+ - u^n = \Delta t\, G_\text{exp}`` of every implicit variable in the [`ImplicitSolverWorkspace`](@ref) of the variable and resets all of their prognostic fields to ``u^n``, and then calls the solver-level `implicit_step!(solver, operator, workspaces, state, model, Δt)` of its solver once; when a variable is solved, the solver forms its explicit stage from its current field and the increment. The solver iterates over linearizations of the operator, which it obtains from the process that owns each variable through the [process hooks](@ref imex-process-hooks):

```@docs; canonical = false
BackwardEuler
Picard
```

The implicit steps of the soil variables are described with their processes: vertical heat conduction in [Implicit vertical heat conduction](@ref), the Richards equation in [Implicit Richards equation](@ref soil-hydrology-implicit-richards), and their coupling in [Coupling of water and energy](@ref imex-water-energy-coupling). Both follow the same pattern. The backward Euler step is linearized in the variable of the closure relation (the temperature or the total hydraulic head) with coefficients lagged at the latest iterate, which gives a tridiagonal system per column (see [Tridiagonal operator](@ref)). The conserved variable is then updated from the implicit fluxes through the interior faces rather than from the linearization, so the column-integrated energy and water change only by the boundary fluxes and sources in ``G_\text{exp}``, independent of the number of iterations.

With `iterations = 1`, this is the linearly implicit scheme used by many land surface models [bonanClimateChangeTEMSoilMoisture2019](@cite). The default is `iterations = 2`. With `iterations = M > 1`, the lagged coefficients are re-evaluated at the new iterate (after re-applying the closure) and the solve is repeated, which is a Picard iteration on the nonlinear system [CeliaEtAl1990](@cite). A single step is still only first order accurate in ``\Delta t``, but additional iterations reduce the error from the lagged capacity near freezing fronts.

### [Process hooks](@id imex-process-hooks)

Each iteration of the [`Picard`](@ref) solver calls four hooks for each implicit variable, which the model forwards to the process that owns the variable, so that processes depend on neither the solver nor the timestepper. The hooks receive the explicit stage ``u^+`` and dispatch on an [`Iteration`](@ref)`{name, target, final}`, which names the variable, the field `target` that its linear system is solved for (see [Implicit Euler with a Picard solver](@ref)), and whether the iteration is the last one:

- [`linearize!`](@ref)`(operator, rhs, state, model, explicit_stage, Δt, ::Iteration{name})` evaluates the lagged coefficients of `operator` and the right-hand side `rhs` at the current iterate, where `explicit_stage` holds ``u^+``;
- [`solve!`](@ref)`(target, operator, rhs, Δt)` solves the linear system for the field `target`; it dispatches on the operator type and needs no process-specific method;
- [`implicit_update!`](@ref)`(state, model, operator, explicit_stage, Δt, ::Iteration{name})` updates the prognostic variable conservatively, from the fluxes implied by the solution;
- [`implicit_closure!`](@ref)`(state, model, ::Iteration{name, target, final})` re-evaluates the closure, where `final` is `true` only in the last iteration.

After the last iteration, the prognostic field must hold ``u^{n+1}``, with the closure variables derived from it consistent. A hook may read another implicitly integrated variable from its prognostic field: since all implicit variables are reset to ``u^n`` before the first solve, it holds ``u^n`` if that variable is solved later and ``u^{n+1}`` if it was solved earlier, never the explicit stage.

```@docs; canonical = false
Iteration
linearize!
implicit_update!
implicit_closure!
```

### Example

We configure a [`SoilModel`](@ref) with an `IMEX(ForwardEuler(Float32), BackwardEuler(Float32))` timestepper, a thawed initial column, and a constant cooling flux at the surface via [`SoilHeatFlux`](@ref) (positive upward, i.e. out of the soil). The default time step of the `IMEX` is that of its explicit partner. The time step of one hour is more than ten times the explicit stability limit of this column (roughly six minutes):

```@example simulation
solver = Picard(Float32; iterations = 2)
timestepper = IMEX(ForwardEuler(Float32; Δt = 3600), BackwardEuler(Float32; solver))
imex_grid = ColumnGrid(CPU(), Float32, ExponentialSpacing(Δz_min = 0.02, Δz_max = 0.5, N = 20))
imex_model = SoilModel(imex_grid; timestepper)
imex_integrator = initialize(
    imex_model;
    boundary_conditions = SoilHeatFlux(80.0f0),
    initializers = (temperature = 0.5f0, saturation_water_ice = 0.8f0),
)
for _ in 1:24
    timestep!(imex_integrator, 3600.0)
end
imex_integrator.state.temperature   # temperature after 24 steps of one hour
```

The surface cools below 0 °C and a freezing front enters the column. The same configuration under [`ForwardEuler`](@ref) would require a time step of a few minutes.

### Choosing `Δt` and `iterations`

The implicit step is unconditionally stable for the diffusion operator (the linear system is an M-matrix, see [Tridiagonal operator](@ref), so the temperature obeys a discrete maximum principle), but it is only first order accurate. The choice of `Δt` is therefore governed by accuracy and by the time scales of the explicitly treated terms (e.g. boundary forcing), not by the diffusive CFL. A useful reference is the explicit limit [`cell_diffusion_timescale`](@ref); steps of tens to hundreds of times that limit are stable, and the error of the solution decreases linearly when `Δt` is halved.

Phase change is the main source of error. With `iterations = 1`, the lagged apparent heat capacity overestimates the cooling of cells that begin to freeze within a step. Using `iterations = 2` reduces the maximum temperature error of a freezing front by roughly a factor of ten in the tests, at the cost of one additional coefficient evaluation, tridiagonal solve, and closure per step. More iterations are not guaranteed to reduce the error further, since the regularized capacity of a cell switches on and off as the iterates enter and leave the phase change. For thawed columns (no phase change), a single iteration is sufficient.

### [Limitations](@id imex-limitations)

- Only the vertical heat conduction, the advection of heat by the soil water flux, and the Richards-equation Darcy flux in the soil can be integrated implicitly. The advection is first-order upwind and therefore numerically diffusive at large advective Courant numbers. Infiltrating water enters at the temperature of the top soil cell rather than at the skin or rain temperature, water removed by evapotranspiration carries no heat, and the surface excess pool stores no heat, i.e. the heat of the excess water leaves the system with it.
- Value (Dirichlet) boundary conditions on `temperature`, such as [`PrescribedSurfaceTemperature`](@ref), are rejected when `internal_energy` is implicit, with an informative error at initialization. Use flux boundary conditions on `internal_energy`, such as [`SoilHeatFlux`](@ref) and [`GeothermalHeatFlux`](@ref), or an explicit timestepper. Gradient (Neumann) boundary conditions on `temperature` are supported; the flux ``-\kappa\, \partial T / \partial z`` they prescribe is treated explicitly, at ``t^n``, like a flux boundary condition.
- Boundary fluxes are evaluated at ``t^n`` and held fixed over the step; only the diffusive fluxes through interior faces are implicit.
- With active soil hydrology (e.g. [`RichardsEq`](@ref)), the energy and water equations are coupled by operator splitting (see [Coupling of water and energy](@ref imex-water-energy-coupling)): the Richards step evaluates the hydraulic conductivity at the liquid water fraction at ``t^n``, so the effect of freezing and thawing on the water flow is lagged by one step. This affects only accuracy; energy and water conservation are unaffected.
- The redistribution of water by [`adjust_saturation_profile!`](@ref) moves only the sensible heat of liquid water, also when the water it moves is frozen.
- Boundary Darcy fluxes, e.g. gravity drainage, are evaluated at ``t^n`` and are not limited by the water available in the boundary cell. At very large steps they can overdraw a column so strongly that it alternates between empty and saturated; the water solution is then inaccurate, and since the hydrology closure moves the excess water of a cell (and its heat) before the heat step advects the heat of the water that caused it, the temperature is then no longer guaranteed to stay within its initial range. A `TimeStepWizard` keeps the step below this bound through the explicit water outflow limit of `cell_advection_timescale` (see [Adaptive time stepping with IMEX](@ref)).
- A prescribed infiltration flux on the saturation, such as [`InfiltrationFlux`](@ref), forces water into the top cell whatever its state, also when it is frozen and conducts no water (the surface runoff of a [`LandModel`](@ref) instead limits the infiltration to what the top cell can take). Water then accumulates in frozen surface cells, and when they thaw within a step of several hours, the default two [`Picard`](@ref) iterations can fail to converge: the iterates alternately drain and overfill the cells near the surface, water is lost, and the solution diverges. In a test with 5 mm/day into freezing soil, this happened at steps of eight hours. Keep implicit steps to a few hours in this situation, or limit the prescribed infiltration to the capacity of the soil.
- The meltwater drained by the implicit snow step reaches the soil in the following step (see [Implicit meltwater drainage](@ref)).
- Value (Dirichlet) boundary conditions on `pressure_head` are rejected when `saturation_water_ice` is implicit; use flux conditions on `saturation_water_ice` (e.g. [`InfiltrationFlux`](@ref)) or gradient conditions on `pressure_head` (e.g. [`FreeDrainage`](@ref)).
- Only the `FreeWater` freeze curve is supported, since the regularized apparent heat capacity is only implemented for it.
- [`Heun`](@ref) is not yet supported as the explicit partner of an [`IMEX`](@ref); use [`ForwardEuler`](@ref).
- The scheme is first order accurate in time.

### Reactant and Enzyme

IMEX time stepping with the [`Picard`](@ref) solver and the [`TridiagonalOperator`](@ref) is tested both under Reactant (`ReactantState` grids, compiled with XLA) and for reverse-mode differentiation with Enzyme. The validity checks of the implicit operators run on the host when the integrator is initialized (in [`implicit_operator`](@ref)), so that the kernels contain no error paths. As with other timesteppers, adaptive stepping with a host-side `TimeStepWizard` does not apply to compiled Reactant runs.

## Transient simulations

`ModelIntegrator` defines dispatches for [`timestep!`](@ref) and [`run!`](@ref) that allow for transient, open-ended simulations starting from the initialized state. We can use [`timestep!`](@ref) to take a single time step,

```@example simulation
timestep!(integrator)
```

and [`run!`](@ref) to advance over a fixed number steps or a specified period,

```@example simulation
# Advance by a fixed number of steps
run!(integrator; steps = 100)

# Advance for a given wall-clock period
run!(integrator; period = Day(30), Δt = 3600.0)
```


## Setting up a `Simulation`

The `timestep!` and `run!` methods allow us to control the integrate the model forward in time, but we only have access to the transient model state at each time step. In order for the model to be useful, we also need to be able to define a finite time period over which to run a simulation and save outputs during that time period.

Fortunately, Terrarium's [`ModelIntegrator`](@ref) implements the Oceananigans model interface. As a result, we can take advantage of the built-in Oceananigans infrastructure for configuring and running [`Simulation`](https://clima.github.io/OceananigansDocumentation/stable/simulations/simulations_overview)s:

```@example simulation
sim = Simulation(integrator; stop_time = 24*3600.0, Δt = 300.0)
run!(sim)
```

Note that `stop_time` is by default expressed in the same units as the `clock`, which is here seconds. However, [`Oceananigans.Units`](https://clima.github.io/OceananigansDocumentation/stable/units) provides convenient conversion factors for other time units (e.g. `days`, `hours`):

```@example simulation
using Oceananigans.Units: days

sim = Simulation(integrator; stop_time = 1days, Δt = 300.0)
```

### Callbacks

Terrarium simulations inherit the full Oceananigans [`Callback`](@extref Oceananigans.Simulations.Callback) machinery.
A callback is a function that is called by [`Simulation`](@extref Oceananigans.Simulations.Simulation) at a given schedule during `run!`:

```@example simulation
using Oceananigans: Callback, IterationInterval

function print_progress(sim)
    t = sim.model.clock.time
    iter = sim.model.clock.iteration
    @info "Iteration $iter, time $t s"
end

sim.callbacks[:progress] = Callback(print_progress, IterationInterval(100))
```

The callback function receives the `Simulation` object, giving it access to the full
`integrator` via `sim.model`, and to the current `state` via `sim.model.state`.

## Adaptive time stepping

Some land components, such as the nonlinear advection-diffusion equations for soil energy, water, and carbon
transport, are subject to stability constraints when using explicit time-stepping schemes like [`ForwardEuler`](@ref)
or [`Heun`](@ref). A suitable constraint is the diffusive Courant–Friedrichs–Lewy (CFL) condition. The standard formulation
of the CFL for advective transport places a limit on the wave propagation speed based on the grid cell length $\Delta x$,
```math
\frac{ \lvert v \rvert \Delta t}{\Delta x} \leq C\,.
```
While diffusive systems do not have an independent velocity $v$, they do have a characteristic length scale based on the diffusion coefficient $D$,
```math
\ell \sim \sqrt{D \Delta t}
```
which we can understand as the distance over which information gets dispersed in a single time step $\Delta t$. Plugging that in as the numerator in CFL yields,
```math
\frac{\sqrt{D \Delta t}}{\Delta x} \leq C
```
and solving for $\Delta t$ we get,
```math
\Delta t \leq \frac{C^2 \Delta x^2}{D}\,.
```
Since $C$ is a constant, we can simply redefine $C_{\text{diff}} = C^2$. The choice of $C$ can be determined by [von Neumann stability analysis](https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis) for any given time stepping scheme; in the case of [`ForwardEuler`](@ref), this leads to $C = \frac{1}{2}$.

Terrarium defines the diffusive CFL via the Oceananigans diagnostic [`cell_diffusion_timescale`](@ref) which is computed as the minimum
over all grid cells of the diffusive timescale ``\tau = \Delta z^2 / D``. For the soil model it is the minimum
of the thermal timescale ``\Delta z^2 C / \kappa`` (heat conduction, with thermal conductivity ``\kappa`` and volumetric heat capacity ``C``)
and, for a Richards-equation hydrology, the hydraulic timescale ``\Delta z^2 (\partial\theta/\partial\psi) / K`` (with hydraulic conductivity
``K`` and specific moisture capacity ``\partial\theta/\partial\psi``).

Because a [`ModelIntegrator`](@ref) implements the Oceananigans model interface, this diagnostic can
be consumed directly by the Oceananigans [`TimeStepWizard`](@extref Oceananigans.Simulations.TimeStepWizard)
callback, which rescales `simulation.Δt` each time it is invoked to target a chosen diffusive CFL
number. The following examples use [`ForwardEuler`](@ref) explicitly, since an [`IMEX`](@ref) timestepper imposes no
diffusive restriction from implicitly integrated terms (see [Adaptive time stepping with IMEX](@ref)). Attach it to a [`Simulation`](@extref Oceananigans.Simulations.Simulation) like any other
callback,

```@example simulation
integrator = initialize(SoilModel(grid; timestepper = ForwardEuler(Float32)))
sim = Simulation(integrator; stop_iteration = 50, Δt = 60.0)

wizard = TimeStepWizard(diffusive_cfl = 0.5, max_change = 1.1)
sim.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

run!(sim)
sim.Δt   # adapted time step (s)
```

or more succinctly via [`conjure_time_step_wizard!`](@extref Oceananigans.Simulations.conjure_time_step_wizard!):

```@example simulation
integrator = initialize(SoilModel(grid; timestepper = ForwardEuler(Float32)))
sim = Simulation(integrator; stop_iteration = 50, Δt = 60.0)
conjure_time_step_wizard!(sim, show_progress = false)
run!(sim)
sim.Δt   # adapted time step (s)
```

Oceananigans also defines a companion advective diagnostic [`cell_advection_timescale`](@ref), which the wizard
multiplies by its `cfl` (0.2 by default). For a soil with Richards-equation hydrology, it is the minimum of two time
scales of explicit transport terms:

- the stability limit ``C \Delta z / (C_\text{w} |j_\text{w}|)`` of the explicit upwind advection of heat by the Darcy flux
  ``j_\text{w}`` (with bulk and water volumetric heat capacities ``C`` and ``C_\text{w}``), `Inf` without water flow;
- the explicit water outflow limit [`water_outflow_timescale`](@ref), ``\phi (\xi - \xi_\text{res}) \Delta z / q_\text{out}``,
  the time in which the explicit terms of the saturation tendency would drain a cell of porosity ``\phi`` and
  saturation ``\xi`` to its residual saturation ``\xi_\text{res}``. The outflow ``q_\text{out}`` (m/s) is the net loss of
  water of the cell per unit area through these terms: the Darcy fluxes that are integrated explicitly, the flux
  boundary conditions on the saturation (e.g. [`InfiltrationFlux`](@ref)), evapotranspiration, and the user-defined
  forcing. Cells without net explicit outflow impose no limit.

The wizard calls these diagnostics after each step, when the auxiliary variables have been updated but the tendencies
are those of the beginning of the step. The outflow limit is therefore evaluated directly from the current state by
kernel functions ([`compute_water_outflow_timescale`](@ref)) rather than read from the stored tendency, so that it
measures the explicit outflow of the next step.

### Adaptive time stepping with IMEX

Implicitly integrated operators impose no stability restriction, so the wizard is not throttled by them: under an
[`IMEX`](@ref) timestepper, `cell_diffusion_timescale` is `Inf` for implicit heat conduction and implicit Richards flow,
and the advective limit of the heat advection is `Inf` when `internal_energy` is implicit. The step is then bounded
by the terms that remain explicit and by `max_Δt`. Without water flow, only `max_Δt` remains:

```@example simulation
wizard_integrator = initialize(SoilModel(imex_grid; timestepper = IMEX(ForwardEuler(Float32), BackwardEuler(Float32))))
sim = Simulation(wizard_integrator; stop_iteration = 50, Δt = 60.0)
wizard = TimeStepWizard(diffusive_cfl = 0.5, max_change = 1.5, max_Δt = 6 * 3600.0)
sim.callbacks[:wizard] = Callback(wizard, IterationInterval(1))
run!(sim)
sim.Δt   # reaches `max_Δt`, far beyond the explicit diffusive limit of a few minutes
```

With a Richards-equation hydrology, the Darcy fluxes through the boundary faces of the column (e.g. gravity drainage
through a [`FreeDrainage`](@ref) or unit-gradient condition on `pressure_head`), the flux boundary conditions, and the
sinks stay explicit at ``t^n`` (see [Implicit Richards equation](@ref soil-hydrology-implicit-richards)). A step longer than the outflow
limit would remove more water from a cell than it holds, and the hydrology closure would then move water from other
cells to fill the deficit, so the column alternates between drained and saturated cells. The outflow limit keeps the
wizard below this bound. In a draining sandy column, it can be as restrictive as the explicit diffusive limit of the
Richards equation, since the boundary drainage carries the full flux through the column; the `cfl` of the wizard sets
the fraction of the available water that a step may remove. Under explicit time stepping of the saturation, the outflow
limit includes the interior Darcy fluxes, but the diffusive limit is then usually the more restrictive one.

If `internal_energy` is routed explicitly while `saturation_water_ice` is implicit (see the warning in
[Coupling of water and energy](@ref imex-water-energy-coupling)), the advective limit of the explicit heat advection
and the diffusive limit of explicit heat conduction both apply.

!!! warning "Richards equation is stiff near saturation"
    As the soil saturates, ``\partial\theta/\partial\psi \to 0`` and the hydraulic timescale tends to
    zero. Explicit integration of the Richardson–Richards equation is therefore extremely stiff in
    the saturated limit, which is the regime that motivates implicit time stepping. Implicit integration of the Richards equation removes this restriction (see [Implicit Richards equation](@ref soil-hydrology-implicit-richards)). The
    `TimeStepWizard`'s `min_change` bound prevents `Δt` from collapsing to zero in a single
    adjustment, but a saturated explicit run may still require very small steps.

!!! note "Reactant"
    Adaptive stepping applies to host-driven `Simulation`s only. Under a `ReactantState` grid the
    time step is compiled into the traced `run!` loop, so a host-side wizard callback does not take
    effect there.

```@docs; canonical = false
cell_diffusion_timescale(integrator::ModelIntegrator)
cell_advection_timescale(integrator::ModelIntegrator)
compute_thermal_diffusion_timescale
compute_hydraulic_diffusion_timescale
```

### Output writers

We can make use of [`Oceananigans.OutputWriters`](https://clima.github.io/OceananigansDocumentation/stable/simulations/output_writers) to flexibly save output to disk in a variety of formats.

The simplest choice output writer is [`JLD2Writer`](@extref Oceananigans.OutputWriters.JLD2Writer) which saves selected [`Field`s](@extref Oceananigans.Fields.Field) to a `.jld2` file at a specified schedule:

```@example simulation
using Oceananigans: JLD2Writer, TimeInterval
using Oceananigans.Units: hours

output_file = "$(tempname()).jld2"
println("Writing output to $(output_file)")

sim = Simulation(integrator; stop_time = 24*3600.0, Δt = 300.0)
sim.output_writers[:soil] = JLD2Writer(
    integrator,
    (temperature = integrator.state.temperature,
     saturation  = integrator.state.saturation_water_ice);
    filename = output_file,
    overwrite_files = true,
    schedule = TimeInterval(2hours),
)

# Re-initialize integrator
Terrarium.initialize!(integrator)

# Run the simulation again
run!(sim)
```

!!! note
    Always call `Terrarium.initialize!(integrator)` **before** calling `run!(sim)` when re-running a simulation; otherwise, the simulation will not run due to the stopping condition already being satisfied.


The saved file can be read back as a [`FieldTimeSeries`](@extref Oceananigans.OutputReaders.FieldTimeSeries-Tuple{JLD2.JLDFile, String}) for post-processing:

```@example simulation
using Oceananigans: FieldTimeSeries

temperature_series = FieldTimeSeries(output_file, "temperature")
temperature_series[end]  # Extract temperature Field at the last saved time
```

Output writers can also accept subtypes `AbstractSchedule` via the `schedule` keyword. Schedules determine the frequency and aggregation of the output `Field`s:

| Schedule type | Description |
|---------------|-------------|
| [`TimeInterval(Δt)`](@extref Oceananigans.Utils.TimeInterval) | Write every `Δt` seconds of simulation time |
| [`IterationInterval(n)`](@extref Oceananigans.Utils.IterationInterval) | Write every `n` timesteps |
| [`AveragedTimeInterval(Δt)`](@extref Oceananigans.OutputWriters.AveragedTimeInterval) | Write time-averaged outputs over windows of `Δt` seconds |

Multiple output writers can be added to the same simulation, e.g. to save different
variables at different frequencies:

```julia
sim.output_writers[:fast]   = JLD2Writer(integrator, (temperature = ...,); schedule = TimeInterval(1hours),  ...)
sim.output_writers[:daily]  = JLD2Writer(integrator, (pressure_head = ...,); schedule = AveragedTimeInterval(1days), ...)
```

## [References](@id "timestepping.refs")

```@bibliography
Pages = ["time_stepping.md"]
Canonical = false
```
