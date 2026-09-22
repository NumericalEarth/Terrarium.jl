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
Terrarium currently provides two explicit time steppers, [`ForwardEuler`](@ref) and [`Heun`](@ref):

```@docs; canonical = false
ForwardEuler
Heun
```

Both are constructed with a default timestep `Δt` in seconds. `Δt` can be overridden at run
time by specifying it when calling `run!` or `timestep!`.

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
number. Attach it to a [`Simulation`](@extref Oceananigans.Simulations.Simulation) like any other
callback,

```@example simulation
integrator = initialize(SoilModel(grid))
sim = Simulation(integrator; stop_iteration = 50, Δt = 60.0)

wizard = TimeStepWizard(diffusive_cfl = 0.5, max_change = 1.1)
sim.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

run!(sim)
sim.Δt   # adapted time step (s)
```

or more succinctly via [`conjure_time_step_wizard!`](@extraref Oceananigans.Simulations.conjure_time_step_wizard!):

```@example simulation
integrator = initialize(SoilModel(grid))
sim = Simulation(integrator; stop_iteration = 50, Δt = 60.0)
conjure_time_step_wizard!(sim, show_progress = false)
run!(sim)
sim.Δt   # adapted time step (s)
```

Oceananigans also defines a companion advective diagnostic [`cell_advection_timescale`](@ref) which is relevant for
processes involving advection.

!!! warning "Richards equation is stiff near saturation"
    As the soil saturates, ``\partial\theta/\partial\psi \to 0`` and the hydraulic timescale tends to
    zero. Explicit integration of the Richardson–Richards equation is therefore extremely stiff in
    the saturated limit, which is the regime that motivates implicit time stepping (planned). The
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
    including = [:grid], # save the grid with the output
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
