# Adaptive timestepping via `cell_diffusion_timescale`

> Status: **complete**. Implements a `cell_diffusion_timescale` diagnostic so Terrarium models can be driven adaptively by the Oceananigans `TimeStepWizard` callback, as a precursor to implicit timestepping.

Date of initial draft: 2026-08-11

Base revision: 9ece81381cb88a1b8a8418c25545ca41d7a0923e

## Originating prompt

> As a precursor to the implementation of implicit timestepping, I would like to first add
> adaptive timestepping in Terrarium using the Oceananigans TimeStepWizard callback. This requires
> the implementation of `cell_diffusion_timescale` in Terrarium models. Please investigate this and
> try implementing it.

## Revision log

> 2026-08-11 — Initial draft.
>
> Clarifications received during scoping:
> - Advection may be ignored for now (land models have no advective transport).
> - Both the thermal and hydraulic (Richards) diffusion timescales should be implemented. The
>   moisture-capacity derivative ∂θ/∂ψ needed for the hydraulic timescale is available analytically
>   from FreezeCurves.jl (`swrc(FreezeCurves.derivative, ψ)`).

## Problem description

Terrarium currently integrates all prognostic variables with fixed-step explicit schemes
(`ForwardEuler`, `Heun`) or an `IMEX` composition. There is no mechanism to adapt the step size to
the numerical stability limit of the (stiff) diffusive soil operators. Explicit integration of the
heat and Richardson–Richards equations is constrained by the diffusive
Courant–Friedrichs–Lewy (CFL) condition

```
Δt ≲ Δz² / (2 D)
```

where `D` is the relevant diffusivity (m²/s). Choosing `Δt` too large produces instability;
choosing it globally conservative wastes work. Adaptive stepping is also a natural stepping stone
toward implicit treatment of the diffusive terms, since it makes the stability-limiting timescale
an explicit, testable quantity.

Oceananigans already provides the machinery: the `TimeStepWizard` callback adjusts
`simulation.Δt` each `n` iterations to satisfy target advective and diffusive CFL numbers. It does
so through two model-level diagnostics,

```julia
advective_Δt = wizard.cfl           * cell_advection_timescale(model)
diffusive_Δt = wizard.diffusive_cfl * cell_diffusion_timescale(model)
new_Δt       = min(advective_Δt, diffusive_Δt)   # then bounded by min/max change and min/max Δt
```

Terrarium's `ModelIntegrator` already implements the `Oceananigans.AbstractModel` interface and can
be wrapped in a `Simulation`, so the missing piece is a `cell_diffusion_timescale` method (and, to
avoid an advective `MethodError`, a trivial `cell_advection_timescale`) dispatched on the Terrarium
model types.

## Background

### The diffusive timescale of each soil operator

**Heat conduction** (`SoilThermodynamics` + `ExplicitTwoPhaseHeatConduction`). The prognostic
variable is the volumetric internal energy `U`; the operator is
`∂U/∂t = ∂z(κ ∂z T)`. Linearising about the sensible energy–temperature relation `U ≈ C T` gives an
effective diffusivity for temperature `α = κ / C`, so the per-cell diffusive timescale is

```
τ_heat = Δz² / α = Δz² C / κ
```

where `κ` is the bulk thermal conductivity (W m⁻¹ K⁻¹) and `C` the bulk volumetric heat capacity
(J m⁻³ K⁻¹). Both are computed pointwise from the soil composition by the existing kernel functions
`compute_thermal_conductivity(props, soil)` and `compute_heat_capacity(props, soil)`; neither is
stored as a `Field`. Using the *sensible* `C` (rather than the apparent capacity, which includes
latent heat near 0 °C) is conservative: the apparent capacity is larger, so the true timescale is
never shorter than `τ_heat`.

**Richards' equation** (`SoilHydrology{RichardsEq}`). The prognostic variable is saturation, and the
operator is `∂θ/∂t = ∂z(K ∂z ψ)`. Rewriting in `θ`-form gives an effective moisture diffusivity
`D = K ∂ψ/∂θ = K / (∂θ/∂ψ)`, so

```
τ_water = Δz² / D = Δz² (∂θ/∂ψ) / K
```

where `K` is the (variably saturated) hydraulic conductivity (m s⁻¹) and `∂θ/∂ψ` the specific
moisture capacity (m⁻¹). `K` is computed pointwise by `hydraulic_conductivity(props, soil)`.
The capacity `∂θ/∂ψ` is the analytic derivative of the soil-water retention curve (SWRC); FreezeCurves
exposes it as `swrc(FreezeCurves.derivative, ψ; θsat)`, evaluated at the matric potential `ψₘ`
reconstructed from the stored total `pressure_head` exactly as in `pressure_to_saturation!`
(`ψₘ = ψ − ψ_hydrostatic − ψ_elevation`).

Note the physically correct stiff limit: as the soil saturates, `∂θ/∂ψ → 0`, so `τ_water → 0`. This
is real — explicit Richards is extremely stiff near saturation, which is precisely the regime that
motivates implicit timestepping. The `TimeStepWizard`'s `min_change` bound prevents `Δt` from
collapsing to zero in a single call, so the step shrinks gracefully rather than stalling.

### Reduction to a scalar

Both timescales are cell-centred quantities. Following the Oceananigans idiom
(`cell_advection_timescale`), each is expressed as a `KernelFunctionOperation{Center,Center,Center}`
over the underlying Oceananigans grid (`get_field_grid(grid)`), and reduced with `minimum(...)`. No
scratch `Field` is allocated. The diagnostic is evaluated on the host inside a `Callback`, *not*
inside the traced/compiled `run_timesteps!` loop, so it need not satisfy the in-kernel Reactant
"no reachable throw" constraints (it must still be GPU-compatible for CUDA reductions, which the
existing pointwise kernel functions already are).

## Summary of changes

### New source file: `src/diagnostics/cell_diffusion_timescale.jl`

Dispatch chain (all internal to the `Terrarium` module):

- `Oceananigans.Diagnostics.cell_diffusion_timescale(integrator::ModelIntegrator)` →
  `cell_diffusion_timescale(integrator.state, integrator.model)`.
- `Oceananigans.Advection.cell_advection_timescale(integrator::ModelIntegrator) = NF(Inf)` — land
  models have no advective transport; returning `Inf` lets a plain `TimeStepWizard(diffusive_cfl=…)`
  work without a custom `cell_advection_timescale`.
- `cell_diffusion_timescale(state, model::SoilModel)` and `(state, model::LandModel)` forward to the
  soil component (`model.soil`, `model.grid`, `model.constants`); the diffusive operators live in
  the soil.
- `cell_diffusion_timescale(state, grid, soil::SoilEnergyWaterCarbon, constants)` returns
  `min(τ_energy, τ_water)` from the energy and hydrology sub-processes.
- Generic fallbacks return `NF(Inf)` (no restriction), mirroring Oceananigans'
  `infinite_diffusion_timescale`.

Per-process methods + kernel functions:

- `cell_diffusion_timescale(state, grid, energy::SoilThermodynamics, soil, constants)` builds a KFO
  around `compute_thermal_diffusion_timescale(i,j,k,grid,fields,energy,hydrology,strat,bgc)` and
  returns its `minimum`.
- `cell_diffusion_timescale(state, grid, hydrology::SoilHydrology{NF,RichardsEq}, soil, constants)`
  builds a KFO around `compute_hydraulic_diffusion_timescale(i,j,k,grid,fields,hydrology,strat,bgc)`
  and returns its `minimum`. Non-`RichardsEq`/`NoFlow` hydrology falls back to `Inf`.

The kernel functions reuse the existing pointwise machinery (`soil_composition`,
`compute_thermal_conductivity`, `compute_heat_capacity`, `hydraulic_conductivity`, `porosity`,
`get_swrc`, `Δzᵃᵃᶜ`). The SWRC derivative value is converted to `NF` to keep the reduction type
stable.

### `src/Terrarium.jl`

- `include("diagnostics/cell_diffusion_timescale.jl")` after the model-integrator include.
- `import Oceananigans.Advection` (for the `cell_advection_timescale` extension).
- Re-export `TimeStepWizard`, `conjure_time_step_wizard!`, `Callback`, `add_callback!`,
  `IterationInterval`, `TimeInterval` so adaptive simulations can be assembled with `using Terrarium`
  alone.

No changes to `Project.toml` `[deps]`/`[weakdeps]`. No changes to the timesteppers themselves (the
wizard drives `Δt` externally through the existing `Simulation` → `time_step!(integrator, Δt)` path).

## Testing and verification

New `test/timestepping/adaptive_timestep.jl` (added to the `Timestepping` testset):

- `cell_diffusion_timescale(integrator)` is finite and positive for a `SoilModel` on a `ColumnGrid`.
- Analytic cross-check on a single homogeneous column: the returned thermal timescale equals
  `minⱼ Δzⱼ² Cⱼ / κⱼ` recomputed independently from the composition, and (when the water timescale is
  larger) the reported value matches the thermal branch.
- A `TimeStepWizard(diffusive_cfl=…)` attached to a `Simulation(integrator)` via `Callback` adjusts
  `simulation.Δt` and the run stays finite.
- `cell_advection_timescale(integrator)` returns `Inf`.

## Documentation changes

- Canonical docstrings (`$TYPEDSIGNATURES`) on all new methods and kernel functions, with the
  governing equations and references.
- A short "Adaptive timestepping" page under `docs/src/running/` describing the diffusive CFL, the
  `TimeStepWizard` usage pattern with a `ModelIntegrator`, and the stiff-saturation caveat. Kernel
  functions listed under a "Kernel functions" section per the doc conventions.

## Known limitations

- The hydraulic timescale evaluates the *unfrozen* SWRC capacity; freezing effects on the moisture
  capacity are neglected (frozen cells have `K → 0`, so `τ_water → ∞` and impose no restriction —
  conservative).
- Uses sensible heat capacity, not the apparent (latent-heat-inclusive) capacity; conservative near
  the freezing front.
- Adaptive stepping under `ReactantState` is out of scope: `Δt` is compiled into the traced loop, so
  a host-side wizard callback does not apply there. The diagnostic itself is written to be
  architecture-agnostic.
- The wizard drives an `Oceananigans.Simulation` wrapping the integrator; the bespoke
  `run!(::ModelIntegrator; steps/period)` entry point does not process callbacks.

## Future work

- Implicit (or IMEX) treatment of the diffusive soil operators, using this timescale to set the
  explicit sub-step and/or as a preconditioner scale.
- Optionally expose a Terrarium-level `conjure_time_step_wizard!` convenience specialised for land
  models (thermal/hydraulic only).
- An apparent-heat-capacity variant of `τ_heat` for sharper (less conservative) steps through the
  phase-change zone.
