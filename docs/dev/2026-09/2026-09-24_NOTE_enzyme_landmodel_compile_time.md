# Enzyme compile time of `LandModel` time steps blocks coupled Speedy + Terrarium adjoints

> Status: **investigation note** (not an implementation plan). Records a measured AD limitation, the
> bisection that located it, and two small source changes that came out of it. The coupled
> SpeedyWeather + Terrarium adjoint still does not compile.

Date: 2026-09-24 (measurements), updated 2026-09-25 (source changes, coupled-step follow-up)

Base revision: 3eb7cff52 (`main`), worktree `../Terrarium-speedy-ad`, branch `bg/speedy-enzyme-lateral-coupling`

## Originating prompt

> Please check out a new worktree from `main` and draft an example script that differentiates
> Terrarium + Speedy using Enzyme (no reactant). The derivative should be of a single land grid cell
> surface soil temperature w.r.t the full spatial field. The goal would be to see whether or not we
> can quantify the lateral coupling of land cells via the atmosphere.

## Problem description

The example script `examples/autodiff/speedy_terrarium_lateral_coupling.jl` takes the reverse-mode
derivative of one land column's surface soil temperature, after a short coupled SpeedyWeather +
Terrarium integration, with respect to the initial soil internal-energy field. The forward model, the
objective, and a finite-difference check all work. The `Enzyme.autodiff` call does not finish
compiling in practical time: at T9 with one atmospheric layer and two soil layers, a single coupled
time step was still inside Enzyme's activity/type analysis after 116 minutes.

## Environment

Julia 1.10.12 (`+lts`), Enzyme 0.13.204, SpeedyWeather 0.22.1, Oceananigans 0.113.1, Checkpointing
0.11.2, Terrarium at the base revision above. CPU only, 128-core node. `Enzyme.Compiler.RunAttributor[]
= false` throughout (the workaround SpeedyWeather's own sensitivity examples use). All probes use
`set_runtime_activity(Reverse)` and `@ad_checkpoint Revolve(N)` unless stated.

Note for Julia 1.10: Pkg ignores `[sources]`, so the example environment must `Pkg.develop` the
checkout explicitly or it silently resolves a registered `Terrarium v0.1.7` (Oceananigans 0.111).

## Measurements

Every entry is one Enzyme reverse pass over the stated function. "Killed" means the process was
terminated by an external `timeout` while still compiling; the backtrace in every killed case was
inside Enzyme's `ActivityAnalysis` / `TypeAnalysis` (`isValueInactiveFromUsers`,
`AugmentWithJuliaObjectType`, `TypeTree::insert`), recursing through `recursivelyHandleSubfunction`.

### Coupled step (`SpeedyWeather.time_step!` with Terrarium land), T9, 1 atmospheric layer, 2 soil layers, 1 step

| variant | result |
|---|---|
| as in the script (`set_runtime_activity`, `Duplicated(model)`) | killed at 116 min |
| plain `Reverse` | killed at 120 min |
| `Const(model)` | killed at 120 min |
| `Enzyme.API.looseTypeAnalysis!(true)` | killed at 120 min |
| **`snow = nothing`** | running when this note was written; see the script's warning |

T21 / 5 layers / 4 soil layers (the script's original default) was killed at a 58 min cap once and stopped manually after ~2 h on a second attempt, still compiling both times.

### Controls and bisection

| function differentiated | grid | result |
|---|---|---|
| SpeedyWeather-only `time_step!`, default land, T9 / 1 layer | ring grid | **1166 s** |
| `SoilModel` `timestep!` ×3 | 1 column | **123 s** |
| `SoilModel` `timestep!` ×3 | 291-column `ColumnRingGrid` | **130 s**, remote gradient exactly 0 |
| `LandModel` default (implicit SEB, `SingleLayerSnow`, surface hydrology; `vegetation = nothing`) ×3 | 291-column `ColumnRingGrid` | killed at 89 min |
| `LandModel` default ×3 | 1 column | killed at 60 min |
| `LandModel`, `PrescribedSkinTemperature`, snow kept ×3 | 1 column | killed at 60 min |
| `LandModel`, **`snow = nothing`**, implicit SEB kept ×3 | 1 column | **3051 s** |
| `LandModel`, `PrescribedSkinTemperature` + `snow = nothing` ×3 | 1 column | **1820 s** |

### Things that did not help

Marking all 28 host-level `compute_auxiliary!` / `compute_tendencies!` definitions `@inline` made no
measurable difference (52.0 min vs. a 50.9 min baseline on the single-column `LandModel` probe). Those
edits were reverted and are not on this branch.

## Conclusions

1. The coupling layer (`SpeedyWeatherTerrariumExt`) and SpeedyWeather itself are not the cause. The
   atmosphere-only step differentiates in ~19 min at this size, and none of the coupled-side knobs
   (runtime activity, `Const(model)`, loose type analysis) changes the outcome.
2. Column count and the `ColumnRingGrid` are not the cause. `SoilModel` compiles in ~2 min on 1 and
   on 291 ring columns alike, with the physically correct purely local gradient.
3. The cost is in the `LandModel` process set, and **`SingleLayerSnow` is the component that makes a
   single-column step exceed an hour**. Removing it alone brings the step to 51 min; removing the
   implicit skin-temperature solve alone does not help; removing both gives 30 min. Even the reduced
   `LandModel` is 15× slower to differentiate than `SoilModel`, so surface hydrology, the SEB flux
   code, and `PrescribedAtmosphere` carry a large cost of their own.
4. A `LandModel` `timestep!` had never been differentiated in Terrarium's test suite; the existing
   tests cover `SoilModel` and `SnowModel` steps and individual `LandModel` kernel functions.

## Consequences for the example script

The script sets `snow = nothing` and documents why. The coupled step does not compile even with that
configuration (see below); the script's warning block states the current status. Prescribing the skin temperature is not an acceptable substitute, since the
implicit skin-temperature solve is the land-atmosphere coupling the experiment is about.

## Source changes made (2026-09-25)

Two changes are on this branch. Both are small, both are justified below, and neither changes any
numerical result: gradients before and after are bit-identical, and the full test suite passes.

### 1. `StateVariables` is now a `mutable struct` with `const` fields

`src/state_variables.jl`. All seven fields are marked `const`, so nothing about the type's semantics
changes: the fields still cannot be reassigned, `Adapt.adapt_structure` and
`ConstructionBase.constructorof` still go through the same constructor, and the struct is still
concretely typed.

**Measured: 4.1× faster Enzyme compile** on the single-column `LandModel` probe (`snow = nothing`,
`vegetation = nothing`, homogeneous soil), 3051 s → 739 s, with an unchanged gradient (3.8095163e-7).

The mechanism is that an immutable non-`isbits` struct has no stable identity for LLVM: SROA is free
to decompose it into flattened SSA values, independently, at every non-inlined call boundary it
crosses. Enzyme then redoes the full type-tree and activity analysis of the whole aggregate at each
of those occurrences, and `StateVariables` is a deeply nested aggregate of `NamedTuple`s of `Field`s
that is passed through nearly every function in a time step. Making it mutable gives it one heap
pointer that persists through the call graph, so that analysis happens once. This matches what the
Enzyme authors say about immutable structs being copied between stack frames.

### 2. `run_timesteps!` dispatches on `Val{true}`/`Val{false}` instead of branching on a `Bool`

`src/timesteppers/model_integrator.jl`. The `if show_progress ... else ... end` branch became two
methods, and the `show_progress` keyword default became the literal `Val(false)`; `Bool` arguments
are still accepted and normalized by `as_val`, so `run!(integrator; show_progress = true)` is
unchanged for users.

This is not a performance change, it is a correctness prerequisite for the coupled adjoint. With
SpeedyWeather loaded, differentiating through `run!` failed with an `IllegalTypeAnalysisException`
inside `SpeedyWeather.speedstring`. SpeedyWeather extends `ProgressMeter.speedstring(::AbstractFloat)`
globally, so Terrarium's own unrelated `@showprogress` bar resolves to SpeedyWeather's method once
both packages are loaded, and that method reads mutable global `Ref`s that Enzyme's strict-aliasing
analysis rejects. The branch is dead at runtime, but a runtime `Bool` is not provably constant across
a non-inlined call boundary, so Enzyme still compiled it. Dispatch removes the dead method from the
call graph entirely. Using `Val(false)` as the keyword *default* is what makes this reliable: Julia's
keyword sugar substitutes that literal verbatim at every call site that omits the keyword, so it does
not depend on constant-propagation heuristics (which empirically did not fire here).

## Coupled step after those changes: still blocked, two further failures

Neither change unblocks the coupled T9 step. Past the `speedstring` failure there are two more:

| configuration | result |
|---|---|
| pre-refactor source + `Enzyme.API.strictAliasing!(false)` | clears `speedstring`, then `EnzymeInternalError` in SpeedyWeather's own `vertical_advection!` after ~23 min |
| refactored source, no workaround | clears `speedstring`, then SIGABRT after ~28 min: `TypeAnalysis.cpp:6519: Assertion 'ty == ty2' failed` via `createSelectInstAdjoint` |

The SIGABRT backtrace points at the refactored `run_timesteps!`/`run!`, but that is an inlining
artifact rather than a defect in the refactor: an isolated probe differentiating a plain `SoilModel`
through exactly the same refactored `run!` path, with no SpeedyWeather loaded, compiles cleanly in
66 s and returns the known-correct gradient (2.2433834e-7). Both remaining failures are inside code
that only becomes reachable once the `speedstring` blocker is cleared, and at least one of them is
squarely inside SpeedyWeather's dynamical core.

Disabling vertical advection is not a way out. SpeedyWeather has no no-op advection scheme (Upwind,
WENO, and Centered are all real physics, called unconditionally from `dynamics_tendencies!`), the one
broader toggle `model.dynamics::Bool` uses the same runtime-`Bool`-across-a-call-boundary pattern that
caused the `speedstring` failure in the first place, and turning off the dynamical core would remove
the atmospheric transport the experiment exists to measure.

## Suggested follow-up (source work, needs its own plan)

- Add a `LandModel` `timestep!` case to `test/differentiability` with a compile-time budget, so this
  regresses visibly. The `StateVariables` mutability result in particular has no test guarding it.
- Report the `vertical_advection!` `EnzymeInternalError` and the `TypeAnalysis` assertion upstream;
  both are outside Terrarium.
- Profile which parts of `SingleLayerSnow` and `ImplicitSkinTemperature` Enzyme's activity analysis
  spends its time on (nested non-inlined calls, the `RootSolvers.find_zero` loop with early exit, the
  `build_residual` closure). Candidate mitigations: inlining or restructuring the residual closure,
  bounding the Newton loop statically, marking clearly inactive arguments `Const` via Enzyme rules,
  and reducing the aggregate size of what is passed into the fused SEB kernel.
- A smooth `SFCC` freeze curve dispatched in the energy closure would remove the `FreeWater` plateau
  that zeroes sensitivities at 0 °C; it is exported but not wired in.
