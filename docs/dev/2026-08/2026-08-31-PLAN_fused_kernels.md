# Fused kernel launches for coupled soil, surface, and vegetation processes

> Status: **in progress** (approved; Phase 1 implemented, Phases 2–4 planned). Replace the per-process
> `compute_auxiliary!` / `compute_tendencies!` kernel launches inside the coupled process types
> (`SoilEnergyWaterCarbon`, `SurfaceHydrology`, `VegetationCarbonCycle`) with a single fused kernel
> each, following the pattern established for `SurfaceEnergyBalance` (commit `2fc7af72a`). Per-process
> launches are kept for standalone use and testing. Phase 1 first relocates the `surface_excess_water`
> prognostic from soil hydrology to surface runoff so the soil fusion in Phase 2 is clean — rev 4
> found this also connects a currently dead tendency path (the pool has no drain in the coupled
> `LandModel` today), not just a scaling fix, which changes what Phase 1's test coverage must include.

Date of initial draft: 2026-08-31

Base revision: b005a836aa2c26b305cc0faa304ba7748e1e4476

## Originating prompt

> I would like to implement more efficient kernel fusing in Terrarium. You can see an example of how
> I did this before in `surface_energy_balance.jl`. The basic idea is to keep per-process kernel
> launches for testing and standalone use, but to create separate coupled kernels in the coupled
> process type. Please try implementing this for `SoilEnergyWaterCarbon` based on the existing kernel
> functions. Feel free to add new mutating variants of kernel functions if necessary. If anything is
> unclear or you hit a design issue, please stop to ask questions.

## Revision log

> 2026-08-31 (rev 1) — Initial draft, scoped to `SoilEnergyWaterCarbon` only. Clarifications during
> scoping:
> - **Do not fuse `initialize!`, `compute_boundary_conditions!`, or `closure!`/`invclosure!`.** Focus
>   only on `compute_auxiliary!` and `compute_tendencies!`.
> - Keep the existing per-process host methods untouched so standalone use, tests, and Enzyme
>   differentiability tests that target them directly keep working.
>
> 2026-08-31 (rev 2) — Restructured per review:
> - **Naming convention.** The fused/coupled `@kernel`s reuse the generic names
>   `compute_auxiliary_kernel!` / `compute_tendencies_kernel!`, dispatching on the coupled process
>   type (as `SingleLayerSnow`, `PALADYNCanopyInterception`, `PALADYNCarbonDynamics`, etc. already do).
>   No bespoke `compute_soil_*_kernel!` names.
> - **No `except = out` in fused kernels.** Passing `get_fields(...; except = out)` strips fields that
>   a *later* sub-process in the same fused kernel needs to *read* (because they are an *output* of an
>   earlier sub-process). Instead pass the full `fields` and accept the duplication between `out` and
>   `fields` (they alias the same `Field` objects, so intra-cell read-after-write works).
> - **Four phases**, adding surface hydrology and vegetation, and moving `surface_excess_water` first:
>   - **Phase 1** — move `surface_excess_water` to runoff; soil hydrology conditionally fills the pool
>     only when an `AbstractSurfaceRunoff` is passed as an optional dependency (standalone: discard).
>   - **Phase 2** — fuse soil (`SoilEnergyWaterCarbon`).
>   - **Phase 3** — fuse surface hydrology (`SurfaceHydrology`).
>   - **Phase 4** — fuse vegetation carbon (`VegetationCarbonCycle`).
>
> 2026-08-31 (rev 3) — Sign-off received on the three open questions:
> - **(1) Phase 1 numerics confirmed as a bug fix.** The `Nz×` over-counting of the
>   `surface_excess_water` tendency is to be corrected (XY runoff kernel), not preserved.
> - **(2) Read the pool via the runoff accessor** (`surface_excess_water(i, j, grid, fields,
>   ::AbstractSurfaceRunoff)`) when runoff is present.
> - **(3) Partial auxiliary fusion is in scope for Phase 4.** `root_distribution` and
>   `plant_available_water` are handled in separate launches from the rest, as already noted. The
>   photosynthesis / stomatal-conductance ordering is **not** an obstacle: `compute_photosynthesis!`
>   only uses `compute_λc(stomcond, vpd)` (a parameter-only call, not stomatal's output field), while
>   `compute_stomatal_conductance` reads `net_assimilation` (photosynthesis's output). So
>   photosynthesis → stomatal conductance is a clean forward dependency that chains inside a single
>   fused kernel.
>
> 2026-08-31 (rev 4) — Correction to Phase 1's problem statement after tracing every call site that
> reaches `compute_tendencies!` for `SoilHydrology{RichardsEq}` (review by Claude):
> - **The `Nz×` over-counting is currently dead code, not a live numerics bug.** `soil_coupled.jl`
>   calls `compute_tendencies!(state, grid, soil.hydrology, soil, constants)` with no `evtr`/`runoff`
>   args, so `runoff` defaults to `nothing`; `SurfaceHydrology.compute_tendencies!` only calls canopy
>   interception's tendency, never runoff's. `runoff` is never non-`nothing` at the point
>   `compute_surface_excess_water_tendency` runs anywhere in `src/`, so `∂S∂t` — and therefore the
>   `Nz×`-scaled tendency — is always exactly `0` in every model configuration that exists today.
> - **The real, live bug is upstream of the scaling error.** `adjust_saturation_profile!` adds to
>   `surface_excess_water` whenever the soil column oversaturates (called from `closure!`/
>   `invclosure!` every step), but — because of the wiring gap above — nothing in the coupled
>   `LandModel` ever drains it. The pool is a monotonically-growing sink with no removal path, fully
>   independent of the `Nz×` factor. Phase 1's rewiring (`SurfaceHydrology.compute_tendencies!` gains a
>   call to the runoff tendency) fixes this wiring gap as a side effect of relocating the tendency, not
>   just the scaling.
> - **Consequence for scope, not conclusion.** This does not change the recommended action — moving
>   the tendency to runoff is still correct and still removes the `Nz×` factor — but Phase 1 is
>   properly described as *connecting a dead tendency path* (which happens to also correct a scaling
>   error once connected), not as re-scaling a term that was already live and wrong. The "Numerics
>   change" section and the Phase 1 test plan below are updated accordingly: the new regression test
>   must exercise the coupled `LandModel` `timestep!` path, since that is the path that was silently
>   broken (unbounded pool growth) and is what most needs protection against regressing.
>
> 2026-08-31 (rev 5) — **Phase 1 implemented** after sign-off. Deviations from the plan text, all
> confirmed during implementation/review:
> - **Tendency sign.** The plan wrote the tendency as `min(∂S∂t, S)`, which is ambiguous on sign. Since
>   `explicit_step!` uses `u += tendency·Δt` and `compute_surface_drainage` returns a *positive* rate
>   `D = S/τ_r` (also consumed by the auxiliary `compute_surface_runoff = F + D − I`), the tendency is
>   implemented as the negative removal `-min(D, S)` so the pool is drawn down. `compute_surface_drainage`
>   keeps its positive-drainage convention (the sign lives in the tendency, not the drainage).
> - **`redistribute_saturation_profile!`.** The per-cell `adjust_saturation_profile!` was factored into a
>   shared `redistribute_saturation_profile!(sat, i, j, grid, hydrology)` that returns the surface excess
>   as a water depth, plus two thin dispatch methods (`::Nothing` discards, `::AbstractSurfaceRunoff`
>   writes the pool) — no short-circuit branch inside the kernel.
> - **Type annotations.** All threaded `runoff` arguments use `Optional{AbstractSurfaceRunoff}`; the
>   per-cell dispatch methods use concrete `::Nothing` / `::AbstractSurfaceRunoff`.
> - **Test adjustment (review).** Rather than only deleting the standalone pool assertion in
>   `soil_hydrology_tests.jl`, the test now *keeps* pool coverage at the soil level: a combined
>   `StateVariables(merge(Variables(hydrology), Variables(runoff)), grid)` state checks that passing a
>   runoff routes the excess into the pool (Case 1b), while the standalone call still discards.
>   Plus standalone runoff-tendency tests (`-min(D, S)`, cap binding, empty pool) and the coupled
>   `LandModel` `timestep!` regression test (pool decays as `S₀(1 − Δt/τ)ⁿ`).

## Problem description

The coupled process types fan out to one host call per sub-process, and each host call issues its own
`launch!` over the grid. Per timestep (per `update_state!`), the default vegetated `LandModel` issues:

| Coupled type | `compute_auxiliary!` launches | `compute_tendencies!` launches |
|---|---|---|
| `SoilEnergyWaterCarbon` (Richards) | 1 XYZ (hydraulics) | 2 XYZ (water, energy) + 1 no-op |
| `SurfaceHydrology` | 3 XY (canopy → ET → runoff) | 1 XY (canopy water) |
| `VegetationCarbonCycle` | 6 XY + 1 XYZ (PAW) + 1 derived XY `compute!` | 3 XY (C, ν, GDD) |

(`root_distribution` declares `root_fraction` as a lazy `FunctionField` and its `compute_auxiliary!` is a
no-op; `vegetation_dynamics.compute_auxiliary!` is also a no-op. So the vegetation auxiliary cost is 6
XY launches plus PAW's XYZ launch and its derived `soil_moisture_limiting_factor` `compute!`.)

Each launch re-reads shared state and re-derives shared quantities (e.g. every soil process rebuilds
`SoilComposition` from `saturation_water_ice` / `liquid_water_fraction` / porosity). On GPUs the launch
overhead and redundant global-memory traffic dominate at the small column counts typical of land
models. The surface energy balance already solved the analogous problem (commit `2fc7af72a`): the
coupled `SurfaceEnergyBalance` owns one fused kernel that calls a **per-process mutating variant** of
each kernel function, each a no-op for "prescribed" schemes. We generalize that structure.

## Background

### The fusion pattern (target shape)

A coupled type defines, per pass, one `@kernel` named `compute_auxiliary_kernel!` /
`compute_tendencies_kernel!` dispatching on the coupled type, whose body calls a **per-cell mutating
variant** of each sub-process's kernel function in dependency order. Each mutating variant takes the
shared `out`/`tendencies` `NamedTuple` and is a no-op for sub-processes with nothing to compute. The
per-process host methods (`compute_auxiliary!`/`compute_tendencies!` on the concrete sub-process) are
retained unchanged for standalone use, tests, and Enzyme.

Two rules (from review):

1. **Reuse the generic kernel names.** `compute_auxiliary_kernel!(out, grid, fields, coupled::T, ...)`
   and `compute_tendencies_kernel!(tend, grid, [clock,] fields, coupled::T, ...)`. This matches the
   dominant convention (`snow_single_layer.jl`, `canopy_interception.jl`, `carbon_dynamics.jl`) and
   coexists with the existing per-process `compute_tendencies_kernel!` methods (different dispatch type).
2. **Never `except = out` in a fused launch.** Collect `out` (the fields the fused kernel writes) and
   pass the *full* `fields` (all sub-process state, including those same fields). Within a cell the
   kernel writes `out.foo[i,j,k]` then a later sub-process reads `fields.foo[i,j,k]` — the same
   `Field` object, so the write is visible. `except = out` would remove `foo` from `fields` and break
   the later read.

### Pre-existing `surface_excess_water` cross-`k` accumulation

`surface_excess_water` is an **XY** (Z-reduced) variable, but the current `RichardsEq`
`compute_tendencies_kernel!` is an **XYZ** kernel that calls
`compute_surface_excess_water_tendency!(tend.surface_excess_water, i, j, k, ...)` once per `k`. Because
Oceananigans maps every `k` of a Z-reduced field to the same cell
(`setindex!(r::ZReducedField, v, i, j, k) = setindex!(r.data, v, i, j, 1)`), the `+=` **accumulates the
tendency across all `Nz` layers** (verified: `f[1,1,k] += 1` for `k=1:Nz` leaves `Nz` in the interior
cell). So the write is `Nz ×` too large *as written*. As traced in rev 4 (see "Numerics change" under
Phase 1), this accumulation is **latent / dead code today** — `runoff` is always `nothing` at the call
site, so the term it scales is exactly `0` and the `Nz` factor never surfaces in any existing model
configuration. It would bite the moment the tendency path is connected, which is precisely what Phase 1
does; moving the tendency into an **XY** runoff kernel means the spurious `Nz` factor has no chance to
reappear once the path goes live.

## Phase 1 — Move `surface_excess_water` to runoff

### Current state

- `SoilHydrology{RichardsEq}` declares `prognostic(:surface_excess_water, XY())`
  (`soil_hydrology_rre.jl`).
- `adjust_saturation_profile!` (XY kernel, `soil_hydrology.jl`) adds top-layer oversaturation into the
  pool (`out.surface_excess_water[i,j,1] += …`). Called from `closure!`/`invclosure!`
  (`soil_hydraulic_closures.jl`).
- `compute_surface_excess_water_tendency!` / `…_tendency` (`soil_hydrology.jl`) compute
  `∂S∂t = min(compute_surface_drainage(runoff, S), S)` and `+=` into the XY tendency from the XYZ
  Richards kernel (the cross-`k` bug above).
- `DirectSurfaceRunoff.compute_surface_runoff!` **reads** the pool via
  `surface_excess_water(i, j, grid, fields, soil_hydrology)` to compute infiltration/runoff.
- Accessor `surface_excess_water(i,j,grid,fields,::SoilHydrology)`: `zero` for `NoFlow`,
  `fields.surface_excess_water[i,j]` for `RichardsEq`.

### Target design

1. **Ownership.** Move `prognostic(:surface_excess_water, XY())` from `SoilHydrology{RichardsEq}` to
   `DirectSurfaceRunoff`. (State fields are merged by name across processes, so the field name is
   unchanged wherever it is read.)
2. **Tendency in runoff.** Move `compute_surface_excess_water_tendency!` / `…_tendency` into
   `direct_surface_runoff.jl` and add a standalone
   `compute_tendencies!(state, grid, runoff::DirectSurfaceRunoff, ...)` that launches an **XY**
   `compute_tendencies_kernel!` writing `tendencies.surface_excess_water[i,j,1] = min(∂S∂t, S)`
   (single XY write — no cross-`k` accumulation). Wire
   `compute_tendencies!(state, grid, hydrology::SurfaceHydrology)` to also call the runoff tendencies
   (it currently only calls canopy interception).
3. **Conditional pool fill.** `adjust_saturation_profile!(state, grid, hydrology, runoff = nothing)`
   builds `out = (saturation_water_ice,)` when `runoff` is `nothing` (standalone: excess discarded) and
   `(saturation_water_ice, surface_excess_water = state.surface_excess_water)` when a runoff is passed.
   The per-cell `adjust_saturation_profile!(out, i, j, grid, hydrology, runoff)` dispatches on
   `runoff::Nothing` (skip pool write) vs `runoff::AbstractSurfaceRunoff` (write pool), so no
   short-circuit branch is needed in the kernel.
4. **Thread the optional dependency.** `closure!`/`invclosure!` for `SoilSaturationPressureClosure` and
   `SoilEnergyWaterCarbon` gain a trailing `runoff = nothing` argument forwarded to
   `adjust_saturation_profile!`. `LandModel.closure!`/`invclosure!` pass `model.surface_hydrology.surface_runoff`;
   `SoilModel.closure!`/`invclosure!` pass nothing (standalone → discard). This is a dependency-threading
   change, **not** a closure fusion (closures stay unfused per scope).
5. **Accessor.** Add `surface_excess_water(i, j, grid, fields, ::AbstractSurfaceRunoff) = fields.surface_excess_water[i, j]`
   and have `compute_surface_runoff!` read the pool through the **runoff** accessor (decided in rev 3).
   The `SoilHydrology` accessor keeps its `NoFlow` → `zero(NF)` method (still used by other readers);
   its `RichardsEq` method is removed once nothing reads the pool through soil (verify with a reference
   search during implementation).

### Numerics change (rev 4: dead tendency path, not a live scaling bug)

Tracing every call site that reaches `compute_tendencies!` for `SoilHydrology{RichardsEq}` shows
`runoff` is never non-`nothing` in any model configuration that exists in `src/` today
(`soil_coupled.jl` calls it with no `evtr`/`runoff` args, and `SurfaceHydrology.compute_tendencies!`
never calls the runoff tendency). So the `Nz × min(∂S∂t, S)` expression is always `Nz × 0 = 0` as
written — it is dead code, not a live miscalculation. The actual live bug is upstream:
`adjust_saturation_profile!` keeps adding to `surface_excess_water` on every oversaturated step via
`closure!`/`invclosure!`, but nothing currently drains it, so the pool grows without bound in the
coupled `LandModel`.

Phase 1 fixes both problems at once: relocating the tendency to an XY `DirectSurfaceRunoff` kernel
that `SurfaceHydrology.compute_tendencies!` actually calls connects the dead path (giving the pool its
first real sink in the coupled model), and doing so as an XY write rather than an XYZ accumulation
means the `Nz×` factor never has a chance to reappear. Both effects should be described together when
this phase is reviewed — "fixes a scaling bug" alone understates the change, since the pool was not
being drained at all before this phase.

### Tests touched by Phase 1

- `test/soil/soil_hydrology_tests.jl:106` asserts `state.surface_excess_water ≈ ∫sat_excess` after a
  standalone `adjust_saturation_profile!`. With the pool moved and standalone = discard, this assertion
  is removed; the test keeps checking the saturation profile is clamped to `[sat_min, 1]`.
- New **coupled `LandModel`** regression test (rev 4): run `timestep!` over several steps with
  `RichardsEq` + `DirectSurfaceRunoff` in an oversaturated configuration and assert
  `surface_excess_water` is actually drawn down (not monotonically growing) and that the drained mass
  over a step equals `∫ min(∂S∂t, S) dt` — this is the path that was silently broken (no drain at all)
  and is what most needs protection against regressing, so a standalone-only test is not sufficient.
- New `test/surface/runoff_tests.jl`: standalone `DirectSurfaceRunoff` tendency equals `min(∂S∂t, S)`;
  coupled soil oversaturation fills the runoff-owned pool via `closure!`.

## Phase 2 — Fuse soil (`SoilEnergyWaterCarbon`)

After Phase 1, soil hydrology's only tendency is `saturation_water_ice` (Richards), and its only
auxiliary is `hydraulic_conductivity` (Richards). Energy contributes one tendency (`internal_energy`)
and no auxiliary; biogeochemistry contributes nothing.

### `compute_tendencies!` — one fused XYZ launch

```julia
function compute_tendencies!(state, grid, soil::SoilEnergyWaterCarbon, constants::PhysicalConstants)
    tendencies = tendency_fields(state, soil)
    fields = get_fields(state, soil)          # full fields, no `except`
    # `launch!(grid, ...)` inserts `grid` as the kernel's second argument automatically
    launch!(grid, XYZ, compute_tendencies_kernel!, tendencies, state.clock, fields, soil, constants)
    return nothing
end

@kernel inbounds = true function compute_tendencies_kernel!(tendencies, grid, clock, fields,
        soil::SoilEnergyWaterCarbon, constants)
    i, j, k = @index(Global, NTuple)
    compute_water_tendencies!(tendencies, i, j, k, grid, clock, fields, soil.hydrology, soil.strat, soil.biogeochem, constants)
    compute_energy_tendencies!(tendencies, i, j, k, grid, fields, soil.energy, soil.hydrology, soil.strat, soil.biogeochem)
    return nothing
end
```

New per-cell mutating variants (added next to the existing single-field kernel functions, which stay):

- `compute_water_tendencies!(tendencies, i, j, k, grid, clock, fields, hydrology::SoilHydrology{NF, NoFlow}, ...)` → `nothing`
  (immobile water has no `saturation_water_ice` tendency field).
- `compute_water_tendencies!(tendencies, i, j, k, grid, clock, fields, hydrology::SoilHydrology{NF, RichardsEq}, strat, bgc, constants)`
  → `compute_saturation_tendency!(tendencies.saturation_water_ice, ...)`.
- The standalone `RichardsEq` `compute_tendencies_kernel!` is refactored to call
  `compute_water_tendencies!` so the two paths cannot drift. `compute_energy_tendencies!` already has
  the tuple-mutating shape and is reused as-is.

### `compute_auxiliary!` — one fused XYZ launch (skipped when empty)

```julia
function compute_auxiliary!(state, grid, soil::SoilEnergyWaterCarbon, constants::PhysicalConstants)
    out = soil_auxiliary_outputs(state, soil)          # (hydraulic_conductivity=…) or (;)
    fields = get_fields(state, soil)                   # full fields, no `except`
    launch!(grid, XYZ, compute_auxiliary_kernel!, out, fields, soil, constants)
    return nothing
end

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields,
        soil::SoilEnergyWaterCarbon, constants)
    i, j, k = @index(Global, NTuple)
    compute_hydraulics!(out, i, j, k, grid, fields, soil.hydrology, soil.strat, soil.biogeochem)
    return nothing
end
```

`soil_auxiliary_outputs` dispatches on the flow type: `(hydraulic_conductivity = state.hydraulic_conductivity,)`
for `RichardsEq`, `(;)` for `NoFlow`. It cannot be `auxiliary_fields(state, soil)` wholesale because
`saturation_water_ice` (auxiliary under `NoFlow`) is an *input* the kernel reads. The water table is
untouched (recomputed in `initialize!`/`closure!`, not `compute_auxiliary!`).

### Debug hooks

Add `debughook!` methods for the two new fused kernels (`checkfinite!(out)`), mirroring the existing
per-process hooks.

## Phase 3 — Fuse surface hydrology (`SurfaceHydrology`)

`SurfaceHydrology` = `canopy_interception` + `evapotranspiration` + `surface_runoff`, all XY.

### `compute_auxiliary!` — one fused XY launch (dependency-ordered)

Current order (canopy → ET → runoff) is preserved inside the fused kernel so each sub-process reads the
previous one's outputs from the shared `fields` (which alias `out`):

```julia
@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields,
        hydrology::SurfaceHydrology, constants, atmos, soil, vegetation, snow)
    i, j = @index(Global, NTuple)
    compute_canopy_auxiliary!(out, i, j, grid, fields, hydrology.canopy_interception, atmos)
    compute_evapotranspiration_fluxes!(out, i, j, grid, fields, hydrology.evapotranspiration, constants, atmos, ...)
    compute_surface_runoff!(out, i, j, grid, fields, hydrology.surface_runoff, hydrology.canopy_interception, get_hydrology(soil), snow)
    return nothing
end
```

The per-cell mutating variants already exist (`compute_canopy_auxiliary!`,
`compute_evapotranspiration_fluxes!`, `compute_surface_runoff!`); the ET variant may need a thin
tuple-mutating wrapper to match the `out`-first convention. Host `compute_auxiliary!` collects the union
of the three sub-processes' auxiliary outputs as `out` and passes full `fields`.

### `compute_tendencies!` — one fused XY launch

After Phase 1, two independent XY tendencies: canopy water and surface excess water.

```julia
@kernel inbounds = true function compute_tendencies_kernel!(tendencies, grid, fields,
        hydrology::SurfaceHydrology, evapotranspiration)
    i, j = @index(Global, NTuple)
    compute_canopy_water_tendency!(tendencies, i, j, grid, fields, hydrology.canopy_interception, evapotranspiration)
    compute_surface_excess_water_tendency!(tendencies, i, j, grid, fields, hydrology.surface_runoff)
    return nothing
end
```

`compute_canopy_water_tendency!` already takes the `tendencies` tuple; the runoff variant (from Phase 1)
is adapted to the tuple form. ET has no tendency.

## Phase 4 — Fuse vegetation carbon (`VegetationCarbonCycle`)

### `compute_tendencies!` — one fused XY launch (clean win)

Three independent XY tendencies (`carbon_vegetation`, `vegetation_area_fraction`, `growing_degree_days`):

```julia
@kernel inbounds = true function compute_tendencies_kernel!(tendencies, grid, fields,
        veg::VegetationCarbonCycle, constants, atmos)
    i, j = @index(Global, NTuple)
    compute_veg_carbon_tendencies!(tendencies, i, j, grid, fields, veg.carbon_dynamics, veg.traits)
    compute_ν_tendencies!(tendencies, i, j, grid, fields, veg.vegetation_dynamics, veg.carbon_dynamics, veg.traits)
    compute_gdd_tendency!(tendencies, i, j, grid, fields, veg.phenology, atmos)
    return nothing
end
```

`compute_veg_carbon_tendencies!`, `compute_ν_tendencies!`, and `compute_gdd_tendency!` already take the
`tend` tuple. `vegetation_dynamics` may be `nothing` (`PrescribedVegetation`) — a `nothing` no-op variant
handles it.

### `compute_auxiliary!` — partial fusion (in scope, rev 3)

The auxiliary sub-processes span two iteration spaces, so they fuse into **two** launches, with the
soil-dependent ones kept separate (rev 3):

1. **`root_distribution`** — `compute_auxiliary!` is already a no-op (`root_fraction` is a lazy
   `FunctionField`); unchanged, no launch.
2. **`plant_available_water`** — stays a **separate XYZ launch** (rev 3). It writes the XYZ
   `plant_available_water` and then derives the XY `soil_moisture_limiting_factor` via `compute!`. This
   must run **before** the fused XY kernel, which reads `soil_moisture_limiting_factor` (photosynthesis
   and stomatal conductance both consume it) — the same ordering the current fan-out already uses.
3. **One fused XY `compute_auxiliary_kernel!`** chaining the five real XY sub-processes in the current
   dependency order:

```julia
@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields,
        veg::VegetationCarbonCycle, constants, atmos, soil)
    i, j = @index(Global, NTuple)
    compute_veg_carbon_auxiliary!(out, i, j, grid, fields, veg.carbon_dynamics, veg.traits)
    compute_phenology!(out, i, j, grid, fields, veg.phenology, atmos)
    # photosynthesis uses only `compute_λc(stomcond, vpd)` (a parameter call), not stomatal's output;
    # stomatal conductance then reads photosynthesis's `net_assimilation` from `fields` (aliases `out`).
    compute_photosynthesis!(out, i, j, grid, fields, veg.photosynthesis, veg.stomatal_conductance, veg.traits, constants, atmos)
    compute_stomatal_conductance!(out, i, j, grid, fields, veg.stomatal_conductance, veg.traits, constants, atmos)
    compute_autotrophic_respiration!(out, i, j, grid, fields, veg.autotrophic_respiration, veg.carbon_dynamics, veg.phenology, veg.traits, atmos)
    return nothing
end
```

The photosynthesis → stomatal-conductance ordering is a clean forward dependency (rev 3): photosynthesis
does not read stomatal's output field, so chaining them in one kernel is correct. `vegetation_dynamics`
has no auxiliary (`compute_auxiliary!` is a no-op) and is omitted. The host `compute_auxiliary!` collects
the union of the five sub-processes' auxiliary outputs as `out` and passes full `fields`. Net: 6 XY
launches → 1 XY launch (PAW's XYZ launch and derived `compute!` unchanged).

## Testing and verification

- **Per-phase numerical equivalence.** For each fused type, a test builds the model, runs
  `initialize!` → `compute_auxiliary!` → `compute_tendencies!`, and asserts the fused outputs match the
  values from calling the per-process host methods directly (the pre-fusion path), to machine precision —
  **except** the Phase 1 `surface_excess_water` tendency, where no pre-fusion coupled value exists to
  compare against (the path was dead: `∂S∂t = 0`, pool undrained). That tendency is instead asserted
  directly against `min(∂S∂t, S)` in the new coupled `LandModel` regression test described in Phase 1.
- **Existing suites green.** `test/soil/*`, `test/surface/*`, `test/vegetation/*`,
  `test/coupled_models/land_model_tests.jl`, `test/timestepping/*`.
- **Differentiability.** `test/differentiability/soil_hydrology_diff.jl` (per-process + full-model
  `timestep!`), plus any vegetation diff tests, stay finite; the `timestep!` adjoint is the key
  Enzyme-safety check for each fused kernel.
- **Reactant.** `test/reactant/` (own env, not under `--check-bounds=yes`) still compiles every fused
  kernel — they inherit only throw-free, allocation-free kernel functions.
- **Launch-count spot check.** Default vegetated model: soil tendencies 2→1 XYZ, soil auxiliaries
  1→1 XYZ (NoFlow 1→0), surface hydrology auxiliaries 3→1 XY and tendencies 1→1 XY, vegetation
  tendencies 3→1 XY and auxiliaries 6→1 XY (PAW's XYZ launch and derived `compute!` unchanged).

## Documentation changes

- `$TYPEDSIGNATURES` docstrings on new mutating variants and fused kernel functions; note in each
  coupled type's docstring that per-process launches remain for standalone use.
- Update the `DirectSurfaceRunoff` and `SoilHydrology` docstrings for the `surface_excess_water`
  ownership change and the optional-runoff dependency of `adjust_saturation_profile!`.
- No new exported API; fused kernels and mutating variants are internal.

## Known limitations

- Phase 2 auxiliary fusion currently only folds in hydrology (energy/bgc auxiliaries are no-ops today);
  the auxiliary-side win is limited to dropping the `NoFlow` no-op launch until more processes diagnose
  auxiliaries.
- Soil `compute_tendencies!` keeps the `(state, grid, soil, constants)` signature; ET enters the soil
  water balance through boundary conditions, so no `evtr`/`runoff` is threaded into the fused soil
  tendencies kernel.
- Phase 4 auxiliary fusion is partial by construction: `plant_available_water` remains a separate XYZ
  launch (it must precede the fused XY kernel, which reads its derived
  `soil_moisture_limiting_factor`), and `root_distribution` stays a lazy `FunctionField` with no launch.

## Future work

- Fuse `compute_boundary_conditions!` halo-filling with the tendencies kernels once the BC refactor
  settles (explicitly out of scope here).
- Fuse the water-table XY scan and any future soil auxiliaries where iteration spaces allow.
- Revisit the `PrescribedVegetation` / `nothing`-component auxiliary paths for a unified fused entry.
- Fold `plant_available_water`'s derived `soil_moisture_limiting_factor` into the XYZ kernel itself
  (replacing the separate `compute!`), which would also let it fuse with other XYZ auxiliaries.

## Decisions (signed off, rev 3)

1. **Phase 1 numerics.** The `Nz×` over-counting of the `surface_excess_water` tendency is a bug and is
   corrected by moving the tendency into an XY runoff kernel.
2. **Phase 1 accessor.** The pool is read through the `AbstractSurfaceRunoff` accessor when runoff is
   present.
3. **Phase 4 scope.** Partial auxiliary fusion is in scope: `root_distribution` and
   `plant_available_water` in separate launches from the fused XY kernel. The photosynthesis /
   stomatal-conductance ordering is not an obstacle (photosynthesis uses only `compute_λc(stomcond, vpd)`,
   a parameter call, while stomatal conductance reads photosynthesis's output).
