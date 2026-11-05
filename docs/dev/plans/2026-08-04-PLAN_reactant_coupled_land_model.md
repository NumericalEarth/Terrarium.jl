# Reactant support for the coupled `LandModel` (soil + snow, no vegetation)

> Status: **completed** for the soil+snow configuration. All six Reactant blockers are resolved —
> blocker #5 was fixed upstream and is available in Oceananigans 0.110.15, so no Terrarium-side
> workaround is needed. `:land_soil_snow` passes against the released version with all 62 fields
> matching, and the full CPU and Reactant suites pass. Vegetation remains unsupported, and the
> *default* `LandModel` still needs `NewtonSolver` to compile (see Known limitations).

Date of initial draft: 2026-08-04

Base revision: e015a29db9d97090edfd112a5eea165640134404

## Originating prompt

> Add a LandModel with Soil (including Richards eq) and Snow but without Vegetation to the Reactant
> model setup and test (and work on) it running correctly with Reactant. Report issues to me.

## Revision log

- 2026-08-04: initial draft, written after the blockers were found and fixed.
- 2026-08-04: the Terrarium-side workaround for blocker #5 (materializing the negated infiltration into
  a `soil_water_flux` coupling field) was **reverted** on request — the underlying `getbc` limitation is
  to be fixed in Oceananigans instead. Written up separately, with a Terrarium-free MWE, in
  `2026-08-04-OCEANANIGANS_getbc_2d_indexing_issue.md`.
- 2026-08-13: **blocker #5 is fixed upstream.** Oceananigans 0.110.15 indexes array-valued boundary
  conditions in 3D (`condition[i, j, 1]`), so the lazy `UnaryOperation` water BC in `LandModel` works
  without the reverted workaround. Compat bounds raised to `0.110.15` in the root and
  `test/reactant` projects; the test environment resolves to Oceananigans 0.110.15 with Reactant
  0.2.278. Note that this required letting Reactant move off its pinned 0.2.272 — that pin, not any
  declared bound, was what previously held Oceananigans at ≤ 0.110.13.
- 2026-08-13: repaired a half-finished `pad_indices` → `field_indices` rename that broke the solvers.
  `NewtonSolver` still called the removed `pad_indices` and shadowed the new function with a local of
  the same name (AGENTS.md pitfall #7), and `FixedPointSolver` splatted the *function* rather than
  calling it. Both `RootSolver` and `FixedPointSolver` also read their target field with the raw 2D
  index tuple; all reads and writes now go through `field_indices` (pitfall #10). The full CPU suite
  passes again.
- 2026-08-13: added an `@assert_kernel` macro (`src/utils/kernel_utils.jl`) — a `Base.@assert`
  replacement that is disabled once Reactant is loaded, so kernels can keep correctness guards on the
  CPU/GPU backends without emitting an un-raisable `llvm.intr.trap`. See the note below on why the
  guard is load-time rather than trace-time.
- 2026-08-13: `:land_soil_snow` **passes against released Oceananigans 0.110.15 with no workaround** —
  all 62 fields match. The full Reactant suite is green (115/115 comparison, 6/6 autodiff), as is the
  CPU suite. The soil+snow half of this plan is complete.
- 2026-08-14: blocker #3's local workaround was reverted (`dcae39485`, "revert to set!"), on the
  assumption that Oceananigans' upstream `set!` improvements (unrelated to blocker #5) had also fixed
  this case. They had not (yet — see below): `set!(state.skin_temperature, state.ground_temperature)`
  still failed with the same `pointer` error on `ConcretePJRTArray`. Root-caused to an upstream bug in
  `OceananigansReactantExt.Fields.broadcastable_dimension`, which is asymmetric and misclassifies this
  location-`Nothing`-vs-windowed-`Center` pair as non-broadcastable, sending `set!` into a
  CPU-interpolation branch that itself hits a missing Reactant.jl `copyto!` overload
  (`SubArray{Array} ← SubArray{ConcretePJRTArray}`, device → host). Reproduced in a Terrarium-free MWE;
  see `2026-08-14-oceananigans_broadcastable_dimension_issue.md` and the accompanying MWE script.
- 2026-08-14: **fixed upstream, on Oceananigans branch `glw/copyable-fields-predicate`** (commit
  `b9ff6dcd`, unifying the base `set!` predicate and the Reactant extension's predicate). Not yet merged
  to `main` or released. `test/reactant/Project.toml` now pins Oceananigans to this branch via
  `[sources]` (previously that section didn't exist in this file at all — only the *root*
  `Project.toml` had a source override, so the test environment silently ignored it and every prior
  local test in `test/reactant` had been running against the plain registry release regardless of any
  branch pin declared elsewhere). The same fix added a `Terrarium = {path = "../.."}` source, since that
  was *also* missing and caused `test/reactant` to resolve `Terrarium` from a stale package-cache
  snapshot rather than the live working tree (silently invalidating local `src/` edits tested through
  that environment, including `NewtonSolver`, which was `UndefVarError` there until fixed). With both
  sources correctly pinned and a from-scratch re-instantiate, blocker #3's local workaround is
  unnecessary: `skin_temperature.jl`'s `initialize!` is back to plain `set!`, and `:land_soil_snow`
  passes 57/57 fields (~7 min). Full details in
  `2026-08-14-oceananigans_broadcastable_dimension_issue.md`.

## Problem description

`test/reactant/` only covered *standalone* models: `SoilModel` with the default `NoFlow` hydrology
(`:soil_heat_column`, `:soil_heat_column_stretched`, `:soil_heat_global`) and a standalone `SnowModel`
(`:snow_column`). The coupled `LandModel` — soil hydrology via the Richardson-Richards equation, a
single-layer snowpack, the surface energy balance with its implicit skin-temperature solve, and the
surface hydrology that couples them — had never been traced. It did not compile.

## Background

Six independent blockers sit between a `LandModel(grid; soil, snow, vegetation = nothing)` and a
working Reactant compile. They are listed in the order they surface, since each one hides the next.

### 1. In-kernel `@assert` with an interpolated message

`saturation_to_pressure!` (`src/processes/soil/hydrology/soil_hydraulic_closures.jl`) asserted
`isfinite(ψm)` with a string-interpolated message. String formatting inside a kernel is not merely a
trap: it drags `IOBuffer`, `jl_string_to_genericmemory`, `ijl_rethrow` and a GC frame into the device
IR. All 25 `InvalidIRError` reasons in the first failure traced back to this single line.

This is the AGENTS.md "no reachable throw paths in kernels" rule. The `@assert` was also redundant: a
`debughook!` for `saturation_to_pressure_kernel!` already runs `checkfinite!` on the output, host-side,
under `TERRARIUM_DEBUG=true`.

**Fix**: drop the `@assert`, and clamp the inverse-SWRC argument from above as well as below
(`clamp(sat * por, sat_res, por)` instead of `max(sat * por, sat_res)`). Oversaturated states would
otherwise evaluate the inverse van Genuchten curve at a negative base and produce the NaN the assertion
was watching for. For legal states (`sat ≤ 1`, guaranteed by `adjust_saturation_profile!`) the clamp is
a no-op.

### 2. Complex arithmetic in the van Genuchten hydraulic conductivity

`hydraulic_conductivity(::AbstractSoilHydraulics{NF, <:VanGenuchten, UnsatKVanGenuchten}, …)` used
`complex()` to tolerate illegal state values, on the reasoning that complex `^` and `sqrt` are total
where their real counterparts raise `DomainError`. Complex math lowers to libdevice symbols that
Reactant cannot resolve:

```
JIT session error: Symbols not found: [ ___nv_cospif, ___nv_hypotf, ___nv_sinpif, ___nv_sincosf ]
CompilationError: MLIR pass pipeline "all" failed   (gpu_compute_hydraulics_kernel!)
```

**Fix**: clamp the effective saturation to the unit interval and use real arithmetic. Every fractional
power then sees a non-negative base, so the `DomainError` path the `complex` trick was avoiding is
unreachable by construction. Verified bit-identical to the complex form for legal states.

### 3. `set!(::Field, ::Field)` across a location mismatch

`initialize!(state, grid, ::ImplicitSkinTemperature, …)` seeded `skin_temperature` (a
`Center, Center, Nothing` surface field) from `ground_temperature`, which is a *window*
(`Center, Center, Center`) of the soil temperature field. `set!` dispatches on that mismatch to
Oceananigans' interpolating fallback, which detours through the CPU and calls `pointer` on a
`ConcretePJRTArray`:

```
ERROR: conversion to pointer not defined for ConcretePJRTArray{Float32, 3, 1}
```

**Status: fixed upstream, unreleased**, on Oceananigans branch `glw/copyable-fields-predicate`. The
root cause was an asymmetric `broadcastable_dimension` check in
`OceananigansReactantExt.Fields`, unrelated to blocker #5, that misclassified this
location-`Nothing`-vs-windowed-`Center` pair as non-broadcastable. With the branch pinned via
`test/reactant/Project.toml`'s `[sources]`, plain `set!` works with no Terrarium-side workaround; the
earlier interior-broadcast workaround was tried, reverted, and finally found unnecessary once the
branch was actually exercised (see the revision log and
`2026-08-14-oceananigans_broadcastable_dimension_issue.md` for the full story, including why the branch
initially *appeared* not to fix it).

### 4. Solvers writing a `Field` with a 2D index

Oceananigans defines `setindex!` on a `Field` for *exactly three* indices. A 2D write falls through to
`Base`'s generic `AbstractArray` path (`_to_subscript_indices` → `axes(::AbstractField)` →
`size(::AbstractGrid, loc, indices)`), which is a dynamic dispatch on the device:

```
InvalidIRError: compiling ... gpu_solve_surface_energy_balance_kernel!
Reason: unsupported call to an unknown function (call to jl_f_throw_methoderror)
```

Both `RootSolver` and `FixedPointSolver` index their target field with the objective's index tuple,
which is `(i, j)` for the `XY` skin-temperature solve. This is AGENTS.md pitfall #10; note that 2D
*reads* happen to work, because `Field` has a catch-all `getindex(f, inds...)` that forwards to the
underlying array — there is no matching `setindex!`.

**Fix**: a `pad_indices(field, indices)` helper in `src/utils/kernel_utils.jl` expands the index tuple
to `ndims(field)` with trailing ones; both solvers use it for every target-field access. It compiles to
nothing — the generated IR is instruction-for-instruction identical to a hand-written `field[i, j, 1]`.

### 5. A lazy `AbstractOperation` as a boundary condition

`LandModel` built the water boundary condition as `InfiltrationFlux(-infiltration)`. `-field` does not
compute anything; it builds a lazy `UnaryOperation`. Boundary conditions are evaluated inside kernels
via `getbc`, which indexes the condition with two indices:

```julia
@inline getbc(bc::BC{<:Any, <:AbstractArray}, i::Integer, j::Integer, args...) = @inbounds bc.condition[i, j]
```

As in #4, that works for a `Field` (catch-all `getindex`) but not for an `AbstractOperation`, which
only defines the 3-index form:

```
InvalidIRError: compiling ... gpu__compute_z_bcs!
Reason: unsupported call to an unknown function (call to jl_f_throw_methoderror)
 [2] axes  @ Oceananigans/src/Fields/abstract_field.jl:62
 [7] getbc @ Oceananigans/src/BoundaryConditions/boundary_condition.jl:182
```

**Status: fixed upstream, released in Oceananigans 0.110.15.** `getbc` now indexes array-valued
conditions in 3D:

```julia
@inline getbc(condition::AbstractArray, i::Integer, j::Integer, grid::AbstractGrid, args...) = @inbounds condition[i, j, 1]
```

which is exactly the fix suggested in `2026-08-04-OCEANANIGANS_getbc_2d_indexing_issue.md`. An
`AbstractOperation` defines the 3-index `getindex`, so the lazy `InfiltrationFlux(-infiltration)`
condition now evaluates inside the kernel without falling back to `axes(::AbstractField)`.

No Terrarium change is required. A workaround had been implemented and verified (materialize the
negation into a `soil_water_flux` coupling field declared by `interface_variables(::LandModel)`
alongside `soil_heat_flux`, refreshed at the end of each `compute_auxiliary!`) but was reverted in
favour of the upstream fix, which avoids an extra field plus refresh step and removes the same trap
for every other operation-valued BC.

### 6. Data-dependent control flow in kernels

Two remaining constructs could not be raised to StableHLO:

- **Masked stores.** `compute_hydraulics!` (`soil_hydrology_rre.jl`) used `if k <= 1 / elseif k >= Nz /
  else` with a store in each branch, plus a second store to face `k + 1` in one branch only:
  `error: masked affine.store is dependent on less dimensions than masked stored value`. Rewritten to
  select the *indices* and issue two unconditional stores (AGENTS.md: `ifelse`, never `if`/`else`, in
  kernels).
- **Convergence-tested loops.** The default skin-temperature solver is `RootSolver`, wrapping
  RootSolvers.jl's Newton iteration. Its `for i in 1:maxiters` loop lowers to an `scf.while` that the
  raise pass rejects: `error: cannot raise op to stablehlo`, located at `RootSolvers.jl:1484` (the `end`
  of that loop) via `solve_skin_temperature!` → `gpu_solve_surface_energy_balance_kernel!`.

**Fix**: the masked stores are fixed in place. For the solver, a new `NewtonSolver` performs a *fixed*
number of iterations carried in the type, so the loop unrolls into straight-line code. The Reactant test
configuration selects it explicitly; **the default is unchanged** (see Known limitations).

### Can RootSolvers.jl be used instead, with a never-satisfied tolerance?

No — tested, twice. `RootSolver(NF; tolerance = RootSolvers.ResidualTolerance(0), max_iterations = 5)`
makes the convergence test `(abs(y) < 0) | (abs(y) < eps(y))` false except at an exact root, so the
iteration always runs its full budget. It still fails to raise. There are two independent causes:

1. **The trip count is a runtime value.** `max_iterations` is an `Int` *field* of `RootSolver`, and the
   solver reaches the kernel as an argument, so the value is data rather than a constant — there is no
   literal for constant propagation to fold. (RootSolvers' `maxiters::Int` positional argument is not
   itself the problem: a literal at the call site would constant-propagate into the `@inline`d
   `_find_zero_newton` normally.)
2. **Making it constant is not sufficient.** Hard-coding the literal `5` at the `find_zero` call site —
   strictly stronger than lifting `max_iterations` to a type parameter — still produces an `scf.while`
   and still fails. The `if !isfinite(y_new); return …; end` line-search abort inside the loop is a
   data-dependent exit, so the loop never becomes a counted `scf.for`; and LLVM will not unroll it
   either, since the body carries the nested Armijo backtracking `while` plus two full evaluations of
   the surface-energy-balance residual.

Making RootSolvers usable here would mean restructuring `_find_zero_newton` upstream: replacing the
in-loop `return`s with a carried `done` flag so the loop runs to completion with a genuinely static trip
count. That is the same shape as `NewtonSolver`, which does it in ~15 lines with no early exits.

## Summary of changes

| File | Change |
| --- | --- |
| `src/processes/soil/hydrology/soil_hydraulic_closures.jl` | Drop in-kernel `@assert`; clamp the inverse-SWRC argument to `[sat_res, por]` |
| `src/processes/soil/hydrology/soil_hydraulic_properties.jl` | Replace `complex` arithmetic in the van Genuchten conductivity with a clamped real form |
| `src/processes/soil/hydrology/soil_hydrology_rre.jl` | `compute_hydraulics!`: branchless index selection, unconditional stores |
| `src/processes/surface/skin_temperature.jl` | `initialize!`: copy interiors instead of `set!` across a location mismatch |
| `src/utils/utils.jl` | `field_indices` helper (replaces the earlier `pad_indices`) |
| `src/utils/kernel_utils.jl` | New `@assert_kernel` macro and its `uses_reactant` guard |
| `src/solvers/root_solvers.jl`, `src/solvers/fixed_point.jl`, `src/solvers/newton.jl` | Index the target field in 3D via `field_indices`, for reads as well as writes |
| `src/solvers/newton.jl` (new) | `NewtonSolver`: fixed, type-level iteration count |
| `src/solvers/solvers.jl` | Include the new solver |
| `src/Terrarium.jl` | Export `@assert_kernel` |
| `ext/TerrariumReactantExt/TerrariumReactantExt.jl` | `uses_reactant(::ReactantMarker) = true` |
| `Project.toml`, `test/reactant/Project.toml` | Raise the Oceananigans compat bound to `0.110.15` for the upstream `getbc` fix |
| `test/reactant/setup.jl`, `test/reactant/runtests.jl` | New `:land_soil_snow` configuration |
| `test/solvers/test_solvers.jl` | `NewtonSolver` tests |
| `docs/dev/2026-08/2026-08-04-OCEANANIGANS_getbc_2d_indexing_issue.md` (+ MWE) | Write-up of blocker #5 for upstream |

`src/models/coupled/land_model.jl` is deliberately **unchanged**: the `soil_water_flux` workaround for
blocker #5 was reverted (see the revision log).

## Testing and verification

- `test/reactant/runtests.jl` gains `:land_soil_snow`: a single column on an `ExponentialSpacing` grid,
  Richards hydrology with a van Genuchten SWRC and `UnsatKVanGenuchten` conductivity, `SingleLayerSnow`,
  no vegetation, under sub-freezing prescribed air temperature. CPU and Reactant are stepped 100 times
  and every prognostic and auxiliary field is compared.
- The pack is held below freezing deliberately: the comparison should not hinge on the exact step at
  which a melt threshold is crossed, which CPU and XLA need not agree on.
- **Result against released Oceananigans 0.110.15, with no Terrarium-side workaround: all 62 fields
  match** (31 initial + 31 stepped). The worst deviations after 100 steps are `max_rel = 9.1e-6`
  (`temperature`), `8.5e-6` (`snow_temperature`) and `6.7e-6` (`soil_heat_flux`), all well inside the
  suite's `rtol = 1e-3`; the initial state is bit-identical apart from `pressure_head` (`1.1e-7`).
  `saturation_water_ice`, `skin_temperature`, `infiltration`, `surface_runoff` and the water-table
  diagnostics are bit-identical even after stepping.
- The whole Reactant suite passes: `Terrarium CPU vs Reactant` 115/115 (8m45s) and
  `Reactant + Enzyme autodiff` 6/6. Note the coupled configuration is slow to compile — a full run
  takes roughly ten minutes.
- An earlier run *with* the (now unnecessary) blocker-#5 workaround gave 59/59 fields matching, with
  worst deviations of `1.7e-5`; the upstream fix reproduces that agreement without the extra field.
- Substituting `NewtonSolver` for the default `RootSolver` changes the CPU answer by at most `6e-7`
  relative (`skin_temperature`) and `8e-6` (`snow_temperature`) after 100 steps; soil temperature,
  saturation and the latent heat flux are bit-identical.
- The full CPU suite passes (`Pkg.test()`, exit 0) after the `field_indices` repair described in the
  revision log. Before that repair the suite had 20 errors, all from the two broken solver call sites.

### Assertions in kernels: `@assert_kernel`

Blocker #1 was resolved by *deleting* an in-kernel `@assert`, which is the correct fix there but leaves
no way to keep a cheap correctness guard in kernel code that also has to compile under Reactant.
`@assert_kernel` fills that gap: it expands to `Base.@assert` normally and to nothing once Reactant is
loaded, so the guard is active on the CPU/GPU backends and its `llvm.intr.trap` never reaches the raise
pass.

The guard is deliberately **load-time** (`uses_reactant()`, overridden by `TerrariumReactantExt`) rather
than trace-time. Two more obvious designs were tried and both fail *inside kernels*:

- `ReactantCore.within_compile()` is constant-folded to `true` only by Reactant's `EnzymeInterpreter`,
  which traces host code. KernelAbstractions kernel bodies are compiled by GPUCompiler, where it folds
  to `false` and the assertion survives — reproduced as
  `error: cannot raise op to stablehlo"llvm.intr.trap"()`.
- Reactant's task-local `raising()` flag *is* set during kernel compilation, but it is a runtime lookup
  and is unreachable from device code:
  `InvalidIRError: unsupported call to an unknown function (call to julia.get_pgcstack)`.

The tradeoff is granularity: loading Reactant disables these assertions process-wide, including in CPU
code that is never traced.

Implementation note: `uses_reactant` dispatches on a `ReactantMarker` singleton rather than being
zero-argument, because an extension may not *overwrite* a method —
`ERROR: Method overwriting is not permitted during Module precompilation`. Dispatching on a marker type
makes the extension's definition a new method instead.

## Known limitations

- **The default `LandModel` would still not compile.** `ImplicitSkinTemperature` defaults
  to `RootSolver`, whose convergence-tested loop is blocker #6. The Reactant configuration overrides it
  with `NewtonSolver`; users of the default configuration hit the raise failure. Making `NewtonSolver`
  the default is recommended (see Future work) but changes results for existing users and so is left for
  review.
- **Vegetation is still unsupported** — see the separate root-fraction `exp(::TracedRArray)` blocker.

## Unrelated issues noticed along the way

- **The default coupled `LandModel` has no snow-aware albedo.** `default_surface_energy_balance` never
  receives the `snow` component and always selects `ConstantAlbedo` (α = 0.3), so a snow-covered column
  absorbs ~70% of the incoming shortwave. With the default forcing (a constant 341 W/m² downwelling
  shortwave, no diurnal cycle) the CPU run settles at a skin temperature of **38 °C over a 0.2 m
  snowpack with air at −2 °C**. `DiagnosticAlbedo` exists for exactly this case (snow-cover-weighted
  blend, α_snow = 0.8) but is not wired into the defaults.
- **The skin temperature is not capped at the melting point over snow**, so nothing bounds the above.

## Future work

- ~~Fix `getbc` upstream in Oceananigans, then re-run `:land_soil_snow` against the released
  version.~~ — done; released in 0.110.15 and reconfirmed (all 62 fields match, no workaround).
- Make `NewtonSolver` the default skin-temperature solver, after checking the effect on the SEB
  regression tests. `src/solvers/newton.jl` carries a TODO noting the longer-term goal of letting
  Reactant use the regular solvers instead, which needs the upstream `_find_zero_newton` restructuring
  described above.
- Adopt `@assert_kernel` at kernel call sites that would benefit from a correctness guard, starting with
  the `isfinite(ψm)` check dropped in blocker #1.
- Add a `LandModel` configuration to `test/reactant/autodiff.jl` once the forward path is verified.
- Select a snow-aware albedo in `default_surface_energy_balance` when a snow component is present.
