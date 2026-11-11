# Upstream issue draft: Oceananigans + Reactant — `RectilinearGrid` with array/range vertical coordinates fails kernel-launch tracing

> **RESOLVED upstream — released in Oceananigans 0.110.9.** Fixed by adapting the grid extents in
> the `RectilinearGrid` `Adapt.adapt_structure` rule: `src/Grids/rectilinear_grid.jl` now emits
> `Adapt.adapt(to, grid.Lx)` (and `Ly`/`Lz`) instead of passing them through unadapted — exactly the
> surviving-traced-scalar candidate flagged in the diagnosis below. With the fix, a `RectilinearGrid`
> with array-valued (`Vector`) or range vertical coordinates traces through kernel launches and
> time-steps correctly under Reactant (verified with the Case A/B reproduction below on Reactant
> 0.2.270). The registry-tagged 0.110.8 does not have the fix; 0.110.9 does.
>
> **Terrarium impact:** the uniform-only workaround (endpoint-tuple `z` for `UniformSpacing`) was
> removed — `z_coordinates` is gone and both grid constructors are back to the plain `Vector` form
> (identical to `main`). All `AbstractVerticalSpacing`s (incl. `ExponentialSpacing`,
> `PrescribedSpacing`) now trace. A `:soil_heat_column_stretched` (`ExponentialSpacing`) config was
> added to `test/reactant/` and passes CPU-vs-Reactant. The `test/reactant` env now just requires
> `Oceananigans = "0.110.9"` in `[compat]` (the interim `[sources]` pin to `main` was dropped once
> 0.110.9 was released).
>
> The original issue text and diagnosis are retained below for reference.

**Where to file:** Oceananigans.jl. Working notes for an independent fix session are in the
appendix at the bottom (drop that section from the actual issue text).

## Summary

Under Reactant, a `RectilinearGrid` traces through kernel launches **only when its vertical
coordinate is given as endpoint tuple** (`z = (zmin, zmax)`). Both an explicit coordinate
`Vector` *and* an `AbstractRange` input are materialized into device arrays at construction;
any kernel launch that receives such a grid as an argument (e.g. the z-halo fill inside
`update_state!`/`time_step!`) then fails during compilation in Reactant's kernel-argument
check. This blocks all stretched-vertical-grid applications (e.g. exponentially spaced layers)
under Reactant, while uniform grids work when — and only when — constructed from endpoint
tuples.

Reproduced with bare Oceananigans (no downstream packages), CPU backend.

## Environment

- Julia 1.12.6 (aarch64, macOS 14), CPU backend (`Reactant.set_default_backend("cpu")`)
- Oceananigans **0.110.7** (latest registry version), Reactant **0.2.270**,
  KernelAbstractions 0.9.42, CUDA.jl loaded (required by Reactant's KA integration even on CPU)

## Reproduction

### Step 1 — construction-level divergence (fast, no compilation)

The three input forms for the same uniform grid diverge in how face coordinates are stored:

```julia
using Oceananigans, Reactant, CUDA
using Oceananigans.Architectures: ReactantState
arch = ReactantState()

for z in [(-2.0f0, 0.0f0),                          # tuple
          range(-2.0f0, 0.0f0, length = 11),        # range
          collect(range(-2.0f0, 0.0f0, length = 11))] # vector
    g = RectilinearGrid(arch, Float32; size = (1, 10), x = (0, 1), z,
                        topology = (Periodic, Flat, Bounded))
    @show typeof(parent(g.z.cᵃᵃᶠ))
end
```

Output (verified):

| `z` input | face coordinate storage |
|---|---|
| tuple `(-2, 0)` | `StepRangeLen{Float32, Float64, Float64, Int64}` |
| `range(-2, 0; length = 11)` | `ConcretePJRTArray{Float32, 1, 1}` |
| `collect(range(…))` | `ConcretePJRTArray{Float32, 1, 1}` |

### Step 2 — compilation failure for the array-backed cases

```julia
using Reactant: @trace

function run4!(model, Δt, Nt)
    @trace track_numbers = false for _ in 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

# A) tuple z — COMPILES AND RUNS
gA = RectilinearGrid(arch; size = 16, z = (-200, 0), topology = (Flat, Flat, Bounded))
mA = HydrostaticFreeSurfaceModel(gA; buoyancy = nothing, tracers = :T)
rA = @compile raise=true raise_first=true sync=true run4!(mA, 60.0, 4)
rA(mA, 60.0, 4)   # OK, clock advances

# B) vector z — FAILS during compilation (same for range z)
zv = collect(range(-200.0f0, 0.0f0, length = 17))
gB = RectilinearGrid(arch, Float32; size = 16, z = zv, topology = (Flat, Flat, Bounded))
mB = HydrostaticFreeSurfaceModel(gB; buoyancy = nothing, tracers = :T)
rB = @compile raise=true raise_first=true sync=true run4!(mB, 60.0, 4)   # error below
```

The same failure occurs with a plain `@compile time_step!(model, Δt)` (no `raise`, no `@trace`).

## Observed error (case B)

```
GPU kernel argument of type RectilinearGrid{TracedRNumber{Float32}, Flat, Flat, Bounded,
  StaticVerticalDiscretization{OffsetVector{TracedRNumber{Float32}, TracedRArray{Float32, 1}}, …}, …}

All TracedRNumber/TracedRArray must have been replaced by their
CuTracedRNumber/CuTracedArray counterparts. A surviving traced value means
some struct in the hierarchy is missing `Adapt.@adapt_structure`, so its fields
were not recursed into during GPU adaptation.
```

raised from `ReactantCUDAExt._check_no_traced_in_kernel_arg` (Reactant
`ext/ReactantCUDAExt.jl:1007`, via `LLVMFunc` `:1022/:1040` and `ka_with_reactant` `:577`),
while launching `Oceananigans.BoundaryConditions.gpu__fill_bottom_and_top_halo!` from
`fill_halo_regions!`.

## Expected

Stretched/array vertical coordinates should trace through kernel launches like the tuple form
does (or, at minimum, `AbstractRange` inputs should be preserved as ranges rather than
materialized, which would make regularly-spaced-but-explicit inputs work).

## Diagnosis notes (partial — pointers, not a confirmed root cause)

- The error's "missing `Adapt.@adapt_structure`" hint appears misleading: adapt rules exist for
  both `RectilinearGrid` (`src/Grids/rectilinear_grid.jl:370`) and
  `StaticVerticalDiscretization` (`src/Grids/vertical_discretization.jl:163`).
- Note that the `RectilinearGrid` adapt rule passes the scalar fields `Lx, Ly, Lz` (and `Nx…Hz`)
  through **unadapted**. Once any coordinate field is a traced array, the grid's `FT` parameter
  is promoted to `TracedRNumber{Float32}` (visible in the error), so those scalar fields are
  traced numbers that survive kernel-argument adaptation. This is one candidate for the
  surviving traced values; the coordinate `OffsetVector{TracedRNumber, TracedRArray}`s are the
  other.
- Contrast with `LatitudeLongitudeGrid`, which works under Reactant with traced *metric* arrays
  (e.g. the `Gu kernel` test in `test/test_reactant.jl` passes): the Reactant extension gives
  LLG a dedicated construction/transfer path (`_to_reactant` in
  `ext/OceananigansReactantExt/Architectures.jl`) plus `traced_type_inner`/`make_tracer` rules.
  `RectilinearGrid` has `traced_type_inner`/`constructorof` rules
  (`ext/OceananigansReactantExt/OceananigansReactantExt.jl`) but no equivalent dedicated
  transfer path; a source comment there notes RG "retains StepRangeLen coordinates which
  sidestep the CuTracedArray VX type constraint" — i.e. the array-coordinate case is known to
  be unsupported.
- The existing Reactant test suite (`test/test_reactant*.jl`) constructs every RectilinearGrid
  from tuples/`extent`, so this case is currently uncovered.
- Version window checked: with Reactant 0.2.243 the same reproduction **segfaults** during
  compilation instead of erroring; 0.2.270 gives the clean error above.

---

## Appendix: working notes for the fix session (not part of the issue)

- **Env recipe:** any project with Oceananigans 0.110.7 + Reactant 0.2.270 + CUDA; on CPU-only
  machines everything works (XLA CPU backend), but `using CUDA` is mandatory — without it kernel
  launches fail earlier with `MethodError: ka_with_reactant(::Nothing, …)`.
  Do **not** run with `--check-bounds=yes`: forced bounds checks make every kernel un-raisable
  (throw paths lower to `llvm.intr.trap`, which cannot be raised to StableHLO).
- **Iteration costs:** Step-1 probe is seconds; each `@compile` attempt is ~30–60 s.
- **Suggested red test:** add a stretched-z variant (`z = collect(…)` or an actual exponential
  spacing) to `test/test_reactant_single_column_models.jl` — the uniform-tuple cases there pass
  today, so a single new grid argument makes the gap CI-visible.
- **Places to look, in order:**
  1. Why range inputs materialize: the grid `generate_coordinate`/validation path in
     `src/Grids/` (tuple → `StepRangeLen`, everything else → array + `on_architecture`).
  2. Kernel-arg adaptation of the traced grid: `ReactantCUDAExt` adaptor rules for
     `TracedRArray`/`TracedRNumber` and how they recurse through
     `RectilinearGrid`/`StaticVerticalDiscretization`/`OffsetArray`; check whether the
     unadapted scalar fields (`Lx…`) or the coordinate arrays are what survives
     (`_check_no_traced_in_kernel_arg` could be extended to print the offending field path).
  3. The LLG-vs-RG asymmetry in `OceananigansReactantExt` (dedicated `on_architecture` +
     `_to_reactant` for LLG only) — porting that treatment to `RectilinearGrid`'s `z` may be
     the whole fix, mirroring `traced_type_inner`'s existing `CZ2` eltype promotion.
- **Downstream context (Terrarium.jl):** works around this by emitting endpoint tuples for
  uniform vertical spacings (`z_coordinates` in `src/grids/vertical_discretization.jl`);
  stretched spacings (`ExponentialSpacing`, `PrescribedSpacing`) remain blocked on this issue.
  Correctness suite to validate against after an upstream fix:
  `julia --project=test/reactant test/reactant/runtests.jl` (add an `ExponentialSpacing`
  config to `test/reactant/setup.jl`).
