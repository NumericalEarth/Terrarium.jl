# Investigating the performance gap between native and Reactant execution

> Status: **planned**. Initial investigation is complete (desk review of the benchmark results and source only, nothing executed); diagnostics and fixes await sign-off.

Date of initial draft: 2026-10-17

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code 2.1.269

Reviewer:

Base revision: `0577a277cd7a4f32c36c01db2f745bcf4af02716` (tip of local `main`, branch `bg/reactant-speedup-investigation`)

## Originating prompt

> Check out a new worktree from main and review the latest benchmark results. Investigate why Terrarium appears to be so much faster with Reactant on both CPU and GPU. The speedup right now seems way higher than it should be and may imply that there are some inefficiencies in Terrarium's kernel launching patterns. As a first step, do not run anything, just formulate some ideas as to what might be going on.

> Draft an implementation plan summarizing the initial investigation and next steps.

## Revision log

- **Rev 1 (2026-10-17, Claude Opus 5.5):** Initial draft from a desk review of `benchmark/README.md`, the benchmark harness, and the stepping path in `src/`.
  No code was executed.

## Problem description

The committed benchmark results (`benchmark/README.md`, collected 2026-10-07 to 2026-10-12) show Reactant running one to two orders of magnitude faster than native execution on the same hardware.

| Configuration | Columns | Native CPU x86 (ms/step) | Reactant CPU x86 (ms/step) | Ratio | Native GPU (ms/step) | Reactant GPU (ms/step) | Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `land` | 128 | 6.39 | 0.098 | 65× | 2.01 | 0.0344 | 58× |
| `land` | 165888 | 6220 | 69.8 | 89× | 92 | 1.13 | 81× |
| `soil_heat` | 128 | 2.73 | 0.0103 | 265× | 0.096 | 0.0105 | 9× |
| `soil_heat` | 165888 | 3380 | 19.1 | 177× | 5.77 | 0.305 | 19× |

XLA is a strong compiler, but a speedup of this size on a 1D column model points to avoidable costs on the native path rather than to anything Reactant does uniquely well.
Native throughput is also low in absolute terms.
Single-threaded `soil_heat` costs roughly 700 ns per cell per time step on the CPU, and native GPU `land` costs about 18 ns per cell per time step at 5 million cells.
Both are far from what a vertical diffusion stencil with pointwise closures should cost.

The goal of this plan is to decompose the gap into its causes with targeted measurements, to correct the parts that are artifacts of the benchmark setup, and then to propose concrete fixes for the native path.

## Background

### How the two paths execute a time step

On the native path, `run_timesteps!` (`src/timesteppers/model_integrator.jl`) is a host loop over `timestep!`.
Every step calls `update_state!` (inputs, `compute_auxiliary!`, `compute_boundary_conditions!`, and `compute_tendencies!`), then one `explicit_step!` launch per prognostic variable (`src/timesteppers/abstract_timestepper.jl`), then `closure!`.
Each process issues its own `launch!` through `launch_kernel!` (`src/grids/grid_utils.jl`), which builds `get_fields`/`tendency_fields` NamedTuples on the host and passes them to Oceananigans' `launch!`.

On the Reactant path, `TerrariumReactantExt.run_timesteps!` traces the same Julia code once inside `Reactant.@trace for`, which yields a single `stablehlo.while` program that XLA then optimizes as a whole.
All host-side Julia work (dispatch, NamedTuple construction, kernel configuration, and argument conversion) happens once at trace time and costs nothing at run time.
XLA can additionally constant-fold anything that depends only on model parameters, fuse producer and consumer kernels, and eliminate loop-carried values that are never read.

### Prior work

The fused-kernels plan (`docs/dev/2026-08/2026-08-31-PLAN_fused_kernels.md`, merged) already reduced the number of auxiliary launches for `SurfaceHydrology` and `VegetationCarbonCycle`.
Its revision 6 found that fusing *tendency* kernels raises register pressure and regresses GPU occupancy, so that route is closed.
The benchmarks reviewed here postdate that work, so the remaining gap is not explained by the auxiliary launch count alone.

### Fixed and per-column costs from the existing data

A linear fit of time per step against column count, using the 128- and 512-column points, separates a fixed per-step cost from a per-column cost.

| Configuration | Fixed cost, CPU x86 | Fixed cost, CPU arm | Fixed cost, GPU | Per-column cost, native CPU x86 | Per-column cost, Reactant CPU x86 |
| --- | --- | --- | --- | --- | --- |
| `land` | ≈1.7 ms | ≈1.8 ms | ≈2.0 ms (flat from 128 to 512 columns) | ≈37 µs | ≈0.75 µs |
| `land_no_vegetation` | ≈0.9 ms | n/a | ≈1.1 to 1.4 ms | ≈34 µs | n/a |
| `soil_heat` | ≈0.14 ms | n/a | ≈0.1 ms | ≈20 µs | n/a |

The fixed cost is nearly the same on CPU and GPU and scales with model complexity.
It therefore cannot be CUDA launch latency, which would not appear on the CPU.
It is host-side Julia work done every step, and it dominates native GPU runs below roughly 10,000 columns.
The per-column cost is a separate problem, and it dominates at large resolutions.

## Hypotheses

Each hypothesis lists the evidence from the desk review and the measurement that would confirm or refute it.

### H1. Per-step host overhead in kernel orchestration

The land model issues roughly 25 `launch!` calls per step from process code, plus one `explicit_step!` launch per prognostic variable, plus halo fills and `compute_z_bcs!` calls.
Each launch rebuilds field NamedTuples, and on the GPU each launch also converts large nested argument types for the device (`cudaconvert`) and looks up the compiled kernel by those types.
If any of `get_fields`, `fastiterate`, or the closures they take is not fully inferred, every step also pays for dynamic dispatch and heap allocations.
About 2 ms spread over about 50 launches is roughly 40 µs per launch, several times the raw CUDA launch latency.
*Test:* D1 and D2.

### H2. Parameter-only transcendental math evaluated per cell

`compute_thermal_conductivity` (`src/processes/soil/energy/soil_thermal_properties.jl`) evaluates `κ_quartz^q * κ_mineral^(1 - q)` and five `x^(1/2)` terms in `InverseQuadratic`, and every one of them depends only on parameters.
Under Reactant, these parameters are trace-time constants, so XLA folds the whole expression away.
Natively, they run on every evaluation, and Julia's `^(::Float32, ::Float32)` computes through Float64.
The `x^(one(x) / 2)` form was chosen over `sqrt` for Reactant kernel compatibility, which makes the native path slower still.
The `soil_heat` gap (177× to 265× on CPU) is larger than the `land` gap, which fits a configuration dominated by this computation.
*Test:* D3 and D5.

### H3. Redundant stencil evaluation

`diffusive_heat_flux` (`src/processes/thermodynamics/heat_conduction.jl`) interpolates κ to each cell face with `ℑzᵃᵃᶠ`, so the tendency kernel evaluates the full conductivity closure four times per cell (two faces, two cells per face).
Together with H2, this is about 28 calls to `^` per cell per step, enough to explain most of the 700 ns per cell on the CPU.
The same pattern (closures re-derived inside stencil operators, and `SoilComposition` rebuilt by every process) probably recurs elsewhere in soil hydrology.
*Test:* D3.

### H4. Missing fusion and dead diagnostic work

Natively, every auxiliary field makes a full round trip through global memory between kernels.
XLA fuses chains of elementwise kernels and keeps intermediates in registers.
XLA can also drop auxiliaries that are overwritten every iteration without being read, such as diagnostic fluxes, whereas the native path computes and stores them on every step.
This is a legitimate optimization, but it means part of the measured gap is work the native path does not need to do.
*Test:* D5.

### H5. Benchmark configurations that are not equivalent

`land_architecture_kwargs` (`benchmark/model_configurations.jl`) gives Reactant runs a fixed five-iteration `NewtonSolver` for skin temperature, while native runs use the tolerance-driven `RootSolver`.
The root solver can iterate many more times per column and causes thread divergence on the GPU.
*Test:* D4.

### H6. XLA's CPU backend is multithreaded

The "1 thread(s)" label in the README refers to Julia threads only.
XLA's CPU runtime may parallelize fused loops across cores.
Similar Reactant throughput on a 4-performance-core Apple M3 and a 128-core AMD EPYC suggests that threading contributes only a modest factor, but this should be pinned down.
*Test:* D4.

### H7. Float64 promotion inside Float32 kernels

`src/processes/vegetation/dynamics/carbon_dynamics.jl` divides by the Float64 literal `365.25 * 24 * 3600` inside kernel code.
The benchmark GPU runs `land` roughly six times slower in Float64 than in Float32, so stray Float64 arithmetic is expensive there.
Under Reactant, the same expression is parameter-only and gets folded (see H2).
*Test:* D6.

### H8. Noisy or state-dependent data points

Several data points do not fit any cost model and should not be over-interpreted until they are reproduced:

- Native GPU `soil_heat` at 4608 columns (0.04 ms/step) is faster than at 128 columns (0.096 ms/step).
- Native GPU time is nearly flat from about 41,000 to 166,000 columns for both `land` and `soil_heat`.
- Native CPU `land` at 4608 columns with 30 layers in Float32 is about four times slower than its neighbors and slower than Float64, on both CPUs.
  This suggests subnormal arithmetic or state-dependent solver iteration counts.

Each run is timed once, so run-to-run variance is unknown.
*Test:* D6 and D7.

## Summary of changes

The work is split into three phases.
Phases 0 and 1 change only the benchmark harness and add diagnostic scripts; they do not change model code or numerics.
Phase 2 consists of model-code fixes chosen from the Phase 1 findings, and each will be added to this plan as a revision for separate approval before implementation.

### Phase 0: benchmark comparability (`benchmark/` only)

1. Repeat each timed run (proposed default: five repetitions) and report the median, keeping the minimum and spread in the JSON store.
   For Reactant, this means calling the compiled program repeatedly, which costs no recompilation.
2. Make solver choice explicit and identical across architectures: use `NewtonSolver(NF; iterations = 5)` for the land configurations on every architecture, or add a suite that runs both solvers natively.
   The reviewer should choose (see Open questions).
3. Record the effective XLA CPU thread count in the machine details, and support a Reactant CPU run restricted to one core, so that single-thread numbers can be compared like for like.
4. Fix the misleading "1 thread(s)" wording for Reactant sections.

### Phase 1: diagnostics (scripts under `benchmark/profiling/`, not part of the regular suite)

- **D1. Host overhead on CPU.**
  At 128 columns, measure `@btime` and `@allocated` for a single `timestep!` on `land`, `land_no_vegetation`, and `soil_heat`.
  Run `@code_warntype` and JET's `@report_opt` on `update_state!(state, model, inputs)` and `explicit_step!` to find runtime dispatch.
  Count `launch!` calls per step.
- **D2. Host and device split on GPU.**
  Run `CUDA.@profile trace = true` over one step at 128 and 165,888 columns.
  Record the number of launches, host time per launch, device time per kernel, and the top kernels by device time.
  For those kernels, record register count and local-memory (spill) usage.
- **D3. Per-cell cost of the soil heat tendency.**
  In a microbenchmark of the `soil_heat` tendency kernel on CPU and GPU, compare ns per cell for: the baseline; mineral and constituent conductivities precomputed on the host; `sqrt` instead of `x^(1/2)`; and κ evaluated once per cell.
  The variants live only in the profiling script, not in `src/`.
- **D4. Fairness controls.**
  Time native `land` with the Newton solver, and Reactant CPU pinned to one core (`taskset` plus the XLA intra-op thread setting; flag names to be checked against the installed XLA version).
- **D5. What XLA does to the program.**
  Dump the optimized HLO for `soil_heat` and `land` (`XLA_FLAGS=--xla_dump_to=...`).
  Count fusions per loop iteration, check whether the conductivity powers are constant-folded, and check whether any auxiliary fields are dropped from the loop.
- **D6. Precision and the Float32 anomaly.**
  Audit `src/` kernels for Float64 literals and inspect `@code_llvm` of the hot Float32 kernels for `double` operations.
  Re-run the 4608-column Float32 `land` case with `set_zero_subnormals(true)` and with logged root-solver iteration counts.
- **D7. Variance.**
  Repeat the anomalous points from H8 with the Phase 0 harness.

The deliverable of Phase 1 is a short findings section added to this plan as Rev 2, apportioning the gap at small and large resolutions among H1 to H8.

### Phase 2: fixes (each requires its own approved revision)

Candidate fixes, in the expected order of payoff:

1. **Hoist parameter-only computations out of kernels (H2).**
   Compute derived constants such as the bulk mineral conductivity and square-rooted constituent conductivities on the host immediately before each launch, as a plain function of the process parameters, and pass the result as an `isbits` kernel argument.
   Computing them per launch rather than storing them in the struct keeps them differentiable with respect to parameters and compatible with `reconstruct`.
2. **Evaluate closures once per cell (H3).**
   Either compute κ at k−1, k, and k+1 once per thread, or store thermal conductivity as an auxiliary field computed once per step.
   The second option trades redundant arithmetic for extra memory traffic and needs to be measured against the first.
3. **Remove host overhead (H1).**
   Fix any type instability or allocation that D1 finds.
   Merge the per-variable `explicit_step!` launches into one launch per variable shape (XYZ, XY, and slab).
   Explore caching the configured kernels and their converted arguments across steps.
4. **Eliminate Float64 promotion (H7)** wherever D6 finds it.
5. **Skip diagnostic-only auxiliaries inside the step loop (H4)**, if D5 shows a material share of the work is dead.
   This changes when diagnostics are up to date and needs a design discussion before it is planned in detail.

## Testing and verification

- Phases 0 and 1 do not change model code; the existing test suite must still pass, and the draft doc build must succeed.
- Each Phase 2 fix must reproduce the pre-change results within floating-point tolerance on the existing soil, snow, and land model tests.
  Fixes that reorder arithmetic (for example `sqrt` versus `^(1/2)`, or hoisted constants) are expected to change results at the level of rounding only.
- Phase 2 fixes touch kernel code that Reactant and Enzyme compile, so the Reactant and Enzyme suites must pass in CI.
  Following the repository rules, they are run locally only if the extensions or their tests are edited.
- Each Phase 2 fix is a major revision of model compute functions, so the benchmarks are re-run on at least `cpu-x86` and `gpu-nvidia`, with the improvement reported against the current README baseline.

## Documentation changes

- `benchmark/README.md` and `docs/src/benchmarks.md` are regenerated from the updated harness, with a note on solver parity and XLA threading in the README preamble.
- Rev 2 of this plan records the Phase 1 findings.
- Any process whose kernel arguments change in Phase 2 has its docstrings and doc page updated to match.

## Known limitations

- The hypotheses come from a desk review only; none has been confirmed by measurement.
- The fixed-cost estimates rest on two data points per configuration from single, unrepeated runs.
- Reactant's constant folding and dead-code elimination are legitimate optimizations, so the native path is not expected to match Reactant exactly; the aim is to remove avoidable overhead, not to close the gap completely.
- Phase 2 fix 3 may be limited by how much host work Oceananigans' `launch!` and CUDA.jl's argument conversion do per call, which is outside Terrarium's control.

## Future work

- A generic mechanism for declaring parameter-derived constants once per process, so that new processes do not reintroduce H2.
- A regression check in the benchmark harness that flags fixed per-step costs above a threshold.
- Revisit CUDA graphs or other launch batching for the native GPU path, if host overhead remains significant after Phase 2.

## Open questions for the reviewer

1. For Phase 0 item 2, should all architectures use the Newton solver, or should native runs report both solvers?
2. Should the Phase 1 profiling scripts be committed under `benchmark/profiling/`, or kept out of the repository?
3. Is Phase 2 fix 5 (skipping diagnostic-only auxiliaries during stepping) acceptable in principle, or should diagnostics always be current after each step?
