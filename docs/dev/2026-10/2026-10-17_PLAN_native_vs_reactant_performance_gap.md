# Investigating the performance gap between native and Reactant execution

> Status: **in progress**. Phases 0 and 1 are complete; Phase 2 items 1, 5, and 6 are implemented on `bg/performance-fixes`, item 7 remains (items 2 to 4 are deferred), and the native CPU baseline and the full benchmark runs are pending Slurm.

Date of initial draft: 2026-10-17

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code 2.1.269

Reviewer: Brian Groenke (brian.groenke@pik-potsdam.de); Phase 0 approved in Rev 6, the plan as a whole in Rev 10

Base revision: `0577a277cd7a4f32c36c01db2f745bcf4af02716` (tip of local `main`, branch `bg/reactant-speedup-investigation`)

## Originating prompt

> Check out a new worktree from main and review the latest benchmark results. Investigate why Terrarium appears to be so much faster with Reactant on both CPU and GPU. The speedup right now seems way higher than it should be and may imply that there are some inefficiencies in Terrarium's kernel launching patterns. As a first step, do not run anything, just formulate some ideas as to what might be going on.

> Draft an implementation plan summarizing the initial investigation and next steps.

## Revision log

- **Rev 1 (2026-10-17, Claude Opus 5.5):** Initial draft from a desk review of `benchmark/README.md`, the benchmark harness, and the stepping path in `src/`.
  No code was executed.
- **Rev 2 (2026-10-17, Claude Opus 5.5):** Added the performance target set by the author: no parity goal, but Reactant should be at most about 10× faster than native execution.
- **Rev 3 (2026-10-17, Claude Opus 5.5):** Recorded the D1 findings and added hypothesis H9 (allocations in kernels and per-step host reductions).
  Added two Phase 2 items for H9 (closure-kernel allocations and the root-fraction reduction) and reordered Phase 2 by expected payoff.
  Made the H3 fix concrete: a cell-centered `thermal_conductivity` auxiliary, implemented on `main` rather than on `bg/implicit-timestepping`.
- **Rev 4 (2026-10-17, Claude Opus 5.5):** Ran D2 to D7 at the author's request and replaced the D1 findings with a full Phase 1 findings section, including a verdict for each hypothesis.
  Corrected two Rev 3 attributions: the kernel allocations come from despecialized pass-through varargs, not from type instability in the closure code, and the host-side reduction is the plant-available-water `Integral`, not the root-fraction normalization.
  Reordered Phase 2 around the measured payoffs, demoted the κ auxiliary to a CPU-only, conditional item, and dropped the H4 item.
  Recorded that Phase 0 was not implemented and added two Phase 0 items (compile order and a fresh native GPU baseline).
- **Rev 5 (2026-10-18, Claude Opus 5.5):** Recorded the author's answers to the open questions: the Newton solver on all architectures; the profiling scripts committed under `benchmark/profiling/`; one pull request covering all 28 varargs kernels; `sqrt` to be checked against Reactant first (done, see Phase 2 item 3); and the κ auxiliary deferred.
- **Rev 6 (2026-10-18, Claude Opus 5.5):** Phase 0 approved by the author and implemented (items 1 to 5); implementation notes are under Phase 0.
  Added a Slurm batch script for long benchmark runs, without cluster-specific settings.
  Checked whether `sqrt` still breaks Reactant (open question 4 of Rev 5): it does not, so Phase 2 item 3 uses `sqrt` on all architectures.
- **Rev 7 (2026-10-18, Claude Opus 5.5):** First Phase 0 GPU re-run on an H100.
  Found that the published native GPU baseline was dominated by a one-off compilation of about 1.8 s in the first timed run of every new grid size, and fixed the harness to absorb it in the warm-up (second attempt, after the first did not hold up on job 2554084).
- **Rev 8 (2026-10-18, Claude Opus 5.5):** Final Phase 0 GPU re-run with the corrected warm-up; Phase 0 complete.
- **Rev 9 (2026-10-18, Claude Opus 5.5):** Phase 2 items 2 (the soil moisture limiting factor in a kernel) and 3 (hoisting parameter-only computations, and `sqrt`) deferred by the author.
- **Rev 10 (2026-10-18, Claude Opus 5.5):** Plan approved by the author. Added the protocol for measuring each Phase 2 change on its own, preceded by a native CPU baseline with the Phase 0 harness.
- **Rev 11 (2026-10-22, Claude Opus 5.5):** Implemented Phase 2 item 1 (31 kernels, a regression test, and before/after measurements).
  Moved the native CPU baseline to a worktree pinned at the pre-Phase 2 commit, with the author's approval, because Slurm could not accept jobs.
- **Rev 12 (2026-10-22, Claude Opus 5.5), approved by the author on 2026-10-24:** Profiled the host side of a native GPU step for item 5 and found that the per-launch cost is about 5.5 µs, not 20 µs, and that two host-side defects outside the item's original scope cost more than all of its planned changes together.
  Proposed to add both fixes to item 5 and to demote the launch merging and the `reset_tendencies!` rewrite.
- **Rev 13 (2026-10-24, Claude Opus 5.5):** Implemented item 5 as rescoped in Rev 12 (compile-time `ntdiff`, the snow `out` fix, a regression test, and before/after measurements: 1.4× to 1.7× on the GPU and up to 1.2× on the CPU at small sizes).
  The re-profile finds that launch merging is not worth doing and identifies field `NamedTuple`s of more than 32 entries as the largest remaining per-launch host cost, recorded under future work.
- **Rev 14 (2026-10-24, Claude Opus 5.5):** Implemented item 6 (bounds checks elided in the surface energy balance kernel; see the item for the scope that turned out to be unnecessary).
  The author decided on two pull requests: the investigation (this plan, the profiling scripts, and the Phase 0 harness) from `bg/reactant-speedup-investigation`, and all Phase 2 fixes from `bg/performance-fixes`, with one full benchmark run for the latter.
  Corrected the item 5 baseline commit, which a reword of the item 1 commits had replaced.
- **Rev 15 (2026-10-24, Claude Opus 5.5):** At the author's request, turned the profiling scripts into general tools: plan-specific names (`d1_` to `d7_`) replaced by descriptive ones, `key=value` options with defaults and `--help` on every tool, and the three scripts that test one specific change moved to `benchmark/profiling/experiments/`.
  The diagnostics map to the tools as listed in the Phase 1 findings.

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

### Performance target

Parity with Reactant is not a goal.
The target is that Reactant is **at most about 10× faster** than native execution, for every benchmark configuration and resolution, on both CPU and GPU.
Against the current ratios, this requires the native path to become roughly 6× (`land`, GPU, 128 columns) to 27× (`soil_heat`, CPU, 128 columns) faster.
The target is measured after Phase 0, so that solver choice and thread count are equal on both sides; otherwise the ratio partly reflects configuration differences rather than Terrarium itself.

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

### H9. Allocations in kernels and per-step host reductions (added in Rev 3)

D1 found that native steps allocate in proportion to the number of columns, which only happens if kernels themselves allocate.
It also found a host-side reduction that runs on every step and dominates `land` at small sizes.
Neither cost exists under Reactant, because tracing resolves all dispatch and the reduction becomes part of the compiled program.
*Evidence:* Phase 1 findings below (D1, D6, and D7).

## Summary of changes

The work is split into three phases.
Phases 0 and 1 change only the benchmark harness and add diagnostic scripts; they do not change model code or numerics.
Phase 2 consists of model-code fixes chosen from the Phase 1 findings, and each will be added to this plan as a revision for separate approval before implementation.

### Phase 0: benchmark comparability (`benchmark/` only)

*Status (Rev 8):* complete.
Phase 1 was run first; its scripts measured what items 1 to 3 control, so the findings do not depend on Phase 0.

1. Repeat each timed run (proposed default: five repetitions) and report the median, keeping the minimum and spread in the JSON store.
   For Reactant, this means calling the compiled program repeatedly, which costs no recompilation.
2. Make solver choice explicit and identical across architectures: use `NewtonSolver(NF; iterations = 5)` for the land configurations on every architecture (decided in Rev 5).
   D4 shows that the choice does not affect native performance, so this is for like-for-like physics only.
3. Record the effective XLA CPU thread count in the machine details, and support a Reactant CPU run restricted to one core, so that single-thread numbers can be compared like for like.
4. Fix the misleading "1 thread(s)" wording for Reactant sections.
5. *(Added in Rev 4.)* Control for compile order (D7): either run each configuration in a fresh process, or warm up on a throwaway grid size before timing.
   Users hit the first-size slowdown too, so the harness should also report the first-size time separately rather than hide it.
   Once Phase 2 item 2 is implemented, the effect should disappear, and the harness can check that it does.
6. *(Added in Rev 4.)* Re-run the native GPU baseline once items 1 to 5 are in place; the published `gpu-nvidia` numbers are stale (D2).

Implementation notes (Rev 6):

- Item 1: `BenchmarkMode` has a `repeats` field (2 in `quick` mode, 5 otherwise); every run is timed that many times from the re-initialized state, the median is reported, and a new Spread column gives (slowest − fastest) / median.
  The JSON store keeps `ms_per_step_min`, `ms_per_step_max`, and `repeats` next to the existing fields, so older records still load.
  A `default` run therefore takes about five times as long as before.
- Item 2: `land_solver_kwargs` in `benchmark/model_configurations.jl` replaces `land_architecture_kwargs` and applies `NewtonSolver(NF; iterations = 5)` on every architecture.
- Items 3 and 4: the run metadata records the Julia thread count, the CPU affinity and OS thread count of the process (Linux only), and a one-sentence `run_summary` used by both the README and the documentation page; older records fall back to the previous wording.
  A Reactant CPU run restricted to a single core (for example with `taskset -c 0`) is stored under its own `reactant-cpu-*-1core` label.
- Item 5: the warm-up approach was chosen over fresh processes, because it costs seconds rather than a Julia start-up and compilation per configuration.
  The new suites `:bench000` (first) and `:bench999` (last) run each configuration at 32 columns, with a tenth of the usual time steps; the first absorbs the first-model slowdown and the pair reports its size.
  They are skipped under Reactant, where they would only add XLA compilations.
- Item 6, first re-run (Slurm job 2553453, 2026-10-18, NVIDIA H100 80GB, Julia 1.12.7): the medians are valid, but the Spread column is not, because the first timed repetition carried a one-off cost of about 1.8 s.
  That cost is independent of the number of steps and recurs for every new grid size; it is a compilation in the first call of the timed-run closure, which the warm-up did not absorb because it went through a different closure.
  The slowest repetitions reproduce the published `gpu-nvidia` numbers (for example 89.1 versus 92.0 ms/step for `land` at 165,888 columns, against a median of 1.26 ms/step), so **the published native GPU baseline consisted almost entirely of this artifact**.
  A first attempt (calling `initialize!` before the warm-up as well) did not remove it: the second re-run (Slurm job 2554084) showed the same maxima, with medians within a few percent of the first re-run.
  The harness now warms up and times through one stepping function (`step!` in `build_runner`); with `benchmark_configuration` on the A40, the spread for `soil_heat` at 8192 columns drops from about 2,000% to 2%, and the remaining spreads (up to 163% for runs of 24 steps) come from pauses of 20 to 70 ms in single repetitions, not from compilation.
- Item 6, final re-run (Slurm job 2554901, 2026-10-18, H100): with the fix, most spreads are 0 to 20%; a few points still show one slow repetition (for example `land` at 41,472 columns, maximum 32.8 against a median of 1.22 ms/step), which is an isolated pause rather than the artifact.
  The medians agree with the two earlier re-runs within a few percent, except `land` at 165,888 columns (1.87 against 1.26 and 1.28 ms/step).
  Against the published `reactant-gpu` numbers, native GPU is 37× (`land`) and 18× (`land_no_vegetation`) behind at 128 columns, 2× to 7× behind from 512 to 18,432 columns, and between 0.8× and 1.7× from 41,472 to 165,888 columns; `soil_heat` is between 0.8× and 4.2× everywhere.
  On the GPU, the 10× target is therefore missed only at the smallest grids, where about 1.2 ms of host overhead per step dominates (Phase 2 item 5).
  With the corrected medians, native GPU is within 1.1× to 1.2× of Reactant at 165,888 columns (`land` 1.26 versus 1.13 ms/step, `soil_heat` 0.37 versus 0.31), within about 7× from 512 to 8192 columns for `land`, and 37× behind at 128 columns, where the roughly 1.2 ms of host overhead per step dominates.
  The compile-order suites show no first-model effect on the GPU (for example 0.56 and 0.57 ms/step for `land_no_vegetation`).
- `benchmark/run_benchmarks.sbatch` runs the suite as a Slurm job; the partition, QoS, account, time limit, and GPU request are passed to `sbatch`, and the Julia executable and repository root come from the environment.

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

The deliverable of Phase 1 is a findings section in this plan, apportioning the gap at small and large resolutions among the hypotheses.

### Phase 1 findings (Rev 4)

All seven diagnostics were run on 2026-10-17 on the AMD EPYC 9554 host with an NVIDIA A40 GPU, using Julia 1.12.7 with one Julia thread, Float32, and 30 layers unless stated otherwise.
The scripts are committed under `benchmark/profiling/`, with their own environment and a README.
Since Rev 15 they are general tools with options; the diagnostics map to them as follows:

| Diagnostic | Tool and options |
| --- | --- |
| D1 | `host_overhead.jl` (defaults) |
| D2 | `gpu_profile.jl` (defaults) |
| D3 | `experiments/soil_heat_tendency_variants.jl variant=V0` to `V4`, with `arch=cpu` and `arch=gpu` |
| D4 | `native_timing.jl solver=root` and `solver=newton`; `reactant_timing.jl` under `taskset -c 0` and unpinned |
| D5 | `reactant_timing.jl` with `XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text"` |
| D6 | `kernel_ir_audit.jl`, `kernel_allocations.jl`, `allocation_sites.jl phase=closure`, `experiments/vararg_specialization.jl`, and `native_timing.jl ftz=true` and `float=Float64` |
| D7 | `native_timing.jl` at neighboring `nlat` and `nz`, `compile_order.jl`, `gc_share.jl`, and `gpu_profile.jl sections=timings` |

Native CPU jobs were pinned to separate cores with `taskset`, and every timing below is the median of three to five repetitions.
Reactant was measured on the CPU only; Reactant GPU ratios below use the published `reactant-gpu` numbers.

#### Summary

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| H1, host overhead per step | **Confirmed on GPU, secondary on CPU** | D2: `land` keeps the A40 idle at small sizes; about 1.2 ms of Julia host time per step for 43 launches and 8 memsets. |
| H2, parameter-only math per cell | **Confirmed, dominant for the soil heat tendency on CPU** | D3: hoisting gives 36× on CPU, 1.2× on GPU. D5: the HLO handed to XLA has no `power` or `sqrt` ops for `soil_heat`; Reactant folds them before XLA. |
| H3, redundant stencil evaluation | **Confirmed on CPU, neutral on GPU** | D3: 4× on CPU; 1.05× on GPU at 165,888 columns and 0.68× at 4608 columns. |
| H4, fusion and dead diagnostics | **Refuted** | D5: XLA runs 12 fused kernels per `soil_heat` step and 198 per `land` step, versus 7 and 43 native launches; loop-carried state is not pruned noticeably. |
| H5, solver mismatch | **Refuted for performance** | D4: native `RootSolver` and `NewtonSolver(5)` time identically at 128, 4608, and 18,432 columns. |
| H6, XLA threading | **Minor** | D4: 64 threads versus one core gives 1.0× to 1.9× on Reactant CPU. |
| H7, Float64 promotion | **Minor** | D6: no Float64 arithmetic in hot kernels apart from the known `carbon_dynamics.jl` literals and one `power_by_squaring` in the SEB kernel. |
| H8, noisy data | **Explained** | D7 and the compile-order test: the published anomalies come from compile order, not from subnormals or solver iterations. |
| H9, allocations and host reductions | **Confirmed, the largest single native CPU cost after H2** | D1, D6: five kernels lose specialization of their pass-through varargs and box values for every cell; a host-side reduction is recompiled badly for the first grid size. |

On equal cores, the native CPU gap is about 200× for `soil_heat` and 42× for `land` at 4608 columns (native 111 and 203 ms/step, single-core Reactant 0.56 and 4.78 ms/step).
For `soil_heat`, the measured causes account for nearly all of the native time: the tendency kernel costs about 69 ms (H2 and H3) and the despecialized closure kernel about 42 ms (H9).
With both fixed, the projected native step is about 0.3 ms, comparable to single-core Reactant.
For `land`, the same causes account for an estimated 150 of 203 ms; the remainder is spread over the other kernels and has not been broken down.

On the GPU, native execution is already within the 10× target at large sizes (`land` 2.8×, `soil_heat` 4.6× at 165,888 columns on the A40), but not at small sizes, where per-step host work dominates (`land_no_vegetation` 22× at 128 columns, `land` about 7× at 512 to 2048 columns).
The Phase 0 GPU re-run on an H100 (Rev 7) confirms this picture and narrows the large-size gap to 1.1× to 1.2×; see the Phase 0 implementation notes.

#### D1. Host overhead on CPU

- Allocations per step grow with the number of columns, about 210 per column for `soil_heat` and more for `land`.
- At 32 columns, `land` spends 18.5 of 20 ms per step in `compute_auxiliary!`, and only 7% of profile samples fall inside kernel bodies.
  Rev 3 attributed this to the root-fraction normalization.
  It is in fact `compute!(state.soil_moisture_limiting_factor)` at the end of the plant-available-water `compute_auxiliary!` (`src/processes/vegetation/hydraulics/plant_available_water.jl`): an Oceananigans `Integral` over the lazy expression `plant_available_water * root_fraction / Δz`, reduced on every step through Oceananigans' generic `Base.sum!` path (`construct_regionally`, `intersect_indices`).
- Launches per step: 39 for `land`, 28 for `land_no_vegetation`, and 6 for `soil_heat`; seven of the 39 are single-variable `explicit_step_xy_kernel!` launches.
- JET's `report_opt` on `update_state!` gives 1, 32, and 43 reports for `soil_heat`, `land_no_vegetation`, and `land`; these have not been triaged.

#### D2. Host and device split on GPU

| Case | Launches and memsets per step | CUDA API time per step | Device busy per step | Wall time per step |
| --- | --- | --- | --- | --- |
| `soil_heat`, 128 columns | 7 | 0.058 ms | 0.016 ms | 0.054 ms |
| `soil_heat`, 165,888 columns | 7 | 0.072 ms | 0.76 ms | 0.78 ms |
| `land`, 128 columns | 43 and 8 | 0.45 ms | 0.15 ms | 3.75 ms (first size compiled, see D7) |
| `land`, 512 to 41,472 columns | 43 and 8 | about 0.45 ms | 0.2 to 0.5 ms | 1.36 to 1.46 ms |
| `land`, 165,888 columns | 43 and 8 | 0.46 ms | 2.28 ms | 3.15 ms |

- The native GPU numbers are much better than the published ones (for example `land_no_vegetation` 2.55 versus 50.9 ms/step at 165,888 columns), so the published `gpu-nvidia` baseline is stale.
- At small sizes, the A40 is idle most of each step; the step is bound by Julia host work between launches, about 20 µs per launch beyond the 4 to 5 µs driver call.
- At 165,888 columns, the most expensive kernels are the soil energy tendency (442 µs, 64 registers) and the soil water tendency (317 µs).
  The SEB kernel uses 232 registers per thread, which limits the A40 to about 8 of 48 warps per SM.
  `reset_tendencies!` costs 8 memsets and 2 `fill` kernels (about 130 µs).
  Simple kernels such as `explicit_step_xyz_kernel!` run near memory bandwidth (about 570 GB/s).
- No kernel uses local-memory loads or stores apart from 10 in a halo kernel, so register spilling is not an issue.
  The plant-available-water integral runs as a `serial_mapreduce_kernel` (56 µs).

#### D3. Per-cell cost of the soil heat tendency

Times are for the soil energy tendency launch alone; all variants reproduce the baseline bitwise.

| Variant | CPU, 4608 columns | GPU, 4608 columns | GPU, 165,888 columns |
| --- | --- | --- | --- |
| Baseline | 499 ns/cell | 1.00× | 1.00× |
| V1, parameter-only conductivities hoisted | 13.9 ns/cell (36×) | 1.38× | 1.23× |
| V2, `sqrt` instead of `x^(1/2)` | 101 ns/cell (4.9×) | 1.26× | 1.16× |
| V3, κ stored once per cell | 123 ns/cell (4.1×) | 0.68× | 1.05× |
| V4, V1 and V3 together | 0.5 ns/cell (1000×) | 0.70× | 1.27× |

V4 is faster than the product of V1 and V3 on the CPU, presumably because the simplified kernel vectorizes; this has not been confirmed from the IR.
On the GPU, V3 costs a separate launch and halo fill, which outweighs the saved arithmetic at small sizes.

#### D4. Fairness controls

- Native `land` with `RootSolver` and with `NewtonSolver(5)` takes 32.0 and 31.7 ms/step at 128 columns, 203.7 and 203.1 at 4608, and 806.7 and 806.5 at 18,432.
- Reactant CPU, single core (`taskset -c 0`) versus 64 threads: `soil_heat` 0.56 versus 0.63 ms/step at 4608 columns and 28.2 versus 14.7 at 165,888; `land` 4.78 versus 3.76 at 4608.
  XLA starts about 400 OS threads when it is not pinned.

#### D5. What XLA does to the program

- For `soil_heat`, the module handed to XLA contains no `power`, `sqrt`, `exp`, or `log` ops: Reactant's own passes have already constant-folded the conductivity expressions.
  The time loop body runs 12 fused loop kernels and carries 22 arrays.
- For `land`, the time loop body runs 198 fused kernels and contains the five-iteration Newton loop; 12 `power`, 28 `exp`, and 10 `log` ops remain, all depending on the state.
- Reactant's advantage is therefore not fusion or dead-code elimination, but constant folding and the absence of host overhead and dynamic dispatch.

#### D6. Precision, specialization, and the Float32 anomaly

- Source scan: the only bare Float64 literals in kernel code are in `carbon_dynamics.jl`.
  The SEB kernel calls a Float64 `power_by_squaring` and makes 15 non-inlined calls to `specific_humidity_difference`.
- PTX: no kernel has more than a handful of f64 instructions; the soil energy tendency kernel contains 28 `pow` calls, as predicted by H2 and H3.
- Specialization: five kernels lose specialization at run time and box their arguments for every cell, because Julia does not specialize a method on `Vararg` arguments that it only passes through.
  At 128 `land` columns (3840 cells), they allocate 27,180 (`energy_to_temperature_kernel!`), 19,440 (`saturation_to_pressure_kernel!`), 11,760 (`compute_hydraulics_kernel!`), 650 (`compute_snow_interface_fluxes_kernel!`), and 391 (`snow_energy_to_temperature_kernel!`) times per step; the other 19 kernel instances allocate nothing.
  Redefining `energy_to_temperature_kernel!` with `args::Vararg{Any, N}) where {N}` takes `soil_heat`'s `closure!` from 1.09 million allocations and 305 ns per cell to zero allocations and 0.49 ns per cell (about 620×), with bitwise-identical temperatures.
  Static inspection with `ka_code_llvm` misses this, because it compiles a fully specialized instance that the runtime dispatch never uses.
- Flushing subnormals to zero changes nothing (202.6 versus 203.7 ms/step), and no prognostic value is subnormal.
  Float64 `land` takes 235.8 ms/step, about 16% slower than Float32, the normal ordering.

#### D7. Variance and the published anomalies

- Repetitions are tight: the spread is within a few percent for all native CPU cases.
- Native CPU `land` scales linearly: 92.5, 203, 360, and 807 ms/step at 2048, 4608, 8192, and 18,432 columns; 145, 203, and 392 ms/step at 20, 30, and 60 layers.
- **The first grid size compiled in a process is slow.**
  Running `land` at 128 and then 512 columns gives 32.7 and 22.1 ms/step; the reverse order gives 82.5 ms/step for 512 columns and 6.1 for 128.
  The extra allocations in the first size are host-side; the kernels allocate the same in both orders.
  The likely cause is the plant-available-water `Integral` reduction, which is inferred fully only when an earlier size has already been compiled; `land_no_vegetation`, which has no such reduction, shows no anomaly on the GPU.
  The published benchmark compiles 4608 columns, 30 layers, and Float32 first, and every published anomaly falls on that type signature.

### Phase 2: fixes (each requires its own approved revision)

Candidate fixes, in the expected order of payoff (reordered in Rev 4 after Phase 1):

1. **Force specialization of pass-through varargs in kernels (H9). Implemented (Rev 11).**
   Change the trailing `args...` of the `@kernel` definitions that only pass their arguments through to `args::Vararg{Any, N}) where {N}`.
   Five kernels are measured to despecialize (D6); there are 28 kernels with the pattern, and all of them are changed in one pull request (decided in Rev 5), so that the outcome no longer depends on Julia's heuristics.
   This is a signature-only change with a measured 620× speedup of `soil_heat`'s `closure!` and bitwise-identical results.
   It must be checked that KernelAbstractions' `@kernel` accepts the `where` clause on both backends and under Reactant.

   *Implementation (Rev 11, branch `bg/vararg-kernel-specialization`):* 31 kernels were changed, not 28: the original count missed kernels whose signatures span several lines, including `saturation_to_pressure_kernel!` and `pressure_to_saturation_kernel!`.
   Three further kernels keep a plain `args...`, because Julia does not allow a typed `Vararg` after optional positional arguments: `solve_surface_energy_balance_kernel!`, `compute_surface_energy_fluxes_kernel!`, and the canopy evapotranspiration `compute_auxiliary_kernel!`.
   None of the three allocates at present; making their optional arguments required would let them use the typed form as well.
   A new test, `test/coupled_models/kernel_allocations.jl`, checks that `closure!` and `compute_auxiliary!` allocate the same at 8 and 32 columns for a `SoilModel` and for a `LandModel` without vegetation but with Richards hydrology; it fails 3 of its 4 cases on the previous code and passes on the new.
   The full test suite passes.
   Measured before and after on the same machine in the same session (native CPU, Julia 1.12.7, one thread, median of five repetitions; `benchmark/profiling/phase2_step_times.jl`):

   | Configuration | Columns | Before (ms/step) | After (ms/step) | Speedup | Allocations per step, before → after |
   | --- | --- | --- | --- | --- | --- |
   | `land` | 128 | 6.98 | 4.20 | 1.66× | 60,771 → 1,350 |
   | `land` | 4608 | 203 | 116 | 1.75× | 2,489,546 → 1,361 |
   | `land_no_vegetation` | 128 | 5.79 | 3.28 | 1.76× | 60,033 → 612 |
   | `land_no_vegetation` | 4608 | 192 | 104 | 1.85× | 2,488,803 → 618 |
   | `soil_heat` | 128 | 3.07 | 1.95 | 1.57× | 27,186 → 6 |
   | `soil_heat` | 4608 | 110 | 70 | 1.58× | 1,090,897 → 7 |

   All 24 kernel instances of a `land` step now allocate nothing (`kernel_allocations.jl`; before: five kernels, 59,421 allocations per step at 128 columns), and the allocations that remain are host-side and independent of the grid size.
   What remains of `soil_heat` at 4608 columns (70 ms) is almost exactly the cost of the soil energy tendency kernel measured in D3 (499 ns per cell), which deferred items 2 and 3 address.
   On the GPU (local A40), there is no change, as expected, because GPU kernels are always compiled fully specialized: `land` takes 1.43 and 1.45 ms/step at 128 columns, `land_no_vegetation` 2.50 and 2.50 at 165,888, and `soil_heat` 1.40 and 1.40 at 165,888, before and after, and host allocations per step are identical (for example 5,112 for `land`).
   The remaining GPU differences (for example `land` at 165,888 columns, 3.45 and 2.95 ms/step, with maxima above 38 ms in both runs) are run-to-run noise; a first GPU comparison that ran alongside the test suite was discarded because the suite also uses the GPU.
2. **Compute the soil moisture limiting factor inside a kernel (H9). Deferred (Rev 9).**
   Replace `compute!(state.soil_moisture_limiting_factor)`, the Oceananigans `Integral` over a lazy operation, with a column sum in an XY kernel function, or fold it into the existing plant-available-water XYZ kernel as the fused-kernels plan already proposes in its future work.
   This removes the dominant host cost of `land` at small sizes, the compile-order anomaly, and the `serial_mapreduce_kernel` on the GPU.
   The root fraction can stay a lazy `FunctionField` read inside the kernel, so it stays differentiable with respect to its parameters.
3. **Hoist parameter-only computations out of kernels (H2). Deferred (Rev 9).**
   Compute derived constants such as the bulk mineral conductivity and square-rooted constituent conductivities on the host immediately before each launch, as a plain function of the process parameters, and pass the result as an `isbits` kernel argument.
   Computing them per launch rather than storing them in the struct keeps them differentiable with respect to parameters and compatible with `reconstruct`.
   Measured: 36× for the soil energy tendency on CPU and 1.2× on GPU.
   Where a parameter-dependent square root remains in a kernel, use `sqrt`, which is 4.9× faster on CPU than `x^(1/2)`.
   The Reactant constraint that motivated `x^(1/2)` no longer holds (checked in Rev 6 with Reactant 0.2.294): `sqrt` now has a method for the traced device scalar `CuTracedRNumber`, and with the soil conductivities promoted to traced scalars as in `examples/autodiff/global_sensitivity.jl`, the `sqrt` and `x^(1/2)` variants compile to identical StableHLO (Reactant rewrites `x^(1/2)` into `sqrt`), and both compile and run forward and in reverse mode.
   So `sqrt` can be used on all architectures, with no native-only dispatch.
4. **Evaluate the conductivity closure once per cell (H3), CPU benefit only. Deferred (Rev 5).**
   Storing κ in a cell-centered `thermal_conductivity` auxiliary gives 4× on CPU on its own and about 28× on top of item 3, but is neutral on the GPU at 165,888 columns and a regression at 4608 columns when it costs an extra launch (D3).
   If implemented, the κ computation must be fused into the existing soil auxiliary launch (`compute_hydraulics!` in `SoilEnergyWaterCarbon`'s `compute_auxiliary!`) so that no launch or halo fill is added.
   The placement and conflict notes from Rev 3 still apply: implement on `main`, place the new kernel function next to `compute_auxiliary!` in `soil_energy.jl`, and leave the reuse in `linearize!` to `bg/implicit-timestepping`.
   An alternative that avoids the extra field is to evaluate κ at k−1, k, and k+1 once per thread, which cuts four evaluations to three; this has not been measured.
   The decision should be revisited after items 1 to 3, since they change what remains of the CPU cost.
5. **Reduce host overhead per step on the GPU (H1). Rescoped in Rev 12.**
   Merge the per-variable `explicit_step!` launches into one launch per variable shape; seven of the 43 `land` launches are single-variable XY steps.
   Replace the 8 memsets and 2 `fill` kernels of `reset_tendencies!` with a single launch.
   Profile the roughly 20 µs of Julia host work per launch (argument conversion and kernel lookup) and explore caching the configured kernels and their converted arguments across steps.
   *Findings (Rev 12):* `benchmark/profiling/launch_overhead.jl` splits one `explicit_step_xy_kernel!` launch of `land` at 128 columns on the A40 into its stages.
   The whole `explicit_step!` path costs 5.5 µs of host time per launch, of which 2.4 µs is the CUDA.jl kernel lookup (`@cuda launch=false`) and 2.7 µs the argument packing and driver call; Oceananigans' `configure_kernel` (0.2 µs) and `convert_to_device` (1 µs) are small.
   A trivial kernel launched directly through CUDA.jl costs 2.1 to 2.9 µs, so Terrarium and Oceananigans add only about 3 µs per launch, and caching configured kernels would save at most about 3 µs × 43 launches, roughly 0.1 ms per step.
   Merging the seven single-variable `explicit_step!` launches saves about 35 µs, and the `reset_tendencies!` path takes about 2% of host time, so both are small.
   A host profile of 2000 `land` steps instead points to two defects:
   - `ntdiff` (`src/utils/tuple_utils.jl`), which `get_fields(...; except = out)` calls in the canopy evapotranspiration, surface energy balance, and snow interface `compute_auxiliary!` methods, filters the key tuple with `filter(∉(excluded), keys)` at run time; for these large `NamedTuple`s the filter is not constant-folded, so the `NamedTuple` is built dynamically on every call (13% of host samples).
     A `@generated` `ntdiff` that computes the kept keys at compile time removes this.
   - In `timestep!(state, model, ::AbstractSnow)` (`src/models/snow/snow_model.jl`), `out = (; snow_water_equivalent, snow_energy) = state` binds `out` to the whole `state`, not to a `NamedTuple` of the two fields, so every step converts the entire `StateVariables` for the launch of `enforce_snow_constraints!` (about 4% of host samples).
   The plant-available-water `Integral` (deferred item 2) took 60% of the samples, but this profile ran on the first model built in the process, which suffers the compile-order slowdown, so that share is an overestimate.

   Measured with both fixes applied as method overrides in a scratch script (Julia 1.12.7, one thread, median of five repetitions, a throwaway model built first):

   | Configuration | Columns | GPU (A40), current → `ntdiff` → both (ms/step) | CPU, current → `ntdiff` → both (ms/step) |
   | --- | --- | --- | --- |
   | `land` | 128 | 1.50 → 1.00 → 0.86 | 4.20 → 3.78 → 3.58 |
   | `land` | 4608 | 1.55 → 1.00 → 0.89 | 113 → 113 → 113 |
   | `land_no_vegetation` | 128 | 0.74 → 0.52 → 0.48 | 3.22 → 3.02 → 3.03 |
   | `land_no_vegetation` | 4608 | 0.75 → 0.53 → 0.49 | 103 → 102 → 103 |

   *Scope of item 5 (Rev 12, approved):* (a) a compile-time `ntdiff`; (b) the snow `out` fix; (c) a re-profile afterwards to decide whether launch merging and the `reset_tendencies!` rewrite are still worth doing.

   *Implementation (Rev 13, branch `bg/host-overhead`):* `ntdiff` is now a `@generated` function that computes the retained keys from the types, and the snow `timestep!` passes a `NamedTuple` of the two fields it updates.
   A new test in `test/utils.jl` checks `ntdiff` on a 40-key `NamedTuple` with `@inferred`; the previous implementation fails it (it infers up to at least 20 keys, but not 40), on Julia 1.10 and 1.12.
   The full test suite passes.
   Measured before (commit `4fa16da47`, a pinned worktree) and after, on the same machine in the same session (Julia 1.12.7, one thread, median of five repetitions; `phase2_step_times.jl`):

   | Configuration | Columns | GPU (A40), before → after (ms/step) | CPU, before → after (ms/step) | Allocations per step on GPU, before → after |
   | --- | --- | --- | --- | --- |
   | `land` | 128 | 1.45 → 0.84 (1.71×) | 4.25 → 3.56 (1.19×) | 5112 → 4117 |
   | `land` | 4608 | 1.45 → 0.89 (1.63×) | 115 → 112 | 5131 → 4135 |
   | `land` | 165,888 | 3.00 → 2.92 | not measured | 5125 → 4131 |
   | `land_no_vegetation` | 128 | 0.72 → 0.53 (1.36×) | 3.25 → 3.04 (1.07×) | 2643 → 2436 |
   | `land_no_vegetation` | 4608 | 0.71 → 0.52 (1.37×) | 103 → 102 | 2656 → 2449 |
   | `land_no_vegetation` | 165,888 | 2.96 → 2.50 | not measured | 2658 → 2452 |
   | `soil_heat` | all | unchanged | unchanged | unchanged |

   `soil_heat` does not use `get_fields(...; except)` or snow, so it is unaffected, as expected.

   *Re-profile (Rev 13):* with a throwaway model built first, the host time of a `land` step at 128 columns on the A40 drops from 1.20 ms to 0.77 ms (`launch_overhead.jl`), and the profile no longer shows a dominant cost.
   In this steady state the plant-available-water `Integral` (deferred item 2) takes only 4% to 7% of host time; its 60% share in Rev 12 was an artifact of profiling the first model built in the process.
   `explicit_step!` takes 12% and `reset_tendencies!` 6% of the remaining host time, about 90 and 45 µs per step, so merging their launches would save at most about 0.1 ms; this is left undone.
   The largest remaining host costs are three single launches, of the canopy evapotranspiration `compute_auxiliary_kernel!` (13%), `solve_surface_energy_balance_kernel!` (8%), and `compute_snow_interface_fluxes!` (8%), at about 50 to 100 µs each instead of about 6 µs.
   Their field `NamedTuple`s, from `get_fields` over several components, have more than 32 entries, and Julia's `map` over a tuple of more than 32 elements falls back to an untyped `Vector{Any}` path; both Oceananigans' `convert_to_device` and CUDA.jl's `cudaconvert` adapt the arguments with such a `map`, so each of these launches converts its fields twice without type inference.
   This only affects GPU (and presumably Reactant tracing) launches, because `convert_to_device` is the identity on the CPU.
   Passing only the fields each kernel reads, or splitting the `NamedTuple`s below 33 entries, would remove it; this is a change to the kernel interfaces and is proposed as future work, not done here.
6. **Slim the surface energy balance kernel. Implemented (Rev 14).**
   Its 232 registers limit GPU occupancy; inline or restructure `specific_humidity_difference`, remove the Float64 `power_by_squaring`, and check whether the 24 bounds-check throw paths can be elided.

   *Findings (Rev 14):* the 232 registers measured in D2 belong to the default skin temperature solver, `RootSolver(NF; max_iterations = 10)`, which wraps the Newton method of RootSolvers.jl with a convergence test and a finite-difference derivative (two residual evaluations per iteration).
   D2 ran before Phase 0 switched the benchmarks to Terrarium's `NewtonSolver(NF; iterations = 5)`, which performs five unrolled iterations without a convergence test; with it, the kernel uses 111 registers.
   Neither variant contains Float64 arithmetic in its PTX on the current code, so there is no `power_by_squaring` left to remove.
   Both variants still carried bounds checks (27 and 39 calls to `throw_boundserror`) despite `@kernel inbounds = true`, because `@inbounds` reaches a field access only if every function in between propagates it.
   In the surface energy balance call graph it was cut off by functions marked `@inline` (which does not propagate `@inbounds`), by two unannotated atmosphere accessors (`shortwave_down`, `longwave_down`), by the `@inline` call operator of `ObjectiveFunction`, and, for `RootSolver`, by RootSolvers.jl itself, which calls Terrarium's residual closures.

   *Implementation (Rev 14, branch `bg/performance-fixes`):* the index-taking functions of the call graph (`compute_sensible_heat_flux`, `compute_latent_heat_flux`, both `compute_skin_temperature` methods, both `aerodynamic_resistance` methods, `shortwave_down`, `longwave_down`) and the `ObjectiveFunction` call operator now use `@propagate_inbounds`, which also inlines.
   The `RootSolver` residual closures apply `@inbounds` themselves, with a comment that `indices` must be valid for the target field, as they are when `solve!` is called from a kernel.
   Restructuring `specific_humidity_difference` was not done: the kernel takes about 4% of a `land` step at 165,888 columns, so further register savings would not show in the step time.
   Measured with `benchmark/profiling/seb_kernel.jl` (A40, `land` at 165,888 columns; kernel time is the median of five repetitions of 50 launches), before (commit `1cdf17ec6`) and after, in the same session:

   | Solver | Registers | Local memory (bytes) | `throw_boundserror` calls | PTX lines | Kernel time (µs) |
   | --- | --- | --- | --- | --- | --- |
   | `NewtonSolver` (benchmarks) | 111 → 106 | 32 → 0 | 27 → 0 | 2452 → 1934 | 134 → 124 (1.08×) |
   | `RootSolver` (default) | 232 → 128 | 32 → 208 | 39 → 0 | 6511 → 5821 | 247 → 216 (1.14×) |

   The 208 bytes of local memory of the `RootSolver` variant are reserved by `ptxas`; the PTX itself has no local loads or stores.
   Native step times (`phase2_step_times.jl`, same session) change by less than the 10% noise threshold, as expected for a kernel that is a small share of the step: for example, `land` on the GPU takes 0.92 and 0.87 ms/step at 128 columns and 3.29 and 3.15 ms/step at 165,888 columns, before and after, and 113.5 and 111.3 ms/step on the CPU at 4608 columns.
   The same pattern (index-taking kernel functions marked only `@inline`) occurs in 44 functions elsewhere in `src`, so other kernels likely keep bounds checks too; this is recorded under future work.
7. **Eliminate the remaining Float64 literals (H7)** in `carbon_dynamics.jl`, as a small cleanup.

Skipping diagnostic-only auxiliaries (the former item 7, H4) is dropped, because D5 found no evidence that XLA removes such work.

## Testing and verification

- Phases 0 and 1 do not change model code; the existing test suite must still pass, and the draft doc build must succeed.
- Each Phase 2 fix must reproduce the pre-change results within floating-point tolerance on the existing soil, snow, and land model tests.
  Fixes that reorder arithmetic (for example `sqrt` versus `^(1/2)`, or hoisted constants) are expected to change results at the level of rounding only.
- Phase 2 fixes touch kernel code that Reactant and Enzyme compile, so the Reactant and Enzyme suites must pass in CI.
  Following the repository rules, they are run locally only if the extensions or their tests are edited.
- Each Phase 2 fix is a major revision of model compute functions, so the benchmarks are re-run on at least `cpu-x86` and `gpu-nvidia`, with the improvement reported against the current README baseline.

### Measuring each Phase 2 change (added in Rev 10)

Each Phase 2 item is measured on its own, in the order in which the items land, against the state just before it (not against `main`), because the changes interact; for example, item 1 also reduces garbage collection elsewhere.

1. The native CPU baseline is re-run with the Phase 0 harness (`cpu-x86`, through `benchmark/run_benchmarks.sbatch`), because the stored `cpu-x86` record predates it; the `gpu-nvidia` record from Slurm job 2554901 is already current.
   Since Slurm could not accept jobs when Phase 2 started, the baseline is run from a separate worktree pinned at commit `0af879626` (`.worktrees/cpu-baseline`; same code as `ade540f81` on the investigation branch), which holds the code before item 1 (Rev 11).
2. For every item, a targeted before/after measurement with the scripts in `benchmark/profiling/`, on the same machine and in the same session:

   | Item | Measurement |
   | --- | --- |
   | 1, varargs specialization | `kernel_allocations.jl` (allocations per kernel must drop to zero), `closure_allocations.jl`, and native CPU and GPU step times for all three configurations |
   | 5, GPU host overhead | `d2_gpu_profile.jl`: launches per step, CUDA API time, and wall time against device time at small sizes |
   | 6, surface energy balance kernel | register count and time of `solve_surface_energy_balance_kernel!` from the `d2_gpu_profile.jl` trace |
   | 7, Float64 literals | the PTX and IR audits (`d2_gpu_profile.jl`, `d6_kernel_ir_audit.jl`) for f64 instructions; no timing change is expected |

3. One full benchmark run per pull request on `cpu-x86` and `gpu-nvidia`.
   Since Rev 14, all Phase 2 fixes go into one pull request (`bg/performance-fixes`), so this is a single run after the last item, against the native CPU baseline of step 1 and the `gpu-nvidia` record of job 2554901.

Differences of less than about 10% are treated as noise: the medians reproduce within a few percent, but single points still show occasional outliers.

## Documentation changes

- `benchmark/README.md` and `docs/src/benchmarks.md` are regenerated from the updated harness, with a note on solver parity and XLA threading in the README preamble.
- This plan records the Phase 1 findings (D1 in Rev 3, the full set in Rev 4).
- Any process whose kernel arguments change in Phase 2 has its docstrings and doc page updated to match.

## Known limitations

- The Phase 1 measurements come from scratch scripts run on one machine (AMD EPYC 9554 and NVIDIA A40); Reactant was re-measured on the CPU only.
- The fixed-cost estimates in the Background section rest on the published single-run data and are superseded by the Phase 1 findings.
- The `land` breakdown accounts for an estimated 150 of 203 ms per step on the CPU; the remainder has not been attributed to individual kernels.
- JET's 43 optimization reports for `land` have not been triaged.
- While checking `sqrt` (Rev 6), the reverse-mode gradient of the mean soil temperature with respect to the traced soil conductivities came out exactly zero, for both `sqrt` and the current `x^(1/2)`, whereas `examples/autodiff/global_sensitivity.jl` reports a nonzero sensitivity.
  This was not investigated further; it may be a regression in the parameter-sensitivity path or a difference in the test setup (`benchmark/profiling/experiments/reactant_sqrt.jl`), and it is unrelated to this plan.
- Reactant's constant folding is a legitimate optimization, so the native path is not expected to match Reactant exactly; the aim is to remove avoidable overhead, not to close the gap completely.
- Phase 2 item 5 may be limited by how much host work Oceananigans' `launch!` and CUDA.jl's argument conversion do per call, which is outside Terrarium's control.

## Future work

- Replace `@inline` with `@propagate_inbounds` on the remaining index-taking kernel functions (44 found in Rev 14), and check the compiled kernels for `throw_boundserror` calls, possibly in a test; item 6 did this only for the surface energy balance call graph.
- Keep the field `NamedTuple`s passed to kernels at 32 entries or fewer, or pass only the fields each kernel reads (found in Rev 13): the canopy evapotranspiration, surface energy balance, and snow interface launches each spend 50 to 100 µs of host time per step adapting such tuples through Julia's untyped `map` fallback.
- A generic mechanism for declaring parameter-derived constants once per process, so that new processes do not reintroduce H2.
- A regression check in the benchmark harness that flags fixed per-step costs above a threshold.
- Revisit CUDA graphs or other launch batching for the native GPU path, if host overhead remains significant after Phase 2.

## Open questions for the reviewer

All open questions were answered by the author in Rev 5:

1. Solver: the Newton solver on all architectures (Phase 0 item 2).
2. Profiling scripts: committed under `benchmark/profiling/`.
3. Varargs signature change: one pull request covering all 28 kernels (Phase 2 item 1).
4. `sqrt`: check first whether it is still a problem under Reactant (Phase 2 item 3).
5. κ auxiliary: deferred (Phase 2 item 4).
