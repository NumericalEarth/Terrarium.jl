# Profiling scripts

Diagnostic scripts used to find out why native execution is so much slower than Reactant.
They implement the Phase 1 diagnostics D1 to D7 of
[`docs/dev/2026-10/2026-10-17_PLAN_native_vs_reactant_performance_gap.md`](../../docs/dev/2026-10/2026-10-17_PLAN_native_vs_reactant_performance_gap.md),
which also records their results.
They are not part of the benchmark suite and are run by hand.

## Setup

The scripts have their own environment, because they need packages (JET, Enzyme, CUDA, and Reactant) that the benchmark environment does not.
From this directory:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

All scripts take their output file (or prefix) as a command-line argument and print the same rows to standard output.
Run them with one Julia thread, and pin native CPU runs to a core so that concurrent runs do not disturb each other, for example:

```
taskset -c 10 julia --project=. -t 1 d4_d6_d7_native_cpu.jl results.md
```

Several scripts redefine methods (KernelAbstractions' CPU launch operator, or Terrarium kernels and kernel functions) to count or test something.
Each script must therefore run in a fresh Julia process.

## Scripts

| Script | Diagnostic | What it measures |
| --- | --- | --- |
| `common.jl` | | Shared helpers: model setup from `../model_configurations.jl`, repeated timing, the Newton solver settings. |
| `d1_host_overhead.jl` | D1 | Whole-step time and allocations at 32, 128, and 512 columns; phase breakdown, allocation sites, and a sampling profile at 32 columns. |
| `d2_gpu_profile.jl` | D2, D6, D7 | GPU: PTX statistics per kernel (Float64 instructions, local memory, `pow` calls), repeated timings across resolutions, and `CUDA.@profile` summaries and a trace with register counts. |
| `d3_soil_heat_tendency_variants.jl` | D3 | Cost per cell of the soil energy tendency kernel for the baseline and four variants (hoisted constants, `sqrt`, κ stored once per cell, and both), on CPU or GPU. |
| `d4_d5_reactant_threads_and_hlo.jl` | D4, D5 | Reactant CPU timings with repetitions and the OS thread count; run it under `taskset -c 0` for a single core, and set `XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text"` to dump the HLO. |
| `d4_d6_d7_native_cpu.jl` | D4, D6, D7 | Native CPU: root solver versus Newton solver, flushing subnormals, Float64, neighboring resolutions and layer counts, and repeated timings. |
| `d6_kernel_ir_audit.jl` | D6 | LLVM IR of every CPU kernel launched in one step (Float64 operations, allocation and dispatch call sites, throw sites), plus JET's `report_opt` on `update_state!`. Note that `ka_code_llvm` compiles a fully specialized instance, so it does not show runtime despecialization; use `kernel_allocations.jl` for that. |
| `kernel_allocations.jl` | D6 | Allocations per launch for every CPU kernel during one `land` step. |
| `closure_allocations.jl` | D6 | Allocation count and allocation sites of `closure!` for `soil_heat`. |
| `vararg_specialization.jl` | D6 | `closure!` before and after redefining `energy_to_temperature_kernel!` with `args::Vararg{Any, N}) where {N}`. |
| `compile_order.jl` | D7 | `land` at the given grid sizes in the given order in one process, which exposes the slowdown of the first size compiled. |
| `gc_share.jl` | D7 | Share of garbage collection in the per-step time. |
| `reactant_sqrt.jl` | | Whether `sqrt` instead of `x^(1/2)` in `InverseQuadratic` compiles and differentiates under Reactant when the soil conductivities are traced scalars. Run it once per variant with `XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text"` to compare the compiled programs. The reported gradients were exactly zero for both variants when last run; see the plan's known limitations. |

## Run times

Most native CPU and GPU scripts take a few minutes; `d1_host_overhead.jl` takes more than 20 minutes, mostly in its allocation and sampling profiles.
`d4_d5_reactant_threads_and_hlo.jl` takes about 20 minutes per `land` configuration, almost all of it XLA compilation.

The results recorded in the plan were measured before `build_model` switched the land configurations to the Newton skin-temperature solver; `common.jl` provides `newton_kwargs` and `root_solver_kwargs` to select either solver explicitly.
