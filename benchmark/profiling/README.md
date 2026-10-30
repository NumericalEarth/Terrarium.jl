# Profiling tools

Command-line tools for finding out where the time of a Terrarium time step goes, on the CPU, on the GPU, and under Reactant.
They complement the benchmark suite in `..`, which measures step times; these tools explain them.
They build the same model configurations as the benchmarks (`../model_configurations.jl`) and are run by hand.

They were written for the investigation in [`docs/dev/2026-10/2026-10-17_PLAN_native_vs_reactant_performance_gap.md`](../../docs/dev/2026-10/2026-10-17_PLAN_native_vs_reactant_performance_gap.md), which records what they found.

## Setup

The tools have their own environment, because they need packages (JET, Enzyme, CUDA, and Reactant) that the benchmark environment does not.
From this directory:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Every tool takes `key=value` options with defaults, and lists them with `--help`:

```
julia --project=. host_overhead.jl --help
julia --project=. native_timing.jl config=land nlat=24 solver=root header=true out=results.md
```

Lists are comma-separated (`nlat=2,4,8`, `configs=land,soil_heat`).
Results are printed to standard output and, with `out=<file>`, appended to that file, mostly as Markdown tables; tools that write several files take `out` as a prefix.
Give concurrent runs separate output files: rows appended to one file from two processes can be lost.

Run the tools with one Julia thread, and pin native CPU runs to a core so that concurrent runs do not disturb each other:

```
taskset -c 10 julia --project=. -t 1 native_timing.jl config=soil_heat nlat=24
```

Several tools redefine methods (KernelAbstractions' CPU launch operator, or Terrarium kernels and kernel functions) to count or test something, so each tool must run in a fresh Julia process.
For the same reason, and because the first model stepped in a process can be slower than later ones (see `compile_order.jl`), compare timings from separate processes or from tools that build a throwaway model first.

## Tools

| Tool | Architecture | What it measures |
| --- | --- | --- |
| `host_overhead.jl` | CPU | Whole-step time and allocations across grid sizes, split into a fixed and a per-column cost; a breakdown of one step into its phases; allocation sites; and a sampling profile with the share of time inside kernel bodies. |
| `native_timing.jl` | CPU, GPU | Step time of one configuration and setting (grid size, layers, precision, skin temperature solver, flushing subnormals) with repetitions, and the number of subnormal prognostic values. |
| `gpu_profile.jl` | GPU | PTX statistics per kernel (Float64 instructions, local memory, `pow` calls), step times across grid sizes, and `CUDA.@profile` summaries and a trace with register counts. |
| `reactant_timing.jl` | Reactant | Compile time and step time with repetitions, the number of OS threads, and optionally the compiled HLO (through `XLA_FLAGS`). |
| `kernel_allocations.jl` | CPU | Allocations per launch of every kernel in one step; finds kernels that are not type stable or have lost specialization. |
| `kernel_ir_audit.jl` | CPU | LLVM IR of every kernel launched in one step (Float64 operations, `pow` and `exp`/`log` calls, allocation, dispatch, and throw sites), and JET's `report_opt` on `update_state!`. |
| `allocation_sites.jl` | CPU | Allocation count and allocation sites of one phase of a step (`closure!`, `compute_auxiliary!`, and so on). |
| `compile_order.jl` | CPU | Step times of one configuration at several grid sizes, in a given order, in one process; exposes timings that depend on what was compiled first. |
| `gc_share.jl` | CPU | Share of garbage collection in the step time. |
| `step_times.jl` | CPU, GPU | Step times and allocations per step of several configurations and grid sizes after a throwaway model, as rows for comparing two versions of the code. |
| `launch_overhead.jl` | GPU | Host time of each stage of one kernel launch (Terrarium, Oceananigans, KernelAbstractions, and CUDA.jl), against a trivial CUDA.jl kernel, and a host profile of many steps. |
| `surface_energy_balance_kernel.jl` | GPU | Registers, local memory, PTX statistics (Float64 instructions, traps, calls such as `throw_boundserror`), and device time of the surface energy balance kernel, with either skin temperature solver. |
| `common.jl` | | Shared helpers: option parsing, model setup, timing with repetitions, and the skin temperature solver settings. Not run directly. |

## Experiments

The tools in `experiments/` each test one specific change by redefining Terrarium methods, so they depend on the internals they replace and may need adapting when those change.
They are kept as worked examples and to reproduce the results recorded in the plan.
Run them from this directory, for example `julia --project=. experiments/reactant_sqrt.jl mode=sqrt`.

| Experiment | What it tests |
| --- | --- |
| `experiments/soil_heat_tendency_variants.jl` | Cost per cell of the soil energy tendency kernel for four alternative implementations (hoisted constants, `sqrt`, conductivity computed once per cell, and both), and how much each changes the tendencies. |
| `experiments/vararg_specialization.jl` | Allocations and time of `closure!` before and after forcing `Vararg` specialization of `energy_to_temperature_kernel!`. |
| `experiments/reactant_sqrt.jl` | Whether `sqrt` instead of `x^(1/2)` in `InverseQuadratic` compiles and differentiates under Reactant when the soil conductivities are traced scalars. |

## Run times

Most native CPU and GPU tools take a few minutes with their defaults.
`host_overhead.jl` takes more than 20 minutes, mostly in its allocation and sampling profiles.
`reactant_timing.jl` takes about 20 minutes for a `land` configuration on the CPU backend, almost all of it XLA compilation.
