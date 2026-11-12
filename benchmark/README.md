# Benchmarks

Performance benchmarks for Terrarium.jl, collected across architectures. Each architecture's results live in its own section below; the overview table compares the resolution sweeps across all architectures benchmarked so far.

Every configuration is run for a fixed number of time steps without output. Initialization and — under Reactant — XLA compilation happen before the clock starts and are reported separately; the timed section covers only the stepping loop. Every run is repeated (twice in `quick` mode, five times otherwise), each time from the initial state; the tables report the median repetition, and the Spread column shows how far the fastest and slowest repetitions lie apart. `quick` mode shortens every run, so treat its numbers as a smoke test rather than a measurement, and use the `long` mode for numbers worth quoting.

The land configurations solve the skin temperature with the fixed-iteration Newton solver on every architecture, so that all architectures integrate the same physics. Natively, the first model stepped in a Julia process can be several times slower than later models with the same types; the compile-order check suites run first to absorb this, and are repeated at the end of the run to report the size of the effect. They are not run under Reactant. Reactant CPU runs use all available cores unless the process is restricted to one core (e.g. with `taskset -c 0`), in which case they are stored under a separate `-1core` label.

### Explanation

- Configuration: model setup, see `model_configurations.jl`
- NF: number format, default `Float32`
- Res: approximate horizontal resolution in degrees latitude
- Columns: number of land columns; the benchmark grids are full Gaussian grids with an all-land mask
- L: number of soil layers, default 30
- Δt: time step (s)
- SYPD: simulated years per wallclock day
- ms/step: wallclock milliseconds per time step (median over the repetitions)
- Spread: (slowest − fastest) / median of the repetitions
- Mcell-steps/s: millions of (column × layer) updates per second — the resolution-independent throughput
- Memory: state and tendency fields, including halos
- Compile: one-off XLA compilation time of the stepping loop (Reactant only)
- Status: `ok`, `skipped` (above the mode's resolution cap), `unstable` (state went non-finite) or `failed`

### Running the benchmarks

From `benchmark/`:

```
julia --project=. manual_benchmarking.jl                # CPU (auto-labelled cpu-arm or cpu-x86)
julia --project=. manual_benchmarking.jl gpu            # CUDA GPU
julia --project=. manual_benchmarking.jl reactant-cpu   # Reactant/XLA, CPU backend (auto-labelled reactant-cpu-arm or reactant-cpu-x86)
julia --project=. manual_benchmarking.jl reactant-gpu   # Reactant/XLA, CUDA backend
```

On a Slurm cluster, `run_benchmarks.sbatch` wraps the same commands; see the comments at its top for the arguments and the cluster-specific options to pass to `sbatch`.

A second argument controls the duration: `quick` (0.25x steps, sweeps capped at 8192 columns), `long` (10x steps), or a numeric timestep multiplier. Each run updates only its own architecture's section here; the other architectures are preserved in `assets/benchmark_results.json`.

## Overview: resolution sweeps across architectures

Simulated years per wallclock day (SYPD) for each model configuration across horizontal resolutions, one column per architecture. `—` means the architecture has not been benchmarked yet, skipped that resolution, or failed to run that configuration (see the per-architecture sections for which). Comparison figures are on the documentation's Benchmarks page.

| Configuration | Columns | cpu-arm | cpu-x86 | gpu-nvidia | reactant-cpu-arm | reactant-cpu-x86 | reactant-gpu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| land | 128 | 334 | 576 | 2778 | 18311 | 16761 | 47766 |
| land | 512 | 114 | 156 | 2742 | 5563 | 4272 | 8877 |
| land | 2048 | 31 | 39 | 2526 | 1717 | 1833 | 8368 |
| land | 4608 | 4.12 | 18 | 2777 | 852 | 822 | 7748 |
| land | 8192 | 7.87 | 10 | 2189 | 390 | 422 | 7384 |
| land | 18432 | 3.25 | 4.46 | 1971 | 208 | 194 | 5508 |
| land | 41472 | 1.51 | 1.98 | 3028 | 87 | 84 | 3946 |
| land | 73728 | 0.77 | 1.11 | 2122 | 36 | 52 | 2757 |
| land | 165888 | 0.39 | 0.49 | 1666 | 8.17 | 24 | 1450 |
| land_no_vegetation | 128 | 428 | 669 | 4867 | 15876 | 13850 | 50896 |
| land_no_vegetation | 512 | 136 | 173 | 4893 | 2342 | 4704 | 9771 |
| land_no_vegetation | 2048 | 33 | 44 | 4825 | 1538 | 1637 | 9129 |
| land_no_vegetation | 4608 | 16 | 19 | 4599 | 708 | 616 | 8330 |
| land_no_vegetation | 8192 | 8.17 | 11 | 4602 | 337 | 475 | 7672 |
| land_no_vegetation | 18432 | 3.75 | 4.88 | 5379 | 238 | 229 | 5727 |
| land_no_vegetation | 41472 | 1.68 | 2.17 | 5303 | 101 | 85 | 4337 |
| land_no_vegetation | 73728 | 0.92 | 1.22 | 3707 | 43 | 55 | 2888 |
| land_no_vegetation | 165888 | 0.41 | 0.54 | 2043 | 19 | 27 | 1550 |
| soil_heat | 128 | 1011 | 973 | 51385 | 170034 | 159726 | 155936 |
| soil_heat | 512 | 265 | 243 | 51226 | 42299 | 14526 | 163367 |
| soil_heat | 2048 | 65 | 61 | 51642 | 8078 | 10623 | 118920 |
| soil_heat | 4608 | 27 | 27 | 52653 | 9025 | 3541 | 83290 |
| soil_heat | 8192 | 14 | 15 | 54847 | 1978 | 1648 | 39899 |
| soil_heat | 18432 | 6.75 | 6.81 | 34000 | 668 | 673 | 28231 |
| soil_heat | 41472 | 3.17 | 3.01 | 18126 | 374 | 383 | 12236 |
| soil_heat | 73728 | 1.86 | 1.7 | 10595 | 151 | 186 | 8601 |
| soil_heat | 165888 | 0.74 | 0.75 | 5027 | 73 | 86 | 5386 |

## Architecture: `cpu-arm`

Created for Terrarium.jl v0.1.7 on Wed, 07 Oct 2026 14:14:59 in `default` mode (1x time steps, timed once, 1 Julia thread(s)).

### Machine details

```julia
julia> versioninfo()
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: macOS (arm64-apple-darwin24.0.0)
  CPU: 8 × Apple M3
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, apple-m3)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 4 virtual cores)
Environment:
  JULIA_NUM_PRECOMPILE_TASKS = 2
```


### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 600 | 217 | 4.1 | 400 | 0.345 | 6.86 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 15 | 106 | 1.3 | 5.76 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 30 | 54.7 | 2.53 | 3.85 MiB |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 600 | 2000 | 334 | 4.92 | 0.78 | 204.14 KiB |
| land | 11.25° | 512 | 600 | 1953 | 114 | 14.4 | 1.07 | 789.14 KiB |
| land | 5.62° | 2048 | 600 | 488 | 31 | 53.3 | 1.15 | 3.06 MiB |
| land | 3.75° | 4608 | 600 | 217 | 4.12 | 399 | 0.347 | 6.86 MiB |
| land | 2.81° | 8192 | 600 | 122 | 7.87 | 209 | 1.18 | 12.20 MiB |
| land | 1.88° | 18432 | 600 | 54 | 3.25 | 505 | 1.09 | 27.43 MiB |
| land | 1.25° | 41472 | 600 | 24 | 1.51 | 1090 | 1.14 | 61.71 MiB |
| land | 0.94° | 73728 | 600 | 20 | 0.77 | 2150 | 1.03 | 109.70 MiB |
| land | 0.62° | 165888 | 600 | 20 | 0.39 | 4170 | 1.19 | 246.81 MiB |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 600 | 2000 | 428 | 3.84 | 1 | 171.16 KiB |
| land_no_vegetation | 11.25° | 512 | 600 | 1953 | 136 | 12.1 | 1.27 | 661.66 KiB |
| land_no_vegetation | 5.62° | 2048 | 600 | 488 | 33 | 49.4 | 1.24 | 2.56 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 16 | 106 | 1.31 | 5.76 MiB |
| land_no_vegetation | 2.81° | 8192 | 600 | 122 | 8.17 | 201 | 1.22 | 10.23 MiB |
| land_no_vegetation | 1.88° | 18432 | 600 | 54 | 3.75 | 438 | 1.26 | 23.00 MiB |
| land_no_vegetation | 1.25° | 41472 | 600 | 24 | 1.68 | 977 | 1.27 | 51.74 MiB |
| land_no_vegetation | 0.94° | 73728 | 600 | 20 | 0.92 | 1780 | 1.24 | 91.98 MiB |
| land_no_vegetation | 0.62° | 165888 | 600 | 20 | 0.41 | 4000 | 1.25 | 206.94 MiB |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 1011 | 1.63 | 2.36 | 114.63 KiB |
| soil_heat | 11.25° | 512 | 600 | 1953 | 265 | 6.21 | 2.48 | 443.13 KiB |
| soil_heat | 5.62° | 2048 | 600 | 488 | 65 | 25.4 | 2.42 | 1.72 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 27 | 60.9 | 2.27 | 3.85 MiB |
| soil_heat | 2.81° | 8192 | 600 | 122 | 14 | 118 | 2.08 | 6.85 MiB |
| soil_heat | 1.88° | 18432 | 600 | 54 | 6.75 | 243 | 2.27 | 15.40 MiB |
| soil_heat | 1.25° | 41472 | 600 | 24 | 3.17 | 519 | 2.4 | 34.65 MiB |
| soil_heat | 0.94° | 73728 | 600 | 20 | 1.86 | 884 | 2.5 | 61.60 MiB |
| soil_heat | 0.62° | 165888 | 600 | 20 | 0.74 | 2220 | 2.24 | 138.59 MiB |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 600 | 651 | 32 | 51.4 | 0.896 | 3.70 MiB |
| land | 3.75° | 4608 | 20 | 600 | 326 | 20 | 83 | 1.11 | 5.28 MiB |
| land | 3.75° | 4608 | 30 | 600 | 217 | 3.93 | 418 | 0.331 | 6.86 MiB |
| land | 3.75° | 4608 | 60 | 600 | 109 | 5.82 | 282 | 0.98 | 11.62 MiB |
| land | 3.75° | 4608 | 100 | 600 | 65 | 4.03 | 407 | 1.13 | 17.95 MiB |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 600 | 217 | 3.86 | 426 | 0.325 | 6.86 MiB |
| land | Float64 | 3.75° | 4608 | 600 | 217 | 10 | 159 | 0.87 | 13.73 MiB |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | 600 | 217 | 3.84 | 428 | 0.323 | 6.86 MiB |
| land | Heun | 3.75° | 4608 | 600 | 217 | 1.85 | 890 | 0.155 | 6.86 MiB |

## Architecture: `cpu-x86`

Created for Terrarium.jl v0.1.10 on Fri, 06 Nov 2026 12:58:40 in `default` mode (1.0x time steps, median of 5 repetitions, 1 Julia thread(s)).

### Machine details

```julia
julia> versioninfo()
Julia Version 1.12.7
Commit 6d172b025e4 (2026-08-15 08:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 9554 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
  GC: Built with stock GC
Threads: 1 default, 0 interactive, 1 GC (on 128 virtual cores)
Environment:
  LD_LIBRARY_PATH = /usr/local/lib:/p/system/packages_rhel9/tools/socat/1.8.1.1/lib64:/p/system/packages_rhel9/tools/socat/1.8.1.1/lib:/usr/local/lib::
```

CPU affinity of the benchmark process: `0-127`; OS threads at the end of the run: 65.


### Compile-order check: first models stepped in the process

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 45.00° | 32 | 600 | 200 | 72 | 23 | 15% | 0.0418 | 58.19 KiB |
| land_no_vegetation | 45.00° | 32 | 600 | 200 | 2343 | 0.701 | 11% | 1.37 | 48.84 KiB |
| soil_heat | 45.00° | 32 | 600 | 200 | 3860 | 0.426 | 0% | 2.26 | 32.51 KiB |

### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 600 | 217 | 18 | 92.4 | 0% | 1.5 | 6.90 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 19 | 84.8 | 0% | 1.63 | 5.79 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 27 | 60.3 | 0% | 2.29 | 3.85 MiB |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 600 | 2000 | 576 | 2.85 | 0% | 1.35 | 205.19 KiB |
| land | 11.25° | 512 | 600 | 1953 | 156 | 10.5 | 0% | 1.46 | 793.19 KiB |
| land | 5.62° | 2048 | 600 | 488 | 39 | 41.7 | 2% | 1.47 | 3.07 MiB |
| land | 3.75° | 4608 | 600 | 217 | 18 | 92.4 | 0% | 1.5 | 6.90 MiB |
| land | 2.81° | 8192 | 600 | 122 | 10 | 164 | 3% | 1.5 | 12.26 MiB |
| land | 1.88° | 18432 | 600 | 54 | 4.46 | 368 | 2% | 1.5 | 27.57 MiB |
| land | 1.25° | 41472 | 600 | 24 | 1.98 | 829 | 0% | 1.5 | 62.02 MiB |
| land | 0.94° | 73728 | 600 | 20 | 1.11 | 1480 | 0% | 1.5 | 110.26 MiB |
| land | 0.62° | 165888 | 600 | 20 | 0.49 | 3320 | 0% | 1.5 | 248.07 MiB |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 600 | 2000 | 669 | 2.46 | 0% | 1.56 | 172.21 KiB |
| land_no_vegetation | 11.25° | 512 | 600 | 1953 | 173 | 9.47 | 0% | 1.62 | 665.71 KiB |
| land_no_vegetation | 5.62° | 2048 | 600 | 488 | 44 | 37.5 | 0% | 1.64 | 2.58 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 19 | 85 | 0% | 1.63 | 5.79 MiB |
| land_no_vegetation | 2.81° | 8192 | 600 | 122 | 11 | 150 | 0% | 1.64 | 10.29 MiB |
| land_no_vegetation | 1.88° | 18432 | 600 | 54 | 4.88 | 337 | 3% | 1.64 | 23.14 MiB |
| land_no_vegetation | 1.25° | 41472 | 600 | 24 | 2.17 | 758 | 0% | 1.64 | 52.06 MiB |
| land_no_vegetation | 0.94° | 73728 | 600 | 20 | 1.22 | 1350 | 0% | 1.64 | 92.54 MiB |
| land_no_vegetation | 0.62° | 165888 | 600 | 20 | 0.54 | 3030 | 0% | 1.64 | 208.20 MiB |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 973 | 1.69 | 0% | 2.27 | 114.63 KiB |
| soil_heat | 11.25° | 512 | 600 | 1953 | 243 | 6.75 | 0% | 2.28 | 443.13 KiB |
| soil_heat | 5.62° | 2048 | 600 | 488 | 61 | 27 | 0% | 2.28 | 1.72 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 27 | 60.3 | 0% | 2.29 | 3.85 MiB |
| soil_heat | 2.81° | 8192 | 600 | 122 | 15 | 108 | 0% | 2.28 | 6.85 MiB |
| soil_heat | 1.88° | 18432 | 600 | 54 | 6.81 | 241 | 0% | 2.29 | 15.40 MiB |
| soil_heat | 1.25° | 41472 | 600 | 24 | 3.01 | 546 | 0% | 2.28 | 34.65 MiB |
| soil_heat | 0.94° | 73728 | 600 | 20 | 1.7 | 965 | 0% | 2.29 | 61.60 MiB |
| soil_heat | 0.62° | 165888 | 600 | 20 | 0.75 | 2180 | 0% | 2.28 | 138.59 MiB |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 600 | 651 | 44 | 37.6 | 0% | 1.22 | 3.73 MiB |
| land | 3.75° | 4608 | 20 | 600 | 326 | 25 | 64.9 | 0% | 1.42 | 5.32 MiB |
| land | 3.75° | 4608 | 30 | 600 | 217 | 18 | 92.3 | 0% | 1.5 | 6.90 MiB |
| land | 3.75° | 4608 | 60 | 600 | 109 | 9.46 | 174 | 0% | 1.59 | 11.65 MiB |
| land | 3.75° | 4608 | 100 | 600 | 65 | 5.81 | 283 | 0% | 1.63 | 17.99 MiB |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 600 | 217 | 18 | 92.2 | 0% | 1.5 | 6.90 MiB |
| land | Float64 | 3.75° | 4608 | 600 | 217 | 16 | 104 | 3% | 1.33 | 13.80 MiB |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | 600 | 217 | 18 | 92.3 | 0% | 1.5 | 6.90 MiB |
| land | Heun | 3.75° | 4608 | 600 | 217 | 8.88 | 185 | 0% | 0.747 | 6.90 MiB |

### Compile-order check: the same models at the end of the run

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 45.00° | 32 | 600 | 200 | 69 | 23.6 | 0% | 0.0406 | 58.19 KiB |
| land_no_vegetation | 45.00° | 32 | 600 | 200 | 2271 | 0.723 | 3% | 1.33 | 48.84 KiB |
| soil_heat | 45.00° | 32 | 600 | 200 | 3861 | 0.425 | 1% | 2.26 | 32.51 KiB |

## Architecture: `gpu-nvidia`

Created for Terrarium.jl v0.1.10 on Fri, 06 Nov 2026 14:25:14 in `default` mode (1.0x time steps, median of 5 repetitions, 1 Julia thread(s)).

### Machine details

```julia
julia> versioninfo()
Julia Version 1.12.7
Commit 6d172b025e4 (2026-08-15 08:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 9554 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
  GC: Built with stock GC
Threads: 1 default, 0 interactive, 1 GC (on 128 virtual cores)
Environment:
  LD_LIBRARY_PATH = /usr/local/lib:/p/system/packages_rhel9/tools/socat/1.8.1.1/lib64:/p/system/packages_rhel9/tools/socat/1.8.1.1/lib:/usr/local/lib::
```

```julia
julia> CUDA.versioninfo()
CUDA toolchain: 
- runtime 13.3.0, artifact installation
- driver 580.126.9 for 13.3
- compiler 13.3.33, artifact installation

CUDA libraries: 
- cuBLAS: 13.6.0
- cuSPARSE: 12.8.2
- cuSOLVER: 12.2.6
- cuFFT: 12.3.0
- cuRAND: 10.4.3
- CUPTI: 2026.2.1 (API 13.3.1)
- NVML: 13.0.0+580.126.9

Julia packages: 
- CUDACore: 6.2.2
- GPUArrays: 11.5.16
- GPUCompiler: 1.23.0
- KernelAbstractions: 0.9.45
- CUDA_Driver_jll: 13.3.1+0
- CUDA_Compiler_jll: 0.4.4+1
- CUDA_Runtime_jll: 0.23.0+1
- NVPTX_LLVM_Backend_jll: 22.1.7+1

Toolchain:
- Julia: 1.12.7
- LLVM: 18.1.7

4 devices:
  0: NVIDIA H100 80GB HBM3 (sm_90, 77.932 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.3 (LLVM: sm_90a / PTX 9.0)
  1: NVIDIA H100 80GB HBM3 (sm_90, 79.175 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.3 (LLVM: sm_90a / PTX 9.0)
  2: NVIDIA H100 80GB HBM3 (sm_90, 79.175 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.3 (LLVM: sm_90a / PTX 9.0)
  3: NVIDIA H100 80GB HBM3 (sm_90, 79.175 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.3 (LLVM: sm_90a / PTX 9.0)
```

CPU affinity of the benchmark process: `0-127`; OS threads at the end of the run: 69.


### Compile-order check: first models stepped in the process

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 45.00° | 32 | 600 | 200 | 348 | 4.72 | 1% | 0.203 | 58.19 KiB |
| land_no_vegetation | 45.00° | 32 | 600 | 200 | 4927 | 0.333 | 41% | 2.88 | 48.84 KiB |
| soil_heat | 45.00° | 32 | 600 | 200 | 56441 | 0.0291 | 1% | 33 | 32.51 KiB |

### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 600 | 217 | 2772 | 0.593 | 28% | 233 | 6.90 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 4817 | 0.341 | 53% | 405 | 5.79 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 56484 | 0.0291 | 1% | 4750 | 3.85 MiB |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 600 | 2000 | 2778 | 0.591 | 2% | 6.49 | 205.19 KiB |
| land | 11.25° | 512 | 600 | 1953 | 2742 | 0.599 | 2% | 25.6 | 793.19 KiB |
| land | 5.62° | 2048 | 600 | 488 | 2526 | 0.65 | 14% | 94.5 | 3.07 MiB |
| land | 3.75° | 4608 | 600 | 217 | 2777 | 0.591 | 4% | 234 | 6.90 MiB |
| land | 2.81° | 8192 | 600 | 122 | 2189 | 0.751 | 13% | 327 | 12.26 MiB |
| land | 1.88° | 18432 | 600 | 54 | 1971 | 0.833 | 64% | 664 | 27.57 MiB |
| land | 1.25° | 41472 | 600 | 24 | 3028 | 0.543 | 30% | 2290 | 62.02 MiB |
| land | 0.94° | 73728 | 600 | 20 | 2122 | 0.774 | 332% | 2860 | 110.26 MiB |
| land | 0.62° | 165888 | 600 | 20 | 1666 | 0.986 | 0% | 5050 | 248.07 MiB |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 600 | 2000 | 4867 | 0.337 | 4% | 11.4 | 172.21 KiB |
| land_no_vegetation | 11.25° | 512 | 600 | 1953 | 4893 | 0.336 | 114% | 45.8 | 665.71 KiB |
| land_no_vegetation | 5.62° | 2048 | 600 | 488 | 4825 | 0.34 | 17% | 180 | 2.58 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 4599 | 0.357 | 16% | 387 | 5.79 MiB |
| land_no_vegetation | 2.81° | 8192 | 600 | 122 | 4602 | 0.357 | 62% | 689 | 10.29 MiB |
| land_no_vegetation | 1.88° | 18432 | 600 | 54 | 5379 | 0.305 | 128% | 1810 | 23.14 MiB |
| land_no_vegetation | 1.25° | 41472 | 600 | 24 | 5303 | 0.31 | 341% | 4020 | 52.06 MiB |
| land_no_vegetation | 0.94° | 73728 | 600 | 20 | 3707 | 0.443 | 44% | 4990 | 92.54 MiB |
| land_no_vegetation | 0.62° | 165888 | 600 | 20 | 2043 | 0.804 | 0% | 6190 | 208.20 MiB |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 51385 | 0.032 | 87% | 120 | 114.63 KiB |
| soil_heat | 11.25° | 512 | 600 | 1953 | 51226 | 0.0321 | 76% | 479 | 443.13 KiB |
| soil_heat | 5.62° | 2048 | 600 | 488 | 51642 | 0.0318 | 20% | 1930 | 1.72 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 52653 | 0.0312 | 11% | 4430 | 3.85 MiB |
| soil_heat | 2.81° | 8192 | 600 | 122 | 54847 | 0.03 | 1545% | 8210 | 6.85 MiB |
| soil_heat | 1.88° | 18432 | 600 | 54 | 34000 | 0.0483 | 1% | 11400 | 15.40 MiB |
| soil_heat | 1.25° | 41472 | 600 | 24 | 18126 | 0.0906 | 1% | 13700 | 34.65 MiB |
| soil_heat | 0.94° | 73728 | 600 | 20 | 10595 | 0.155 | 1% | 14300 | 61.60 MiB |
| soil_heat | 0.62° | 165888 | 600 | 20 | 5027 | 0.327 | 0% | 15200 | 138.59 MiB |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 600 | 651 | 2789 | 0.589 | 7% | 78.2 | 3.73 MiB |
| land | 3.75° | 4608 | 20 | 600 | 326 | 2778 | 0.591 | 18% | 156 | 5.32 MiB |
| land | 3.75° | 4608 | 30 | 600 | 217 | 2803 | 0.586 | 9% | 236 | 6.90 MiB |
| land | 3.75° | 4608 | 60 | 600 | 109 | 2827 | 0.581 | 81% | 476 | 11.65 MiB |
| land | 3.75° | 4608 | 100 | 600 | 65 | 3014 | 0.545 | 372% | 845 | 17.99 MiB |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 600 | 217 | 2751 | 0.597 | 12% | 231 | 6.90 MiB |
| land | Float64 | 3.75° | 4608 | 600 | 217 | 2447 | 0.671 | 37% | 206 | 13.80 MiB |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | 600 | 217 | 2845 | 0.577 | 25% | 239 | 6.90 MiB |
| land | Heun | 3.75° | 4608 | 600 | 217 | 1011 | 1.62 | 15% | 85.1 | 6.90 MiB |

### Compile-order check: the same models at the end of the run

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 45.00° | 32 | 600 | 200 | 325 | 5.06 | 2% | 0.19 | 58.19 KiB |
| land_no_vegetation | 45.00° | 32 | 600 | 200 | 4760 | 0.345 | 5% | 2.78 | 48.84 KiB |
| soil_heat | 45.00° | 32 | 600 | 200 | 55337 | 0.0297 | 3% | 32.3 | 32.51 KiB |

## Architecture: `reactant-cpu-arm`

Created for Terrarium.jl v0.1.7 on Wed, 07 Oct 2026 19:20:44 in `default` mode (1x time steps, timed once, 1 Julia thread(s)).

### Machine details

```julia
julia> versioninfo()
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: macOS (arm64-apple-darwin24.0.0)
  CPU: 8 × Apple M3
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, apple-m3)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 4 virtual cores)
Environment:
  JULIA_NUM_PRECOMPILE_TASKS = 2
```

Reactant backend: `cpu` (selected with `Reactant.set_default_backend("cpu")`).


### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 600 | 217 | 768 | 2.14 | 64.6 | 6.86 MiB | 557.8 s |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 765 | 2.15 | 64.4 | 5.76 MiB | 182.8 s |
| soil_heat | 3.75° | 4608 | 600 | 217 | 3512 | 0.468 | 296 | 3.85 MiB | 15.7 s |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 600 | 2000 | 18311 | 0.0897 | 42.8 | 204.14 KiB | 523.0 s |
| land | 11.25° | 512 | 600 | 1953 | 5563 | 0.295 | 52 | 789.14 KiB | 507.6 s |
| land | 5.62° | 2048 | 600 | 488 | 1717 | 0.957 | 64.2 | 3.06 MiB | 506.1 s |
| land | 3.75° | 4608 | 600 | 217 | 852 | 1.93 | 71.7 | 6.86 MiB | 15.2 s |
| land | 2.81° | 8192 | 600 | 122 | 390 | 4.22 | 58.3 | 12.20 MiB | 535.8 s |
| land | 1.88° | 18432 | 600 | 54 | 208 | 7.91 | 69.9 | 27.43 MiB | 549.7 s |
| land | 1.25° | 41472 | 600 | 24 | 87 | 19 | 65.6 | 61.71 MiB | 606.0 s |
| land | 0.94° | 73728 | 600 | 20 | 36 | 45.3 | 48.9 | 109.70 MiB | 756.0 s |
| land | 0.62° | 165888 | 600 | 20 | 8.17 | 201 | 24.8 | 246.81 MiB | 928.5 s |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 600 | 2000 | 15876 | 0.103 | 37.1 | 171.16 KiB | 167.4 s |
| land_no_vegetation | 11.25° | 512 | 600 | 1953 | 2342 | 0.701 | 21.9 | 661.66 KiB | 245.0 s |
| land_no_vegetation | 5.62° | 2048 | 600 | 488 | 1538 | 1.07 | 57.5 | 2.56 MiB | 286.0 s |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 708 | 2.32 | 59.6 | 5.76 MiB | 15.4 s |
| land_no_vegetation | 2.81° | 8192 | 600 | 122 | 337 | 4.88 | 50.4 | 10.23 MiB | 247.0 s |
| land_no_vegetation | 1.88° | 18432 | 600 | 54 | 238 | 6.9 | 80.2 | 23.00 MiB | 280.8 s |
| land_no_vegetation | 1.25° | 41472 | 600 | 24 | 101 | 16.3 | 76.5 | 51.74 MiB | 238.3 s |
| land_no_vegetation | 0.94° | 73728 | 600 | 20 | 43 | 38.4 | 57.6 | 91.98 MiB | 277.7 s |
| land_no_vegetation | 0.62° | 165888 | 600 | 20 | 19 | 87 | 57.2 | 206.94 MiB | 366.8 s |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 170034 | 0.00966 | 397 | 114.63 KiB | 12.8 s |
| soil_heat | 11.25° | 512 | 600 | 1953 | 42299 | 0.0388 | 396 | 443.13 KiB | 16.4 s |
| soil_heat | 5.62° | 2048 | 600 | 488 | 8078 | 0.203 | 302 | 1.72 MiB | 13.9 s |
| soil_heat | 3.75° | 4608 | 600 | 217 | 9025 | 0.182 | 759 | 3.85 MiB | 1.6 s |
| soil_heat | 2.81° | 8192 | 600 | 122 | 1978 | 0.83 | 296 | 6.85 MiB | 15.1 s |
| soil_heat | 1.88° | 18432 | 600 | 54 | 668 | 2.46 | 225 | 15.40 MiB | 17.7 s |
| soil_heat | 1.25° | 41472 | 600 | 24 | 374 | 4.4 | 283 | 34.65 MiB | 23.0 s |
| soil_heat | 0.94° | 73728 | 600 | 20 | 151 | 10.9 | 203 | 61.60 MiB | 29.0 s |
| soil_heat | 0.62° | 165888 | 600 | 20 | 73 | 22.4 | 222 | 138.59 MiB | 48.7 s |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 600 | 651 | 1796 | 0.915 | 50.4 | 3.70 MiB | 513.3 s |
| land | 3.75° | 4608 | 20 | 600 | 326 | 1372 | 1.2 | 76.9 | 5.28 MiB | 512.4 s |
| land | 3.75° | 4608 | 30 | 600 | 217 | 981 | 1.67 | 82.6 | 6.86 MiB | 18.3 s |
| land | 3.75° | 4608 | 60 | 600 | 109 | 483 | 3.4 | 81.3 | 11.62 MiB | 519.5 s |
| land | 3.75° | 4608 | 100 | 600 | 65 | 315 | 5.22 | 88.3 | 17.95 MiB | 517.9 s |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 600 | 217 | 546 | 3.01 | 46 | 6.86 MiB | 17.5 s |
| land | Float64 | 3.75° | 4608 | 600 | 217 | 567 | 2.9 | 47.7 | 13.73 MiB | 428.9 s |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | 600 | 217 | 962 | 1.71 | 81 | 6.86 MiB | 16.1 s | ok |
| land | Heun | 3.75° | 4608 | — | — | — | — | — | — | — | failed: TypeError |

## Architecture: `reactant-cpu-x86`

Created for Terrarium.jl v0.1.9-DEV on Mon, 12 Oct 2026 19:39:21 in `default` mode (1x time steps, timed once, 1 Julia thread(s)).

### Machine details

```julia
julia> versioninfo()
Julia Version 1.12.2
Commit ca9b6662be4 (2025-11-20 16:25 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 9554 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 128 virtual cores)
Environment:
  LD_LIBRARY_PATH = /usr/local/lib:/usr/local/lib:
```

Reactant backend: `cpu` (selected with `Reactant.set_default_backend("cpu")`).


### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 600 | 217 | 726 | 2.26 | 61.1 | 6.86 MiB | 978.4 s |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 696 | 2.36 | 58.5 | 5.76 MiB | 314.3 s |
| soil_heat | 3.75° | 4608 | 600 | 217 | 2379 | 0.691 | 200 | 3.85 MiB | 18.3 s |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 600 | 2000 | 16761 | 0.098 | 39.2 | 204.14 KiB | 935.0 s |
| land | 11.25° | 512 | 600 | 1953 | 4272 | 0.385 | 39.9 | 789.14 KiB | 947.5 s |
| land | 5.62° | 2048 | 600 | 488 | 1833 | 0.896 | 68.6 | 3.06 MiB | 949.6 s |
| land | 3.75° | 4608 | 600 | 217 | 822 | 2 | 69.2 | 6.86 MiB | 20.6 s |
| land | 2.81° | 8192 | 600 | 122 | 422 | 3.9 | 63.1 | 12.20 MiB | 970.8 s |
| land | 1.88° | 18432 | 600 | 54 | 194 | 8.48 | 65.2 | 27.43 MiB | 1007.0 s |
| land | 1.25° | 41472 | 600 | 24 | 84 | 19.6 | 63.6 | 61.71 MiB | 1076.6 s |
| land | 0.94° | 73728 | 600 | 20 | 52 | 31.5 | 70.3 | 109.70 MiB | 1182.5 s |
| land | 0.62° | 165888 | 600 | 20 | 24 | 69.8 | 71.3 | 246.81 MiB | 1466.7 s |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 600 | 2000 | 13850 | 0.119 | 32.4 | 171.16 KiB | 304.6 s |
| land_no_vegetation | 11.25° | 512 | 600 | 1953 | 4704 | 0.349 | 44 | 661.66 KiB | 305.9 s |
| land_no_vegetation | 5.62° | 2048 | 600 | 488 | 1637 | 1 | 61.2 | 2.56 MiB | 308.5 s |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 616 | 2.67 | 51.8 | 5.76 MiB | 11.9 s |
| land_no_vegetation | 2.81° | 8192 | 600 | 122 | 475 | 3.46 | 71 | 10.23 MiB | 319.7 s |
| land_no_vegetation | 1.88° | 18432 | 600 | 54 | 229 | 7.19 | 77 | 23.00 MiB | 336.2 s |
| land_no_vegetation | 1.25° | 41472 | 600 | 24 | 85 | 19.2 | 64.7 | 51.74 MiB | 377.3 s |
| land_no_vegetation | 0.94° | 73728 | 600 | 20 | 55 | 30 | 73.8 | 91.98 MiB | 433.4 s |
| land_no_vegetation | 0.62° | 165888 | 600 | 20 | 27 | 60.7 | 82 | 206.94 MiB | 591.1 s |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 159726 | 0.0103 | 373 | 114.63 KiB | 15.7 s |
| soil_heat | 11.25° | 512 | 600 | 1953 | 14526 | 0.113 | 136 | 443.13 KiB | 15.8 s |
| soil_heat | 5.62° | 2048 | 600 | 488 | 10623 | 0.155 | 397 | 1.72 MiB | 16.3 s |
| soil_heat | 3.75° | 4608 | 600 | 217 | 3541 | 0.464 | 298 | 3.85 MiB | 2.2 s |
| soil_heat | 2.81° | 8192 | 600 | 122 | 1648 | 0.997 | 247 | 6.85 MiB | 18.1 s |
| soil_heat | 1.88° | 18432 | 600 | 54 | 673 | 2.44 | 227 | 15.40 MiB | 21.3 s |
| soil_heat | 1.25° | 41472 | 600 | 24 | 383 | 4.29 | 290 | 34.65 MiB | 28.2 s |
| soil_heat | 0.94° | 73728 | 600 | 20 | 186 | 8.84 | 250 | 61.60 MiB | 38.1 s |
| soil_heat | 0.62° | 165888 | 600 | 20 | 86 | 19.1 | 260 | 138.59 MiB | 66.4 s |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 600 | 651 | 1800 | 0.913 | 50.5 | 3.70 MiB | 967.4 s |
| land | 3.75° | 4608 | 20 | 600 | 326 | 1252 | 1.31 | 70.2 | 5.28 MiB | 970.1 s |
| land | 3.75° | 4608 | 30 | 600 | 217 | 863 | 1.9 | 72.6 | 6.86 MiB | 21.4 s |
| land | 3.75° | 4608 | 60 | 600 | 109 | 412 | 3.99 | 69.3 | 11.62 MiB | 958.1 s |
| land | 3.75° | 4608 | 100 | 600 | 65 | 195 | 8.41 | 54.8 | 17.95 MiB | 969.2 s |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 600 | 217 | 684 | 2.4 | 57.6 | 6.86 MiB | 21.2 s |
| land | Float64 | 3.75° | 4608 | 600 | 217 | 547 | 3 | 46 | 13.73 MiB | 721.9 s |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | 600 | 217 | 850 | 1.93 | 71.5 | 6.86 MiB | 21.3 s |
| land | Heun | 3.75° | 4608 | 600 | 217 | 345 | 4.77 | 29 | 6.86 MiB | 636.9 s |

## Architecture: `reactant-gpu`

Created for Terrarium.jl v0.1.9-DEV on Fri, 09 Oct 2026 21:25:41 in `default` mode (1x time steps, timed once, 1 Julia thread(s)).

### Machine details

```julia
julia> versioninfo()
Julia Version 1.12.2
Commit ca9b6662be4 (2025-11-20 16:25 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 9554 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 128 virtual cores)
Environment:
  LD_LIBRARY_PATH = /usr/local/lib:/usr/local/lib:
```

```julia
julia> CUDA.versioninfo()
CUDA toolchain: 
- runtime 13.3.0, artifact installation
- driver 580.126.9 for 13.3
- compiler 13.3.33, artifact installation

CUDA libraries: 
- cuBLAS: 13.6.0
- cuSPARSE: 12.8.2
- cuSOLVER: 12.2.6
- cuFFT: 12.3.0
- cuRAND: 10.4.3
- CUPTI: 2026.2.1 (API 13.3.1)
- NVML: 13.0.0+580.126.9

Julia packages: 
- CUDACore: 6.2.2
- GPUArrays: 11.5.16
- GPUCompiler: 1.23.0
- KernelAbstractions: 0.9.44
- CUDA_Driver_jll: 13.3.1+0
- CUDA_Compiler_jll: 0.4.4+1
- CUDA_Runtime_jll: 0.23.0+1
- NVPTX_LLVM_Backend_jll: 22.1.7+1

Toolchain:
- Julia: 1.12.2
- LLVM: 18.1.7

1 device:
  0: NVIDIA H100 80GB HBM3 (sm_90, 19.239 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.3 (LLVM: sm_90a / PTX 9.0)
```

Reactant backend: `gpu` (selected with `Reactant.set_default_backend("gpu")`).


### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 600 | 217 | 8555 | 0.192 | 720 | 6.86 MiB | 986.6 s |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 8979 | 0.183 | 756 | 5.76 MiB | 319.4 s |
| soil_heat | 3.75° | 4608 | 600 | 217 | 67258 | 0.0244 | 5660 | 3.85 MiB | 18.5 s |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 600 | 2000 | 47766 | 0.0344 | 112 | 204.14 KiB | 945.3 s |
| land | 11.25° | 512 | 600 | 1953 | 8877 | 0.185 | 83 | 789.14 KiB | 950.6 s |
| land | 5.62° | 2048 | 600 | 488 | 8368 | 0.196 | 313 | 3.06 MiB | 962.7 s |
| land | 3.75° | 4608 | 600 | 217 | 7748 | 0.212 | 652 | 6.86 MiB | 21.9 s |
| land | 2.81° | 8192 | 600 | 122 | 7384 | 0.222 | 1100 | 12.20 MiB | 981.8 s |
| land | 1.88° | 18432 | 600 | 54 | 5508 | 0.298 | 1850 | 27.43 MiB | 1020.6 s |
| land | 1.25° | 41472 | 600 | 24 | 3946 | 0.416 | 2990 | 61.71 MiB | 1092.2 s |
| land | 0.94° | 73728 | 600 | 20 | 2757 | 0.596 | 3710 | 109.70 MiB | 1198.1 s |
| land | 0.62° | 165888 | 600 | 20 | 1450 | 1.13 | 4390 | 246.81 MiB | 1489.3 s |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 600 | 2000 | 50896 | 0.0323 | 119 | 171.16 KiB | 313.2 s |
| land_no_vegetation | 11.25° | 512 | 600 | 1953 | 9771 | 0.168 | 91.4 | 661.66 KiB | 313.2 s |
| land_no_vegetation | 5.62° | 2048 | 600 | 488 | 9129 | 0.18 | 341 | 2.56 MiB | 315.3 s |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 8330 | 0.197 | 701 | 5.76 MiB | 13.4 s |
| land_no_vegetation | 2.81° | 8192 | 600 | 122 | 7672 | 0.214 | 1150 | 10.23 MiB | 323.1 s |
| land_no_vegetation | 1.88° | 18432 | 600 | 54 | 5727 | 0.287 | 1930 | 23.00 MiB | 342.0 s |
| land_no_vegetation | 1.25° | 41472 | 600 | 24 | 4337 | 0.379 | 3280 | 51.74 MiB | 385.4 s |
| land_no_vegetation | 0.94° | 73728 | 600 | 20 | 2888 | 0.569 | 3890 | 91.98 MiB | 442.7 s |
| land_no_vegetation | 0.62° | 165888 | 600 | 20 | 1550 | 1.06 | 4700 | 206.94 MiB | 599.2 s |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 155936 | 0.0105 | 365 | 114.63 KiB | 15.7 s |
| soil_heat | 11.25° | 512 | 600 | 1953 | 163367 | 0.0101 | 1530 | 443.13 KiB | 16.1 s |
| soil_heat | 5.62° | 2048 | 600 | 488 | 118920 | 0.0138 | 4450 | 1.72 MiB | 16.3 s |
| soil_heat | 3.75° | 4608 | 600 | 217 | 83290 | 0.0197 | 7010 | 3.85 MiB | 2.2 s |
| soil_heat | 2.81° | 8192 | 600 | 122 | 39899 | 0.0412 | 5970 | 6.85 MiB | 18.1 s |
| soil_heat | 1.88° | 18432 | 600 | 54 | 28231 | 0.0582 | 9500 | 15.40 MiB | 21.7 s |
| soil_heat | 1.25° | 41472 | 600 | 24 | 12236 | 0.134 | 9270 | 34.65 MiB | 28.8 s |
| soil_heat | 0.94° | 73728 | 600 | 20 | 8601 | 0.191 | 11600 | 61.60 MiB | 39.0 s |
| soil_heat | 0.62° | 165888 | 600 | 20 | 5386 | 0.305 | 16300 | 138.59 MiB | 67.4 s |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 600 | 651 | 8288 | 0.198 | 232 | 3.70 MiB | 953.3 s |
| land | 3.75° | 4608 | 20 | 600 | 326 | 7578 | 0.217 | 425 | 5.28 MiB | 969.6 s |
| land | 3.75° | 4608 | 30 | 600 | 217 | 7540 | 0.218 | 635 | 6.86 MiB | 23.1 s |
| land | 3.75° | 4608 | 60 | 600 | 109 | 6216 | 0.264 | 1050 | 11.62 MiB | 967.1 s |
| land | 3.75° | 4608 | 100 | 600 | 65 | 5491 | 0.299 | 1540 | 17.95 MiB | 976.7 s |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 600 | 217 | 7359 | 0.223 | 619 | 6.86 MiB | 23.1 s |
| land | Float64 | 3.75° | 4608 | 600 | 217 | 4894 | 0.336 | 412 | 13.73 MiB | 720.4 s |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | 600 | 217 | 7234 | 0.227 | 609 | 6.86 MiB | 23.3 s |
| land | Heun | 3.75° | 4608 | 600 | 217 | 3760 | 0.437 | 316 | 6.86 MiB | 628.9 s |

