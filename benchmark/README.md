# Benchmarks

Performance benchmarks for Terrarium.jl, collected across architectures. Each architecture's results live in its own section below; the overview table compares the resolution sweeps across all architectures benchmarked so far.

Every configuration is run for a fixed number of time steps without output. Initialization and — under Reactant — XLA compilation happen before the clock starts and are reported separately; the timed section covers only the stepping loop. Every run is repeated (twice in `quick` mode, five times otherwise), each time from the initial state; the tables report the median repetition, and the Spread column shows how far the fastest and slowest repetitions lie apart. `quick` mode shortens every run, so treat its numbers as a smoke test rather than a measurement, and use the `long` mode for numbers worth quoting.

The land configurations solve the skin temperature with the fixed-iteration Newton solver on every architecture, so that all architectures integrate the same physics. The land configurations and the `soil` configuration solve the soil heat conduction and the Richards equation implicitly, with `IMEX(ForwardEuler, BackwardEuler)` and two Picard iterations per step; `soil_heat` (immobile soil water) is integrated explicitly with `ForwardEuler`. Earlier results integrated the land configurations explicitly with `ForwardEuler` at Δt = 600 s, but the explicit Richards equation is unstable at time steps longer than about one minute, so those results timed an invalid simulation and overstate the speed of the explicit land model. Compared with explicit time steps of one minute, the implicit land model is about 4× faster on the CPU and 5–8× faster on the GPU. Natively, the first model stepped in a Julia process can be several times slower than later models with the same types; the compile-order check suites run first to absorb this, and are repeated at the end of the run to report the size of the effect. They are not run under Reactant. Reactant CPU runs use all available cores unless the process is restricted to one core (e.g. with `taskset -c 0`), in which case they are stored under a separate `-1core` label.

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
- Status: `ok`, `skipped` (above the mode's resolution cap), `unstable` (state went non-finite) or `failed`; `ok` only means that the state stayed finite, since the harness does not validate the solution

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
| land | 128 | 334 | 247 | 2022 | 18311 | 16761 | 47766 |
| land | 512 | 114 | 66 | 2093 | 5563 | 4272 | 8877 |
| land | 2048 | 31 | 17 | 2101 | 1717 | 1833 | 8368 |
| land | 4608 | 4.12 | 7.52 | 2100 | 852 | 822 | 7748 |
| land | 8192 | 7.87 | 4.2 | 2052 | 390 | 422 | 7384 |
| land | 18432 | 3.25 | 1.86 | 2001 | 208 | 194 | 5508 |
| land | 41472 | 1.51 | 0.82 | 1827 | 87 | 84 | 3946 |
| land | 73728 | 0.77 | 0.46 | 1429 | 36 | 52 | 2757 |
| land | 165888 | 0.39 | 0.2 | 762 | 8.17 | 24 | 1450 |
| land_no_vegetation | 128 | 428 | 266 | 3482 | 15876 | 13850 | 50896 |
| land_no_vegetation | 512 | 136 | 68 | 3455 | 2342 | 4704 | 9771 |
| land_no_vegetation | 2048 | 33 | 17 | 3434 | 1538 | 1637 | 9129 |
| land_no_vegetation | 4608 | 16 | 7.53 | 3394 | 708 | 616 | 8330 |
| land_no_vegetation | 8192 | 8.17 | 4.19 | 3314 | 337 | 475 | 7672 |
| land_no_vegetation | 18432 | 3.75 | 1.87 | 3277 | 238 | 229 | 5727 |
| land_no_vegetation | 41472 | 1.68 | 0.85 | 2357 | 101 | 85 | 4337 |
| land_no_vegetation | 73728 | 0.92 | 0.47 | 1578 | 43 | 55 | 2888 |
| land_no_vegetation | 165888 | 0.41 | 0.21 | 805 | 19 | 27 | 1550 |
| soil | 128 | — | 1143 | 28498 | — | — | — |
| soil | 512 | — | 285 | 27336 | — | — | — |
| soil | 2048 | — | 71 | 26486 | — | — | — |
| soil | 4608 | — | 32 | 24082 | — | — | — |
| soil | 8192 | — | 18 | 21642 | — | — | — |
| soil | 18432 | — | 7.81 | 16332 | — | — | — |
| soil | 41472 | — | 3.46 | 10951 | — | — | — |
| soil | 73728 | — | 1.94 | 7135 | — | — | — |
| soil | 165888 | — | 0.86 | 3467 | — | — | — |
| soil_heat | 128 | 1011 | 898 | 96056 | 170034 | 159726 | 155936 |
| soil_heat | 512 | 265 | 224 | 95010 | 42299 | 14526 | 163367 |
| soil_heat | 2048 | 65 | 56 | 90644 | 8078 | 10623 | 118920 |
| soil_heat | 4608 | 27 | 25 | 72173 | 9025 | 3541 | 83290 |
| soil_heat | 8192 | 14 | 14 | 60059 | 1978 | 1648 | 39899 |
| soil_heat | 18432 | 6.75 | 6.23 | 38700 | 668 | 673 | 28231 |
| soil_heat | 41472 | 3.17 | 2.76 | 21116 | 374 | 383 | 12236 |
| soil_heat | 73728 | 1.86 | 1.56 | 12173 | 151 | 186 | 8601 |
| soil_heat | 165888 | 0.74 | 0.69 | 5716 | 73 | 86 | 5386 |

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

Created for Terrarium.jl v0.1.11-DEV on Fri, 13 Nov 2026 05:21:58 in `default` mode (1.0x time steps, median of 5 repetitions, 1 Julia thread(s)).

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
| land | 45.00° | 32 | 900 | 200 | 856 | 2.88 | 76% | 0.334 | 63.68 KiB |
| land_no_vegetation | 45.00° | 32 | 900 | 200 | 967 | 2.55 | 3% | 0.377 | 54.48 KiB |
| soil | 45.00° | 32 | 3600 | 200 | 4674 | 2.11 | 12% | 0.455 | 48.69 KiB |
| soil_heat | 45.00° | 32 | 600 | 200 | 3581 | 0.459 | 0% | 2.09 | 32.51 KiB |

### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 900 | 217 | 7.67 | 321 | 1% | 0.43 | 7.55 MiB |
| land_no_vegetation | 3.75° | 4608 | 900 | 217 | 7.8 | 316 | 1% | 0.438 | 6.46 MiB |
| soil | 3.75° | 4608 | 3600 | 217 | 33 | 303 | 0% | 0.457 | 5.77 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 25 | 65.9 | 0% | 2.1 | 3.85 MiB |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 900 | 2000 | 247 | 9.98 | 4% | 0.385 | 224.55 KiB |
| land | 11.25° | 512 | 900 | 1953 | 66 | 37.1 | 1% | 0.414 | 868.05 KiB |
| land | 5.62° | 2048 | 900 | 488 | 17 | 146 | 1% | 0.42 | 3.36 MiB |
| land | 3.75° | 4608 | 900 | 217 | 7.52 | 328 | 1% | 0.422 | 7.55 MiB |
| land | 2.81° | 8192 | 900 | 122 | 4.2 | 587 | 1% | 0.419 | 13.42 MiB |
| land | 1.88° | 18432 | 900 | 54 | 1.86 | 1330 | 1% | 0.417 | 30.17 MiB |
| land | 1.25° | 41472 | 900 | 24 | 0.82 | 3000 | 1% | 0.415 | 67.88 MiB |
| land | 0.94° | 73728 | 900 | 20 | 0.46 | 5410 | 1% | 0.409 | 120.67 MiB |
| land | 0.62° | 165888 | 900 | 20 | 0.2 | 12300 | 1% | 0.406 | 271.49 MiB |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 900 | 2000 | 266 | 9.27 | 1% | 0.414 | 192.10 KiB |
| land_no_vegetation | 11.25° | 512 | 900 | 1953 | 68 | 36.4 | 1% | 0.422 | 742.60 KiB |
| land_no_vegetation | 5.62° | 2048 | 900 | 488 | 17 | 147 | 1% | 0.417 | 2.88 MiB |
| land_no_vegetation | 3.75° | 4608 | 900 | 217 | 7.53 | 327 | 1% | 0.423 | 6.46 MiB |
| land_no_vegetation | 2.81° | 8192 | 900 | 122 | 4.19 | 589 | 1% | 0.418 | 11.48 MiB |
| land_no_vegetation | 1.88° | 18432 | 900 | 54 | 1.87 | 1320 | 1% | 0.42 | 25.81 MiB |
| land_no_vegetation | 1.25° | 41472 | 900 | 24 | 0.85 | 2910 | 0% | 0.428 | 58.07 MiB |
| land_no_vegetation | 0.94° | 73728 | 900 | 20 | 0.47 | 5250 | 1% | 0.421 | 103.23 MiB |
| land_no_vegetation | 0.62° | 165888 | 900 | 20 | 0.21 | 12000 | 1% | 0.416 | 232.25 MiB |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 898 | 1.83 | 0% | 2.1 | 114.63 KiB |
| soil_heat | 11.25° | 512 | 600 | 1953 | 224 | 7.33 | 0% | 2.1 | 443.13 KiB |
| soil_heat | 5.62° | 2048 | 600 | 488 | 56 | 29.3 | 0% | 2.1 | 1.72 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 25 | 65.9 | 0% | 2.1 | 3.85 MiB |
| soil_heat | 2.81° | 8192 | 600 | 122 | 14 | 117 | 0% | 2.1 | 6.85 MiB |
| soil_heat | 1.88° | 18432 | 600 | 54 | 6.23 | 264 | 0% | 2.1 | 15.40 MiB |
| soil_heat | 1.25° | 41472 | 600 | 24 | 2.76 | 594 | 0% | 2.09 | 34.65 MiB |
| soil_heat | 0.94° | 73728 | 600 | 20 | 1.56 | 1060 | 0% | 2.1 | 61.60 MiB |
| soil_heat | 0.62° | 165888 | 600 | 20 | 0.69 | 2370 | 0% | 2.1 | 138.59 MiB |

### Soil model with Richards flow, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil | 22.50° | 128 | 3600 | 2000 | 1143 | 8.62 | 1% | 0.445 | 171.69 KiB |
| soil | 11.25° | 512 | 3600 | 1953 | 285 | 34.5 | 1% | 0.445 | 663.69 KiB |
| soil | 5.62° | 2048 | 3600 | 488 | 71 | 139 | 0% | 0.443 | 2.57 MiB |
| soil | 3.75° | 4608 | 3600 | 217 | 32 | 311 | 1% | 0.445 | 5.77 MiB |
| soil | 2.81° | 8192 | 3600 | 122 | 18 | 553 | 0% | 0.445 | 10.26 MiB |
| soil | 1.88° | 18432 | 3600 | 54 | 7.81 | 1260 | 1% | 0.438 | 23.07 MiB |
| soil | 1.25° | 41472 | 3600 | 24 | 3.46 | 2850 | 0% | 0.437 | 51.90 MiB |
| soil | 0.94° | 73728 | 3600 | 20 | 1.94 | 5080 | 1% | 0.435 | 92.26 MiB |
| soil | 0.62° | 165888 | 3600 | 20 | 0.86 | 11500 | 0% | 0.434 | 207.57 MiB |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 900 | 651 | 20 | 123 | 1% | 0.375 | 4.03 MiB |
| land | 3.75° | 4608 | 20 | 900 | 326 | 11 | 233 | 1% | 0.395 | 5.79 MiB |
| land | 3.75° | 4608 | 30 | 900 | 217 | 7.21 | 342 | 1% | 0.405 | 7.55 MiB |
| land | 3.75° | 4608 | 60 | 900 | 109 | 3.76 | 656 | 2% | 0.422 | 12.83 MiB |
| land | 3.75° | 4608 | 100 | 900 | 65 | 2.27 | 1090 | 2% | 0.424 | 19.87 MiB |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 900 | 217 | 7.36 | 335 | 1% | 0.413 | 7.55 MiB |
| land | Float64 | 3.75° | 4608 | 900 | 217 | 6.15 | 401 | 1% | 0.345 | 15.10 MiB |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | IMEX(ForwardEuler, BackwardEuler) | 3.75° | 4608 | 900 | 217 | 7.38 | 334 | 1% | 0.414 | 7.55 MiB |
| land | IMEX(Heun, BackwardEuler) | 3.75° | 4608 | 900 | 217 | 5.6 | 440 | 0% | 0.314 | 7.55 MiB |

### Compile-order check: the same models at the end of the run

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 45.00° | 32 | 900 | 200 | 860 | 2.87 | 16% | 0.335 | 63.68 KiB |
| land_no_vegetation | 45.00° | 32 | 900 | 200 | 987 | 2.5 | 4% | 0.385 | 54.48 KiB |
| soil | 45.00° | 32 | 3600 | 200 | 4373 | 2.25 | 12% | 0.426 | 48.69 KiB |
| soil_heat | 45.00° | 32 | 600 | 200 | 3580 | 0.459 | 0% | 2.09 | 32.51 KiB |

## Architecture: `gpu-nvidia`

Created for Terrarium.jl v0.1.11-DEV on Thu, 12 Nov 2026 23:56:36 in `default` mode (1.0x time steps, median of 5 repetitions, 1 Julia thread(s)).

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
- runtime 13.4.0, artifact installation
- driver 580.126.9 for CUDA 13.5, forward-compatible
- compiler 13.5.37, artifact installation

CUDA libraries: 
- cuBLAS: 13.8.0
- cuSPARSE: 12.8.6
- cuSOLVER: 12.3.5
- cuFFT: 12.4.0
- cuRAND: 10.4.4
- CUPTI: 2026.3.1 (API 13.4.1)
- NVML: 13.0.0+580.126.9

Julia packages: 
- CUDACore: 6.4.1
- GPUArrays: 11.5.16
- GPUCompiler: 2.10.0
- KernelAbstractions: 0.9.46
- CUDA_Driver_jll: 13.5.0+0
- CUDA_Compiler_jll: 0.6.5+0
- CUDA_Runtime_jll: 0.25.0+3
- NVPTX_LLVM_Backend_jll: 23.1.2+2

Toolchain:
- Julia: 1.12.7
- LLVM: 18.1.7

4 devices:
  0: NVIDIA H100 80GB HBM3 (sm_90, 76.708 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.4 (LLVM: sm_90a / PTX 9.2)
  1: NVIDIA H100 80GB HBM3 (sm_90, 79.175 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.4 (LLVM: sm_90a / PTX 9.2)
  2: NVIDIA H100 80GB HBM3 (sm_90, 79.175 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.4 (LLVM: sm_90a / PTX 9.2)
  3: NVIDIA H100 80GB HBM3 (sm_90, 79.175 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.4 (LLVM: sm_90a / PTX 9.2)
```

CPU affinity of the benchmark process: `0-127`; OS threads at the end of the run: 69.


### Compile-order check: first models stepped in the process

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 45.00° | 32 | 900 | 200 | 2148 | 1.15 | 3% | 0.837 | 63.68 KiB |
| land_no_vegetation | 45.00° | 32 | 900 | 200 | 3497 | 0.705 | 3% | 1.36 | 54.48 KiB |
| soil | 45.00° | 32 | 3600 | 200 | 28086 | 0.351 | 0% | 2.74 | 48.69 KiB |
| soil_heat | 45.00° | 32 | 600 | 200 | 94845 | 0.0173 | 2% | 55.4 | 32.51 KiB |

### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 900 | 217 | 2035 | 1.21 | 15% | 114 | 7.55 MiB |
| land_no_vegetation | 3.75° | 4608 | 900 | 217 | 3340 | 0.738 | 13% | 187 | 6.46 MiB |
| soil | 3.75° | 4608 | 3600 | 217 | 24058 | 0.41 | 0% | 337 | 5.77 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 72425 | 0.0227 | 1% | 6090 | 3.85 MiB |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 900 | 2000 | 2022 | 1.22 | 1% | 3.15 | 224.55 KiB |
| land | 11.25° | 512 | 900 | 1953 | 2093 | 1.18 | 1% | 13 | 868.05 KiB |
| land | 5.62° | 2048 | 900 | 488 | 2101 | 1.17 | 157% | 52.4 | 3.36 MiB |
| land | 3.75° | 4608 | 900 | 217 | 2100 | 1.17 | 6% | 118 | 7.55 MiB |
| land | 2.81° | 8192 | 900 | 122 | 2052 | 1.2 | 16% | 205 | 13.42 MiB |
| land | 1.88° | 18432 | 900 | 54 | 2001 | 1.23 | 69% | 449 | 30.17 MiB |
| land | 1.25° | 41472 | 900 | 24 | 1827 | 1.35 | 48% | 922 | 67.88 MiB |
| land | 0.94° | 73728 | 900 | 20 | 1429 | 1.72 | 18% | 1280 | 120.67 MiB |
| land | 0.62° | 165888 | 900 | 20 | 762 | 3.24 | 161% | 1540 | 271.49 MiB |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 900 | 2000 | 3482 | 0.708 | 4% | 5.43 | 192.10 KiB |
| land_no_vegetation | 11.25° | 512 | 900 | 1953 | 3455 | 0.713 | 5% | 21.5 | 742.60 KiB |
| land_no_vegetation | 5.62° | 2048 | 900 | 488 | 3434 | 0.718 | 37% | 85.6 | 2.88 MiB |
| land_no_vegetation | 3.75° | 4608 | 900 | 217 | 3394 | 0.726 | 20% | 190 | 6.46 MiB |
| land_no_vegetation | 2.81° | 8192 | 900 | 122 | 3314 | 0.743 | 66% | 331 | 11.48 MiB |
| land_no_vegetation | 1.88° | 18432 | 900 | 54 | 3277 | 0.752 | 13% | 735 | 25.81 MiB |
| land_no_vegetation | 1.25° | 41472 | 900 | 24 | 2357 | 1.05 | 145% | 1190 | 58.07 MiB |
| land_no_vegetation | 0.94° | 73728 | 900 | 20 | 1578 | 1.56 | 84% | 1420 | 103.23 MiB |
| land_no_vegetation | 0.62° | 165888 | 900 | 20 | 805 | 3.06 | 0% | 1630 | 232.25 MiB |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 96056 | 0.0171 | 200% | 225 | 114.63 KiB |
| soil_heat | 11.25° | 512 | 600 | 1953 | 95010 | 0.0173 | 7% | 888 | 443.13 KiB |
| soil_heat | 5.62° | 2048 | 600 | 488 | 90644 | 0.0181 | 1% | 3390 | 1.72 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 72173 | 0.0228 | 0% | 6070 | 3.85 MiB |
| soil_heat | 2.81° | 8192 | 600 | 122 | 60059 | 0.0274 | 0% | 8990 | 6.85 MiB |
| soil_heat | 1.88° | 18432 | 600 | 54 | 38700 | 0.0424 | 1% | 13000 | 15.40 MiB |
| soil_heat | 1.25° | 41472 | 600 | 24 | 21116 | 0.0778 | 2% | 16000 | 34.65 MiB |
| soil_heat | 0.94° | 73728 | 600 | 20 | 12173 | 0.135 | 0% | 16400 | 61.60 MiB |
| soil_heat | 0.62° | 165888 | 600 | 20 | 5716 | 0.287 | 0% | 17300 | 138.59 MiB |

### Soil model with Richards flow, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil | 22.50° | 128 | 3600 | 2000 | 28498 | 0.346 | 132% | 11.1 | 171.69 KiB |
| soil | 11.25° | 512 | 3600 | 1953 | 27336 | 0.361 | 11% | 42.6 | 663.69 KiB |
| soil | 5.62° | 2048 | 3600 | 488 | 26486 | 0.372 | 0% | 165 | 2.57 MiB |
| soil | 3.75° | 4608 | 3600 | 217 | 24082 | 0.409 | 39% | 338 | 5.77 MiB |
| soil | 2.81° | 8192 | 3600 | 122 | 21642 | 0.455 | 0% | 540 | 10.26 MiB |
| soil | 1.88° | 18432 | 3600 | 54 | 16332 | 0.603 | 0% | 916 | 23.07 MiB |
| soil | 1.25° | 41472 | 3600 | 24 | 10951 | 0.9 | 0% | 1380 | 51.90 MiB |
| soil | 0.94° | 73728 | 3600 | 20 | 7135 | 1.38 | 0% | 1600 | 92.26 MiB |
| soil | 0.62° | 165888 | 3600 | 20 | 3467 | 2.84 | 0% | 1750 | 207.57 MiB |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 900 | 651 | 2017 | 1.22 | 6% | 37.7 | 4.03 MiB |
| land | 3.75° | 4608 | 20 | 900 | 326 | 1987 | 1.24 | 9% | 74.3 | 5.79 MiB |
| land | 3.75° | 4608 | 30 | 900 | 217 | 2039 | 1.21 | 15% | 114 | 7.55 MiB |
| land | 3.75° | 4608 | 60 | 900 | 109 | 1964 | 1.25 | 14% | 220 | 12.83 MiB |
| land | 3.75° | 4608 | 100 | 900 | 65 | 1658 | 1.49 | 98% | 310 | 19.87 MiB |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 900 | 217 | 2016 | 1.22 | 5% | 113 | 7.55 MiB |
| land | Float64 | 3.75° | 4608 | 900 | 217 | 1783 | 1.38 | 6% | 100 | 15.10 MiB |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | IMEX(ForwardEuler, BackwardEuler) | 3.75° | 4608 | 900 | 217 | 2041 | 1.21 | 8% | 115 | 7.55 MiB |
| land | IMEX(Heun, BackwardEuler) | 3.75° | 4608 | 900 | 217 | 915 | 2.69 | 19% | 51.4 | 7.55 MiB |

### Compile-order check: the same models at the end of the run

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 45.00° | 32 | 900 | 200 | 2053 | 1.2 | 25% | 0.8 | 63.68 KiB |
| land_no_vegetation | 45.00° | 32 | 900 | 200 | 3384 | 0.728 | 1% | 1.32 | 54.48 KiB |
| soil | 45.00° | 32 | 3600 | 200 | 28307 | 0.348 | 0% | 2.76 | 48.69 KiB |
| soil_heat | 45.00° | 32 | 600 | 200 | 93787 | 0.0175 | 1% | 54.8 | 32.51 KiB |

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

