# Benchmarks

Performance benchmarks for Terrarium.jl, collected across architectures. Each architecture's results live in its own section below; the overview table compares the resolution sweeps across all architectures benchmarked so far.

Every configuration is run for a fixed number of time steps without output. Initialization and — under Reactant — XLA compilation happen before the clock starts and are reported separately; the timed section covers only the stepping loop. A run takes a few seconds and is timed once, so it measures the mean rather than the minimum: deviations of ±20% between repetitions are normal, and `quick` mode — which shortens every run — is noisier still, so treat its numbers as a smoke test rather than a measurement. Use the `long` mode for numbers worth quoting.

### Explanation

- Configuration: model setup, see `model_configurations.jl`
- NF: number format, default `Float32`
- Res: approximate horizontal resolution in degrees latitude
- Columns: number of land columns; the benchmark grids are full Gaussian grids with an all-land mask
- L: number of soil layers, default 30
- Δt: time step (s)
- SYPD: simulated years per wallclock day
- ms/step: wallclock milliseconds per time step
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

A second argument controls the duration: `quick` (0.25x steps, sweeps capped at 8192 columns), `long` (10x steps), or a numeric timestep multiplier. Each run updates only its own architecture's section here; the other architectures are preserved in `assets/benchmark_results.json`.

## Overview: resolution sweeps across architectures

Simulated years per wallclock day (SYPD) for each model configuration across horizontal resolutions, one column per architecture. `—` means the architecture has not been benchmarked yet, skipped that resolution, or failed to run that configuration (see the per-architecture sections for which). Comparison figures are on the documentation's Benchmarks page.

| Configuration | Columns | cpu-arm | cpu-x86 | gpu-nvidia | reactant-cpu-arm | reactant-cpu-x86 | reactant-gpu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| land | 128 | 334 | 273 | 316 | 18311 | — | — |
| land | 512 | 114 | 76 | 316 | 5563 | — | — |
| land | 2048 | 31 | 19 | 242 | 1717 | — | — |
| land | 4608 | 4.12 | 3.35 | 189 | 852 | — | — |
| land | 8192 | 7.87 | 4.86 | 142 | 390 | — | — |
| land | 18432 | 3.25 | 2.18 | 85 | 208 | — | — |
| land | 41472 | 1.51 | 0.96 | 44 | 87 | — | — |
| land | 73728 | 0.77 | 0.54 | 38 | 36 | — | — |
| land | 165888 | 0.39 | 0.24 | 38 | 8.17 | — | — |
| land_no_vegetation | 128 | 428 | 304 | 1435 | 15876 | — | — |
| land_no_vegetation | 512 | 136 | 80 | 1429 | 2342 | — | — |
| land_no_vegetation | 2048 | 33 | 20 | 984 | 1538 | — | — |
| land_no_vegetation | 4608 | 16 | 8.81 | 602 | 708 | — | — |
| land_no_vegetation | 8192 | 8.17 | 4.99 | 426 | 337 | — | — |
| land_no_vegetation | 18432 | 3.75 | 2.23 | 89 | 238 | — | — |
| land_no_vegetation | 41472 | 1.68 | 1.01 | 115 | 101 | — | — |
| land_no_vegetation | 73728 | 0.92 | 0.56 | 94 | 43 | — | — |
| land_no_vegetation | 165888 | 0.41 | 0.25 | 99 | 19 | — | — |
| soil_heat | 128 | 1011 | 578 | 12048 | 170034 | 84969 | 110785 |
| soil_heat | 512 | 265 | 148 | 11593 | 42299 | 11925 | 111800 |
| soil_heat | 2048 | 65 | 37 | 6419 | 8078 | 6839 | 94046 |
| soil_heat | 4608 | 27 | 17 | 18447 | 9025 | 1506 | 64078 |
| soil_heat | 8192 | 14 | 9.26 | 3467 | 1978 | 862 | 41518 |
| soil_heat | 18432 | 6.75 | 4.12 | 1715 | 668 | 443 | 20687 |
| soil_heat | 41472 | 3.17 | 1.82 | 392 | 374 | 152 | 9516 |
| soil_heat | 73728 | 1.86 | 1.02 | 644 | 151 | 85 | 8047 |
| soil_heat | 165888 | 0.74 | 0.46 | 596 | 73 | 42 | 4342 |

## Architecture: `cpu-arm`

Created for Terrarium.jl v0.1.7 on Wed, 07 Oct 2026 14:14:59 in `default` mode (1x time steps, 1 thread(s)).

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

Created for Terrarium.jl v0.1.4 on Sun, 09 Aug 2026 20:35:28 in `default` mode (1x time steps, 1 thread(s)).

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


### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 600 | 217 | 3.49 | 471 | 0.293 | 6.86 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 9.37 | 175 | 0.789 | 5.76 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 17 | 96.1 | 1.44 | 3.85 MiB |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 600 | 2000 | 273 | 6.01 | 0.639 | 204.14 KiB |
| land | 11.25° | 512 | 600 | 1953 | 76 | 21.5 | 0.715 | 789.14 KiB |
| land | 5.62° | 2048 | 600 | 488 | 19 | 85.9 | 0.715 | 3.06 MiB |
| land | 3.75° | 4608 | 600 | 217 | 3.35 | 490 | 0.282 | 6.86 MiB |
| land | 2.81° | 8192 | 600 | 122 | 4.86 | 338 | 0.728 | 12.20 MiB |
| land | 1.88° | 18432 | 600 | 54 | 2.18 | 755 | 0.733 | 27.43 MiB |
| land | 1.25° | 41472 | 600 | 24 | 0.96 | 1710 | 0.728 | 61.71 MiB |
| land | 0.94° | 73728 | 600 | 20 | 0.54 | 3030 | 0.73 | 109.70 MiB |
| land | 0.62° | 165888 | 600 | 20 | 0.24 | 6880 | 0.724 | 246.81 MiB |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 600 | 2000 | 304 | 5.4 | 0.711 | 171.16 KiB |
| land_no_vegetation | 11.25° | 512 | 600 | 1953 | 80 | 20.6 | 0.747 | 661.66 KiB |
| land_no_vegetation | 5.62° | 2048 | 600 | 488 | 20 | 82.7 | 0.743 | 2.56 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 8.81 | 186 | 0.742 | 5.76 MiB |
| land_no_vegetation | 2.81° | 8192 | 600 | 122 | 4.99 | 329 | 0.746 | 10.23 MiB |
| land_no_vegetation | 1.88° | 18432 | 600 | 54 | 2.23 | 738 | 0.749 | 23.00 MiB |
| land_no_vegetation | 1.25° | 41472 | 600 | 24 | 1.01 | 1630 | 0.764 | 51.74 MiB |
| land_no_vegetation | 0.94° | 73728 | 600 | 20 | 0.56 | 2910 | 0.76 | 91.98 MiB |
| land_no_vegetation | 0.62° | 165888 | 600 | 20 | 0.25 | 6560 | 0.758 | 206.94 MiB |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 578 | 2.84 | 1.35 | 114.63 KiB |
| soil_heat | 11.25° | 512 | 600 | 1953 | 148 | 11.1 | 1.39 | 443.13 KiB |
| soil_heat | 5.62° | 2048 | 600 | 488 | 37 | 44.4 | 1.38 | 1.72 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 17 | 99.4 | 1.39 | 3.85 MiB |
| soil_heat | 2.81° | 8192 | 600 | 122 | 9.26 | 177 | 1.39 | 6.85 MiB |
| soil_heat | 1.88° | 18432 | 600 | 54 | 4.12 | 399 | 1.39 | 15.40 MiB |
| soil_heat | 1.25° | 41472 | 600 | 24 | 1.82 | 903 | 1.38 | 34.65 MiB |
| soil_heat | 0.94° | 73728 | 600 | 20 | 1.02 | 1610 | 1.38 | 61.60 MiB |
| soil_heat | 0.62° | 165888 | 600 | 20 | 0.46 | 3560 | 1.4 | 138.59 MiB |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 600 | 651 | 22 | 74.8 | 0.616 | 3.70 MiB |
| land | 3.75° | 4608 | 20 | 600 | 326 | 12 | 138 | 0.67 | 5.28 MiB |
| land | 3.75° | 4608 | 30 | 600 | 217 | 8.2 | 200 | 0.69 | 6.86 MiB |
| land | 3.75° | 4608 | 60 | 600 | 109 | 4.23 | 388 | 0.713 | 11.62 MiB |
| land | 3.75° | 4608 | 100 | 600 | 65 | 2.63 | 626 | 0.737 | 17.95 MiB |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 600 | 217 | 3.13 | 524 | 0.264 | 6.86 MiB |
| land | Float64 | 3.75° | 4608 | 600 | 217 | 6.7 | 245 | 0.564 | 13.73 MiB |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | 600 | 217 | 8.04 | 204 | 0.676 | 6.86 MiB |
| land | Heun | 3.75° | 4608 | 600 | 217 | 4.14 | 396 | 0.349 | 6.86 MiB |

## Architecture: `gpu-nvidia`

Created for Terrarium.jl v0.1.4 on Sun, 09 Aug 2026 20:02:07 in `default` mode (1x time steps, 1 thread(s)).

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
- CUDACore: 6.2.1
- GPUArrays: 11.5.9
- GPUCompiler: 1.23.0
- KernelAbstractions: 0.9.42
- CUDA_Driver_jll: 13.3.0+1
- CUDA_Compiler_jll: 0.4.4+1
- CUDA_Runtime_jll: 0.23.0+1
- NVPTX_LLVM_Backend_jll: 22.1.7+1

Toolchain:
- Julia: 1.12.2
- LLVM: 18.1.7

1 device:
  0: NVIDIA H100 80GB HBM3 (sm_90, 77.995 GiB / 79.647 GiB available)
     compiles to sm_90a / PTX 9.3 (LLVM: sm_90a / PTX 9.0)
```


### Model configurations, default resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 600 | 217 | 156 | 10.5 | 13.1 | 6.86 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 676 | 2.43 | 56.9 | 5.76 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 3683 | 0.446 | 310 | 3.85 MiB |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | 600 | 2000 | 316 | 5.19 | 0.739 | 204.14 KiB |
| land | 11.25° | 512 | 600 | 1953 | 316 | 5.2 | 2.95 | 789.14 KiB |
| land | 5.62° | 2048 | 600 | 488 | 242 | 6.78 | 9.07 | 3.06 MiB |
| land | 3.75° | 4608 | 600 | 217 | 189 | 8.7 | 15.9 | 6.86 MiB |
| land | 2.81° | 8192 | 600 | 122 | 142 | 11.6 | 21.2 | 12.20 MiB |
| land | 1.88° | 18432 | 600 | 54 | 85 | 19.2 | 28.8 | 27.43 MiB |
| land | 1.25° | 41472 | 600 | 24 | 44 | 37.5 | 33.2 | 61.71 MiB |
| land | 0.94° | 73728 | 600 | 20 | 38 | 43 | 51.5 | 109.70 MiB |
| land | 0.62° | 165888 | 600 | 20 | 38 | 43.1 | 115 | 246.81 MiB |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | 600 | 2000 | 1435 | 1.14 | 3.35 | 171.16 KiB |
| land_no_vegetation | 11.25° | 512 | 600 | 1953 | 1429 | 1.15 | 13.4 | 661.66 KiB |
| land_no_vegetation | 5.62° | 2048 | 600 | 488 | 984 | 1.67 | 36.8 | 2.56 MiB |
| land_no_vegetation | 3.75° | 4608 | 600 | 217 | 602 | 2.73 | 50.7 | 5.76 MiB |
| land_no_vegetation | 2.81° | 8192 | 600 | 122 | 426 | 3.85 | 63.8 | 10.23 MiB |
| land_no_vegetation | 1.88° | 18432 | 600 | 54 | 89 | 18.4 | 30 | 23.00 MiB |
| land_no_vegetation | 1.25° | 41472 | 600 | 24 | 115 | 14.3 | 86.9 | 51.74 MiB |
| land_no_vegetation | 0.94° | 73728 | 600 | 20 | 94 | 17.4 | 127 | 91.98 MiB |
| land_no_vegetation | 0.62° | 165888 | 600 | 20 | 99 | 16.7 | 299 | 206.94 MiB |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 12048 | 0.136 | 28.2 | 114.63 KiB |
| soil_heat | 11.25° | 512 | 600 | 1953 | 11593 | 0.142 | 108 | 443.13 KiB |
| soil_heat | 5.62° | 2048 | 600 | 488 | 6419 | 0.256 | 240 | 1.72 MiB |
| soil_heat | 3.75° | 4608 | 600 | 217 | 18447 | 0.0891 | 1550 | 3.85 MiB |
| soil_heat | 2.81° | 8192 | 600 | 122 | 3467 | 0.474 | 519 | 6.85 MiB |
| soil_heat | 1.88° | 18432 | 600 | 54 | 1715 | 0.958 | 577 | 15.40 MiB |
| soil_heat | 1.25° | 41472 | 600 | 24 | 392 | 4.19 | 297 | 34.65 MiB |
| soil_heat | 0.94° | 73728 | 600 | 20 | 644 | 2.55 | 867 | 61.60 MiB |
| soil_heat | 0.62° | 165888 | 600 | 20 | 596 | 2.76 | 1810 | 138.59 MiB |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | 600 | 651 | 268 | 6.13 | 7.52 | 3.70 MiB |
| land | 3.75° | 4608 | 20 | 600 | 326 | 222 | 7.41 | 12.4 | 5.28 MiB |
| land | 3.75° | 4608 | 30 | 600 | 217 | 190 | 8.64 | 16 | 6.86 MiB |
| land | 3.75° | 4608 | 60 | 600 | 109 | 79 | 20.8 | 13.3 | 11.62 MiB |
| land | 3.75° | 4608 | 100 | 600 | 65 | 96 | 17.2 | 26.8 | 17.95 MiB |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | 600 | 217 | 169 | 9.7 | 14.2 | 6.86 MiB |
| land | Float64 | 3.75° | 4608 | 600 | 217 | 169 | 9.7 | 14.3 | 13.73 MiB |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | 600 | 217 | 302 | 5.43 | 25.4 | 6.86 MiB |
| land | Heun | 3.75° | 4608 | 600 | 217 | 97 | 17 | 8.12 | 6.86 MiB |

## Architecture: `reactant-cpu-arm`

Created for Terrarium.jl v0.1.7 on Wed, 07 Oct 2026 19:20:44 in `default` mode (1.0x time steps, 1 thread(s)).

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

Created for Terrarium.jl v0.1.4 on Sun, 09 Aug 2026 20:12:54 in `default` mode (1x time steps, 1 thread(s)).

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

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | — | — | — | — | — | — | — | failed: MethodError |
| land_no_vegetation | 3.75° | 4608 | — | — | — | — | — | — | — | failed: ErrorException |
| soil_heat | 3.75° | 4608 | 600 | 217 | 1484 | 1.11 | 125 | 3.85 MiB | 31.6 s | ok |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | — | — | — | — | — | — | failed: MethodError |
| land | 11.25° | 512 | — | — | — | — | — | — | failed: MethodError |
| land | 5.62° | 2048 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |
| land | 2.81° | 8192 | — | — | — | — | — | — | failed: ErrorException |
| land | 1.88° | 18432 | — | — | — | — | — | — | failed: ErrorException |
| land | 1.25° | 41472 | — | — | — | — | — | — | failed: ErrorException |
| land | 0.94° | 73728 | — | — | — | — | — | — | failed: ErrorException |
| land | 0.62° | 165888 | — | — | — | — | — | — | failed: ErrorException |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 11.25° | 512 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 5.62° | 2048 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 3.75° | 4608 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 2.81° | 8192 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 1.88° | 18432 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 1.25° | 41472 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 0.94° | 73728 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 0.62° | 165888 | — | — | — | — | — | — | failed: ErrorException |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 84969 | 0.0193 | 199 | 114.63 KiB | 14.6 s |
| soil_heat | 11.25° | 512 | 600 | 1953 | 11925 | 0.138 | 112 | 443.13 KiB | 14.6 s |
| soil_heat | 5.62° | 2048 | 600 | 488 | 6839 | 0.24 | 256 | 1.72 MiB | 14.7 s |
| soil_heat | 3.75° | 4608 | 600 | 217 | 1506 | 1.09 | 127 | 3.85 MiB | 1.1 s |
| soil_heat | 2.81° | 8192 | 600 | 122 | 862 | 1.9 | 129 | 6.85 MiB | 15.0 s |
| soil_heat | 1.88° | 18432 | 600 | 54 | 443 | 3.71 | 149 | 15.40 MiB | 15.7 s |
| soil_heat | 1.25° | 41472 | 600 | 24 | 152 | 10.8 | 115 | 34.65 MiB | 16.7 s |
| soil_heat | 0.94° | 73728 | 600 | 20 | 85 | 19.3 | 114 | 61.60 MiB | 18.4 s |
| soil_heat | 0.62° | 165888 | 600 | 20 | 42 | 39.5 | 126 | 138.59 MiB | 23.2 s |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | 20 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | 30 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | 60 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | 100 | — | — | — | — | — | — | failed: MethodError |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |
| land | Float64 | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |
| land | Heun | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |

## Architecture: `reactant-gpu`

Created for Terrarium.jl v0.1.4 on Sun, 09 Aug 2026 20:11:03 in `default` mode (1x time steps, 1 thread(s)).

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
- CUDACore: 6.2.1
- GPUArrays: 11.5.9
- GPUCompiler: 1.23.0
- KernelAbstractions: 0.9.42
- CUDA_Driver_jll: 13.3.0+1
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

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | — | — | — | — | — | — | — | failed: MethodError |
| land_no_vegetation | 3.75° | 4608 | — | — | — | — | — | — | — | failed: ErrorException |
| soil_heat | 3.75° | 4608 | 600 | 217 | 65941 | 0.0249 | 5550 | 3.85 MiB | 31.5 s | ok |

### Land model, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 22.50° | 128 | — | — | — | — | — | — | failed: MethodError |
| land | 11.25° | 512 | — | — | — | — | — | — | failed: MethodError |
| land | 5.62° | 2048 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |
| land | 2.81° | 8192 | — | — | — | — | — | — | failed: ErrorException |
| land | 1.88° | 18432 | — | — | — | — | — | — | failed: ErrorException |
| land | 1.25° | 41472 | — | — | — | — | — | — | failed: ErrorException |
| land | 0.94° | 73728 | — | — | — | — | — | — | failed: ErrorException |
| land | 0.62° | 165888 | — | — | — | — | — | — | failed: ErrorException |

### Land model without vegetation, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_no_vegetation | 22.50° | 128 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 11.25° | 512 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 5.62° | 2048 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 3.75° | 4608 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 2.81° | 8192 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 1.88° | 18432 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 1.25° | 41472 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 0.94° | 73728 | — | — | — | — | — | — | failed: ErrorException |
| land_no_vegetation | 0.62° | 165888 | — | — | — | — | — | — | failed: ErrorException |

### Soil heat conduction, horizontal resolution

| Configuration | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soil_heat | 22.50° | 128 | 600 | 2000 | 110785 | 0.0148 | 259 | 114.63 KiB | 14.4 s |
| soil_heat | 11.25° | 512 | 600 | 1953 | 111800 | 0.0147 | 1050 | 443.13 KiB | 14.4 s |
| soil_heat | 5.62° | 2048 | 600 | 488 | 94046 | 0.0175 | 3520 | 1.72 MiB | 14.6 s |
| soil_heat | 3.75° | 4608 | 600 | 217 | 64078 | 0.0256 | 5390 | 3.85 MiB | 1.1 s |
| soil_heat | 2.81° | 8192 | 600 | 122 | 41518 | 0.0396 | 6210 | 6.85 MiB | 14.9 s |
| soil_heat | 1.88° | 18432 | 600 | 54 | 20687 | 0.0794 | 6960 | 15.40 MiB | 15.5 s |
| soil_heat | 1.25° | 41472 | 600 | 24 | 9516 | 0.173 | 7210 | 34.65 MiB | 16.6 s |
| soil_heat | 0.94° | 73728 | 600 | 20 | 8047 | 0.204 | 10800 | 61.60 MiB | 18.4 s |
| soil_heat | 0.62° | 165888 | 600 | 20 | 4342 | 0.378 | 13200 | 138.59 MiB | 23.1 s |

### Number of soil layers

| Configuration | Res | Columns | L | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | 3.75° | 4608 | 10 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | 20 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | 30 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | 60 | — | — | — | — | — | — | failed: MethodError |
| land | 3.75° | 4608 | 100 | — | — | — | — | — | — | failed: MethodError |

### Number format, Float32 vs Float64

| Configuration | NF | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | Float32 | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |
| land | Float64 | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |

### Time stepper

| Configuration | Variant | Res | Columns | Δt | Steps | SYPD | ms/step | Mcell-steps/s | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land | ForwardEuler | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |
| land | Heun | 3.75° | 4608 | — | — | — | — | — | — | failed: MethodError |

