# Benchmark suite for Terrarium

> Status: **in progress**. Harness, model configuration registry, results store, README generator and
> documentation page are implemented and verified end to end on `cpu-arm`; the Reactant path of the
> harness is verified too (it compiles and times `:soil_heat`, and reports the land configurations'
> compile failures rather than dying). The remaining architectures — `cpu-x86`, `gpu-nvidia`,
> `reactant-cpu`, `reactant-gpu` — still need a full run on their respective machines.

Date of initial draft: 2026-08-03

Base revision: e015a29db9d97090edfd112a5eea165640134404

## Originating prompt

> We want to set up a benchmark suite for Terrarium.
> Inspect SpeedyWeather.jl's benchmark for an example, you can find them in the repository in the
> benchmark folder.
>
> * We need to run on ARM (normal), ARM (Reactant), x86 (normal), x86 (Reactant), GPU (normal),
>   GPU (Reactant)
> * Reactant needs to use `Reactant.set_default_backend("gpu")`
> * We need a list of model configurations to run. They should take in the number of gridpoints /
>   resolution
> * We want a similar documentation upload of the benchmarks

> Also introduce a modifier to run the tests longer for a more robust test, or short for a quick
> benchmark

## Revision log

- 2026-08-03: initial draft.
- 2026-08-03: model configuration list narrowed (in review) to the coupled `LandModel` with Richards
  hydrology and snow, plus a no-vegetation variant; `:soil_heat` retained only as a Reactant
  reference configuration. Resolution sweep extended to `nlat_half = 144` (~166k columns) on a
  synthetic all-land mask. The stale `test/benchmarks/gpu/` scripts are left untouched.
- 2026-08-03: two pre-existing defects had to be worked around in the benchmark configurations
  before the coupled land model could be benchmarked at all; both are described under
  [Defects found while building this](#defects-found-while-building-this) and both are worked around
  in `benchmark/model_configurations.jl` rather than fixed in `src/` (out of scope here).
- 2026-08-03: `:bench500` benchmarks `ForwardEuler` against `Heun` only. `IMEX` needs an implicit
  sub-stepper and Terrarium has no concrete one yet — nothing implements
  `timestepping(…) == Implicit()`.
- 2026-08-09: the shared `benchmark_texture`/`benchmark_soil`/`benchmark_vegetation` helpers were
  dropped; each `build_model` method now spells out its own soil, texture and vegetation, and no
  longer passes values that are already component defaults. Fixed the vegetation process name in
  `:land` (`VegetationCarbon` → `VegetationCarbonCycle`), which had made every `:land` run fail with
  an `UndefVarError` on all architectures.
- 2026-08-09: dropped the yr⁻¹→s⁻¹ rescaling of the PALADYN rates from `:land` entirely.
  `compute_Λ_loc` converts `γL`/`γR`/`γS` upstream since `903530eb`, so the benchmark was applying
  that conversion a second time; `γv_min` is left at its default too. See defect 1 below.
- 2026-10-07 (revision 6): Reactant CPU results are labelled by host CPU, `reactant-cpu-arm` or
  `reactant-cpu-x86` (like `cpu-arm`/`cpu-x86`), so runs on different machines no longer overwrite each
  other; the stored EPYC entry was renamed to `reactant-cpu-x86`. Under Reactant the land
  configurations use the fixed-iteration `NewtonSolver` for the skin temperature, because the default
  `RootSolver` cannot be raised to StableHLO; the other architectures keep the default solver. The
  land configurations also needed the horizontal grid metrics of `ColumnRingGrid` to be forwarded to
  its `RectilinearGrid` (fixed in `src/` on its own branch, `mg/fix-columnringgrid-metrics`).

## Problem description

Terrarium has no reproducible performance benchmark. The only existing artefact,
`test/benchmarks/gpu/soil_heat_hydrology_global.jl`, is a one-off scaling script written against the
pre-`ModelIntegrator` API (`StateVariables(model)`, `timestep!(state, Δt)`) and no longer runs.

At the same time Terrarium has four execution paths that must each stay fast — plain CPU, CUDA GPU,
Reactant/XLA on CPU, Reactant/XLA on GPU — across two CPU ISAs (ARM and x86), i.e. six benchmark
targets. There is currently no way to state what any of them costs, no baseline against which a
performance regression could be noticed, and no published numbers for users deciding which backend
to run on.

## Background

SpeedyWeather.jl solves the same problem in `SpeedyWeather/benchmark`:

- `manual_benchmarking.jl <arch>` runs the suites for one architecture,
- merges the results into a shared `assets/benchmark_results.json` keyed by architecture label,
- regenerates `README.md` from the *whole* store, so a run on one machine never clobbers another's,
- and `docs/generate_benchmarks_page.jl` renders the same JSON into a `Benchmarks` documentation
  page with cross-architecture comparison figures at doc-build time.

That design is adopted here essentially unchanged, because the awkward part of this task — six
architectures that can never be measured on one machine at one time — is exactly what the JSON store
plus per-architecture regeneration solves.

## Summary of changes

### `benchmark/` (new)

| File | Purpose |
| --- | --- |
| `Project.toml` | Benchmark environment; `Terrarium` via `[sources]` path `..` |
| `model_configurations.jl` | `Val`-dispatched registry of resolution-parametrized model configurations |
| `benchmark_suite.jl` | `BenchmarkSuite` type, timing protocol, markdown writers |
| `define_benchmarks.jl` | The suites: which configurations at which resolutions |
| `manual_benchmarking.jl` | Driver: architecture/mode parsing, backend loading, JSON merge, README regeneration |
| `assets/benchmark_results.json` | Committed results store, one record per architecture |
| `README.md` | Generated: overview table + one section per architecture |

### Architecture matrix

Selected by the first CLI argument. The label is the JSON key, the README section title and the
documentation table column.

| Argument | Label | Architecture | Backends loaded |
| --- | --- | --- | --- |
| *(none)* / `cpu` | `cpu-arm` / `cpu-x86` (from `Sys.ARCH`) | `CPU()` | — |
| `gpu` | `gpu-nvidia` | `GPU()` | `CUDA` |
| `reactant-cpu` | `reactant-cpu` | `ReactantState()` | `CUDA`, `Reactant` + `set_default_backend("cpu")` |
| `reactant-gpu` | `reactant-gpu` | `ReactantState()` | `CUDA`, `Reactant` + `set_default_backend("gpu")` |

`CUDA` is loaded for *both* Reactant variants: it provides the KernelAbstractions↔Reactant glue and
is required even on the CPU backend. All backend packages are loaded before `using Terrarium` so the
package extensions register.

### Duration modifier

Second CLI argument, so one suite serves both a smoke check and a publication run:

```
julia --project=. manual_benchmarking.jl gpu quick   # 0.25x steps, sweep capped at 8192 columns
julia --project=. manual_benchmarking.jl gpu         # default, each timed run a few seconds
julia --project=. manual_benchmarking.jl gpu long    # 10x steps, full sweep
julia --project=. manual_benchmarking.jl gpu 3.5     # explicit timestep multiplier, full sweep
```

`BenchmarkMode` carries `timestep_multiplier` and `max_npoints`. The mode is recorded in the JSON
`meta` and printed in the README and documentation heading for each architecture: SYPD is a rate and
so is comparable across modes, but the noise level is not, and the two must not be silently mixed.

### Model configurations

`build_model(::Val{:name}, arch, NF; nlat_half, nz)` returns
`(; model, boundary_conditions, initializers, Δt)`, following the registry pattern already used by
`test/reactant/setup.jl`. `arch`, `nlat_half` and `nz` are the only things that vary between runs.

Horizontal resolution is `RingGrids.FullGaussianGrid(nlat_half)` with the default all-land mask
(`npoints = 8·nlat_half²`), so no asset download is needed on any benchmark machine — the same
choice `test/reactant/setup.jl` makes for CI. Vertical resolution is `ExponentialSpacing(N = nz)`.

- `:land` — fully coupled `LandModel`: `VegetationCarbon`, `SoilEnergyWaterCarbon` with
  `SoilHydrology(NF, RichardsEq())`, `SingleLayerSnow`, and the default surface energy balance,
  surface hydrology and prescribed atmosphere. `Δt = 600 s`; the coupled model raises a `DomainError`
  in the turbulent-flux thermodynamics for Δt ≳ 1800 s.
- `:land_no_vegetation` — identical but `vegetation = nothing`, which also selects
  `BareGroundEvaporation` and `NoCanopyInterception`. The pair isolates the cost of the
  vegetation/canopy path.
- `:soil_heat` — the minimal heat-conduction `SoilModel` from `test/reactant/setup.jl`. This is a
  *reference* configuration, not part of the headline comparison: it is the only configuration
  currently proven to compile under Reactant, so an empty Reactant column can be attributed to the
  land physics rather than to the harness.

### Suites

| Key | Title | Sweep |
| --- | --- | --- |
| `:bench100` | Model configurations, default resolution | all configurations, `nlat_half = 24`, `nz = 30` |
| `:bench200` | Land model, horizontal resolution | `:land`, `nlat_half ∈ [4, 8, 16, 24, 32, 48, 72, 96, 144]` — the cross-architecture overview |
| `:bench201` | Land model without vegetation, horizontal resolution | `:land_no_vegetation`, same sweep |
| `:bench300` | Vertical resolution | `:land`, `nz ∈ [10, 20, 30, 60, 100]` |
| `:bench400` | Number format | `:land`, `Float32` vs `Float64` |
| `:bench500` | Time stepper | `:land`, `ForwardEuler` / `Heun` / `IMEX` |

Configurations above the mode's `max_npoints` are skipped, logged, and rendered as `—`. Nothing is
dropped silently.

### Timing protocol

Per configuration: build and `initialize` (timed separately, reported as *init*, excluded from
SYPD); compute state memory by summing `length(parent(f)) · sizeof(eltype(f))` over the model's
state and tendency fields (`Base.summarysize` is not used — it undercounts device arrays by orders
of magnitude); warm-up run; re-`initialize!`; timed run with `time()` around it and an architecture
synchronization before the clock is stopped.

- CPU: no synchronization needed.
- GPU: `CUDA.synchronize()`.
- Reactant: `run!` recompiles on every call (its `compiled_run!` argument is never handed back to the
  caller), so the harness compiles once itself, exactly as `TerrariumReactantExt` does —
  `@compile raise = true raise_first = true sync = true run_timesteps!(integrator, Δt, nsteps, false)` —
  reports the compile time as its own metric, and then calls the compiled function twice (warm-up,
  timed). `sync = true` makes the timed call synchronous.

Step counts follow `nsteps = multiplier · clamp(round(Int, C / (npoints·nz)), 20, 2000)`, with `C`
chosen so a mid-size CPU configuration takes a few seconds.

Any failure of a single configuration is caught, logged with the exception, and recorded as `—`, so
one non-compiling configuration cannot cost an entire architecture its results.

### Metrics

SYPD (simulated years per wallclock day, `Δt·nsteps / (elapsed·365.25)` — the same definition
SpeedyWeather uses, so the two models' tables are directly comparable), ms/step, throughput in
cell-steps per second, state memory, initialization time, and (Reactant only) compile time.

### Documentation

`docs/generate_benchmarks_page.jl` is included from `docs/make.jl` before `makedocs` and writes
`docs/src/benchmarks.md` from the JSON store, together with CairoMakie figures (SYPD vs number of
columns, log-log, one line per architecture, one figure per configuration, with reference lines at
5°/2°/1°/0.5°). Section headings are suffixed with the architecture label so Documenter's `@ref`
resolver cannot confuse them with other pages. A missing or empty JSON store yields a placeholder
page, so the documentation still builds on a fresh checkout. `docs/Project.toml` gains `JSON3`;
`docs/make.jl` gains a top-level `"Benchmarks"` page.

## Defects found while building this

Neither is caused by the benchmark suite; both block the coupled land model outright and are worked
around in `benchmark/model_configurations.jl`, with the workaround documented at the point of use.
Both deserve their own fix (and their own tests) in a separate PR.

### 1. Vegetation turnover rates are in yr⁻¹ but integrated in seconds

`PALADYNCarbonDynamics.γL`, `.γR`, `.γS` (`src/processes/vegetation/dynamics/carbon_dynamics.jl`) and
`PALADYNVegetationDynamics.γv_min` (`…/vegetation_dynamics.jl`) are declared with `units = u"yr^-1"`
but are used unconverted in tendencies that the time steppers integrate in seconds. The litterfall
term is `Λ_loc = (γL/SLA + γR/SLA + γS·awl)·LAI_b = 0.16·LAI_b`, and with `LAI_b = C_veg/2.2` this
makes the carbon pool decay at `0.0727 s⁻¹` — an e-folding time of 14 seconds instead of 14 years.

Consequence: with an explicit time stepper the pool oscillates with a growth factor of `1 + λΔt`,
i.e. −3.4 per step at Δt = 60 s. Measured `carbon_vegetation` for the coupled model at Δt = 60 s:

```
0.1 → −0.376 → 1.28 → −4.29 → 14.4 → −48.6 → 163 → −549 → (abort)
```

The model then aborts inside a kernel on `@assert isfinite(β) && 0 <= β <= 1` in
`compute_stomatal_conductance`. Stability would require Δt ≲ 25 s. The unit tests do not catch this
because `test/coupled_models/land_model_tests.jl` takes a single time step. (`γv_min` already carries
a `# TODO this parameter is yearly` comment.)

Benchmark workaround (2026-08-03, since removed): the `:land` configuration rescaled the four rates
by `1/(365.25·24·3600)`.

Superseded 2026-08-09. `compute_Λ_loc` converts `γL`, `γR` and `γS` to s⁻¹ itself as of `903530eb`
(2026-08-07), so rescaling them in the benchmark had become a *double* conversion — litterfall was
effectively switched off. `:land` now takes the `PALADYNCarbonDynamics` and
`PALADYNVegetationDynamics` defaults and rescales nothing. `compute_γv` still returns `γv_min`
unconverted, so the yr⁻¹/s⁻¹ mismatch remains open for that one rate; it damps the vegetation
fraction on a ~500 s timescale but does not abort the run, because `ν_star` floors at `ν_seed` and
the tendency degenerates to a linear drift rather than a growing oscillation.

### 2. The default soil texture makes the vegetation soil-moisture factor non-finite

`SoilTexture` defaults to pure sand (`clay = 0`). For `SoilHydraulicsSURFEX` — the default hydraulics
— both `field_capacity = 0.089·(100·clay)^0.35` and `wilting_point = 0.03713·√(100·clay)` are then
zero, so the plant available water `(θ − θ_wp)/(θ_fc − θ_wp)` is `0/0`. `NaN` survives the
`max(min(1, ·), 0)` clamp, β is `NaN`, and the same assertion aborts the run — during initialization
this time. So `LandModel(grid)` with default vegetation and default soil cannot be initialized.

Benchmark workaround: the land configurations prescribe a loam (`sand = 0.4, clay = 0.2`, so
`silt = 0.4`).

A fix could give the SURFEX hydraulics a floor on `θ_fc − θ_wp`, or default `SoilTexture` to a
non-degenerate loam. Either way the assertion should also move out of the kernel: a reachable throw
path in a kernel is banned by `AGENTS.md` because it cannot be raised under Reactant.

### 3. Residual: vegetation carbon drifts negative under the default forcing

With the rates rescaled the coupled model integrates stably for at least 100 steps at Δt = 600 s, but
`carbon_vegetation` falls linearly (roughly −1.8·10⁻⁴ kg m⁻² s⁻¹) under the default constant
atmospheric forcing, i.e. autotrophic respiration greatly exceeds assimilation, and turns negative
within a day of simulated time. This does not affect the benchmark (the work per time step is
unchanged and no branch depends on the sign), but it suggests a further calibration or unit problem
in the NPP path that is worth a look.

## Testing and verification

Benchmarks are not part of CI (they are wallclock measurements on dedicated hardware). Verification
is manual:

1. `julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'`
2. `cd benchmark && julia --project=. manual_benchmarking.jl cpu quick` — populates the `cpu-arm`
   (or `cpu-x86`) section of `benchmark/README.md` and writes `assets/benchmark_results.json`.
3. `julia --project=. manual_benchmarking.jl reactant-cpu quick` — exercises the compile path.
4. `julia --project=docs docs/make.jl --draft` — renders `docs/src/benchmarks.md` and its figures.
5. Running twice must change only the record for the architecture just measured.

## Documentation changes

- New generated page `docs/src/benchmarks.md`, registered as a top-level `Benchmarks` entry.
- New `benchmark/README.md` (generated) describing the metric definitions and how to reproduce.
- `AGENTS.md` gains a short `## Benchmarks` section: how to run the suite, and the convention that
  the committed README numbers are the regression baseline (±20% is noise; larger deviations should
  be reported).

## Known limitations

- **Neither land configuration compiles under Reactant today** (measured 2026-08-03, `reactant-cpu`,
  128 columns × 10 layers):

  | Configuration | Reactant CPU |
  | --- | --- |
  | `:soil_heat` | compiles in 24.6 s, then 0.011 ms/step (149,000 SYPD) |
  | `:land_no_vegetation` | `InvalidIRError` compiling `gpu_saturation_to_pressure_kernel!` (the soil hydrology saturation→pressure closure) |
  | `:land` | `MethodError: no method matching exp(::Reactant.TracedRArray{Float32, 3})` — the static root-fraction path |

  Both failures are in Terrarium's physics kernels, not in the harness: the same harness compiles and
  times `:soil_heat` on the same run. The `:land` failure is the known root-fraction gap
  (`FunctionField` `sum(dims = 3)` in `StaticExponentialRootDistribution` evaluating the closure on the
  whole traced array). The `:land_no_vegetation` failure is new information — the Richards-equation
  hydrology closure is a Reactant blocker independently of vegetation. The in-kernel `@assert` in
  `compute_stomatal_conductance` is a third one, which vegetation configurations would hit after the
  root-fraction issue is fixed.

  This is why `:soil_heat` is in the registry: it is what distinguishes "Reactant cannot compile this
  physics yet" from "the benchmark harness is broken".
- **Two workarounds carried in the benchmark configurations** for the defects described above. They
  must be removed once those are fixed upstream; neither affects the measured cost per time step.
- **Single-threaded CPU numbers.** Thread scaling is not swept; runs inherit whatever
  `JULIA_NUM_THREADS` the machine provides, and the value is recorded in the machine info block.
- **Synthetic all-land mask.** Column counts are ~3x a real land-sea mask at the same `nlat_half`.
  This is deliberate (reproducibility, no asset download) but means absolute SYPD is pessimistic
  relative to a realistic land simulation.
- **Timings are means, not minima.** As in SpeedyWeather, a run of several seconds is timed once;
  ±20% variation between repetitions is normal. The `long` mode exists for when that is not good
  enough.

## Future work

- `test/benchmarks/gpu/soil_heat_hydrology_global.jl` and its `.sbatch` companion are written
  against the pre-`ModelIntegrator` API and no longer run. They are left untouched by this change;
  they should either be ported to the new harness or removed in a separate PR.
- Automated regression checking (compare against the committed JSON in CI on a fixed runner).
- A thread-scaling suite for the CPU architectures.
- Per-kernel benchmarks (analogous to SpeedyWeather's individual dynamics functions suite) once the
  full-model numbers are stable.