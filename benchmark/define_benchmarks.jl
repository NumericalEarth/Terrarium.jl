# The benchmark suites. Keys are sorted for a deterministic README/documentation section order.
#
# `:bench200` is the headline sweep: its results feed the cross-architecture overview table and the
# comparison figures in the documentation. Keep that in mind before changing it — the stored results
# of other architectures were collected with the resolutions defined here.

benchmarks = Dict{Symbol, BenchmarkSuite}()

## Default resolution used wherever a suite varies something other than resolution.
const DEFAULT_NLAT_HALF = 24        # 4608 columns, ~3.75°
const DEFAULT_NZ = 30

## Horizontal resolution sweep: 128 columns (~22°) up to 165,888 columns (~0.6°).
const RESOLUTION_SWEEP = [4, 8, 16, 24, 32, 48, 72, 96, 144]

## Compile-order check, first suite of the run: each configuration on a small grid that no other suite
## uses. Natively, the first model stepped in a Julia process can be several times slower than later
## models with the same types (see docs/dev/2026-10/2026-10-17_PLAN_native_vs_reactant_performance_gap.md,
## D7). Running these first absorbs that slowdown, so that it does not distort the suites below, and
## `:bench999` repeats them at the end of the run to report its size. Not run under Reactant, where it
## would only add XLA compilations.
const COMPILE_ORDER_NLAT_HALF = 2   # 32 columns
const COMPILE_ORDER_TIMESTEP_FACTOR = 0.1

benchmarks[:bench000] = BenchmarkSuite(
    title = "Compile-order check: first models stepped in the process",
    nruns = length(CONFIGURATIONS),
    config = collect(CONFIGURATIONS),
    nlat_half = fill(COMPILE_ORDER_NLAT_HALF, length(CONFIGURATIONS)),
    timestep_multiplier = COMPILE_ORDER_TIMESTEP_FACTOR,
    native_only = true,
)

## All configurations at one resolution — the "what does each model cost" table.
benchmarks[:bench100] = BenchmarkSuite(
    title = "Model configurations, default resolution",
    nruns = length(CONFIGURATIONS),
    config = collect(CONFIGURATIONS),
)

## Headline sweep: the fully coupled land model across horizontal resolutions.
benchmarks[:bench200] = BenchmarkSuite(
    title = "Land model, horizontal resolution",
    nruns = length(RESOLUTION_SWEEP),
    config = fill(:land, length(RESOLUTION_SWEEP)),
    nlat_half = copy(RESOLUTION_SWEEP),
)

## Same sweep without vegetation, to isolate the cost of the canopy path.
benchmarks[:bench201] = BenchmarkSuite(
    title = "Land model without vegetation, horizontal resolution",
    nruns = length(RESOLUTION_SWEEP),
    config = fill(:land_no_vegetation, length(RESOLUTION_SWEEP)),
    nlat_half = copy(RESOLUTION_SWEEP),
)

## Soil heat conduction only: the Reactant reference configuration (see model_configurations.jl).
benchmarks[:bench202] = BenchmarkSuite(
    title = "Soil heat conduction, horizontal resolution",
    nruns = length(RESOLUTION_SWEEP),
    config = fill(:soil_heat, length(RESOLUTION_SWEEP)),
    nlat_half = copy(RESOLUTION_SWEEP),
)

## Vertical resolution at fixed horizontal resolution.
let nz = [10, 20, 30, 60, 100]
    benchmarks[:bench300] = BenchmarkSuite(
        title = "Number of soil layers",
        nruns = length(nz),
        nz = copy(nz),
    )
end

## Number format.
benchmarks[:bench400] = BenchmarkSuite(
    title = "Number format, Float32 vs Float64",
    nruns = 2,
    NF = [Float32, Float64],
)

## Time steppers. `IMEX` is not benchmarked: it needs an implicit sub-stepper and Terrarium does not
## yet provide a concrete one (`timestepping(…) == Implicit()` has no implementation).
benchmarks[:bench500] = BenchmarkSuite(
    title = "Time stepper",
    nruns = 2,
    model_kwargs = [(timestepper = ForwardEuler,), (timestepper = Heun,)],
    label = ["ForwardEuler", "Heun"],
)

## Compile-order check, last suite of the run: the same models as `:bench000`, built anew at the end.
benchmarks[:bench999] = BenchmarkSuite(
    title = "Compile-order check: the same models at the end of the run",
    nruns = length(CONFIGURATIONS),
    config = collect(CONFIGURATIONS),
    nlat_half = fill(COMPILE_ORDER_NLAT_HALF, length(CONFIGURATIONS)),
    timestep_multiplier = COMPILE_ORDER_TIMESTEP_FACTOR,
    native_only = true,
)
