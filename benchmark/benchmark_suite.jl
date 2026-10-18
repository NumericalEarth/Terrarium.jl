# Benchmark harness: suite definition, timing protocol and markdown rendering.
#
# One `BenchmarkSuite` is a list of runs that share a table. Each run builds a model configuration
# (see `model_configurations.jl`) at a given resolution, times a fixed number of time steps, and
# records SYPD, per-step time, throughput, memory and (under Reactant) compile time.

using BenchmarkTools: prettymemory
using Printf: @sprintf
using Statistics: median

import KernelAbstractions
using Oceananigans.Architectures: device

# --- run length --------------------------------------------------------------------------

# Timestep budget in cell-steps: the number of time steps for a run is chosen so that a run touches
# roughly this many (column × layer) cells in total, clamped to a sane range. Calibrated so that a
# mid-size configuration (~1.4e5 cells) takes a few seconds on one CPU core; adjust here if the
# default mode drifts away from that on new hardware.
const TIMESTEP_BUDGET = 3.0e7
const MIN_TIMESTEPS = 20
const MAX_TIMESTEPS = 2000

"""
    n_timesteps(ncells, multiplier = 1)

Number of time steps for a timed run over `ncells` cells. `multiplier` scales the clamped result, so
a `quick` run shortens and a `long` run lengthens even the floored/ceilinged configurations.
"""
function n_timesteps(ncells::Integer, multiplier::Real = 1)
    nsteps = clamp(round(Int, TIMESTEP_BUDGET / ncells), MIN_TIMESTEPS, MAX_TIMESTEPS)
    return max(2, round(Int, multiplier * nsteps))
end

# --- benchmark mode ----------------------------------------------------------------------

"""
How long the suite should run. `timestep_multiplier` scales every timed run; `max_ncolumns` caps the
resolution sweeps (configurations above it are skipped and reported as `—`); `repeats` is the number of
timed repetitions of every run, of which the median is reported.
"""
struct BenchmarkMode
    name::String
    timestep_multiplier::Float64
    max_ncolumns::Int
    repeats::Int
end

const QUICK_MODE = BenchmarkMode("quick", 0.25, 8192, 2)      # nlat_half ≤ 32
const DEFAULT_MODE = BenchmarkMode("default", 1.0, typemax(Int), 5)
const LONG_MODE = BenchmarkMode("long", 10.0, typemax(Int), 5)

"""
    parse_mode(arg)

Parse the duration modifier: `quick`, `default`/empty, `long`, or a bare number used directly as the
timestep multiplier (with no resolution cap).
"""
function parse_mode(arg::AbstractString)
    a = lowercase(strip(arg))
    (isempty(a) || a == "default") && return DEFAULT_MODE
    a == "quick" && return QUICK_MODE
    a == "long" && return LONG_MODE
    multiplier = tryparse(Float64, a)
    isnothing(multiplier) && error("Unknown mode `$arg`. Use one of: quick, default, long, or a numeric multiplier.")
    multiplier > 0 || error("Timestep multiplier must be > 0, got $multiplier.")
    return BenchmarkMode(@sprintf("%gx", multiplier), multiplier, typemax(Int), DEFAULT_MODE.repeats)
end

# --- suite -------------------------------------------------------------------------------

Base.@kwdef mutable struct BenchmarkSuite
    "Table heading in the README and documentation page"
    title::String

    "Number of runs in this suite"
    nruns::Int = 1

    ## --- what to run (one entry per run) ---
    config::Vector{Symbol} = fill(:land, nruns)
    NF::Vector{DataType} = fill(Float32, nruns)
    nlat_half::Vector{Int} = fill(24, nruns)
    nz::Vector{Int} = fill(30, nruns)
    "Extra kwargs splatted into the model constructor; `Type`/`Function` values are called with `NF`"
    model_kwargs::Vector = fill(NamedTuple(), nruns)
    "Optional label for the varying quantity of this suite, e.g. the time stepper name"
    label::Vector{String} = fill("", nruns)
    "Extra factor on the mode's timestep multiplier, for suites that only need short runs"
    timestep_multiplier::Float64 = 1.0
    "Skip this suite under Reactant, where each configuration costs a long XLA compilation"
    native_only::Bool = false

    ## --- results (filled by `run_benchmark_suite!`) ---
    ncolumns::Vector{Int} = fill(0, nruns)
    Δt::Vector{Float64} = fill(NaN, nruns)
    nsteps::Vector{Int} = fill(0, nruns)
    sypd::Vector{Float64} = fill(NaN, nruns)
    "Median over the timed repetitions"
    ms_per_step::Vector{Float64} = fill(NaN, nruns)
    ms_per_step_min::Vector{Float64} = fill(NaN, nruns)
    ms_per_step_max::Vector{Float64} = fill(NaN, nruns)
    repeats::Vector{Int} = fill(0, nruns)
    "Throughput in million cell-steps per second"
    throughput::Vector{Float64} = fill(NaN, nruns)
    memory::Vector{Int} = fill(0, nruns)
    init_time::Vector{Float64} = fill(NaN, nruns)
    compile_time::Vector{Float64} = fill(NaN, nruns)
    "`ok`, `skipped`, `unstable`, or `failed: …`"
    status::Vector{String} = fill("", nruns)

    ## --- shared settings, set by the driver ---
    architecture::Any = CPU()
    mode::BenchmarkMode = DEFAULT_MODE
end

# --- device helpers ----------------------------------------------------------------------

"""
    device_synchronize(arch)

Wait for all pending device work so that a wallclock measurement covers it.
"""
device_synchronize(arch) = KernelAbstractions.synchronize(device(arch))

"Steps taken before the timed run, to move JIT compilation and first kernel launches out of it."
const WARMUP_STEPS = 3

"""
    build_runner(arch, integrator, Δt, nsteps)

Return `(; step!, compile_time)`, where `step!(n)` advances `integrator` by `n` steps of size `Δt`.
The harness warms up with `step!(WARMUP_STEPS)` and times `step!(nsteps)`. Both go through the same
function on purpose: when the warm-up and the timed run used two different closures, the first timed
repetition of every new grid size carried a one-off compilation (about 2 s on an H100 node), which
the warm-up had not absorbed. `reactant_runner.jl` adds the `ReactantState` method, which compiles the
stepping loop once for a fixed step count (so `n` is ignored and its warm-up is a full run) and
reports the compile time.
"""
function build_runner(arch, integrator, Δt, nsteps)
    step!(n) = run!(integrator; steps = n, Δt)
    return (step! = step!, compile_time = NaN)
end

# Recursive walk over the (possibly namespaced) NamedTuples returned by `get_fields`.
field_bytes(f::Field) = length(parent(f)) * sizeof(eltype(f))
field_bytes(nt::NamedTuple) = sum(field_bytes, values(nt); init = 0)
field_bytes(_) = 0

"""
    state_memory(integrator)

Total bytes of the model's state and tendency fields, including halos.

`Base.summarysize` is deliberately not used: it counts the wrapper of a device array rather than its
contents and undercounts GPU/Reactant states by orders of magnitude.
"""
function state_memory(integrator)
    state, model = integrator.state, integrator.model
    try
        return field_bytes(get_fields(state, model)) + field_bytes(Terrarium.tendency_fields(state, model))
    catch err
        @warn "Could not measure state memory; falling back to Base.summarysize" exception = err
        return Base.summarysize(state)
    end
end

all_finite(f::Field) = all(isfinite, Array(parent(f)))
all_finite(nt::NamedTuple) = all(all_finite, values(nt))
all_finite(_) = true

"""
    state_is_finite(integrator)

`true` if every prognostic field is finite. A configuration that blows up still produces a timing,
but the number describes a diverging simulation, so it is flagged as `unstable` in the tables.
"""
function state_is_finite(integrator)
    try
        return all_finite(Terrarium.prognostic_fields(integrator.state, integrator.model))
    catch err
        @warn "Could not check state for finiteness" exception = err
        return true
    end
end

# --- the timed run -----------------------------------------------------------------------

"""
    benchmark_configuration(config, arch, NF; nlat_half, nz, model_kwargs, multiplier, repeats)

Build, initialize, warm up and time one model configuration. Returns a NamedTuple of metrics.

Initialization and (under Reactant) compilation are measured separately and excluded from the
timing: the timed section covers only the stepping loop, which is what SYPD describes. The stepping
loop is timed `repeats` times, each from the re-initialized state; SYPD, ms/step, and throughput
describe the median repetition, and the fastest and slowest repetitions are returned as well.
"""
function benchmark_configuration(
        config::Symbol, arch, ::Type{NF};
        nlat_half::Integer, nz::Integer, model_kwargs = (;), multiplier::Real = 1, repeats::Integer = 1
    ) where {NF}

    cfg = build_model(config, arch, NF; nlat_half, nz, model_kwargs)
    Δt = NF(cfg.Δt)
    ncols = ncolumns(nlat_half)
    nsteps = n_timesteps(ncols * nz, multiplier)

    init_time = @elapsed integrator = initialize(
        cfg.model;
        boundary_conditions = cfg.boundary_conditions,
        initializers = cfg.initializers,
    )
    device_synchronize(arch)
    memory = state_memory(integrator)

    runner = build_runner(arch, integrator, cfg.Δt, nsteps)

    ## Warm up: JIT (and, on the GPU, the first kernel launches) must not land in the timed run.
    ## The warm-up starts from `initialize!` and calls the same stepping function as the timed run, so
    ## that it compiles everything the timed repetitions need (see `build_runner`).
    Terrarium.initialize!(integrator)
    device_synchronize(arch)
    runner.step!(min(nsteps, WARMUP_STEPS))
    device_synchronize(arch)

    ## Each repetition restarts from the initial state, so that every one starts where the warm-up did.
    ## This is a convenience, not a requirement (the amount of work per step does not depend on the
    ## state), so a failure is reported once and the timing still goes ahead.
    elapsed_times = Float64[]
    reinitialize_failed = false
    for _ in 1:repeats
        if !reinitialize_failed
            try
                Terrarium.initialize!(integrator)
                device_synchronize(arch)
            catch err
                @warn "Could not re-initialize before the timed run; timing continues from the current state" exception = err
                reinitialize_failed = true
            end
        end
        t0 = time()
        runner.step!(nsteps)
        device_synchronize(arch)
        push!(elapsed_times, time() - t0)
    end
    elapsed = median(elapsed_times)

    ## SYPD: simulated seconds per wallclock second, converted to years per day.
    sypd = Δt * nsteps / (elapsed * 365.25)
    ms_per_step = 1.0e3 * elapsed / nsteps
    ms_per_step_min = 1.0e3 * minimum(elapsed_times) / nsteps
    ms_per_step_max = 1.0e3 * maximum(elapsed_times) / nsteps
    throughput = ncols * nz * nsteps / elapsed / 1.0e6

    status = state_is_finite(integrator) ? "ok" : "unstable"
    return (;
        ncolumns = ncols, Δt, nsteps, sypd, ms_per_step, ms_per_step_min, ms_per_step_max, repeats,
        throughput, memory, init_time, compile_time = runner.compile_time, status,
    )
end

"""
    run_benchmark_suite!(suite)

Run every configuration in `suite`, filling in its result vectors. Configurations above the mode's
resolution cap are skipped, and a configuration that errors (e.g. fails to compile under Reactant)
is recorded as failed — neither loses the rest of the suite.
"""
function run_benchmark_suite!(suite::BenchmarkSuite)
    arch = suite.architecture
    for i in 1:suite.nruns
        config = suite.config[i]
        NF = suite.NF[i]
        nlat_half = suite.nlat_half[i]
        nz = suite.nz[i]
        ncols = ncolumns(nlat_half)
        suite.ncolumns[i] = ncols

        if suite.native_only && arch isa ReactantState
            @info "  ↳ skipping $config at nlat_half=$nlat_half: this suite is not run under Reactant"
            suite.status[i] = "skipped"
            continue
        end

        if ncols > suite.mode.max_ncolumns
            @info "  ↳ skipping $config at nlat_half=$nlat_half ($ncols columns): above the $(suite.mode.name) mode cap of $(suite.mode.max_ncolumns) columns"
            suite.status[i] = "skipped"
            continue
        end

        @info "  ↳ $config | $NF | nlat_half=$nlat_half ($ncols columns) | nz=$nz"
        try
            result = benchmark_configuration(
                config, arch, NF;
                nlat_half, nz,
                model_kwargs = suite.model_kwargs[i],
                multiplier = suite.mode.timestep_multiplier * suite.timestep_multiplier,
                repeats = suite.mode.repeats,
            )
            suite.Δt[i] = result.Δt
            suite.nsteps[i] = result.nsteps
            suite.sypd[i] = result.sypd
            suite.ms_per_step[i] = result.ms_per_step
            suite.ms_per_step_min[i] = result.ms_per_step_min
            suite.ms_per_step_max[i] = result.ms_per_step_max
            suite.repeats[i] = result.repeats
            suite.throughput[i] = result.throughput
            suite.memory[i] = result.memory
            suite.init_time[i] = result.init_time
            suite.compile_time[i] = result.compile_time
            suite.status[i] = result.status
            @info @sprintf("     %.3g SYPD | %.3g ms/step (median of %d, spread %s) | %.3g Mcell-steps/s | %s",
                result.sypd, result.ms_per_step, result.repeats, format_spread(result.ms_per_step_min, result.ms_per_step_max, result.ms_per_step),
                result.throughput, result.status)
        catch err
            @warn "Benchmark failed for $config at nlat_half=$nlat_half, nz=$nz on $(typeof(arch)); recording as failed" exception = (err, catch_backtrace())
            suite.status[i] = "failed: $(nameof(typeof(err)))"
        end
    end
    return suite
end

# --- rendering ---------------------------------------------------------------------------

"""
    format_sypd(s)

One decimal digit below 10, integer above, `—` for anything non-finite.
"""
function format_sypd(s)
    (s isa Number && isfinite(s)) || return "—"
    return s < 10 ? string(round(s; digits = 2)) : string(round(Int, s))
end

function format_metric(x, sigdigits = 3)
    (x isa Number && isfinite(x)) || return "—"
    rounded = round(x; sigdigits)
    return isinteger(rounded) ? string(round(Int, rounded)) : string(rounded)
end
format_integer(x) = (x isa Number && isfinite(x)) ? string(round(Int, x)) : "—"
format_seconds(x) = (x isa Number && isfinite(x)) ? @sprintf("%.1f s", x) : "—"
format_memory(bytes) = bytes > 0 ? prettymemory(bytes) : "—"

"""
    format_spread(min, max, median)

Spread of the timed repetitions, `(max - min) / median`, as a percentage; `—` if not available.
"""
function format_spread(lo, hi, mid)
    all(x -> x isa Number && isfinite(x), (lo, hi, mid)) && mid > 0 || return "—"
    return string(round(Int, 100 * (hi - lo) / mid), "%")
end

varies(v) = any(x -> x != v[1], v)

"""
    write_results(io, suite)

Write one markdown table for `suite`. Columns that do not vary within the suite are omitted (the
defaults are documented in the README preamble), which keeps each table down to what it is about.
"""
function write_results(io, suite::BenchmarkSuite)
    write(io, "\n### $(suite.title)\n\n")

    print_NF = varies(suite.NF)
    print_nz = varies(suite.nz)
    print_label = any(!isempty, suite.label)
    print_compile = any(isfinite, suite.compile_time)
    print_status = any(s -> s != "ok", suite.status)

    header = "| Configuration "
    header *= print_NF ? "| NF " : ""
    header *= print_label ? "| Variant " : ""
    header *= "| Res | Columns "
    header *= print_nz ? "| L " : ""
    header *= "| Δt | Steps | SYPD | ms/step | Spread | Mcell-steps/s | Memory "
    header *= print_compile ? "| Compile " : ""
    header *= print_status ? "| Status " : ""
    header *= "|"

    ncols_table = length(findall('|', header)) - 1
    write(io, header, "\n")
    write(io, repeat("| --- ", ncols_table), "|\n")

    for i in 1:suite.nruns
        row = "| $(suite.config[i]) "
        row *= print_NF ? "| $(suite.NF[i]) " : ""
        row *= print_label ? "| $(suite.label[i]) " : ""
        row *= @sprintf("| %.2f° | %d ", resolution_degrees(suite.nlat_half[i]), suite.ncolumns[i])
        row *= print_nz ? "| $(suite.nz[i]) " : ""
        row *= "| $(format_integer(suite.Δt[i])) "
        row *= "| $(suite.nsteps[i] > 0 ? string(suite.nsteps[i]) : "—") "
        row *= "| $(format_sypd(suite.sypd[i])) "
        row *= "| $(format_metric(suite.ms_per_step[i])) "
        row *= "| $(format_spread(suite.ms_per_step_min[i], suite.ms_per_step_max[i], suite.ms_per_step[i])) "
        row *= "| $(format_metric(suite.throughput[i])) "
        row *= "| $(format_memory(suite.memory[i])) "
        row *= print_compile ? "| $(format_seconds(suite.compile_time[i])) " : ""
        row *= print_status ? "| $(suite.status[i]) " : ""
        write(io, row, "|\n")
    end
    return
end

"""
    overview_data(suite)

Structured results of the headline resolution sweep, stored in the JSON so that the cross-
architecture overview table and the documentation figures can be regenerated without re-running.
"""
function overview_data(suite::BenchmarkSuite)
    ## JSON has no NaN/Inf literal — emit `null` for non-finite metrics (skipped or failed runs).
    json_safe(x) = (x isa Number && isfinite(x)) ? x : nothing
    return Dict(
        "config" => string.(suite.config),
        "nlat_half" => collect(suite.nlat_half),
        "ncolumns" => collect(suite.ncolumns),
        "nz" => collect(suite.nz),
        "sypd" => map(json_safe, suite.sypd),
        "ms_per_step" => map(json_safe, suite.ms_per_step),
        "ms_per_step_min" => map(json_safe, suite.ms_per_step_min),
        "ms_per_step_max" => map(json_safe, suite.ms_per_step_max),
        "repeats" => collect(suite.repeats),
        "throughput" => map(json_safe, suite.throughput),
        "memory" => map(json_safe, suite.memory),
        "compile_time" => map(json_safe, suite.compile_time),
        "status" => collect(suite.status),
    )
end
