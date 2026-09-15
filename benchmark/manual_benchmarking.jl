#=
Run the Terrarium benchmark suite on one architecture and merge the result into a shared `README.md`
without overwriting the results of the other architectures.

Run `julia --project=. manual_benchmarking.jl --help` for the argument documentation.

Results are merged into `assets/benchmark_results.json`, keyed by architecture label. `README.md` is
then regenerated from the whole store, so a run on one machine never clobbers another's numbers. The
documentation page `docs/src/benchmarks.md` is generated from the same JSON at doc-build time; see
`docs/generate_benchmarks_page.jl`.
=#

import Pkg
cd(@__DIR__)
Pkg.activate(".")

using ArgParse

# Argument parsing happens before any backend package is loaded, so that `--help` is cheap and so that
# only the packages needed for the requested architecture are brought in.
const ARCH_CHOICES = ["cpu", "gpu", "reactant-cpu", "reactant-gpu"]
const MODE_CHOICES = ["quick", "default", "long"]

# `parse_mode` (benchmark_suite.jl) does the real parsing, but it only runs after the backend packages
# are loaded, so the mode is validated here to fail fast on a typo.
function valid_mode(arg::AbstractString)
    a = lowercase(strip(arg))
    a in MODE_CHOICES && return true
    multiplier = tryparse(Float64, a)
    return !isnothing(multiplier) && multiplier > 0
end

settings = ArgParseSettings(;
    prog = "manual_benchmarking.jl",
    description = "Run the Terrarium benchmark suite on one architecture and merge the result into " *
        "`assets/benchmark_results.json` and `README.md`.",
)
@add_arg_table! settings begin
    "arch"
    help = "Architecture to benchmark, one of: " * join(ARCH_CHOICES, ", ")
    arg_type = String
    default = "cpu"
    range_tester = arg -> lowercase(arg) in ARCH_CHOICES
    "mode"
    help = "Duration modifier: `quick` (0.25x steps, sweeps capped), `default`, `long` (10x steps), " *
        "or a positive numeric timestep multiplier"
    arg_type = String
    default = "default"
    range_tester = valid_mode
    "--suites"
    help = "Comma-separated suite keys to run (e.g. `bench200,bench202`). Default: all suites. " *
        "A partial run replaces this architecture's stored results with just these suites."
    arg_type = String
    default = ""
end
const PARSED_ARGS = parse_args(ARGS, settings)

const ARCH_ARG = lowercase(PARSED_ARGS["arch"])
const MODE_ARG = PARSED_ARGS["mode"]
const SUITES_ARG = PARSED_ARGS["suites"]

# Backend packages must be loaded BEFORE `using Terrarium` so its extensions register.
if ARCH_ARG == "gpu" || ARCH_ARG == "reactant-cpu" || ARCH_ARG == "reactant-gpu"
    using CUDA
end
if ARCH_ARG == "reactant-cpu" || ARCH_ARG == "reactant-gpu"
    using Reactant
    Reactant.set_default_backend(ARCH_ARG == "reactant-gpu" ? "gpu" : "cpu")
end

using Terrarium
using Dates, InteractiveUtils, JSON3, Printf

# `arg` is one of `ARCH_CHOICES`; ArgParse has already rejected anything else.
function pick_architecture(arg::AbstractString)
    if arg == "gpu"
        CUDA.functional() || error("No functional CUDA device found for architecture `gpu`.")
        return (GPU(), "gpu-nvidia")
    elseif arg == "reactant-cpu"
        return (ReactantState(), "reactant-cpu")
    elseif arg == "reactant-gpu"
        return (ReactantState(), "reactant-gpu")
    else
        arch_str = String(Sys.ARCH)
        label = (startswith(arch_str, "aarch") || arch_str == "arm64") ? "cpu-arm" : "cpu-x86"
        return (CPU(), label)
    end
end

const ARCH, ARCH_LABEL = pick_architecture(ARCH_ARG)

include("model_configurations.jl")
include("benchmark_suite.jl")
include("define_benchmarks.jl")

const MODE = parse_mode(MODE_ARG)

# The `ReactantState` runner needs the `@compile` macro, so it is only parsed once Reactant is loaded.
if ARCH isa ReactantState
    include("reactant_runner.jl")
end

@info "Running Terrarium benchmarks for `$ARCH_LABEL` ($(typeof(ARCH))) in `$(MODE.name)` mode"

for suite in values(benchmarks)
    suite.architecture = ARCH
    suite.mode = MODE
end

# Deterministic suite order, so README sections are stable across runs. `--suites` narrows the run to
# a subset; because each run replaces this architecture's whole stored record, a partial run drops the
# suites it did not run — warn when that is the case.
const ALL_SUITE_KEYS = sort!(collect(keys(benchmarks)))
const SUITE_KEYS = if isempty(strip(SUITES_ARG))
    ALL_SUITE_KEYS
else
    requested = [Symbol(strip(s)) for s in split(SUITES_ARG, ',') if !isempty(strip(s))]
    unknown = setdiff(requested, collect(keys(benchmarks)))
    isempty(unknown) || error("Unknown suite(s): $(join(unknown, ", ")). Known suites: $(join(ALL_SUITE_KEYS, ", ")).")
    requested
end
SUITE_KEYS == ALL_SUITE_KEYS ||
    @warn "Running only $(join(SUITE_KEYS, ", ")): this replaces $ARCH_LABEL's stored results with just these suites."

for key in SUITE_KEYS
    @info "→ Suite $key: $(benchmarks[key].title)"
    run_benchmark_suite!(benchmarks[key])
end

# The resolution sweeps that make up the cross-architecture overview table and the documentation
# figures. One row per (configuration, resolution).
const OVERVIEW_SUITES = [:bench200, :bench201, :bench202]

function arch_markdown()
    io = IOBuffer()
    for key in SUITE_KEYS
        write_results(io, benchmarks[key])
    end
    return String(take!(io))
end

function machine_info()
    io = IOBuffer()
    write(io, "```julia\njulia> versioninfo()\n")
    versioninfo(io)
    write(io, "```\n")
    if ARCH isa GPU || ARCH_LABEL == "reactant-gpu"
        write(io, "\n```julia\njulia> CUDA.versioninfo()\n")
        try
            CUDA.versioninfo(io)
        catch err
            write(io, "(CUDA.versioninfo() failed: $err)\n")
        end
        write(io, "```\n")
    end
    if ARCH isa ReactantState
        backend = ARCH_LABEL == "reactant-gpu" ? "gpu" : "cpu"
        write(io, "\nReactant backend: `$backend` (selected with `Reactant.set_default_backend(\"$backend\")`).\n")
    end
    return String(take!(io))
end

"""
    git_revision()

Short HEAD SHA of the Terrarium checkout the benchmark runs from, suffixed `-dirty` when the working
tree has uncommitted changes. Returns `"unknown"` if git is unavailable or the directory is not a
repository, so a missing revision never aborts a benchmark run.
"""
function git_revision()
    repo = normpath(joinpath(@__DIR__, ".."))
    try
        sha = strip(read(`git -C $repo rev-parse --short HEAD`, String))
        dirty = !isempty(strip(read(`git -C $repo status --porcelain`, String)))
        return dirty ? sha * "-dirty" : sha
    catch err
        @warn "Could not determine the git revision of $repo" exception = err
        return "unknown"
    end
end

arch_record = Dict(
    "meta" => Dict(
        "terrarium_version" => string(pkgversion(Terrarium)),
        "git_revision" => git_revision(),
        "timestamp" => Dates.format(Dates.now(), Dates.RFC1123Format),
        "arch_type" => string(typeof(ARCH)),
        "mode" => MODE.name,
        "timestep_multiplier" => MODE.timestep_multiplier,
        "threads" => Threads.nthreads(),
        "machine_info" => machine_info(),
    ),
    "markdown" => arch_markdown(),
    "overview" => Dict(string(key) => overview_data(benchmarks[key]) for key in OVERVIEW_SUITES if key in SUITE_KEYS),
)

# --- merge into the JSON store -----------------------------------------------------------

const ASSETS_DIR = joinpath(@__DIR__, "assets")
mkpath(ASSETS_DIR)
const RESULTS_JSON = joinpath(ASSETS_DIR, "benchmark_results.json")

function load_results()
    isfile(RESULTS_JSON) || return Dict{String, Any}()
    try
        return Dict{String, Any}(JSON3.read(read(RESULTS_JSON, String), Dict{String, Any}))
    catch err
        @warn "Could not parse $RESULTS_JSON; starting fresh" exception = err
        return Dict{String, Any}()
    end
end

all_results = load_results()
all_results[ARCH_LABEL] = arch_record

open(RESULTS_JSON, "w") do io
    JSON3.pretty(io, all_results)
end
@info "Wrote results for $ARCH_LABEL → $RESULTS_JSON"

# --- regenerate README.md from the store -------------------------------------------------

const ARCH_ORDER = ["cpu-arm", "cpu-x86", "gpu-nvidia", "reactant-cpu", "reactant-gpu"]

function sorted_arch_labels(results)
    known = filter(in(keys(results)), ARCH_ORDER)
    extra = sort!([k for k in keys(results) if !(k in ARCH_ORDER)])
    return vcat(known, extra)
end

function write_preamble(md)
    write(md, "# Benchmarks\n\n")
    write(md, "Performance benchmarks for Terrarium.jl, collected across architectures. ")
    write(md, "Each architecture's results live in its own section below; the overview table compares the ")
    write(md, "resolution sweeps across all architectures benchmarked so far.\n\n")

    write(md, "Every configuration is run for a fixed number of time steps without output. ")
    write(md, "Initialization and — under Reactant — XLA compilation happen before the clock starts and are ")
    write(md, "reported separately; the timed section covers only the stepping loop. ")
    write(md, "A run takes a few seconds and is timed once, so it measures the mean rather than the minimum: ")
    write(md, "deviations of ±20% between repetitions are normal, and `quick` mode — which shortens every run — ")
    write(md, "is noisier still, so treat its numbers as a smoke test rather than a measurement. ")
    write(md, "Use the `long` mode for numbers worth quoting.\n\n")

    write(md, "### Explanation\n\n")
    write(md, "- Configuration: model setup, see `model_configurations.jl`\n")
    write(md, "- NF: number format, default `Float32`\n")
    write(md, "- Res: approximate horizontal resolution in degrees latitude\n")
    write(md, "- Columns: number of land columns; the benchmark grids are full Gaussian grids with an all-land mask\n")
    write(md, "- L: number of soil layers, default $DEFAULT_NZ\n")
    write(md, "- Δt: time step (s)\n")
    write(md, "- SYPD: simulated years per wallclock day\n")
    write(md, "- ms/step: wallclock milliseconds per time step\n")
    write(md, "- Mcell-steps/s: millions of (column × layer) updates per second — the resolution-independent throughput\n")
    write(md, "- Memory: state and tendency fields, including halos\n")
    write(md, "- Compile: one-off XLA compilation time of the stepping loop (Reactant only)\n")
    write(md, "- Status: `ok`, `skipped` (above the mode's resolution cap), `unstable` (state went non-finite) or `failed`\n\n")

    write(md, "### Running the benchmarks\n\n")
    write(md, "From `benchmark/`:\n\n")
    write(md, "```\n")
    write(md, "julia --project=. manual_benchmarking.jl                # CPU (auto-labelled cpu-arm or cpu-x86)\n")
    write(md, "julia --project=. manual_benchmarking.jl gpu            # CUDA GPU\n")
    write(md, "julia --project=. manual_benchmarking.jl reactant-cpu   # Reactant/XLA, CPU backend\n")
    write(md, "julia --project=. manual_benchmarking.jl reactant-gpu   # Reactant/XLA, CUDA backend\n")
    write(md, "```\n\n")
    write(md, "A second argument controls the duration: `quick` (0.25x steps, sweeps capped at 8192 columns), ")
    write(md, "`long` (10x steps), or a numeric timestep multiplier. ")
    write(md, "A third, optional `--suites bench200,bench202` restricts the run to specific suites (default: all); ")
    write(md, "a partial run replaces that architecture's stored results with just the suites it ran. ")
    write(md, "Each run records the Terrarium git revision it was produced at, shown in the section header below. ")
    write(md, "Each run updates only its own architecture's section here; the other architectures are preserved ")
    write(md, "in `assets/benchmark_results.json`.\n\n")
    return
end

function write_overview(md, all_results, labels)
    write(md, "## Overview: resolution sweeps across architectures\n\n")
    write(md, "Simulated years per wallclock day (SYPD) for each model configuration across horizontal ")
    write(md, "resolutions, one column per architecture. `—` means the architecture has not been benchmarked ")
    write(md, "yet, skipped that resolution, or failed to run that configuration (see the per-architecture ")
    write(md, "sections for which). Comparison figures are on the documentation's Benchmarks page.\n\n")

    ## Union of (config, ncolumns) rows over all architectures and overview suites.
    rows = Tuple{String, Int}[]
    for label in labels, key in string.(OVERVIEW_SUITES)
        ov = get(get(all_results[label], "overview", Dict()), key, nothing)
        ov === nothing && continue
        for i in eachindex(ov["config"])
            r = (String(ov["config"][i]), Int(ov["ncolumns"][i]))
            r in rows || push!(rows, r)
        end
    end
    sort!(rows; by = r -> (r[1], r[2]))

    write(md, "| Configuration | Columns | " * join(labels, " | ") * " |\n")
    write(md, "| --- | --- | " * join(fill("---", length(labels)), " | ") * " |\n")
    for (config, ncols) in rows
        cells = String[]
        for label in labels
            push!(cells, format_sypd(lookup_sypd(all_results[label], config, ncols)))
        end
        write(md, "| $config | $ncols | " * join(cells, " | ") * " |\n")
    end
    write(md, "\n")
    return
end

"""
    lookup_sypd(record, config, ncolumns)

SYPD of one (configuration, resolution) point in one architecture's record, or `nothing`.
"""
function lookup_sypd(record, config::AbstractString, ncols::Integer)
    overview = get(record, "overview", nothing)
    overview === nothing && return nothing
    for (_, ov) in overview
        for i in eachindex(ov["config"])
            if String(ov["config"][i]) == config && Int(ov["ncolumns"][i]) == ncols
                return ov["sypd"][i]
            end
        end
    end
    return nothing
end

function write_arch_section(md, label, record)
    meta = record["meta"]
    write(md, "## Architecture: `$label`\n\n")
    write(md, "Created for Terrarium.jl v$(meta["terrarium_version"]) on $(meta["timestamp"]) ")
    write(md, "in `$(meta["mode"])` mode ($(meta["timestep_multiplier"])x time steps, $(meta["threads"]) thread(s)), ")
    write(md, "at git revision `$(get(meta, "git_revision", "unknown"))`.\n\n")
    write(md, "### Machine details\n\n")
    write(md, meta["machine_info"])
    write(md, "\n")
    write(md, record["markdown"])
    write(md, "\n")
    return
end

const README_PATH = joinpath(@__DIR__, "README.md")
labels = sorted_arch_labels(all_results)

open(README_PATH, "w") do md
    write_preamble(md)
    write_overview(md, all_results, labels)
    for label in labels
        write_arch_section(md, label, all_results[label])
    end
end
@info "Regenerated $README_PATH for architectures: $(join(labels, ", "))"
