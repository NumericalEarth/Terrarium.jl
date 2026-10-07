#=
Generate `docs/src/benchmarks.md` from the benchmark results JSON produced by
`benchmark/manual_benchmarking.jl`. Called from `docs/make.jl` before `makedocs`, so that the page is
an ordinary static markdown file by the time Documenter sees it.

The page is structured as:
  - introduction
  - one scaling figure per model configuration (SYPD vs number of columns, one line per architecture)
  - a cross-architecture overview table
  - one section per architecture, mirroring the markdown stored in the JSON

If the JSON is missing or empty, a placeholder page is written telling the reader how to populate it,
so that `make.jl` stays runnable on a fresh checkout.
=#

using CairoMakie
using JSON3
using Terrarium

const BENCHMARK_RESULTS_PATH = joinpath(pkgdir(Terrarium), "benchmark", "assets", "benchmark_results.json")
const DOCS_ASSETS_DIR = joinpath(@__DIR__, "src", "assets", "benchmarks")
const DOCS_PAGE_PATH = joinpath(@__DIR__, "src", "benchmarks.md")

## Stable architecture ordering — matches benchmark/manual_benchmarking.jl
const ARCH_ORDER = ["cpu-arm", "cpu-x86", "gpu-nvidia", "reactant-cpu-arm", "reactant-cpu-x86", "reactant-gpu"]

## Reference resolutions drawn as vertical lines in the scaling figures.
const REFERENCE_RESOLUTIONS = [(5.0, "5°"), (2.0, "2°"), (1.0, "1°"), (0.5, "0.5°")]

reference_ncolumns(degrees) = 8 * round(Int, 180 / (2 * degrees))^2

function sorted_arch_labels(results)
    known = filter(in(keys(results)), ARCH_ORDER)
    extra = sort!([k for k in keys(results) if !(k in ARCH_ORDER)])
    return vcat(known, extra)
end

function load_results()
    isfile(BENCHMARK_RESULTS_PATH) || return Dict{String, Any}()
    try
        return Dict{String, Any}(JSON3.read(read(BENCHMARK_RESULTS_PATH, String), Dict{String, Any}))
    catch err
        @warn "Could not parse $BENCHMARK_RESULTS_PATH; rendering an empty benchmarks page" exception = err
        return Dict{String, Any}()
    end
end

"""
    overview_series(record, config)

`(ncolumns, sypd)` pairs of one configuration in one architecture's record, sorted by resolution and
with non-finite entries (skipped or failed runs) dropped.
"""
function overview_series(record, config::AbstractString)
    overview = get(record, "overview", nothing)
    overview === nothing && return (Int[], Float64[])
    xs, ys = Int[], Float64[]
    for (_, suite) in overview
        for i in eachindex(suite["config"])
            String(suite["config"][i]) == config || continue
            s = suite["sypd"][i]
            (s isa Number && isfinite(s) && s > 0) || continue
            push!(xs, Int(suite["ncolumns"][i]))
            push!(ys, Float64(s))
        end
    end
    order = sortperm(xs)
    return (xs[order], ys[order])
end

"""
    configurations(results, labels)

Every configuration name appearing in any architecture's overview data.
"""
function configurations(results, labels)
    names = String[]
    for label in labels
        overview = get(results[label], "overview", nothing)
        overview === nothing && continue
        for (_, suite) in overview, c in suite["config"]
            String(c) in names || push!(names, String(c))
        end
    end
    return sort!(names)
end

"""
    write_scaling_figure(results, labels, config, png_path)

SYPD against number of land columns for one configuration, one line per architecture, on log-log
axes. Returns the path written to, or `nothing` if no architecture has data for this configuration.
"""
function write_scaling_figure(results, labels, config, png_path)
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1];
        xlabel = "Number of land columns",
        ylabel = "SYPD (simulated years per wallclock day)",
        title = "Configuration: $config",
        xscale = log10,
        yscale = log10,
    )
    palette = Makie.wong_colors()

    ## Reference resolutions first, so the data lines are drawn on top. Only those inside the
    ## plotted range are drawn: a line far outside it would stretch the log axis and squash the data.
    ymax = 0.0
    xmin, xmax = typemax(Int), 0
    for label in labels
        xs, ys = overview_series(results[label], config)
        isempty(ys) && continue
        ymax = max(ymax, maximum(ys))
        xmin = min(xmin, minimum(xs))
        xmax = max(xmax, maximum(xs))
    end
    for (degrees, text) in REFERENCE_RESOLUTIONS
        x = reference_ncolumns(degrees)
        (ymax > 0 && xmin <= x <= xmax) || continue
        vlines!(ax, [x]; linestyle = :dash, color = (:black, 0.4))
        text!(ax, x * 1.05, ymax; text, align = (:left, :top), fontsize = 12, color = (:black, 0.6))
    end

    plotted = false
    for (j, label) in enumerate(labels)
        xs, ys = overview_series(results[label], config)
        isempty(xs) && continue
        scatterlines!(ax, xs, ys; label, color = palette[mod1(j, length(palette))], linewidth = 2, markersize = 10)
        plotted = true
    end
    plotted || return nothing
    axislegend(ax; position = :lb)
    save(png_path, fig)
    return png_path
end

format_sypd_cell(s) = (s isa Number && isfinite(s)) ? (s < 10 ? string(round(s; digits = 2)) : string(round(Int, s))) : "—"

function lookup_sypd(record, config::AbstractString, ncolumns::Integer)
    overview = get(record, "overview", nothing)
    overview === nothing && return nothing
    for (_, suite) in overview
        for i in eachindex(suite["config"])
            if String(suite["config"][i]) == config && Int(suite["ncolumns"][i]) == ncolumns
                return suite["sypd"][i]
            end
        end
    end
    return nothing
end

function write_overview_table(io, results, labels)
    rows = Tuple{String, Int}[]
    for label in labels
        overview = get(results[label], "overview", nothing)
        overview === nothing && continue
        for (_, suite) in overview, i in eachindex(suite["config"])
            r = (String(suite["config"][i]), Int(suite["ncolumns"][i]))
            r in rows || push!(rows, r)
        end
    end
    sort!(rows; by = r -> (r[1], r[2]))

    write(io, "| Configuration | Columns | " * join(labels, " | ") * " |\n")
    write(io, "| --- | --- | " * join(fill("---", length(labels)), " | ") * " |\n")
    for (config, ncolumns) in rows
        cells = [format_sypd_cell(lookup_sypd(results[label], config, ncolumns)) for label in labels]
        write(io, "| `$config` | $ncolumns | " * join(cells, " | ") * " |\n")
    end
    write(io, "\n")
    return
end

"""
    rewrite_arch_markdown(md, label)

Suffix every `###`/`####` heading with the architecture label, so that the slugs are unique across
the documentation and Documenter's `@ref` resolver cannot confuse a suite title with a page title.
"""
function rewrite_arch_markdown(md::AbstractString, label::AbstractString)
    io = IOBuffer()
    for line in eachline(IOBuffer(md), keep = true)
        stripped = chomp(line)
        if startswith(stripped, "### ") || startswith(stripped, "#### ")
            write(io, stripped, " — ", label, "\n")
        else
            write(io, line)
        end
    end
    return String(take!(io))
end

function write_empty_page(path)
    open(path, "w") do io
        write(io, "# Benchmarks\n\n")
        write(io, "No benchmark results have been collected yet. To populate this page, run from `benchmark/`:\n\n")
        write(io, "```\n")
        write(io, "julia --project=. manual_benchmarking.jl                # CPU\n")
        write(io, "julia --project=. manual_benchmarking.jl gpu            # CUDA GPU\n")
        write(io, "julia --project=. manual_benchmarking.jl reactant-cpu   # Reactant/XLA, CPU backend (auto-labelled reactant-cpu-arm or reactant-cpu-x86)\n")
        write(io, "julia --project=. manual_benchmarking.jl reactant-gpu   # Reactant/XLA, CUDA backend\n")
        write(io, "```\n\n")
        write(io, "Results are stored in `benchmark/assets/benchmark_results.json` and read from there by the documentation build.\n")
    end
    return
end

function generate_benchmarks_page()
    mkpath(DOCS_ASSETS_DIR)
    results = load_results()

    if isempty(results)
        @info "No benchmark results found; writing a placeholder benchmarks page"
        write_empty_page(DOCS_PAGE_PATH)
        return
    end

    labels = sorted_arch_labels(results)

    figures = Tuple{String, String}[]   # (configuration, asset path used in the markdown)
    for config in configurations(results, labels)
        filename = "scaling_$(config).png"
        out = write_scaling_figure(results, labels, config, joinpath(DOCS_ASSETS_DIR, filename))
        out === nothing || push!(figures, (config, joinpath("assets", "benchmarks", filename)))
    end

    open(DOCS_PAGE_PATH, "w") do io
        write(io, "# Benchmarks\n\n")
        write(io, "Performance of Terrarium.jl across architectures and resolutions. ")
        write(io, "This page is generated at documentation-build time from `benchmark/assets/benchmark_results.json`, ")
        write(io, "which is updated by `benchmark/manual_benchmarking.jl`. ")
        write(io, "Running the benchmark on another architecture adds a series to each figure, a column to the ")
        write(io, "overview table, and a section to the bottom of this page.\n\n")

        write(io, "Every configuration runs for a fixed number of time steps without output. ")
        write(io, "Initialization and - under Reactant - XLA compilation happen before the clock starts and are ")
        write(io, "reported separately; the timed section covers only the stepping loop. ")
        write(io, "A run is timed once and takes a few seconds, so treat the numbers as indicative: ")
        write(io, "±20% between repetitions is normal.\n\n")

        write(io, "The benchmark grids are full Gaussian grids in which every point is an active land column (i.e. a. rock planet configuration), ")
        write(io, "so a real land–sea mask at the same resolution has roughly a third as many columns. ")
        write(io, "The model configurations themselves are defined in `benchmark/model_configurations.jl`.\n\n")

        write(io, "## Scaling with resolution\n\n")
        for (config, link) in figures
            write(io, "![$config scaling across architectures]($link)\n\n")
        end

        write(io, "## Overview: SYPD across architectures\n\n")
        write(io, "Simulated years per wallclock day for each configuration and resolution. ")
        write(io, "`—` means the architecture has not been benchmarked yet, skipped that resolution, or could ")
        write(io, "not run that configuration; the per-architecture sections below say which.\n\n")
        write_overview_table(io, results, labels)

        for label in labels
            record = results[label]
            meta = record["meta"]
            write(io, "## Architecture: `$label`\n\n")
            write(io, "Created for Terrarium.jl v$(meta["terrarium_version"]) on $(meta["timestamp"]) ")
            write(io, "in `$(meta["mode"])` mode ($(meta["timestep_multiplier"])x time steps, $(meta["threads"]) thread(s)).\n\n")
            write(io, "### Machine details — $label\n\n")
            write(io, meta["machine_info"])
            write(io, "\n")
            write(io, rewrite_arch_markdown(record["markdown"], label))
            write(io, "\n")
        end
    end

    @info "Generated $DOCS_PAGE_PATH ($(length(labels)) architecture(s), $(length(figures)) figure(s))"
    return
end

generate_benchmarks_page()
