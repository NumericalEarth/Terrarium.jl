const USAGE = """
Host-side cost of native CPU time steps.

Usage: julia --project=. host_overhead.jl [key=value ...]

Reports, for each configuration:
  1. the whole `timestep!` time and allocations at each grid size, with a least-squares fit of the
     time into a fixed cost per step and a cost per column;
  2. a breakdown of one step into its phases (`reset_tendencies!`, `update_inputs!`,
     `compute_auxiliary!`, ...), at the smallest grid size;
and, for one configuration:
  3. the allocation sites of one step, grouped by the innermost Terrarium or Oceananigans frame;
  4. a sampling profile, with the share of samples inside kernel bodies, and the flat profile written to
     `<out>.flat.txt` (or `host_overhead.flat.txt` if `out` is empty).

Options:
  configs=land,land_no_vegetation,soil_heat   configurations (see ../model_configurations.jl)
  nlat=2,4,8                                  grid sizes (`nlat_half`) for sections 1 and 2; at least two
  profile_config=land                         configuration for sections 3 and 4
  profile_steps=400                           steps sampled in section 4
  out=                                        file to append the results to (default: standard output only)

Sections 3 and 4 take most of the run time (more than 20 minutes for the defaults).
"""
include(joinpath(@__DIR__, "common.jl"))
using Profile

const OPTIONS = parse_options(
    (configs = [:land, :land_no_vegetation, :soil_heat], nlat = [2, 4, 8], profile_config = :land, profile_steps = 400, out = ""),
    USAGE
)
length(OPTIONS.nlat) >= 2 || error("`nlat` needs at least two grid sizes for the fit")

med_ms(b) = median(b).time / 1.0e6

function phases(integrator, Δt)
    state, model, inputs = integrator.state, integrator.model, integrator.inputs
    grid = get_grid(model)
    ts = Terrarium.get_timestepper(model)
    names = Terrarium.prognostic_names(state)
    return (
        reset_tendencies = () -> Terrarium.reset_tendencies!(state),
        update_inputs = () -> Terrarium.update_inputs!(state, grid, inputs),
        compute_auxiliary = () -> Terrarium.compute_auxiliary!(state, model),
        compute_boundary_conditions = () -> Terrarium.compute_boundary_conditions!(state, model),
        compute_tendencies = () -> Terrarium.compute_tendencies!(state, model),
        explicit_step = () -> Terrarium.explicit_step!(state, grid, ts, Δt, names),
        model_timestep = () -> Terrarium.timestep!(state, model, ts, Δt),
        closure = () -> Terrarium.closure!(state, model),
    )
end

"Innermost Terrarium or Oceananigans frame of an allocation's stack trace."
function allocation_site(a)
    frames = filter(fr -> occursin("Terrarium", string(fr.file)) || occursin("Oceananigans", string(fr.file)), a.stacktrace)
    isempty(frames) && return "(no Terrarium or Oceananigans frame)"
    fr = first(frames)
    return "$(fr.func) @ $(basename(string(fr.file))):$(fr.line)"
end

"Split raw profile data into one vector of instruction pointers per sample."
function profile_samples(data)
    samples = Vector{Vector{UInt64}}()
    current = UInt64[]
    for ip in data
        if ip == 0
            isempty(current) || push!(samples, current)
            current = UInt64[]
        else
            push!(current, ip)
        end
    end
    return samples
end

with_output(OPTIONS.out) do io
    say(io, environment_summary())
    smallest = minimum(OPTIONS.nlat)

    say(io, "\n## 1. Whole timestep! (median, evals = 1)")
    say(io, "| config | columns | ms/step | allocs/step | KiB/step |")
    say(io, "| --- | --- | --- | --- | --- |")
    fits = Dict{Symbol, Vector{Tuple{Int, Float64}}}()
    for config in OPTIONS.configs, nlat in OPTIONS.nlat
        integrator, Δt = setup(config, CPU(); nlat_half = nlat)
        b = @benchmark timestep!($integrator, $Δt; finalize = false) evals = 1 seconds = 4
        push!(get!(fits, config, Tuple{Int, Float64}[]), (ncolumns(nlat), med_ms(b)))
        say(io, @sprintf("| %s | %d | %.4g | %d | %.1f |", config, ncolumns(nlat), med_ms(b), b.allocs, b.memory / 1024))
    end
    say(io, "\nLeast-squares fit ms/step = fixed + per column × columns:")
    for config in OPTIONS.configs
        x = [Float64(p[1]) for p in fits[config]]
        y = [p[2] for p in fits[config]]
        slope = cov(x, y) / var(x)
        intercept = mean(y) - slope * mean(x)
        say(io, @sprintf("- %s: fixed ≈ %.3f ms/step, per column ≈ %.2f µs", config, intercept, 1.0e3 * slope))
    end

    say(io, "\n## 2. Phase breakdown at $(ncolumns(smallest)) columns (median)")
    for config in OPTIONS.configs
        integrator, Δt = setup(config, CPU(); nlat_half = smallest)
        say(io, "\n### $config")
        say(io, "| phase | ms | allocs | KiB |")
        say(io, "| --- | --- | --- | --- |")
        total = 0.0
        for (name, f) in pairs(phases(integrator, Δt))
            f()
            b = @benchmark $f() evals = 1 seconds = 1.5
            total += med_ms(b)
            say(io, @sprintf("| %s | %.4g | %d | %.1f |", name, med_ms(b), b.allocs, b.memory / 1024))
        end
        say(io, @sprintf("| **sum** | %.4g | | |", total))
    end

    config = OPTIONS.profile_config
    say(io, "\n## 3. Allocation sites, $config at $(ncolumns(smallest)) columns, one step (top 25 by count)")
    integrator, Δt = setup(config, CPU(); nlat_half = smallest)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = 1 timestep!(integrator, Δt; finalize = false)
    allocs = Profile.Allocs.fetch().allocs
    say(io, "total sampled allocations: $(length(allocs))")
    counts = Dict{String, Tuple{Int, Int}}()
    for a in allocs
        k = allocation_site(a)
        n, s = get(counts, k, (0, 0))
        counts[k] = (n + 1, s + a.size)
    end
    for (k, (n, s)) in first(sort(collect(counts); by = x -> -x[2][1]), 25)
        say(io, @sprintf("- %5d allocs, %8d B: %s", n, s, k))
    end
    say(io, "\nTypes allocated (top 15):")
    type_counts = Dict{String, Int}()
    for a in allocs
        k = first(string(a.type), 140)
        type_counts[k] = get(type_counts, k, 0) + 1
    end
    for (k, n) in first(sort(collect(type_counts); by = x -> -x[2]), 15)
        say(io, "- $n × $k")
    end

    say(io, "\n## 4. Sampling profile, $config at $(ncolumns(smallest)) columns, $(OPTIONS.profile_steps) steps")
    Profile.clear()
    Profile.init(n = 10^7, delay = 0.0005)
    @profile for _ in 1:OPTIONS.profile_steps
        timestep!(integrator, Δt; finalize = false)
    end
    data, lidict = Profile.retrieve()
    samples = profile_samples(data)
    ## KernelAbstractions names the CPU function generated by `@kernel` with a `cpu_` prefix
    in_kernel = 0
    in_run = 0
    for s in samples
        names = [string(fr.func) for ip in s for fr in get(lidict, ip, Base.StackTraces.StackFrame[])]
        any(n -> startswith(n, "cpu_"), names) && (in_kernel += 1)
        any(n -> n == "__run" || n == "__thread_run", names) && (in_run += 1)
    end
    nsamples = max(1, length(samples))
    say(
        io, @sprintf(
            "samples: %d; inside @kernel bodies (cpu_*): %.1f%%; inside KernelAbstractions' __run: %.1f%%",
            length(samples), 100 * in_kernel / nsamples, 100 * in_run / nsamples
        )
    )
    flat = (isempty(OPTIONS.out) ? "host_overhead" : OPTIONS.out) * ".flat.txt"
    open(flat, "w") do pio
        Profile.print(IOContext(pio, :displaysize => (10000, 400)); format = :flat, sortedby = :count, C = false, mincount = 20)
    end
    say(io, "flat profile written to $flat")
end
