# Shared helpers for the profiling tools in this directory (see README.md).
#
# Every tool takes `key=value` options on the command line, with defaults given by the tool, and prints
# its usage with `--help`. `arch=gpu` loads CUDA before Terrarium, so it must be passed on the command
# line (not set later) for tools that run on the GPU.
if any(==("arch=gpu"), ARGS)
    using CUDA
end
using Terrarium
using BenchmarkTools, Printf, Statistics
import KernelAbstractions as KA
import Oceananigans

include(joinpath(@__DIR__, "..", "model_configurations.jl"))

## Command-line options

parse_option(::Type{String}, s) = String(s)
parse_option(::Type{Symbol}, s) = Symbol(s)
parse_option(::Type{Int}, s) = parse(Int, s)
parse_option(::Type{Float64}, s) = parse(Float64, s)
parse_option(::Type{Bool}, s) = parse(Bool, s)
parse_option(::Type{Vector{T}}, s) where {T} = isempty(s) ? T[] : T[parse_option(T, x) for x in split(s, ",")]

"""
    parse_options(defaults::NamedTuple, usage::String)

Parse `key=value` command-line arguments into a `NamedTuple` with the keys and value types of `defaults`.
Lists are given comma-separated (`nlat=2,4,8`). Prints `usage` and exits on `--help` or `-h`, and errors on
unknown keys or arguments that are not of the form `key=value`.
"""
function parse_options(defaults::NamedTuple, usage::String)
    if any(in(("--help", "-h")), ARGS)
        println(usage)
        exit(0)
    end
    parsed = Dict{Symbol, Any}(pairs(defaults))
    for arg in ARGS
        key, sep, value = partition_option(arg)
        sep || error("expected a `key=value` argument, got `$arg`; see `--help`")
        haskey(parsed, key) || error("unknown option `$key`; valid options are $(join(keys(defaults), ", "))")
        parsed[key] = parse_option(typeof(defaults[key]), value)
    end
    return NamedTuple{keys(defaults)}(Tuple(parsed[k] for k in keys(defaults)))
end

function partition_option(arg)
    i = findfirst('=', arg)
    isnothing(i) && return Symbol(arg), false, ""
    return Symbol(arg[1:(i - 1)]), true, arg[(i + 1):end]
end

## Output

"""
    with_output(f, out)

Call `f(io)` with an `IO` that `say` writes to: the file `out` (appended to), or `devnull` if `out` is empty.
`say` always prints to standard output as well.
"""
with_output(f, out::AbstractString) = isempty(out) ? f(devnull) : open(f, out, "a")

say(io, s) = (println(io, s); flush(io); println(s); flush(stdout))

## Models

architecture(name) = name == "gpu" ? GPU() : name == "cpu" ? CPU() : error("unknown architecture `$name`; use `cpu` or `gpu`")
float_type(name) = name == "Float32" ? Float32 : name == "Float64" ? Float64 : error("unknown float type `$name`")

"""
    solver_kwargs(solver, NF)

Model keyword arguments that select the skin temperature solver of the land configurations:
`:default` keeps whatever `build_model` uses (`NewtonSolver` with five iterations, see `land_solver_kwargs`
in ../model_configurations.jl), `:newton` selects it explicitly, and `:root` selects the model default,
`RootSolver`, which wraps the Newton method of RootSolvers.jl with a convergence test.
"""
function solver_kwargs(solver::Symbol, ::Type{NF}) where {NF}
    solver == :default && return (;)
    solver == :newton && return (
        surface_energy_balance = SurfaceEnergyBalance(NF; skin_temperature = ImplicitSkinTemperature(NF; solver = Terrarium.NewtonSolver(NF; iterations = 5))),
    )
    solver == :root && return (surface_energy_balance = SurfaceEnergyBalance(NF),)
    return error("unknown solver `$solver`; use `default`, `newton`, or `root`")
end

"""
    setup(config, arch, NF = Float32; nlat_half, nz = 30, model_kwargs = (;), warmup = 3)

Build and initialize the benchmark configuration `config` (see ../model_configurations.jl) and run `warmup`
steps, so that the returned integrator is compiled. Returns `(integrator, Δt)`.
"""
function setup(config, arch, ::Type{NF} = Float32; nlat_half, nz = 30, model_kwargs = (;), warmup = 3) where {NF}
    cfg = build_model(config, arch, NF; nlat_half, nz, model_kwargs)
    integrator = initialize(cfg.model; boundary_conditions = cfg.boundary_conditions, initializers = cfg.initializers)
    Δt = NF(cfg.Δt)
    for _ in 1:warmup
        timestep!(integrator, Δt; finalize = false)
    end
    sync(arch)
    return integrator, Δt
end

"""
    absorb_first_model_slowdown(arch)

Build and step a small throwaway `land` model. The first model stepped in a native process can be several
times slower than later ones (see `compile_order.jl`), so tools that compare timings call this first.
"""
function absorb_first_model_slowdown(arch)
    integrator, Δt = setup(:land, arch; nlat_half = 2)
    time_steps(integrator, Δt, arch; nsteps = 10, reps = 1)
    return nothing
end

sync(arch) = KA.synchronize(Oceananigans.Architectures.device(arch))

"A number of steps that takes roughly a similar time at every grid size, between `lo` and `hi`."
default_nsteps(nlat_half, nz; lo = 20, hi = 2000) = clamp(round(Int, 3.0e7 / (ncolumns(nlat_half) * nz)), lo, hi)

"Median and spread of `reps` timings of `nsteps` steps each, in ms/step."
function time_steps(integrator, Δt, arch; nsteps, reps)
    times = Float64[]
    for _ in 1:reps
        sync(arch)
        t = @elapsed begin
            for _ in 1:nsteps
                timestep!(integrator, Δt; finalize = false)
            end
            sync(arch)
        end
        push!(times, 1.0e3 * t / nsteps)
    end
    return (median = median(times), min = minimum(times), max = maximum(times), all = times)
end

fmt_times(t) = @sprintf("%.4g (min %.4g, max %.4g)", t.median, t.min, t.max)

"Run-time environment: Julia version, Julia threads, and the CPU affinity of this process (Linux only)."
function environment_summary()
    affinity = try
        strip(read(`taskset -cp $(getpid())`, String))
    catch
        "unknown"
    end
    return "Julia $(VERSION), $(Threads.nthreads()) thread(s), $affinity"
end
