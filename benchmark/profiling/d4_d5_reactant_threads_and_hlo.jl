# D4 (Reactant CPU thread count) and D5 (HLO dump, via XLA_FLAGS set by the caller).
# usage: julia --project=. d4_d5_reactant_threads_and_hlo.jl <config> <nlat_half> <reps> <label> <outfile>
using CUDA
using Reactant
Reactant.set_default_backend("cpu")
include(joinpath(@__DIR__, "common.jl"))
const CONFIG, NLAT, REPS, LABEL, OUT = Symbol(ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4], ARGS[5]

os_threads() = parse(Int, split(only(filter(startswith("Threads:"), readlines("/proc/self/status"))))[2])
affinity() = strip(read(`taskset -cp $(getpid())`, String))

const ARCH = ReactantState()
cfg = build_model(CONFIG, ARCH, Float32; nlat_half = NLAT, nz = 30)
integrator = initialize(cfg.model; boundary_conditions = cfg.boundary_conditions, initializers = cfg.initializers)
Δt = Float32(cfg.Δt)
nsteps = clamp(round(Int, 3.0e7 / (ncolumns(NLAT) * 30)), 20, 2000)

compile_time = @elapsed compiled = @compile raise = true raise_first = true sync = true Terrarium.run_timesteps!(integrator, Δt, nsteps, false)
compiled(integrator, Δt, nsteps, false)  # warm-up
times = Float64[]
for _ in 1:REPS
    Terrarium.initialize!(integrator)
    t = @elapsed compiled(integrator, Δt, nsteps, false)
    push!(times, 1e3 * t / nsteps)
end
finite = all(f -> !(f isa Oceananigans.Fields.AbstractField) || all(isfinite, Array(parent(f))), values(Terrarium.prognostic_fields(integrator.state, integrator.model)))
open(OUT, "a") do io
    say(io, @sprintf("| %s | %s | %d | %d | %d | %.1f s | %s | %d | %s | %s |", LABEL, CONFIG, ncolumns(NLAT), nsteps, REPS, compile_time,
        fmt_times((median = median(times), min = minimum(times), max = maximum(times))), os_threads(), affinity(), finite))
end
