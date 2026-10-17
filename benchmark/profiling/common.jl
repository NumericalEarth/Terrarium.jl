# Shared helpers for the profiling scripts (see README.md and
# docs/dev/2026-10/2026-10-17_PLAN_native_vs_reactant_performance_gap.md).
using Terrarium
using BenchmarkTools, Printf, Statistics
import KernelAbstractions as KA
import Oceananigans

include(joinpath(@__DIR__, "..", "model_configurations.jl"))

say(io, s) = (println(io, s); flush(io); println(s); flush(stdout))

# `build_model` uses the Newton skin-temperature solver for the land configurations (see
# `land_solver_kwargs` in ../model_configurations.jl); these select either solver explicitly.
newton_kwargs(::Type{NF}) where {NF} = (
    surface_energy_balance = SurfaceEnergyBalance(NF; skin_temperature = ImplicitSkinTemperature(NF; solver = Terrarium.NewtonSolver(NF; iterations = 5))),
)
root_solver_kwargs(::Type{NF}) where {NF} = (surface_energy_balance = SurfaceEnergyBalance(NF),)

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

sync(arch) = KA.synchronize(Oceananigans.Architectures.device(arch))

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
        push!(times, 1e3 * t / nsteps)
    end
    return (median = median(times), min = minimum(times), max = maximum(times), all = times)
end

fmt_times(t) = @sprintf("%.4g (min %.4g, max %.4g)", t.median, t.min, t.max)
