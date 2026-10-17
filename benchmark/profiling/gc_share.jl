# Share of garbage collection in the per-step time of native CPU runs.
# usage: julia --project=. gc_share.jl
include(joinpath(@__DIR__, "common.jl"))
for (config, nlat, nsteps) in ((:land, 4, 200), (:land, 24, 20), (:soil_heat, 24, 40))
    integrator, Δt = setup(config, CPU(); nlat_half = nlat)
    GC.gc()
    r = @timed for _ in 1:nsteps
        timestep!(integrator, Δt; finalize = false)
    end
    b = @benchmark timestep!($integrator, $Δt; finalize = false) evals = 1 seconds = 3
    println(@sprintf("%-10s %6d columns: consecutive %.3g ms/step (GC %.0f%% of it, %.1f MiB/step) | BenchmarkTools median %.3g ms, mean %.3g ms",
        config, ncolumns(nlat), 1e3 * r.time / nsteps, 100 * r.gctime / r.time, r.bytes / nsteps / 2^20, median(b).time / 1e6, mean(b).time / 1e6))
end
