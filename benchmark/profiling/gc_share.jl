const USAGE = """
Share of garbage collection in native CPU step times.

Usage: julia --project=. gc_share.jl [key=value ...]

Times `nsteps` consecutive steps with `@timed` and reports the share of garbage collection and the memory
allocated per step, next to BenchmarkTools' median and mean. A large difference between the consecutive
time and the median points to garbage collection that BenchmarkTools' sampling hides.

Options:
  configs=land,soil_heat   configurations (see ../model_configurations.jl)
  nlat=4,24                grid sizes (`nlat_half`); every configuration is run at every size
  nsteps=0                 consecutive steps; 0 picks a number that takes similar time at every size
  out=                     file to append the results to (default: standard output only)
"""
include(joinpath(@__DIR__, "common.jl"))

const OPTIONS = parse_options((configs = [:land, :soil_heat], nlat = [4, 24], nsteps = 0, out = ""), USAGE)

with_output(OPTIONS.out) do io
    say(io, environment_summary())
    say(io, "| config | columns | steps | consecutive ms/step | GC share | MiB/step | median ms/step | mean ms/step |")
    say(io, "| --- | --- | --- | --- | --- | --- | --- | --- |")
    for config in OPTIONS.configs, nlat in OPTIONS.nlat
        integrator, Δt = setup(config, CPU(); nlat_half = nlat)
        nsteps = OPTIONS.nsteps > 0 ? OPTIONS.nsteps : default_nsteps(nlat, 30; hi = 200)
        GC.gc()
        r = @timed for _ in 1:nsteps
            timestep!(integrator, Δt; finalize = false)
        end
        b = @benchmark timestep!($integrator, $Δt; finalize = false) evals = 1 seconds = 3
        say(io, @sprintf("| %s | %d | %d | %.3g | %.0f%% | %.1f | %.3g | %.3g |",
            config, ncolumns(nlat), nsteps, 1.0e3 * r.time / nsteps, 100 * r.gctime / r.time, r.bytes / nsteps / 2^20,
            median(b).time / 1.0e6, mean(b).time / 1.0e6))
    end
end
