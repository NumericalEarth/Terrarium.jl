const USAGE = """
Effect of compilation order on native CPU step times.

Usage: julia --project=. compile_order.jl [key=value ...]

Builds and times one configuration at the given grid sizes, in the given order, in one process. The first
model stepped in a native process can be several times slower than later ones, because methods compiled
for it (for example, Oceananigans' reductions) are specialized less well; running the same sizes in
different orders, each in a fresh process, shows whether a timing depends on what was compiled before.

Options:
  config=land      configuration (see ../model_configurations.jl)
  order=4,24       grid sizes (`nlat_half`), in the order in which they are built
  out=             file to append the results to (default: standard output only)
"""
include(joinpath(@__DIR__, "common.jl"))

const OPTIONS = parse_options((config = :land, order = [4, 24], out = ""), USAGE)

with_output(OPTIONS.out) do io
    say(io, environment_summary())
    say(io, "| order | columns | ms/step (median) | allocs/step | closure! allocs | closure! allocs per cell |")
    say(io, "| --- | --- | --- | --- | --- | --- |")
    for nlat in OPTIONS.order
        integrator, Δt = setup(OPTIONS.config, CPU(); nlat_half = nlat)
        state, model = integrator.state, integrator.model
        Terrarium.closure!(state, model)
        nclosure = @allocations Terrarium.closure!(state, model)
        timestep!(integrator, Δt; finalize = false)
        nstep = @allocations timestep!(integrator, Δt; finalize = false)
        b = @benchmark timestep!($integrator, $Δt; finalize = false) evals = 1 seconds = 2
        ncells = ncolumns(nlat) * 30
        say(io, @sprintf("| %s | %d | %.3g | %d | %d | %.2f |",
            join(OPTIONS.order, ","), ncolumns(nlat), median(b).time / 1.0e6, nstep, nclosure, nclosure / ncells))
    end
end
