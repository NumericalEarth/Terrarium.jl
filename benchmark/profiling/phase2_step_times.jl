# Native step times of all three configurations, for before/after comparisons of Phase 2 changes.
# usage: julia --project=. phase2_step_times.jl <cpu|gpu> <label> <outfile>
# Give concurrent runs separate output files: rows appended to one file from two processes can be lost.
const ARCHNAME, LABEL, OUT = ARGS[1], ARGS[2], ARGS[3]
ARCHNAME == "gpu" && using CUDA
include(joinpath(@__DIR__, "common.jl"))

const ARCH = ARCHNAME == "gpu" ? GPU() : CPU()
## grid sizes and steps per repetition; the GPU also covers a large grid
const CASES = ARCHNAME == "gpu" ? ((4, 200), (24, 200), (144, 20)) : ((4, 200), (24, 20))

## The first model stepped in a native process can be several times slower than later ones (see the
## compile-order check in ../define_benchmarks.jl), so a throwaway model absorbs that first.
let (integrator, Δt) = setup(:land, ARCH; nlat_half = 2)
    time_steps(integrator, Δt, ARCH; nsteps = 10, reps = 1)
end

open(OUT, "a") do io
    for config in CONFIGURATIONS, (nlat_half, nsteps) in CASES
        integrator, Δt = setup(config, ARCH; nlat_half)
        allocs = @allocations timestep!(integrator, Δt; finalize = false)
        t = time_steps(integrator, Δt, ARCH; nsteps, reps = 5)
        say(io, @sprintf("| %s | %s | %s | %d | %s | %d |", LABEL, ARCHNAME, config, ncolumns(nlat_half), fmt_times(t), allocs))
    end
end
