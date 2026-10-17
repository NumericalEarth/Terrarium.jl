# Compile-order effect: time `land` at the given grid sizes, in the given order, in one process.
# usage: julia --project=. compile_order.jl <nlat_half,nlat_half,...>
include(joinpath(@__DIR__, "common.jl"))
order = parse.(Int, split(ARGS[1], ","))
for nlat in order
    integrator, Δt = setup(:land, CPU(); nlat_half = nlat)
    state, model = integrator.state, integrator.model
    Terrarium.closure!(state, model)
    nclosure = @allocations Terrarium.closure!(state, model)
    timestep!(integrator, Δt; finalize = false)
    nstep = @allocations timestep!(integrator, Δt; finalize = false)
    b = @benchmark timestep!($integrator, $Δt; finalize = false) evals = 1 seconds = 2
    println(@sprintf("order %-6s nlat_half=%d (%5d columns): %.3g ms/step median, %8d allocs/step, closure! %8d allocs (%.2f per cell)",
        ARGS[1], nlat, ncolumns(nlat), median(b).time / 1e6, nstep, nclosure, nclosure / (ncolumns(nlat) * 30)))
end
