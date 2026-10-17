# Effect of forcing `Vararg` specialization on `energy_to_temperature_kernel!` (allocations, time, results).
# usage: julia --project=. vararg_specialization.jl
include(joinpath(@__DIR__, "common.jl"))
function measure(label, state, model, ncells)
    Terrarium.closure!(state, model)
    n = @allocations Terrarium.closure!(state, model)
    b = @benchmark Terrarium.closure!($state, $model) evals = 1 seconds = 3
    println(@sprintf("%-28s closure!: %8d allocs/call, %.2f ns/cell (median)", label, n, median(b).time / ncells))
end
integrator, Δt = setup(:soil_heat, CPU(); nlat_half = 24)
state, model = integrator.state, integrator.model
ncells = ncolumns(24) * 30
measure("baseline (args...)", state, model, ncells)
before = copy(Array(interior(state.temperature)))
Core.eval(Terrarium, quote
    @kernel inbounds = true function energy_to_temperature_kernel!(out, grid, fields, args::Vararg{Any, N}) where {N}
        i, j, k = @index(Global, NTuple)
        energy_to_temperature!(out, i, j, k, grid, fields, args...)
    end
end)
Base.invokelatest(measure, "Vararg{Any, N} where N", state, model, ncells)
after = Array(interior(state.temperature))
println("max abs temperature difference after fix: ", maximum(abs, after .- before))
