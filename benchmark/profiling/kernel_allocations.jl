# Allocations per launch for every CPU kernel in one `land` step (finds kernels that lose specialization).
# usage: julia --project=. kernel_allocations.jl
import KernelAbstractions as KA
const ALLOCS = Dict{String, Vector{Int}}()
const RECORD = Ref(false)
function (obj::KA.Kernel{KA.CPU})(args...; ndrange = nothing, workgroupsize = nothing)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    length(KA.blocks(iterspace)) == 0 && return nothing
    if RECORD[]
        n = @allocations KA.__run(obj, ndrange, iterspace, args, dynamic, obj.backend.static)
        name = string(nameof(obj.f)) * " (" * string(nameof(parentmodule(typeof(args[1]).name.module))) * ", nargs=$(length(args)))"
        push!(get!(ALLOCS, name, Int[]), n)
    else
        KA.__run(obj, ndrange, iterspace, args, dynamic, obj.backend.static)
    end
    return nothing
end
include(joinpath(@__DIR__, "common.jl"))
for config in (:land,)
    integrator, Δt = setup(config, CPU(); nlat_half = 4)
    ncells = ncolumns(4) * 30
    RECORD[] = true
    timestep!(integrator, Δt; finalize = false)  # compile the recording path
    empty!(ALLOCS)
    timestep!(integrator, Δt; finalize = false)
    RECORD[] = false
    println("$config, $(ncolumns(4)) columns ($ncells cells): allocations per launch during one step")
    for (k, v) in sort(collect(ALLOCS); by = x -> -sum(x[2]))
        println(@sprintf("  %8d  %s  (launches: %d)", sum(v), k, length(v)))
    end
end
