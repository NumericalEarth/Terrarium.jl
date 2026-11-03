const USAGE = """
Allocations per launch of every native CPU kernel during one time step.

Usage: julia --project=. kernel_allocations.jl [key=value ...]

A kernel that allocates is not type stable or has lost specialization (for example, Julia does not
specialize on a trailing `args...` that is only passed through, and then boxes the arguments for every
grid cell). Such allocations grow with the grid size, so compare two grid sizes to tell them apart from
fixed costs. Kernels are listed by total allocations, with their module and number of arguments.

Options:
  configs=land     configurations (see ../model_configurations.jl)
  nlat=4           grid size (`nlat_half`)
  out=             file to append the results to (default: standard output only)
"""
import KernelAbstractions as KA

## Count allocations per launch while `RECORD[]` is set. This wraps KernelAbstractions' CPU launch
## operator before anything is compiled, so that no world-age issues arise.
const ALLOCS = Dict{String, Vector{Int}}()
const RECORD = Ref(false)
function (obj::KA.Kernel{KA.CPU})(args...; ndrange = nothing, workgroupsize = nothing)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    length(KA.blocks(iterspace)) == 0 && return nothing
    if RECORD[]
        n = @allocations KA.__run(obj, ndrange, iterspace, args, dynamic, obj.backend.static)
        name = string(nameof(obj.f)) * " (" * string(nameof(parentmodule(typeof(args[1]).name.module))) * ", $(length(args)) arguments)"
        push!(get!(ALLOCS, name, Int[]), n)
    else
        KA.__run(obj, ndrange, iterspace, args, dynamic, obj.backend.static)
    end
    return nothing
end

include(joinpath(@__DIR__, "common.jl"))
const OPTIONS = parse_options((configs = [:land], nlat = 4, out = ""), USAGE)

with_output(OPTIONS.out) do io
    say(io, environment_summary())
    for config in OPTIONS.configs
        integrator, Δt = setup(config, CPU(); nlat_half = OPTIONS.nlat)
        RECORD[] = true
        timestep!(integrator, Δt; finalize = false)  # compiles the recording path
        empty!(ALLOCS)
        timestep!(integrator, Δt; finalize = false)
        RECORD[] = false
        say(io, "\n$config, $(ncolumns(OPTIONS.nlat)) columns: allocations per kernel during one step")
        say(io, "| allocations | launches | kernel |")
        say(io, "| --- | --- | --- |")
        for (name, counts) in sort(collect(ALLOCS); by = x -> -sum(x[2]))
            say(io, "| $(sum(counts)) | $(length(counts)) | `$name` |")
        end
    end
end
