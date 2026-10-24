# Host cost of a single native GPU kernel launch, split into its stages, and a host profile of a `land`
# step at a small grid size where the GPU is idle most of the time (Phase 2 item 5, H1).
# usage: julia --project=. launch_overhead.jl <outprefix>
using CUDA
using Profile
include(joinpath(@__DIR__, "common.jl"))
import Oceananigans.Architectures: convert_to_device, device
import Oceananigans.Utils: configure_kernel
const OUT = ARGS[1]
const ARCH = GPU()

## a throwaway model absorbs the first-model slowdown, as in phase2_step_times.jl
let (integrator, Δt) = setup(:land, ARCH; nlat_half = 2)
    time_steps(integrator, Δt, ARCH; nsteps = 10, reps = 1)
end
integrator, Δt = setup(:land, ARCH; nlat_half = 4)
state, model = integrator.state, integrator.model
grid = Terrarium.get_grid(model)
timestepper = Terrarium.get_timestepper(model)

"Median host time of `f()` in µs, with the GPU synchronized before each sample but not after."
host_time(f) = 1.0e6 * median(@benchmark($f(), setup = sync(ARCH), evals = 1)).time * 1.0e-9

# A single-variable XY prognostic, as stepped by the seven `explicit_step_xy_kernel!` launches of `land`
function find_xy_prognostic(state)
    for name in Terrarium.prognostic_names(state)
        field = state.prognostic[name]
        field isa Oceananigans.AbstractField{<:Any, <:Any, Nothing} && return field, state.tendencies[name]
    end
    for ns in state.namespaces
        found = find_xy_prognostic(ns)
        isnothing(found) || return found
    end
    return nothing
end
u, ∂u∂t = find_xy_prognostic(state)
dt = Terrarium.convert_dt(eltype(grid), Δt)

kernel! = Terrarium.explicit_step_xy_kernel!
args = (u, grid, ∂u∂t, timestepper, dt)
location = Oceananigans.instantiated_location(u)
configure() = configure_kernel(ARCH, grid, Val(:xy), kernel!, nothing, Val(false); location, reduced_dimensions = ())
loop!, _ = configure()
dargs = convert_to_device(ARCH, args)
cuargs = cudaconvert(dargs)
ndrange, workgroupsize, iterspace, _ = KA.launch_config(loop!, nothing, nothing)
ctx = KA.mkcontext(loop!, ndrange, iterspace)
maxthreads = prod(KA.get(KA.workgroupsize(loop!)))
compiled = @cuda launch = false always_inline = true maxthreads = maxthreads loop!.f(ctx, dargs...)
threads, blocks = length(KA.workitems(iterspace)), length(KA.blocks(iterspace))

## the CUDA.jl floor: a trivial kernel on one array, launched directly
trivial_kernel(x) = (@inbounds x[1] = 0; nothing)
x = CUDA.zeros(Float32, 1)
trivial = @cuda launch = false trivial_kernel(x)

stages = [
    "`Terrarium.explicit_step!` on one XY field (whole path)" => () -> Terrarium.explicit_step!(u, ∂u∂t, grid, timestepper, Δt),
    "`Oceananigans.launch!` with Terrarium's arguments" => () -> Oceananigans.launch!(ARCH, grid, Val(:xy), kernel!, args...),
    "`configure_kernel` (work layout and KA kernel object)" => configure,
    "`convert_to_device` of the arguments" => () -> convert_to_device(ARCH, args),
    "`cudaconvert` of the converted arguments" => () -> cudaconvert(dargs),
    "KA kernel call with converted arguments" => () -> loop!(dargs...),
    "`@cuda launch=false` (kernel lookup, includes `cudaconvert`)" => () -> @cuda(launch = false, always_inline = true, maxthreads = maxthreads, loop!.f(ctx, dargs...)),
    "launch of the compiled kernel (argument packing and driver call)" => () -> compiled(ctx, dargs...; threads, blocks),
    "trivial kernel, `@cuda` with lookup" => () -> @cuda(trivial_kernel(x)),
    "trivial kernel, precompiled launch" => () -> trivial(x),
]

open(OUT * "_stages.md", "w") do io
    say(io, "`land` at $(ncolumns(4)) columns, kernel `explicit_step_xy_kernel!` on $(summary(u))")
    say(io, "")
    say(io, "| Stage | Host time (µs) | Allocations |")
    say(io, "| --- | --- | --- |")
    for (name, f) in stages
        f()
        allocs = @allocations f()
        say(io, @sprintf("| %s | %.2f | %d |", name, host_time(f), allocs))
    end
    step() = timestep!(integrator, Δt; finalize = false)
    step()
    say(io, "")
    say(io, @sprintf("Host time of a whole `timestep!`, not synchronized: %.1f µs", host_time(step)))
    say(io, @sprintf("Wall time of a whole `timestep!`, synchronized: %.1f µs", 1.0e6 * median(@benchmark(($step(); sync(ARCH)), evals = 1)).time * 1.0e-9))
end

## host profile of many steps; launches are asynchronous, so the samples show the host work
step_many() = (for _ in 1:2000; timestep!(integrator, Δt; finalize = false); end; sync(ARCH))
step_many()
Profile.clear()
Profile.init(n = 10^7, delay = 1.0e-4)
@profile step_many()
open(OUT * "_profile_flat.txt", "w") do io
    Profile.print(IOContext(io, :displaysize => (5000, 400)); format = :flat, sortedby = :count, mincount = 20)
end
open(OUT * "_profile_tree.txt", "w") do io
    Profile.print(IOContext(io, :displaysize => (5000, 400)); format = :tree, maxdepth = 60, mincount = 50, noisefloor = 2)
end
