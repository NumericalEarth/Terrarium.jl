const USAGE = """
Host cost of one native GPU kernel launch, split into its stages, and a host profile of many steps.

Usage: julia --project=. launch_overhead.jl [key=value ...]

At small grid sizes the GPU is idle for most of a step and the step time is the host time spent between
launches. This tool times, on the host, each stage of launching `explicit_step_xy_kernel!` on one
single-variable XY prognostic field: Terrarium's `explicit_step!`, Oceananigans' `launch!`,
`configure_kernel`, `convert_to_device`, `cudaconvert`, the KernelAbstractions call, the CUDA.jl kernel
lookup, and the launch itself; and, for reference, a trivial CUDA.jl kernel. It writes the table to
`<out>_stages.md`, with the host and wall time of a whole step.

It then samples the host while running `profile_steps` steps (launches are asynchronous, so the samples
show host work) and writes flat and tree profiles to `<out>_profile_flat.txt` and `<out>_profile_tree.txt`.

Options:
  config=land            configuration (see ../model_configurations.jl); must have an XY prognostic field
  nlat=4                 grid size (`nlat_half`)
  profile_steps=2000     steps sampled for the host profile
  out=launch_overhead    prefix of the output files
"""
using CUDA
using Profile
include(joinpath(@__DIR__, "common.jl"))
import Oceananigans.Architectures: convert_to_device
import Oceananigans.Utils: configure_kernel

const OPTIONS = parse_options((config = :land, nlat = 4, profile_steps = 2000, out = "launch_overhead"), USAGE)
const ARCH = GPU()
const OUT = OPTIONS.out

absorb_first_model_slowdown(ARCH)
integrator, Δt = setup(OPTIONS.config, ARCH; nlat_half = OPTIONS.nlat)
state, model = integrator.state, integrator.model
grid = Terrarium.get_grid(model)
timestepper = Terrarium.get_timestepper(model)

"Median host time of `f()` in µs, with the GPU synchronized before each sample but not after."
host_time(f) = 1.0e6 * median(@benchmark($f(), setup = sync(ARCH), evals = 1)).time * 1.0e-9

"The first single-variable XY prognostic field and its tendency, searching namespaces depth first."
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
found = find_xy_prognostic(state)
isnothing(found) && error("configuration `$(OPTIONS.config)` has no XY prognostic field")
u, ∂u∂t = found
dt = Terrarium.convert_dt(eltype(grid), Δt)

## replicate the stages of `Oceananigans.launch!` and of KernelAbstractions' CUDA launch
kernel! = Terrarium.explicit_step_xy_kernel!
args = (u, grid, ∂u∂t, timestepper, dt)
location = Oceananigans.instantiated_location(u)
configure() = configure_kernel(ARCH, grid, Val(:xy), kernel!, nothing, Val(false); location, reduced_dimensions = ())
loop!, _ = configure()
dargs = convert_to_device(ARCH, args)
ndrange, workgroupsize, iterspace, _ = KA.launch_config(loop!, nothing, nothing)
ctx = KA.mkcontext(loop!, ndrange, iterspace)
maxthreads = prod(KA.get(KA.workgroupsize(loop!)))
always_inline = KA.backend(loop!).always_inline
compiled = @cuda launch = false always_inline = always_inline maxthreads = maxthreads loop!.f(ctx, dargs...)
threads, blocks = length(KA.workitems(iterspace)), length(KA.blocks(iterspace))

## the CUDA.jl floor: a trivial kernel on one array, launched directly
trivial_kernel(x) = (@inbounds x[1] = 0; nothing)
x = CUDA.zeros(Float32, 1)
trivial = @cuda launch = false trivial_kernel(x)

stages = [
    "`Terrarium.explicit_step!` on one XY field (whole path)" => () -> Terrarium.explicit_step!(u, ∂u∂t, grid, timestepper, Δt),
    "`Oceananigans.launch!` with Terrarium's arguments" => () -> Oceananigans.launch!(ARCH, grid, Val(:xy), kernel!, args...),
    "`configure_kernel` (work layout and KernelAbstractions kernel object)" => configure,
    "`convert_to_device` of the arguments" => () -> convert_to_device(ARCH, args),
    "`cudaconvert` of the converted arguments" => () -> cudaconvert(dargs),
    "KernelAbstractions kernel call with converted arguments" => () -> loop!(dargs...),
    "`@cuda launch=false` (kernel lookup, includes `cudaconvert`)" => () -> @cuda(launch = false, always_inline = always_inline, maxthreads = maxthreads, loop!.f(ctx, dargs...)),
    "launch of the compiled kernel (argument packing and driver call)" => () -> compiled(ctx, dargs...; threads, blocks),
    "trivial kernel, `@cuda` with lookup" => () -> @cuda(trivial_kernel(x)),
    "trivial kernel, precompiled launch" => () -> trivial(x),
]

open(OUT * "_stages.md", "w") do io
    say(io, environment_summary())
    say(io, "`$(OPTIONS.config)` at $(ncolumns(OPTIONS.nlat)) columns, kernel `explicit_step_xy_kernel!` on $(summary(u))")
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

function step_many()
    for _ in 1:OPTIONS.profile_steps
        timestep!(integrator, Δt; finalize = false)
    end
    return sync(ARCH)
end
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
say(stdout, "profiles written to $(OUT)_profile_flat.txt and $(OUT)_profile_tree.txt")
