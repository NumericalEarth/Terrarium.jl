# Resources and device time of the surface energy balance kernel of `land` on the GPU (Phase 2 item 6).
# usage: julia --project=. seb_kernel.jl <label> <outprefix> [newton|root]
# The skin temperature solver defaults to Newton, as in the benchmarks; `root` selects the default `RootSolver`.
using CUDA
include(joinpath(@__DIR__, "common.jl"))
import Oceananigans.Architectures: convert_to_device
import Oceananigans.Utils: configure_kernel
const LABEL, OUT = ARGS[1], ARGS[2]
const SOLVER = length(ARGS) > 2 ? ARGS[3] : "newton"
const ARCH = GPU()

const F64_ARITH = r"\b(add|sub|mul|div|fma|neg|abs|min|max|sqrt|rcp|rsqrt|setp\.[a-z]+|ex2|lg2)(\.[a-z]+)*\.f64\b"
const CVT_F64 = r"\bcvt(\.[a-z]+)*\.f64\.f32\b|\bcvt(\.[a-z]+)*\.f32\.f64\b"
count_matches(re, s) = length(collect(eachmatch(re, s)))

model_kwargs = SOLVER == "root" ? root_solver_kwargs(Float32) : newton_kwargs(Float32)
integrator, Δt = setup(:land, ARCH; nlat_half = 144, model_kwargs)
state, model = integrator.state, integrator.model
grid = Terrarium.get_grid(model)
seb, constants, atmos = model.surface_energy_balance, model.constants, model.atmosphere
hydrology, snow = model.surface_hydrology, model.snow

## the arguments of the launch in `solve_surface_energy_balance!`
out = (skin_temperature = state.skin_temperature, Terrarium.auxiliary_fields(state, seb)...)
fields = Terrarium.get_fields(state, seb, atmos, hydrology, snow)
kernel! = Terrarium.solve_surface_energy_balance_kernel!
args = (out, grid, fields, seb, constants, atmos, hydrology, snow)
loop!, _ = configure_kernel(ARCH, grid, Val(:xy), kernel!, nothing, Val(false); location = Oceananigans.instantiated_location(out), reduced_dimensions = ())
dargs = convert_to_device(ARCH, args)
ndrange, workgroupsize, iterspace, _ = KA.launch_config(loop!, nothing, nothing)
ctx = KA.mkcontext(loop!, ndrange, iterspace)
maxthreads = prod(KA.get(KA.workgroupsize(loop!)))
## the same compiler options as the KernelAbstractions launch
const ALWAYS_INLINE = KA.backend(loop!).always_inline
compiled = @cuda launch = false always_inline = ALWAYS_INLINE maxthreads = maxthreads loop!.f(ctx, dargs...)

tt = Tuple{typeof(ctx), map(a -> typeof(cudaconvert(a)), dargs)...}
ptx = sprint(io -> CUDA.code_ptx(io, loop!.f, tt; kernel = true, always_inline = ALWAYS_INLINE, maxthreads))
llvm = sprint(io -> CUDA.code_llvm(io, loop!.f, tt; kernel = true, always_inline = ALWAYS_INLINE, maxthreads, optimize = true))
write(OUT * "_seb.ptx", ptx)
write(OUT * "_seb.ll", llvm)

solve!() = Terrarium.solve_surface_energy_balance!(state, grid, seb, constants, atmos, hydrology, snow)
solve!()
CUDA.synchronize()
nlaunch = 50
times = map(1:5) do _
    1.0e3 * CUDA.@elapsed(for _ in 1:nlaunch; solve!(); end) / nlaunch
end
step_time = time_steps(integrator, Δt, ARCH; nsteps = 20, reps = 5)

open(OUT * "_seb.md", "a") do io
    mem = CUDA.memory(compiled)
    say(io, @sprintf("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %.1f (min %.1f) | %s |",
        "$LABEL ($SOLVER)", CUDA.registers(compiled), mem.local, CUDA.maxthreads(compiled),
        count_matches(F64_ARITH, ptx), count_matches(CVT_F64, ptx), count_matches(r"\btrap;", ptx),
        count_matches(r"\bcall(\.uni)?\b", ptx), count('\n', ptx),
        1.0e3 * median(times), 1.0e3 * minimum(times), fmt_times(step_time)))
end
