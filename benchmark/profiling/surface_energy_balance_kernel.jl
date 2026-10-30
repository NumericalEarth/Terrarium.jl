const USAGE = """
Compiled resources and device time of the surface energy balance kernel on the GPU.

Usage: julia --project=. surface_energy_balance_kernel.jl [key=value ...]

Compiles `solve_surface_energy_balance_kernel!` with the same options as its KernelAbstractions launch and
appends one Markdown table row: registers per thread, local memory per thread (bytes), the maximum threads
per block, counts in its PTX of Float64 arithmetic, Float32↔Float64 conversions, `trap` instructions, and
calls (non-inlined device functions, such as `throw_boundserror` for bounds checks that were not elided),
the PTX length, the device time of one `solve_surface_energy_balance!` (median of five repetitions of
`launches` launches, with the minimum), and the ms/step of the whole model.
With `dump=<prefix>`, it also writes the kernel's PTX and LLVM IR to `<prefix>.ptx` and `<prefix>.ll`.

Options:
  config=land      configuration with a surface energy balance (see ../model_configurations.jl)
  solver=newton    skin temperature solver: `newton` (as in the benchmarks) or `root` (the model default,
                   `RootSolver`); see `solver_kwargs` in common.jl
  nlat=144         grid size (`nlat_half`)
  launches=50      launches per timed repetition
  dump=            prefix for the PTX and LLVM IR files (default: not written)
  label=           free-text label for the first column
  header=false     print the table header first
  out=             file to append the row to (default: standard output only)
"""
using CUDA
include(joinpath(@__DIR__, "common.jl"))
import Oceananigans.Architectures: convert_to_device
import Oceananigans.Utils: configure_kernel

const OPTIONS = parse_options(
    (config = :land, solver = :newton, nlat = 144, launches = 50, dump = "", label = "", header = false, out = ""),
    USAGE
)
const ARCH = GPU()

const F64_ARITH = r"\b(add|sub|mul|div|fma|neg|abs|min|max|sqrt|rcp|rsqrt|setp\.[a-z]+|ex2|lg2)(\.[a-z]+)*\.f64\b"
const CVT_F64 = r"\bcvt(\.[a-z]+)*\.f64\.f32\b|\bcvt(\.[a-z]+)*\.f32\.f64\b"
count_matches(rx, s) = length(collect(eachmatch(rx, s)))

integrator, Δt = setup(OPTIONS.config, ARCH; nlat_half = OPTIONS.nlat, model_kwargs = solver_kwargs(OPTIONS.solver, Float32))
state, model = integrator.state, integrator.model
grid = Terrarium.get_grid(model)
seb, constants, atmos = model.surface_energy_balance, model.constants, model.atmosphere
hydrology, snow = model.surface_hydrology, model.snow
solver_name = nameof(typeof(seb.skin_temperature.solver))

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
always_inline = KA.backend(loop!).always_inline
compiled = @cuda launch = false always_inline = always_inline maxthreads = maxthreads loop!.f(ctx, dargs...)

tt = Tuple{typeof(ctx), map(a -> typeof(cudaconvert(a)), dargs)...}
ptx = sprint(io -> CUDA.code_ptx(io, loop!.f, tt; kernel = true, always_inline, maxthreads))
if !isempty(OPTIONS.dump)
    write(OPTIONS.dump * ".ptx", ptx)
    write(OPTIONS.dump * ".ll", sprint(io -> CUDA.code_llvm(io, loop!.f, tt; kernel = true, always_inline, maxthreads, optimize = true)))
end

solve!() = Terrarium.solve_surface_energy_balance!(state, grid, seb, constants, atmos, hydrology, snow)
solve!()
CUDA.synchronize()
times = map(1:5) do _
    1.0e3 * CUDA.@elapsed(
        for _ in 1:OPTIONS.launches
            solve!()
        end
    ) / OPTIONS.launches
end
step_time = time_steps(integrator, Δt, ARCH; nsteps = 20, reps = 5)

with_output(OPTIONS.out) do io
    if OPTIONS.header
        say(io, environment_summary())
        say(io, "| label | solver | columns | registers | local bytes | max threads/block | f64 arith | f32↔f64 cvt | trap | call | PTX lines | kernel time (µs, median (min)) | ms/step |")
        say(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    end
    mem = CUDA.memory(compiled)
    say(io, @sprintf("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %.1f (%.1f) | %s |",
        OPTIONS.label, solver_name, ncolumns(OPTIONS.nlat), CUDA.registers(compiled), mem.local, CUDA.maxthreads(compiled),
        count_matches(F64_ARITH, ptx), count_matches(CVT_F64, ptx), count_matches(r"\btrap;", ptx),
        count_matches(r"\bcall(\.uni)?\b", ptx), count('\n', ptx),
        1.0e3 * median(times), 1.0e3 * minimum(times), fmt_times(step_time)))
end
