const USAGE = """
Experiment: effect of forcing `Vararg` specialization on a native CPU kernel.

Usage: julia --project=. experiments/vararg_specialization.jl [key=value ...]

Julia does not specialize a function on a trailing `args...` that it only passes through, so a kernel
written that way boxes its arguments for every grid cell on the CPU. This experiment measures
`closure!` of `soil_heat` (allocations per call and median ns per cell), redefines
`energy_to_temperature_kernel!` with `args::Vararg{Any, N}) where {N}`, measures again, and reports the
largest change of the resulting temperatures, which should be zero.

If the kernel already has the typed signature, both measurements are the same. The redefinition copies the
kernel's body, so it must be updated if that body changes. `kernel_allocations.jl` finds kernels that
allocate in general.

Options:
  nlat=24     grid size (`nlat_half`)
  out=        file to append the results to (default: standard output only)
"""
include(joinpath(@__DIR__, "..", "common.jl"))

const OPTIONS = parse_options((nlat = 24, out = ""), USAGE)

function measure(io, label, state, model, ncells)
    Terrarium.closure!(state, model)
    n = @allocations Terrarium.closure!(state, model)
    b = @benchmark Terrarium.closure!($state, $model) evals = 1 seconds = 3
    return say(io, @sprintf("%-28s closure!: %8d allocs/call, %.2f ns/cell (median)", label, n, median(b).time / ncells))
end

with_output(OPTIONS.out) do io
    say(io, environment_summary())
    integrator, Δt = setup(:soil_heat, CPU(); nlat_half = OPTIONS.nlat)
    state, model = integrator.state, integrator.model
    ncells = ncolumns(OPTIONS.nlat) * 30
    measure(io, "current signature", state, model, ncells)
    before = copy(Array(interior(state.temperature)))
    Core.eval(
        Terrarium, quote
            @kernel inbounds = true function energy_to_temperature_kernel!(out, grid, fields, args::Vararg{Any, N}) where {N}
                i, j, k = @index(Global, NTuple)
                energy_to_temperature!(out, i, j, k, grid, fields, args...)
            end
        end
    )
    Base.invokelatest(measure, io, "Vararg{Any, N} where {N}", state, model, ncells)
    after = Array(interior(state.temperature))
    say(io, "largest absolute temperature difference: $(maximum(abs, after .- before))")
end
