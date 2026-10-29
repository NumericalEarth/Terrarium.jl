# Reactant-specific stepping runner. Included by the driver only for the `reactant-*` architectures,
# after `using Reactant` — the `@compile` macro must be resolvable when this file is parsed.
#
# `run!` on a `ReactantState` integrator compiles the stepping loop on every call: its `compiled_run!`
# argument lets a caller supply a previously compiled program, but the compiled program is never
# handed back (see `ext/TerrariumReactantExt/integrator.jl`). Timing `run!` would therefore time the
# XLA compiler, not the model. We compile once here — with exactly the flags the extension uses — and
# report the compile time as its own metric, since for Reactant it is a headline number in itself.
#
# TODO: This should be regarded as a temporary approach, the next Reactant related revision should rather
# allow the compiled call to be handed over directly without redefining this here

function build_runner(::ReactantState, integrator, Δt, nsteps)
    compile_time = @elapsed compiled_run! = @compile raise = true raise_first = true sync = true run_timesteps!(integrator, Δt, nsteps, false)
    @info @sprintf("     compiled the %d-step loop in %.1f s", nsteps, compile_time)
    ## The compiled program has a fixed step count, so `n` is ignored and the warm-up is a full run.
    ## `sync = true` makes the call block until the device is done, so the wallclock around it is the
    ## execution time.
    step!(n) = compiled_run!(integrator, Δt, nsteps, false)
    return (step! = step!, compile_time = compile_time)
end
