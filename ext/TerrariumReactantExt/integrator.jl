const ReactantIntegrator{NF} = ModelIntegrator{NF, <:RARCH} where {NF}

# Compiled time stepping.
#
# `run!` on a device integrator compiles the whole stepping loop into a single StableHLO
# program (via `@trace for`) and runs it. The compiled function can be reused across runs by
# passing it back as the `compiled_run!` keyword. The traced loop body calls the timestepper-level
# `timestep!` (`timestep!(integrator, timestepper, Δt)`), which is not overridden here. This
# avoids recursively re-entering the compiling wrapper.

# Reactant specialization of `Terrarium.run_timesteps!` (generic host loop defined in src). Here
# `Reactant.@trace` compiles the loop to a single `stablehlo.while` (rather than unrolling `Nt`
# copies of the step into the trace); this is the function compiled by `run!` and mirrors
# Oceananigans' `run_timesteps!` test helper.
#
# `checkpointing` is forwarded to `@trace` and controls reverse-mode-AD memory: `false` (default)
# stores every intermediate state; a scheme such as `Reactant.Periodic(n)`/`Reactant.Binomial(n)`
# stores only `n` checkpoints and recomputes the rest. It has no effect on a pure forward run but
# matters when this function is differentiated (see the autodiff example).
function Terrarium.run_timesteps!(integrator::ReactantIntegrator{NF}, Δt, Nt, checkpointing = false) where {NF}
    timestepper = get_timestepper(integrator.model)
    Δt = Terrarium.convert_dt(NF, Δt)
    Reactant.@trace checkpointing = checkpointing track_numbers = false for _ in 1:Nt
        Terrarium.timestep!(integrator, timestepper, Δt)
    end
    # Update auxiliary variables for the final timestep.
    Terrarium.compute_auxiliary!(integrator.state, integrator.model)
    return nothing
end

"""
    run!(integrator; steps, period, Δt, checkpointing = false, compiled_run! = nothing)

Run a `ReactantState` integrator. If `compiled_run!` is `nothing`, the stepping loop is compiled
here with Reactant (`raise=true` lowers the KernelAbstractions kernels — halo fills,
tendency/closure kernels — so their traced grid/array arguments are adapted correctly).
The compiled function is returned-through by being reusable: pass it back as `compiled_run!` on
subsequent runs with the same `Δt`, step count, and `checkpointing` to skip recompilation.

`checkpointing` selects the reverse-mode-AD checkpointing strategy for the traced loop — `false`
(default), or a scheme such as `Reactant.Periodic(n)`/`Reactant.Binomial(n)`. It has no effect on
a forward run but bounds the memory of a reverse pass when the compiled function is differentiated.
"""
function Oceananigans.Simulations.run!(
        integrator::ReactantIntegrator{NF};
        steps::Union{Int, Nothing} = nothing,
        period::Union{Terrarium.Period, Nothing} = nothing,
        Δt = Terrarium.default_dt(get_timestepper(integrator.model)),
        checkpointing = false,
        compiled_run! = nothing,
    ) where {NF}
    Δt = Terrarium.convert_dt(NF, Δt)
    nsteps = Terrarium.get_steps(steps, period, Δt)
    if isnothing(compiled_run!)
        @info "Reactant: compiling run_timesteps! (Δt=$Δt, steps=$nsteps, checkpointing=$checkpointing)"
        compiled_run! = @compile raise = true raise_first = true sync = true run_timesteps!(integrator, Δt, nsteps, checkpointing)
    end
    compiled_run!(integrator, Δt, nsteps, checkpointing)
    return integrator
end

# Single step: the same compiled path with one iteration (no persistent compilation cache — a
# repeated single-step loop should use `run!` instead, which compiles the loop once).
function Terrarium.timestep!(integrator::ReactantIntegrator, Δt; finalize = true)
    run!(integrator; steps = 1, Δt)
    return nothing
end

#
# We here override `default_clock` for Reactant device models to get a Reactant-backed `Clock`
Terrarium.default_clock(model::ReactantModel) = Oceananigans.TimeSteppers.Clock(ground_domain(get_grid(model)))

# A compiled run updates the clock's `ConcreteRNumber` time in place, so the integrator's initial
# time must be a copy of its current value rather than a reference to the same object. Assigning
# the plain number back (in `reset_clock!`) converts it to the clock's `ConcreteRNumber` type again.
Terrarium.initial_clock_time(time::Reactant.ConcreteRNumber) = Reactant.to_number(time)
