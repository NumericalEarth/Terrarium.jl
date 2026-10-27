# # Differentiating Terrarium.jl with Reactant
#
# The [differentiation example](@ref) shows how to take gradients of Terrarium simulations with
# [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) on the CPU, using
# [Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl) to keep the
# reverse pass memory-bounded. Here we do the same thing through
# [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl): the whole gradient computation is traced
# to MLIR/StableHLO and compiled with XLA, and checkpointing is expressed directly with Reactant's
# `@trace checkpointing=true` loop instead of Checkpointing.jl.
#
# As everywhere with Reactant in Terrarium, the only change to the model itself is the
# architecture: we build the grid on `ReactantState()`.

using Terrarium
using Reactant, CUDA   # CUDA is required by Reactant's kernel integration, even on CPU
using Enzyme

import CairoMakie as Makie

# We set up a single soil column on the device, with the same exponentially stretched vertical grid
# as the CPU example.

grid = ColumnGrid(ReactantState(), Float32, ExponentialSpacing())
model = SoilModel(grid)
bcs = PrescribedSurfaceTemperature(:T_ub, 1.0f0)   # constant 1 °C surface temperature
initializers = (temperature = (x, z) -> -1.0f0 - 0.05f0 * z,)
integrator = initialize(model; boundary_conditions = bcs, initializers)

# We differentiate a scalar objective — the mean-square soil temperature after `nsteps` — with
# respect to the initial state. The time loop calls the raw per-step functions (`timestep!` with
# the model's timestepper, then `compute_auxiliary!`) rather than `run!`, because `run!` itself
# compiles and we want to compile the *whole* gradient in one go. [`run_timesteps!`](@ref) is the
# stepping loop underlying `run!`; on `ReactantState` it traces the loop and takes a `checkpointing`
# scheme. Passing `Reactant.Periodic(n)` stores `n` checkpoints and recomputes the rest during the
# reverse pass, keeping memory bounded for long integrations (the same scheme is accepted by `run!`
# via its `checkpointing` keyword).

function loss(integrator, Δt, nsteps, checkpointing)
    run_timesteps!(integrator, Δt, nsteps, checkpointing)
    T = interior(integrator.state.temperature)
    return sum(T .^ 2) / length(T)
end

# `Enzyme.autodiff` in reverse mode computes the objective together with its gradient. We pass the
# integrator as `Duplicated`, so its shadow `dintegrator` accumulates the sensitivity of the loss
# with respect to every state variable — in particular the initial internal energy `U₀`.

function grad_loss!(integrator, dintegrator, Δt, nsteps, checkpointing)
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(integrator, dintegrator),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps),
        Enzyme.Const(checkpointing),
    )
    return loss_value
end

# We allocate the shadow with `make_zero` and compile the gradient with Reactant. As with a forward
# `run!`, the first call compiles (this can take a few minutes); afterwards it is fast.

dintegrator = Enzyme.make_zero(integrator)
Δt = 600.0f0
nsteps = 200
checkpointing = Reactant.Periodic(isqrt(nsteps))   # ≈ √n checkpoints

compiled_grad! = @compile raise = true raise_first = true sync = true grad_loss!(integrator, dintegrator, Δt, nsteps, checkpointing)
loss_value = compiled_grad!(integrator, dintegrator, Δt, nsteps, checkpointing)

# The sensitivity ``\partial \text{loss} / \partial U_0`` of the objective with respect to the
# initial internal energy of each soil layer now lives in `dintegrator`. We move it to the CPU to
# plot it against depth.

dU = Array(interior(dintegrator.state.internal_energy))[1, 1, :]
zs = znodes(integrator.state.temperature)

f = Makie.Figure()
Makie.Axis(f[1, 1], ylabel = "Soil depth / m", xlabel = "Sensitivity ∂loss/∂U₀")
Makie.scatterlines!(f[1, 1], dU, zs)
f
