# # Differentiating Terrarium.jl
#
# We build Terrarium with differentiability in mind. This means that you are able to take derivatives of outputs of Terrarium with automatic differentiation (AD). AD enables us to use e.g. an automated, objective calibration of model parameters, but also the direct integration of neural networks and other machine learning (ML) methods into our model. For this purpose we ensure compatibility on [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). Enzyme.jl can perform both a reverse-mode AD (typical for most ML applications), and a forward-mode AD (more typical in classical sensitivity analysis).
#
# When differentiating through a model integration, AD would usually need to keep track of every single intermediate value that contributes to our output. For long integrations this quickly becomes infeasible due to its high memory demand. Therefore we support checkpointing schemes from [Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl) for these cases that only save selected intermediate values and recompute all other intermediate values when needed. For Enzyme.jl this also has another very practical advantage: the first compile time for the gradient is much lower.
#
# Without much further ado, let us look into how we can differentiate Terrarium hands-on and perform a small sensitivity analysis of a one column soil model. First, we set up our model as usual:

using Terrarium

using Accessors
using Enzyme
using Enzyme: Forward, Reverse, set_runtime_activity
using Checkpointing
using CairoMakie
using Statistics

arch = CPU()
NF = Float32
grid = ColumnGrid(arch, NF, ExponentialSpacing())
initializer = SoilInitializer(NF)
model = SoilModel(grid; timestepper = ForwardEuler(NF), initializer = initializer)
# constant surface temperature of 1°C
bcs = PrescribedSurfaceTemperature(:T_ub, NF(1.0))
integrator = initialize(model, boundary_conditions = bcs)

# So far, this is just our usual setup. In this case, for a soil column with a prescribed surface temperature.
#
# Now, we set up our AD checkpointing scheme for the timestepping. Here we choose a [Revolve scheme](https://dl.acm.org/doi/10.1145/347837.347846) that saves intermediate values at every single time step. Note that when we save at every single time step the different available schemes don't actually differ from each other.

scheme = Revolve(1)

# Next we prepare to differentiate with Enzyme. For a comprehensive introduction to Enzyme, please see [their documentation](https://enzymead.github.io/Enzyme.jl/stable/).
#
# We want to perform a sensitivity analysis of the temperature of the second lowest soil layer ``T_f`` at the end of our simulation with respect to the initial conditions of our simulation ``\mathbf{U}_0``, ``\mathbf{T}_0``, where ``\mathbf{U}`` is the internal energy.
#
# Enzyme's `autodiff` is it's core function that we can use to compute vector-Jacobian products (vJP) for the reverse-mode AD of our `run!` function that integrates our model using the `integrator` that we initialized. We define a function that returns the temperature of the second lowest soil layer after a time integration of 10 steps.

dintegrator = make_zero(integrator)

function layer_temperature(integrator)
    run!(integrator, scheme, 10)
    return interior(integrator.state.temperature)[1, 1, 2]
end

# While doing that we allocated a shadow memory `dintegrator` for Enzyme in which it can accumulate the vJP (see Enzyme docs for more information).
# We just need to call `autodiff` now. Some notes on its arguments:
# - `set_runtime_activity(Reverse)` tells Enzyme to use reverse-mode AD while enabling runtime activity (see [here](https://enzymead.github.io/Enzyme.jl/dev/faq/#faq-runtime-activity) for details)
# - `Active` tells Enzyme that we want to take the derivative of the scalar output of `layer_temperature` with respect to the function's input. The cotangent of the function output is set to 1.0 in this way for the vJP calculation.
# - `Duplicated(integrator, dintegrator)` `dintegrator` is shadow memory that Enzyme uses to accumulate the vJP of the `integrator` state variables.
#
# Executing this for the first time, might take a few minutes. Subsequent executions will be very fast though.

@time autodiff(set_runtime_activity(Reverse), layer_temperature, Active, Duplicated(integrator, dintegrator))

# Let's look at the results that were accumulated in our shadow memory `dintegrator` by Enzyme and plot them!

dU = interior(dintegrator.state.internal_energy)[1, 1, :]
dT = interior(dintegrator.state.temperature)[1, 1, :]
zs = znodes(integrator.state.temperature)

# For temperature:

f = Makie.Figure()
Makie.Axis(f[1, 1], ylabel = "Soil depth", xlabel = "Sensitivity dT_f/dT_0")
Makie.scatterlines!(f[1, 1], dT, zs)
f

# For internal energy:

f2 = Makie.Figure()
Makie.Axis(f2[1, 1], ylabel = "Soil depth", xlabel = "Sensitivity dT_f/dU_0")
Makie.scatterlines!(f2[1, 1], dU, zs)
f2

# As expected the sensitivity is the highest locally, with the same and neighboring soil layers contributing and no sensitivity wrt higher soil layers for our still rather short integration.

# ## Parameter sensitivities
#
# We can also compute sensitivities with respect to *model parameters* rather than initial
# conditions. Which mode to use depends on how many parameters we are asking about: forward mode
# costs one pass per parameter and is the natural choice for a single one, while reverse mode
# returns the derivatives with respect to *all* parameters from a single pass. We do both below.
#
# ### Forward mode: a single parameter
#
# Since we are interested in only a single parameter here — the soil quartz thermal
# conductivity ``\kappa_\text{quartz}`` — forward-mode AD is the natural choice: it
# propagates one tangent vector forward through the computation in a single pass, without
# the memory overhead of reverse mode.
# As a consequence, there is no need to use Checkpointing.jl for this example.
# To make sure that ``\kappa_\text{quartz}`` has a physical effect, we construct the model with a soil consisting of 100% sand.
grid = ColumnGrid(arch, NF, UniformSpacing(Δz = 0.1, N = 100))
initializer = SoilInitializer(NF)
text = SoilTexture(NF; sand = NF(1.0))
strat = HomogeneousSoilStratigraphy(NF; texture = text)
soil = SoilEnergyWaterCarbon(NF; strat)
model = SoilModel(grid; timestepper = ForwardEuler(NF), initializer = initializer)

# We re-initialize a fresh integrator so the state starts at ``t = 0`` and set a constant surface temperature of 1°C
bcs = PrescribedSurfaceTemperature(:T_ub, NF(1.0))
integrator = initialize(model, boundary_conditions = bcs)
dintegrator = make_zero(integrator)

# For physical interpretation, we first plot the initial temperature profile of the soil column:

f3 = Makie.Figure()
Makie.Axis(f3[1, 1], ylabel = "Soil depth (m)", xlabel = "Temperature (°C)")
Makie.scatter!(f3[1, 1], integrator.state.temperature)
f3

# By setting a surface temperature of 1°C, the soil will be heated up from above.
# At the bottom, there is a no flux boundary condition.
# As the initial profile gets warmer with depth, the bottom will start cooling down towards thermal equilibrium.
# Only for ``\kappa_\text{quartz}`` we set the tangent to 1.0, so that the jacobian-vector products (jVPs) computed by forward-mode AD can accumulate the sensitivity to it.

@reset dintegrator.model.soil.energy.thermal_properties.conductivities.quartz = NF(1.0)

# Define a function which returns the entire temperature profile after a time integration of 400 time steps.

function final_temperatures(integrator)
    run!(integrator; steps = 400) # No checkpointing!
    return interior(integrator.state.temperature)[1, 1, :]
end

# Now we call `autodiff` in Forward mode over this function. Some notes on the arguments:
# - As for reverse-mode, we use `set_runtime_activity(Forward)` to enable runtime activity.
# - `Duplicated` to denote that we want to take the derivative of `final_temperatures`
# - `Duplicated(integrator, dintegrator)` to make clear that we want to differentiate with respect to the `integrator` state variables, and that the vJP (the gradient) should be accumulated in `dintegrator`.

(∂T_∂κ_quartz,) = autodiff(set_runtime_activity(Forward), final_temperatures, Duplicated, Duplicated(integrator, dintegrator))

# The output `∂T_∂κ_quartz` holds the sensitivities ``\partial T(k) / \partial \kappa_\text{quartz}`` for each layer ``k``.
# We can plot the final temperature profile and the sensitivities as a function of soil depth:

zs = znodes(integrator.state.temperature)

# Final temperature profile:

f4 = Makie.Figure()
Makie.Axis(f4[1, 1], ylabel = "Soil depth (m)", xlabel = "Temperature (°C)")
Makie.scatter!(f4[1, 1], integrator.state.temperature)
f4

# Sensitivities:

f5 = Makie.Figure()
Makie.Axis(f5[1, 1], ylabel = "Soil depth (m)", xlabel = "Sensitivity ∂T/∂κ_quartz  (K / (W·m⁻¹⋅K⁻¹))")
Makie.scatterlines!(f5[1, 1], ∂T_∂κ_quartz, zs)
f5

# The behavior of the sensitivity is physically consistent: the higher the conductivity, the quicker the heating from above (with the surface temperature higher than the initial soil temperature) can propagate downwards. Logically, there is only sensitivity for layers that have already seen a change in temperature during this (short) time integration.

# ### Reverse mode: the full parameter set
#
# Picking one parameter by hand does not scale to calibration, where we want the gradient with
# respect to everything the model exposes. Terrarium collects the parameters of a model, including
# those of all nested components, with `parameters`:

params = Terrarium.parameters(model)

# The result is a table of every parameter, its value, its bounds, and a description. To
# differentiate with respect to them we need them as a flat vector, which `vec` provides as a
# `ComponentVector`, i.e. a plain vector of values that still knows the name of each entry:

θ = vec(params)
parameter_names = Terrarium.ParameterEditing.labels(θ)

# Now we define the quantity we want the sensitivities of: the mean soil temperature after 50 time
# steps. The parameter vector enters the model through `reconstruct`, which rebuilds the model
# components with the given values. This happens *inside* the differentiated function, which is
# what allows Enzyme to track each parameter from the vector into the model components and back out
# through the integration.

function mean_temperature(θ, integrator, steps)
    perturbed_model = Terrarium.ParameterEditing.reconstruct(integrator.model, θ)
    perturbed = ModelIntegrator(
        integrator.clock,
        perturbed_model,
        integrator.inputs,
        integrator.state,
        integrator.initializers,
    )
    run!(perturbed; steps)
    return mean(interior(perturbed.state.temperature))
end

# `dθ` is the shadow in which Enzyme accumulates the gradient, one entry per parameter. Note that
# we integrate without checkpointing here: rebuilding the model from a parameter vector inside the
# differentiated function is currently incompatible with Checkpointing.jl, so keep the number of
# time steps modest when differentiating with respect to parameters.

dθ = make_zero(θ)
dintegrator = make_zero(integrator)
@time autodiff(
    set_runtime_activity(Reverse),
    mean_temperature,
    Active,
    Duplicated(θ, dθ),
    Duplicated(integrator, dintegrator),
    Const(50),
)

# A single reverse pass has given us all 19 derivatives. Parameters carry different units and span
# several orders of magnitude, so comparing the raw derivatives is misleading; we instead scale each
# one by its parameter value, giving the response to a *fractional* change in that parameter,
# ``\theta_i \, \partial \bar{T} / \partial \theta_i``, in kelvin:

scaled_sensitivities = θ .* dθ

f6 = Makie.Figure(size = (900, 450))
ax = Makie.Axis(
    f6[1, 1],
    ylabel = "Sensitivity θᵢ ∂T̄/∂θᵢ (K)",
    xlabel = "Parameter",
    xticks = (1:length(θ), replace.(parameter_names, "soil." => "")),
    xticklabelrotation = π / 4,
)
Makie.scatterlines!(ax, 1:length(θ), Array(scaled_sensitivities))
f6

# The thermal parameters dominate, as expected for a purely conductive warming experiment: raising
# any of the thermal conductivities warms the column faster, while raising the heat capacities slows
# it down, which is exactly the sign pattern we see. The remaining parameters are identically zero
# here, and that is informative rather than suspicious: the soil is unfrozen, so the ice properties
# never enter; the hydrology is configured as `NoFlow`, so no water moves; and the soil carries no
# organic fraction, so its organic properties have nothing to act on.
#
# These examples should just demonstrate the technical possibilities of Terrarium.jl in an easy and fast to compute setup, stay tuned for more complex examples.
