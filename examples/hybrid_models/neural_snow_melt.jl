# # Hybrid modeling: a neural-network snow melt trained with Reactant + Enzyme
#
# This example is a **technical demonstration** of *hybrid* (physics + machine-learning) modeling
# in Terrarium: we take the degree-day snow model from the [snow example](@ref snow_ddm_example)
# and replace its analytic melt law with a small neural network, which we train to reproduce the
# original melt using [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (compilation) and
# [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) (reverse-mode gradients). It is deliberately
# a toy problem - the NN just relearns a function we already know - but it shows the full
# machinery: generating training data from a Terrarium process, training an MLP with Reactant +
# Enzyme, and evaluating that MLP *inside a kernel* as a drop-in replacement process.
#
# ## What we are (and are not) doing
#
# * The degree-day melt is `M(T) = max(0, k·(T − T_melt))` — a rectified linear function of
#   temperature. A tiny MLP can represent it. We first train the MLP offline on `(T, M(T))` pairs,
#.  and then continue the training online within a Terrarium model later.
# * The `NeuralSnowMelt` process evaluates the trained MLP **per grid point inside a
#   KernelAbstractions kernel**. Plain `Lux.apply` cannot be compiled inside a kernel, so we use
#   [KernelLux.jl](https://github.com/maximilian-gelbrecht/KernelLux.jl), whose `apply_in_kernel`
#   is a device-safe drop-in for `Lux.apply` for a `Chain` of `Dense` layers.

using Terrarium
using Reactant, CUDA          # CUDA is required by Reactant's kernel integration, even on CPU
using Lux, Enzyme, Optimisers, Random
using KernelLux               # device-safe `apply_in_kernel` / `kernelize` for Lux in kernels
using StaticArrays: SA
using Statistics: mean, std
using KernelAbstractions: @index, @kernel
using Terrarium: launch!, XY, interior, AbstractProcess
using Terrarium: Adapt          # Adapt is re-exported by Terrarium (transitive dep)
import Lux.MLDataDevices as MLDD
using Terrarium: Snow, Surface

# ## The reference degree-day model
#
# This is the analytic process we want to emulate (as in the snow example). We also expose its
# melt law `melt(process, T)` directly, so we can generate training targets from it.

@kwdef struct DegreeDaySnow{NF} <: AbstractProcess{NF}
    "Degree-day factor [m/(K s)]"
    k::NF = 5.0e-3 / (24 * 60 * 60)
    "Melting point of snow on the ground [°C]"
    T_melt::NF = 0.0
end

DegreeDaySnow(::Type{NF}) where {NF} = DegreeDaySnow{NF}()

"Analytic melt rate `M(T) = max(0, k (T − T_melt))` [m/s]."
@inline melt(p::DegreeDaySnow, T) = max(zero(T), p.k * (T - p.T_melt))

# ## The neural snow melt process
#
# `NeuralSnowMelt` mirrors `DegreeDaySnow` — same variables, same tendency `dS/dt = P − M` — but
# computes the melt `M` from an MLP with a single hidden layer (`1 → H → 1`, tanh). It stores the
# Lux model together with its parameters/state, and the input/output normalization constants used
# during training, so the kernel can evaluate the network with `KernelLux.apply_in_kernel`.

# The `model` is the `Lux.Chain` (immutable/isbits — just layer dims and activations); `ps`/`st`
# are the Lux parameter and state objects, on whatever architecture the network runs on.
struct NeuralSnowMelt{NF, M, PS, ST} <: AbstractProcess{NF}
    "The Lux MLP (`Chain` of `Dense` layers)"
    model::M
    "Lux parameters of the melt MLP (`NamedTuple` of layer weights/biases)"
    ps::PS
    "Lux state of the melt MLP"
    st::ST
    "Input normalization for temperature (°C)"
    T_mean::NF
    T_std::NF
    "Output scale mapping the network's O(1) output back to a melt rate [m/s]"
    M_scale::NF
end
Adapt.@adapt_structure NeuralSnowMelt   # so the parameter arrays move with the architecture

# `NeuralSnowMeltBatched` holds exactly the same data as `NeuralSnowMelt`, but its
# `compute_tendencies!` (below) evaluates the MLP as a single *batched* `Lux.apply` over all
# columns at once — **without** a `launch!`/kernel. This is the second demonstration approach in
# the online-training section: see the discussion there for the trade-offs versus the in-kernel
# process. It stores the plain Lux parameters (no `kernelize` — the batched forward is ordinary
# array math, not in-kernel scalar code).
struct NeuralSnowMeltBatched{NF, M, PS, ST} <: AbstractProcess{NF}
    "The Lux MLP (`Chain` of `Dense` layers)"
    model::M
    "Lux parameters of the melt MLP (`NamedTuple` of layer weights/biases)"
    ps::PS
    "Lux state of the melt MLP"
    st::ST
    "Input normalization for temperature (°C)"
    T_mean::NF
    T_std::NF
    "Output scale mapping the network's O(1) output back to a melt rate [m/s]"
    M_scale::NF
end
Adapt.@adapt_structure NeuralSnowMeltBatched

# All three processes declare exactly the same state variables, so any of them can be dropped into
# the model.
const SnowVars{NF} = Union{DegreeDaySnow{NF}, NeuralSnowMelt{NF}, NeuralSnowMeltBatched{NF}}
Terrarium.variables(::SnowVars{NF}) where {NF} = (
    Terrarium.input(:air_temperature, Surface(XY()), default = NF(0), units = u"°C", desc = "Near-surface air temperature in °C"),
    Terrarium.input(:snow_fall, Snow(XY()), default = NF(0), units = u"m/s", desc = "Snow fall rate in m/s"),
    Terrarium.prognostic(:snow_storage, Snow(XY()), units = u"m", desc = "Snow water equivalent in m"),
)

# ## The model
#
# A minimal `AbstractModel` holding one snow-melt process (either kind).

@kwdef struct SnowModel{NF, Grid <: Terrarium.AbstractLandGrid{NF}, Pro, Init, TS} <: Terrarium.AbstractModel{NF, Grid}
    grid::Grid
    snow_melt::Pro = DegreeDaySnow(eltype(grid))
    initializer::Init = DefaultInitializer(eltype(grid))
    timestepper::TS = ForwardEuler(eltype(grid))
end
Terrarium.variables(model::SnowModel) = Terrarium.variables(model.snow_melt)
Terrarium.compute_auxiliary!(state, ::SnowModel) = nothing
Terrarium.compute_tendencies!(state, model::SnowModel) = compute_tendencies!(state, model.grid, model.snow_melt)

# The snow storage is non-negative; clip it after each step (as in the snow example).
function Terrarium.timestep!(state::StateVariables, ::SnowModel, ::Terrarium.AbstractTimeStepper, Δt)
    interior(state.snow_storage) .= max.(interior(state.snow_storage), 0)
    return nothing
end

# ### Tendencies for the analytic process

function Terrarium.compute_tendencies!(state, grid, snow_melt::DegreeDaySnow)
    fields = get_fields(state, snow_melt)
    launch!(grid, XY, ddm_snow_flux_kernel!, state.tendencies, fields, snow_melt)
    return nothing
end
@kernel function ddm_snow_flux_kernel!(tend, grid, fields, snow_melt)
    i, j = @index(Global, NTuple)
    @inbounds tend.snow_storage[i, j, end] = ddm_snow_flux(i, j, fields, snow_melt)
end
@inline function ddm_snow_flux(i, j, fields, snow_melt::DegreeDaySnow)
    P = @inbounds fields.snow_fall[i, j, end]
    T = @inbounds fields.air_temperature[i, j, end]
    return P - melt(snow_melt, T)
end

# ### Tendencies for the NN-trained process

function Terrarium.compute_tendencies!(state, grid, snow_melt::NeuralSnowMelt)
    fields = get_fields(state, snow_melt)
    launch!(grid, XY, nn_snow_flux_kernel!, state.tendencies, fields, snow_melt)
    return nothing
end
@kernel function nn_snow_flux_kernel!(tend, grid, fields, snow_melt)
    i, j = @index(Global, NTuple)
    @inbounds tend.snow_storage[i, j, end] = nn_snow_flux(i, j, fields, snow_melt)
end
@inline function nn_snow_flux(i, j, fields, p::NeuralSnowMelt)
    P = @inbounds fields.snow_fall[i, j, end]
    T = @inbounds fields.air_temperature[i, j, end]
    Tn = (T - p.T_mean) / p.T_std
    ## KernelLux evaluates the MLP inside the kernel; the input is a 1-element static vector.
    y, _ = apply_in_kernel(p.model, SA[Tn], p.ps, p.st)
    M = y[1] * p.M_scale          # un-scale network output to a melt rate
    return P - M
end

# ### Tendencies for the *batched* NN-based process (no kernel launch)
#
# This is the second demonstration approach. Instead of launching a kernel that evaluates the MLP
# once per grid point, we gather all the columns' temperatures into a single `(1, N)` batch, run
# the network with one ordinary (batched) `Lux.apply` — a matmul + activation — and scatter the
# resulting melt back into the tendency field. There is no `launch!`: the whole tendency is a
# vectorized array operation. This traces natively under Reactant (no `raise` needed for it — it is
# plain array math, exactly Reactant's normal Lux workflow) and differentiates with Enzyme the
# same way the offline training does. For a large network the batched matmul is far more efficient
# than a per-point kernel; the trade-off is that the tendency is computed for the whole grid at
# once, so it does not naturally compose with per-point physics that would live inside a kernel.
function Terrarium.compute_tendencies!(state, grid, snow_melt::NeuralSnowMeltBatched)
    fields = get_fields(state, snow_melt)
    T = interior(fields.air_temperature)                                   # inputs over all columns
    Tn = (T .- snow_melt.T_mean) ./ snow_melt.T_std                        # normalize (materializes)
    M_norm, _ = Lux.apply(snow_melt.model, reshape(Tn, 1, length(Tn)), snow_melt.ps, snow_melt.st)
    M = reshape(M_norm, size(T)) .* snow_melt.M_scale                      # un-scale to a melt rate
    interior(state.tendencies.snow_storage) .= interior(fields.snow_fall) .- M
    return nothing
end

# ## Generate training data from the reference process
#
# We sample temperatures over a realistic range and evaluate the degree-day melt to get targets.
# Inputs and targets are normalized; the normalization
# constants are stored in the `NeuralSnowMelt` process and undone at inference.

const NF = Float32
reference = DegreeDaySnow(NF)

T_train = reshape(collect(range(NF(-30), NF(30), length = 512)), 1, :)   # (1, N) row of temperatures
M_train = melt.(Ref(reference), T_train)                                  # (1, N) target melt rates

T_mean = mean(T_train)
T_std = std(T_train)
M_scale = maximum(M_train)                     # melt is O(1e-6); scale targets to O(1) for training
T_norm = (T_train .- T_mean) ./ T_std
M_norm = M_train ./ M_scale

# ## Train the MLP with Reactant + Enzyme
#
# We build a small MLP, move it (and the data) to the Reactant device `rdev`, and train with
# `Lux.Training.single_train_step!` using the Enzyme AD backend. On the Reactant device this
# compiles the train step with XLA; each step computes the loss and its gradient with Enzyme and
# updates the parameters with Adam.

const H = 16
mlp = Lux.Chain(Lux.Dense(1 => H, tanh), Lux.Dense(H => 1))

rdev = MLDD.reactant_device()
rng = Random.Xoshiro(0)
ps, st = Lux.setup(rng, mlp) |> rdev # move the parameters and state to the device
data = (rdev(T_norm), rdev(M_norm)) # move the data to the device

function train_mlp(mlp, ps, st, data; epochs = 2000, lr = NF(1.0e-2))
    tstate = Lux.Training.TrainState(mlp, ps, st, Optimisers.Adam(lr))
    local loss
    for epoch in 1:epochs
        _, loss, _, tstate = Lux.Training.single_train_step!(AutoEnzyme(), MSELoss(), data, tstate)
        (epoch == 1 || epoch % 500 == 0) &&
            println("  epoch $epoch   loss = $(Reactant.to_number(loss))")
    end
    return tstate
end

println("Training the neural melt law (Reactant + Enzyme):")
tstate = train_mlp(mlp, ps, st, data)

# ## Assemble the trained NN-based process
#
# We move the trained parameters back to the CPU (we run the hybrid model on the CPU below) and
# build the `NeuralSnowMelt` process.
cdev = MLDD.cpu_device()
neural_melt(model, ps, st) =
    NeuralSnowMelt(model, kernelize(cdev(ps)), cdev(st), T_mean, T_std, M_scale)
neural = neural_melt(mlp, tstate.parameters, tstate.states)

# Quick check of the learned melt law against the analytic one, via the same `apply_in_kernel`:
function nn_melt(p::NeuralSnowMelt, T)
    Tn = (T - p.T_mean) / p.T_std
    y, _ = apply_in_kernel(p.model, SA[Tn], p.ps, p.st)
    return y[1] * p.M_scale
end

T_check = range(NF(-30), NF(30), length = 200)
melt_error = maximum(abs, nn_melt.(Ref(neural), T_check) .- melt.(Ref(reference), T_check))
println("Max |M_nn − M| over [-30, 30] °C: ", melt_error, "  (analytic max melt ", M_scale, ")")

# ## Run both models and compare
#
# We build a small global column grid, force it with a latitude-dependent temperature and uniform
# snowfall, start fully snow-covered, and integrate both the analytic and neural snow models. The
# two snow fields should agree to within the network's fit error.

using RingGrids

rings = RingGrids.FullGaussianGrid(8)
grid = ColumnRingGrid(UniformSpacing(Δz = 0.1, N = 1), rings)         # CPU, all-land
_lons, _lats = RingGrids.get_lonlats(rings)                  # per-column lon/lat in radians
air_temperature = NF.(25 .- 30 .* abs.(_lats) ./ (π / 2))    # warm equator (~25°C), cold poles (~-5°C)
snow_fall = fill(NF(1.0e-7), length(_lats))                  # uniform light snowfall [m/s]

## input sources take a RingGrids.Field over the grid's rings
inputs = InputSources(
    InputSource(grid, RingGrids.Field(air_temperature, rings), name = :air_temperature, units = u"°C"),
    InputSource(grid, RingGrids.Field(snow_fall, rings), name = :snow_fall, units = u"m/s"; domain = Snow()),
)
initializers = (snow_storage = NF(0.5),)                      # start with 0.5 m everywhere

function run_snow(process; period = Day(30), Δt = 3600.0)
    model = SnowModel(grid; snow_melt = process, timestepper = ForwardEuler(NF))
    integrator = initialize(model; inputs, initializers)
    run!(integrator; period, Δt)
    return Array(interior(integrator.state.snow_storage))[:, 1, 1]
end

snow_ddm = run_snow(reference)
snow_nn = run_snow(neural)

snow_diff = maximum(abs, snow_ddm .- snow_nn)
println("Ran both snow models for 30 days on $(length(_lats)) columns.")
println("Max |S_ddm − S_nn| after offline training: ", snow_diff, " m  (mean snow ", mean(snow_ddm), " m)")

# ## Online fine-tuning: training *through* the full Terrarium model
#
# The training above was *offline*: we fit the melt law `M(T)` pointwise, independently of the
# simulation. *Online* training instead differentiates through the model's own time integration.
# Here we do this with the full `SnowModel`, which contains the neural network,
# on the Reactant device: we roll the model forward, compare the
# simulated snow to a reference trajectory, and let Enzyme backpropagate through the compiled
# rollout to the network weights. This is the differentiable-simulation idea, and it is what
# Reactant + Enzyme make possible.
#
# It can help even in this toy setting: the offline objective fits `M(T)` pointwise, whereas the
# online objective matches the simulated snow trajectory directly — the quantity we care about —
# so a small pointwise melt error that would accumulate over the rollout gets corrected.
#
# ### Two ways to evaluate the network — same result
#
# We run the online fine-tuning **twice**, with the two processes defined above, to show to both them
# as they both have possible applicaations for hybrid land modelling with Terrarium.
#
# 1. **In-kernel** (`NeuralSnowMelt`): the network is evaluated per grid point inside a
#    KernelAbstractions kernel (`apply_in_kernel`). This composes naturally with per-point,
#    kernel-based physics, but for a large network the per-point evaluation is comparatively slow.
# 2. **Batched, kernel-free** (`NeuralSnowMeltBatched`): the network is evaluated for all columns
#    at once with a single batched `Lux.apply` and no kernel launch. For a large network this
#    batched matmul is far more efficient, but the tendency is a whole-grid vectorized operation
#    that does not fold into a per-point physics kernel.
#
# Both are differentiated through the identical `run_timesteps!` rollout and, starting from the
# same offline weights, converge to essentially the same finetuned model.

const Δt_ft = 86400.0f0    # 1-day steps
const Nt_ft = 15           # 15-day rollout

# We run the full model on the Reactant device, with the input sources on a `ReactantState` grid.

device_grid = ColumnRingGrid(ReactantState(), NF, UniformSpacing(Δz = 0.1, N = 1), rings)
device_inputs = InputSources(
    InputSource(device_grid, RingGrids.Field(air_temperature, rings), name = :air_temperature, units = u"°C"),
    InputSource(device_grid, RingGrids.Field(snow_fall, rings), name = :snow_fall, units = u"m/s"; domain = Snow()),
)
build_device_integrator(process) = initialize(
    SnowModel(device_grid; snow_melt = process, timestepper = ForwardEuler(NF));
    inputs = device_inputs, initializers = (snow_storage = NF(0.5),),
)

# The target is the snow field the *analytic* model produces after the same rollout, and the reset
# state `S0` (0.5 m everywhere); both are process-independent, so we get them once from a reference.

reference_integrator = build_device_integrator(DegreeDaySnow(NF))
S0 = copy(interior(reference_integrator.state.snow_storage))     # reset state, 0.5 m everywhere
run!(reference_integrator; steps = Nt_ft, Δt = Δt_ft)
S_ref = copy(interior(reference_integrator.state.snow_storage))

# The loss resets the snow to `S0`, steps the full `SnowModel` with `run_timesteps!`, and measures
# the mismatch to the reference. `snow_grad!` differentiates it with Enzyme; the network-weight
# gradients land in `dintegrator.model.snow_melt.ps`. Both are generic over the process type — they
# see only the `integrator` — so the same functions drive both approaches. (We zero the shadow
# *outside* the compiled function — `make_zero!` inside it cannot reconstruct the immutable model.)

function snow_loss(integrator, S0, S_ref, Nt)
    copyto!(interior(integrator.state.snow_storage), S0)     # reset the rollout
    run_timesteps!(integrator, Δt_ft, Nt)                    # step the full Terrarium model
    return mean(abs2, interior(integrator.state.snow_storage) .- S_ref)
end
function snow_grad!(integrator, dintegrator, S0, S_ref, Nt)
    _, loss = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal), snow_loss, Enzyme.Active,
        Enzyme.Duplicated(integrator, dintegrator),
        Enzyme.Const(S0), Enzyme.Const(S_ref), Enzyme.Const(Nt)
    )
    return loss
end

# Adam update of the network parameters. Because rolling the melt through `Nt·Δt ≈ 10⁶` seconds
# amplifies its gradient enormously, an adaptive optimizer (which rescales by the gradient
# magnitude) is far more robust here than plain gradient descent. We optimize the Lux parameters
# `ps` directly; the optimizer runs eagerly on the host over the device arrays — which are shared
# with `integrator.model.snow_melt.ps`, so the model updates in place. (Eager, not traced, so
# Adam's scalar state is a plain number.) The compiled gradient function is passed in so the same
# loop works for either process.
function finetune!(compiled_grad!, integrator, S0, S_ref; epochs = 100, lr = NF(1.0e-3), label = "")
    params = integrator.model.snow_melt.ps
    opt = Optimisers.setup(Optimisers.Adam(lr), params)
    for epoch in 1:epochs
        dintegrator = Enzyme.make_zero(integrator)           # fresh zero shadow each step
        loss = compiled_grad!(integrator, dintegrator, S0, S_ref, Nt_ft)
        grads = dintegrator.model.snow_melt.ps
        Optimisers.update!(opt, params, grads)
        (epoch == 1 || epoch % 25 == 0) &&
            println("  [$label] epoch $epoch   snow loss = $(Reactant.to_number(loss))")
    end
    return nothing
end

# Drive one approach end to end: build the device integrator for `device_process`, compile its
# gradient, finetune, then rebuild an equivalent CPU process (via `cpu_process_from`) from the
# finetuned weights and run the hybrid model to get the snow field to compare against the analytic.
function finetune_and_run(device_process, cpu_process_from; label)
    integrator = build_device_integrator(device_process)
    dintegrator = Enzyme.make_zero(integrator)
    compiled_grad! = Reactant.@compile raise = true raise_first = true sync = true snow_grad!(
        integrator, dintegrator, S0, S_ref, Nt_ft
    )
    println("Online fine-tuning through the full SnowModel [$label] (Reactant + Enzyme):")
    finetune!(compiled_grad!, integrator, S0, S_ref; label)
    cpu_process = cpu_process_from(integrator.model.snow_melt.ps, integrator.model.snow_melt.st)
    return run_snow(cpu_process)
end

# Each approach starts from an independent copy of the offline-trained weights (a device→host→device
# round trip), so fine-tuning one does not disturb the other. The in-kernel CPU process uses
# `kernelize`d parameters; the batched CPU process uses plain arrays (ordinary `Lux.apply`).
fresh_offline_ps() = rdev(cdev(tstate.parameters))

snow_nn_ft_kernel = finetune_and_run(
    NeuralSnowMelt(mlp, fresh_offline_ps(), tstate.states, T_mean, T_std, M_scale),
    (ps, st) -> neural_melt(mlp, ps, st);
    label = "in-kernel",
)
snow_nn_ft_batched = finetune_and_run(
    NeuralSnowMeltBatched(mlp, fresh_offline_ps(), tstate.states, T_mean, T_std, M_scale),
    (ps, st) -> NeuralSnowMeltBatched(mlp, cdev(ps), cdev(st), T_mean, T_std, M_scale);
    label = "batched",
)

println("Max |S_ddm − S_nn| after online fine-tuning [in-kernel]: ", maximum(abs, snow_ddm .- snow_nn_ft_kernel), " m")
println("Max |S_ddm − S_nn| after online fine-tuning [batched]:   ", maximum(abs, snow_ddm .- snow_nn_ft_batched), " m")
println("Max |S_in-kernel − S_batched| between the two approaches: ", maximum(abs, snow_nn_ft_kernel .- snow_nn_ft_batched), " m")

# The physics-based melt law has been replaced by a trained neural network — fit offline and then
# finetuned online by differentiating through the full Terrarium `SnowModel` with Reactant +
# Enzyme. We demonstrated the two ways to embed the network — evaluated per grid point inside a
# kernel, or batched across the grid without a kernel — which give essentially the same result and
# trade off physics integration against efficiency for large networks. This is the essential
# building block of hybrid land modeling in Terrarium.
