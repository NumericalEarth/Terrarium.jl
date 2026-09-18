# # Global sensitivity analysis with Reactant + Enzyme
#
# This example computes *global sensitivity maps*: Both parameter sensitivty and state sensitivty.
# We start by computing the sensitivity of the global mean surface
# soil temperature after one simulated day with respect to the initial soil state at **every**
# land grid point and soil layer. Reverse-mode AD delivers this entire map in a *single* gradient
# evaluation — the cost of the reverse pass is independent of the number of inputs, which is what
# makes global (all-points-at-once) sensitivity analysis feasible.
#
# We follow the same Reactant + Enzyme recipe as the
# [Reactant differentiation example](@ref): the model is built on `ReactantState()`, the stepping
# loop [`run_timesteps!`](@ref) is traced with a checkpointing scheme, and `Enzyme.autodiff`
# differentiates through the whole compiled rollout. The physical setup mirrors the ERA5-forced
# global soil heat example (`examples/simulations/soil_heat_global_era5.jl`).

using Terrarium
using Reactant, CUDA   # CUDA is required by Reactant's kernel integration, even on CPU
using Enzyme

using Rasters, NCDatasets
using CairoMakie
using Statistics: quantile, mean

import RingGrids

# ## Grid and forcing
#
# We build the same ~1° global land grid as the ERA5 example: all grid points with more than 50%
# land become independent soil columns. Reactant currently requires uniform vertical spacing (see
# the [Reactant page](@ref)), so we use uniformly spaced layers instead of `ExponentialSpacing`.

## Load land-sea mask at ~1° resolution
land_sea_frac_10km = RingGrids.Field(ERA5LandInvariants(), "lsm")
land_sea_frac_N72 = RingGrids.interpolate(ring_grid, land_sea_frac_10km)
land_mask = land_sea_frac_N72 .> 0.5 # select only grid points with > 50% land

Nz = 30 # number of soil layers
grid = ColumnRingGrid(ReactantState(), Float32, UniformSpacing(Δz = 0.2f0, N = Nz), land_mask)

# Load ERA-5 2 meter air temperature at ~1° resolution as the surface forcing. The compiled
# stepping loop cannot yet update time-dependent `FieldTimeSeries` inputs from the host, so we
# take the first time slice as a constant-in-time forcing field (a `FieldInputSource`, as in the
# hybrid modeling example).

# The full-year hourly file is large (~1.5 GB, 8760 time steps); load it lazily and materialize
# only the first time slice, which we use as a constant-in-time forcing.
Tair_raster = Raster(Terrarium.get_asset(ERA5LandForcings()), name = "t2m", lazy = true)
Tair0 = convert.(Float32, replace_missing(Tair_raster[Ti(1)], NaN)) .- 273.15f0
Tair_field = RingGrids.FullGaussianField(Matrix(Tair0), input_as = Matrix)
Tair_forcing = InputSource(grid, Tair_field, name = :Tair, units = u"°C")

# Masked surface temperatures ordered by the grid's active (land) columns, for the initializer
# below (column index `x` runs over the land columns).
Tsurf_0 = Tair_field[findall(land_mask)]

# ## Model setup
#
# As everywhere with Reactant in Terrarium, the model construction is standard — only the grid's
# architecture differs. We give the soil a **loam** texture (sand < 1): it is a representative
# composition and, in particular, it makes the non-quartz mineral thermal conductivity influence the
# bulk conductivity, which the parameter sensitivity below relies on. The same soil is used
# throughout.

stratigraphy = HomogeneousSoilStratigraphy(
    eltype(grid), texture = SoilTexture(eltype(grid); sand = 0.4f0, clay = 0.3f0, silt = 0.3f0)
)
model = SoilModel(
    grid; soil = SoilEnergyWaterCarbon(eltype(grid); strat = stratigraphy),
    timestepper = ForwardEuler(eltype(grid))
)
boundary_conditions = PrescribedSurfaceTemperature(:Tair)
initializers = (
    ## steady-ish state initial condition for temperature
    temperature = (x, z) -> Tsurf_0[Int(round(x)) + 1] - 0.02f0 * z,
    saturation_water_ice = 1.0f0,
)
inputs = InputSources(Tair_forcing)
integrator = initialize(model; inputs, initializers, boundary_conditions)

# ## The objective and its gradient
#
# The scalar objective is the global mean surface (top-layer) soil temperature after `nsteps`.
# [`run_timesteps!`](@ref) is the stepping loop underlying `run!`; on `ReactantState` it traces
# the loop and takes a `checkpointing` scheme that bounds the memory of the reverse pass.

function mean_surface_temperature(integrator, Δt, nsteps, checkpointing)
    run_timesteps!(integrator, Δt, nsteps, checkpointing)
    Tsurf = interior(integrator.state.temperature)[:, :, end]
    return sum(Tsurf) / length(Tsurf)
end

# `Enzyme.autodiff` in reverse mode computes the objective together with its gradient. The
# integrator is passed as `Duplicated`, so its shadow `dintegrator` accumulates the sensitivity
# of the objective with respect to every state variable — in particular the prognostic internal
# energy `U₀` of every column and layer.

function grad_mean_surface_temperature!(integrator, dintegrator, Δt, nsteps, checkpointing)
    _, value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        mean_surface_temperature, Enzyme.Active,
        Enzyme.Duplicated(integrator, dintegrator),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps),
        Enzyme.Const(checkpointing),
    )
    return value
end

# We allocate the shadow with `make_zero` and compile the whole gradient with Reactant. As with a
# forward `run!`, the first call compiles (this can take a few minutes); afterwards it is fast.

dintegrator = Enzyme.make_zero(integrator)
Δt = 600.0f0
nsteps = 144                                       # one simulated day
checkpointing = Reactant.Periodic(isqrt(nsteps))   # ≈ √n checkpoints

compiled_grad! = @compile raise = true raise_first = true sync = true grad_mean_surface_temperature!(
    integrator, dintegrator, Δt, nsteps, checkpointing
)
T̄_surf = compiled_grad!(integrator, dintegrator, Δt, nsteps, checkpointing)
println("Global mean surface soil temperature after 1 day: $(Reactant.to_number(T̄_surf)) °C")

# ## The global sensitivity map
#
# A single reverse pass gave us ``\partial \bar{T}_\text{surf} / \partial U_0`` for all land
# columns and soil layers at once. We move the shadow to the CPU and plot it.

dU = Array(interior(dintegrator.state.internal_energy))   # (Ncolumns, 1, Nz)
cpu_grid = on_architecture(CPU(), grid)
## scatter all layers back onto the rings (columns × layers), then take the top (surface) layer as
## a horizontal field for plotting.
dU_ring = RingGrids.Field(dU[:, 1, :], cpu_grid)
dU_surface = dU_ring[:, end]                              # top-layer sensitivity, a horizontal field

# The sensitivity is positive almost everywhere (more initial internal energy ⇒ warmer soil ⇒
# higher global mean), but a small number of columns sitting near the freezing point have an
# outsized response: there latent heat of fusion makes the effective heat capacity very large and
# ``\partial T / \partial U`` spikes. Those outliers are physical, but on a linear color scale
# they would pin every other column to one end of the colorbar. We therefore clip the color range
# to the 2nd–98th percentile so the spatial structure of the bulk is visible.
lo, hi = quantile(filter(isfinite, Array(dU_surface)), (0.02, 0.98))

mkpath("plots")
fig = heatmap(
    dU_surface;
    title = "Sensitivity of global mean surface soil temperature to initial internal energy",
    size = (900, 450),
    colorrange = (lo, hi),
    colormap = :viridis,
    highclip = :magenta,
)
Makie.save("plots/global_sensitivity_initialU_era5land.png", fig)
fig

# The same gradient also tells us how deep into the soil a single day of surface forcing
# "reaches": we average the magnitude of the sensitivity over all land columns by depth. The
# sensitivity falls off by many orders of magnitude within the first metre — a single day of
# surface forcing barely perturbs the deep soil — so we plot the magnitude on a log axis.

zs = znodes(integrator.state.temperature)
dU_depth = vec(sum(abs, dU[:, 1, :], dims = 1)) ./ size(dU, 1)
## floor at a tiny fraction of the maximum so the log axis is well-defined without flattening the
## many-orders-of-magnitude decay with depth (values span ~1e-12 at the surface to ~1e-26 deep).
floor_value = maximum(dU_depth) * 1.0f-20
dU_depth_plot = max.(dU_depth, floor_value)

f = Makie.Figure()
Makie.Axis(
    f[1, 1], ylabel = "Soil depth / m", xlabel = "mean |∂T̄_surf/∂U₀|",
    xscale = log10,
)
Makie.scatterlines!(f[1, 1], dU_depth_plot, zs)
Makie.save("plots/global_sensitivity_depth_profile.png", f)
f

# ## Parameter sensitivity: soil mineral thermal conductivity
#
# The maps above are sensitivities to the *initial state*. The **same** reverse pass also carries
# sensitivities to *model parameters* — the shadow `dintegrator.model` accumulates them alongside the
# state gradients, exactly as the neural-network weights are differentiated in the hybrid modeling
# example. Here we compute ``\partial \bar{T} / \partial \kappa_\text{mineral}``, the sensitivity of
# the column-mean soil temperature to the thermal conductivity of the non-quartz (silt/clay) mineral
# grains.
#
# A scalar physical parameter needs one extra step compared with an array input. On a `ReactantState`
# grid the scalars stored in the model are baked into the compiled program as **constants**, so their
# reverse-mode shadow would come back identically zero. To differentiate with respect to
# ``\kappa_\text{mineral}`` we therefore **promote** the soil conductivities to *traced device scalars*
# with `Reactant.to_rarray(...; track_numbers = true)`, so they enter the compiled program as tracked
# inputs. Eager initialization cannot carry a traced scalar through a process struct, so we use a
# *promote-after-init* pattern: initialize with the ordinary (concrete) conductivities, then rebuild
# the integrator around a model whose conductivities are the promoted, traced copy — reusing the
# already-initialized state.
#
# !!! note "Two ingredients make the traced scalar reach the kernel"
#     (1) The soil thermal-property structs are `Adapt`-adaptable, so the promoted scalar is converted
#     to a device value *inside* the tendency kernel instead of remaining a host tracer. (2) The
#     `InverseQuadratic` bulk-conductivity weighting uses the float power `x^(one(x)/2)` rather than
#     `sqrt`, which has no method for a traced device scalar. Both live in `SoilThermalProperties`.

# We differentiate the **column-mean** soil temperature rather than the surface temperature: over one
# day the surface layer is held by the atmospheric boundary condition and is essentially insensitive to
# soil conductivity, whereas the conductivity controls how heat is redistributed through the column.

function mean_column_temperature(integrator, Δt, nsteps, checkpointing)
    run_timesteps!(integrator, Δt, nsteps, checkpointing)
    T = interior(integrator.state.temperature)
    return sum(T) / length(T)
end

function grad_mean_column_temperature!(integrator, dintegrator, Δt, nsteps, checkpointing)
    _, value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        mean_column_temperature, Enzyme.Active,
        Enzyme.Duplicated(integrator, dintegrator),
        Enzyme.Const(Δt), Enzyme.Const(nsteps), Enzyme.Const(checkpointing),
    )
    return value
end

# The same loam soil model as above, but parameterized by its soil thermal conductivities so we can
# build it once with concrete values (to initialize) and once with promoted, traced values (for AD).
soil_conductivity_model(conductivities) = SoilModel(
    grid; timestepper = ForwardEuler(eltype(grid)),
    soil = SoilEnergyWaterCarbon(
        eltype(grid); strat = stratigraphy,
        energy = SoilThermodynamics(
            eltype(grid); thermal_properties = SoilThermalProperties(eltype(grid); conductivities),
        ),
    ),
)

## initialize with concrete conductivities, then swap in the promoted (traced) copy, reusing the state
param_integrator0 = initialize(
    soil_conductivity_model(SoilThermalConductivities(eltype(grid)));
    inputs, initializers, boundary_conditions,
)
promoted_conductivities = Reactant.to_rarray(SoilThermalConductivities(eltype(grid)); track_numbers = true)
param_integrator = ModelIntegrator(
    param_integrator0.clock, soil_conductivity_model(promoted_conductivities),
    param_integrator0.inputs, param_integrator0.state, param_integrator0.initializers,
)

param_dintegrator = Enzyme.make_zero(param_integrator)
compiled_param_grad! = @compile raise = true raise_first = true sync = true grad_mean_column_temperature!(
    param_integrator, param_dintegrator, Δt, nsteps, checkpointing
)
compiled_param_grad!(param_integrator, param_dintegrator, Δt, nsteps, checkpointing)

# The parameter gradient lives in the model shadow — we read it straight from the conductivities
# shadow. It is nonzero: raising the mineral (or quartz) grain conductivity speeds heat exchange
# through the column and changes the daily-mean soil temperature.
∂T̄_∂κ_mineral = Reactant.to_number(param_dintegrator.model.soil.energy.thermal_properties.conductivities.mineral)
∂T̄_∂κ_quartz = Reactant.to_number(param_dintegrator.model.soil.energy.thermal_properties.conductivities.quartz)
println("∂T̄_col/∂κ_mineral = $(∂T̄_∂κ_mineral) K / (W m⁻¹ K⁻¹)")
println("∂T̄_col/∂κ_quartz  = $(∂T̄_∂κ_quartz) K / (W m⁻¹ K⁻¹)")

# ## Grid-point-wise parameter sensitivity: forward-mode AD
#
# The reverse pass above collapsed the whole soil state into one scalar objective and returned one
# number. Forward-mode AD answers the complementary question in a single pass: how does that *one*
# scalar parameter ``\kappa_\text{mineral}`` perturb the soil temperature at *every* grid point and
# depth? Reverse mode is the efficient choice for many-inputs → one-output (the initial-condition maps
# above); forward mode is efficient for **one-input → many-outputs**, which is exactly a scalar-parameter
# sensitivity *map*.
#
# We seed a unit tangent for ``\kappa_\text{mineral}`` into an otherwise-zero shadow integrator
# (`make_zero` gives the all-zero tangent; we set just the `mineral` component to one) and propagate it
# forward through the rollout. `run_timesteps!` mutates the integrator in place, so after the forward
# pass the shadow's temperature field holds ``\partial T / \partial \kappa_\text{mineral}`` at every
# grid point.

## Seed a unit κ_mineral tangent into an otherwise-zero shadow integrator. `make_zero` zeroes every
## parameter and state field; `setproperties` then sets only the `mineral` conductivity tangent to one,
## rebuilding the immutable model chain around it.
function seed_mineral_tangent(integrator)
    d = Enzyme.make_zero(integrator)
    energy = d.model.soil.energy
    thermal_properties = energy.thermal_properties
    ## set only the mineral conductivity tangent to one; every other tangent stays zero
    conductivities = Terrarium.setproperties(
        thermal_properties.conductivities, (; mineral = one(thermal_properties.conductivities.mineral)),
    )
    thermal_properties = Terrarium.setproperties(thermal_properties, (; conductivities))
    energy = Terrarium.setproperties(energy, (; thermal_properties))
    soil = Terrarium.setproperties(d.model.soil, (; energy))
    model = Terrarium.setproperties(d.model, (; soil))
    return ModelIntegrator(d.clock, model, d.inputs, d.state, d.initializers)
end

## a fresh promoted integrator — the reverse pass above advanced its own copy to day 1
fwd_integrator0 = initialize(
    soil_conductivity_model(SoilThermalConductivities(eltype(grid)));
    inputs, initializers, boundary_conditions,
)
# this reuses the traced conductivities we set up above for the global sensitivity example
fwd_integrator = ModelIntegrator(
    fwd_integrator0.clock, soil_conductivity_model(promoted_conductivities),
    fwd_integrator0.inputs, fwd_integrator0.state, fwd_integrator0.initializers,
)
fwd_dintegrator = seed_mineral_tangent(fwd_integrator)

# Forward mode needs no checkpointing (there is no reverse pass to bound); `Enzyme.Duplicated` carries
# the seed and the temperature tangent lands in `fwd_dintegrator` in place.
forward_rollout!(integrator, Δt, nsteps) = (run_timesteps!(integrator, Δt, nsteps); nothing)
function forward_temperature_map!(integrator, dintegrator, Δt, nsteps)
    Enzyme.autodiff(
        Enzyme.Forward, forward_rollout!, Enzyme.Const,
        Enzyme.Duplicated(integrator, dintegrator), Enzyme.Const(Δt), Enzyme.Const(nsteps),
    )
    return nothing
end

compiled_forward_map! = @compile raise = true raise_first = true sync = true forward_temperature_map!(
    fwd_integrator, fwd_dintegrator, Δt, nsteps
)
compiled_forward_map!(fwd_integrator, fwd_dintegrator, Δt, nsteps)

dTdκ = Array(interior(fwd_dintegrator.state.temperature))   # (Ncolumns, 1, Nz) = ∂T/∂κ_mineral

# Consistency check: averaging the map over all grid points recovers the reverse-mode global scalar,
# because ``\partial \bar{T}_\text{col} / \partial \kappa = \frac{1}{N} \sum_\text{grid points}
# \partial T / \partial \kappa``. The two agree to ~7 digits.
println("mean(∂T/∂κ_mineral map) = $(mean(dTdκ))   (reverse-mode scalar: $(∂T̄_∂κ_mineral))")

# The **surface** layer is held by the atmospheric boundary condition, so ``\partial T / \partial
# \kappa \approx 0`` there (as the scalar section noted). We map the horizontal pattern at a shallow
# subsurface level (≈0.5 m); the full vertical structure is in the depth profile below. The map is
# signed — raising ``\kappa_\text{mineral}`` warms some columns and cools others depending on the local
# thermal gradient and how close the soil sits to the freezing point — so we use a diverging colormap
# centered at zero.
zs = znodes(fwd_integrator.state.temperature)
k_plot = argmin(abs.(zs .+ 0.5f0))                          # layer nearest 0.5 m depth
cpu_grid = on_architecture(CPU(), grid)
dTdκ_ring = RingGrids.Field(dTdκ[:, 1, :], cpu_grid)
dTdκ_layer = dTdκ_ring[:, k_plot]                           # horizontal field at ≈0.5 m
clim = quantile(filter(isfinite, abs.(Array(dTdκ_layer))), 0.98)

fig_param = heatmap(
    dTdκ_layer;
    title = "∂(soil T at $(round(zs[k_plot], digits = 2)) m) / ∂κ_mineral  (forward-mode AD)",
    size = (900, 450),
    colorrange = (-clim, clim),
    colormap = :balance,
)
Makie.save("plots/global_parameter_sensitivity_kappa_mineral.png", fig_param)
fig_param

# Averaging the magnitude of the map over all land columns by depth shows the vertical structure. In
# this one-day experiment the response is small through most of the column and rises sharply toward the
# **base**: the surface is pinned by the boundary condition while the lower boundary is insulating
# (zero-flux), so the deep soil relaxes its (slightly super-adiabatic) initial temperature profile at a
# rate the conductivity controls. The shallower response mapped above is smaller in magnitude but
# carries the geographic pattern of the forcing.
dTdκ_depth = vec(sum(abs, dTdκ[:, 1, :], dims = 1)) ./ size(dTdκ, 1)
f_param = Makie.Figure()
Makie.Axis(f_param[1, 1], ylabel = "Soil depth / m", xlabel = "mean |∂T/∂κ_mineral|")
Makie.scatterlines!(f_param[1, 1], dTdκ_depth, zs)
Makie.save("plots/global_parameter_sensitivity_depth_profile.png", f_param)
f_param
