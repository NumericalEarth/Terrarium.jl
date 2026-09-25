# # Lateral coupling of land columns through the atmosphere
#
# Terrarium is a *column* land model: every land grid cell integrates its own 1D soil
# column, and nothing in Terrarium itself moves heat or water sideways between columns.
# When Terrarium is coupled to SpeedyWeather, however, the columns stop being independent.
# Each column exchanges sensible heat, moisture, and radiation with the atmosphere, the
# atmosphere transports those anomalies horizontally, and they are deposited again on
# other land columns. The only path from one soil column to another runs through the air.
#
# This example measures that path directly. We take the adjoint (reverse-mode) derivative
# of the surface soil temperature of **one** land grid cell, at the end of a short coupled
# integration, with respect to the **full spatial field** of the initial soil state. The soil
# prognostic variable in Terrarium is the internal energy ``U`` (J/m³), from which the
# temperature is diagnosed, so the derivative is taken with respect to ``U``:
#
# ```math
# \mathbf{g}(j) = \frac{\partial T_\text{soil}(i^*, t_N)}{\partial U(j, t_0)},
# \qquad j = 1, \dots, N_\text{land}
# ```
#
# A purely local column model would give ``\mathbf{g}(j) = 0`` for every ``j \neq i^*``.
# Anything nonzero off the target cell is lateral coupling mediated by SpeedyWeather, and
# the spatial structure of ``\mathbf{g}`` tells us how far and in which direction the
# coupling reaches over the integration window.
#
# Reverse mode is the natural choice here: one scalar output, ``N_\text{land}`` inputs, so a
# single adjoint pass gives the entire sensitivity map. Forward mode would need one pass per
# land column.
#
# !!! warning "Does not currently compile"
#     This is a research script, not a doc-built example. As of 2026-09-25 the `autodiff` call
#     below **does not compile**, at any configuration tried, down to T9 with a single atmospheric
#     layer and two soil layers (Julia 1.10.12, Enzyme 0.13.204, SpeedyWeather 0.22.1). The forward
#     model, the objective, the spin-up, the target-cell selection, and the finite-difference check
#     all work as written; treat everything from the `autodiff` call onwards as the intended
#     analysis rather than a result anyone has seen.
#
#     What is known, from the bisection in
#     `docs/dev/2026-09/2026-09-24_NOTE_enzyme_landmodel_compile_time.md`:
#
#     * **Compile time** is dominated by the Terrarium process set, not by the coupling or by
#       SpeedyWeather. A SpeedyWeather-only step at this size differentiates in ~19 min, a
#       `SoilModel` on the same 291-column ring grid in ~2 min (with the expected purely local
#       gradient), while the default `LandModel` on a *single* column does not finish in an hour.
#       `SingleLayerSnow` is the component responsible, so this script runs with `snow = nothing`.
#       Making `StateVariables` a mutable struct (on this branch) cut the reduced single-column
#       `LandModel` step by 4.1×, to ~12 min.
#     * **Two hard failures** remain past that, both outside Terrarium. Differentiating through
#       `Terrarium.run!` with SpeedyWeather loaded first hit an `IllegalTypeAnalysisException` in
#       `SpeedyWeather.speedstring`, reached through a dead progress-bar branch; that one is fixed
#       on this branch by dispatching `run_timesteps!` on `Val{true}`/`Val{false}`. Past it, the
#       coupled step fails either with an `EnzymeInternalError` inside SpeedyWeather's own
#       `vertical_advection!` (~23 min) or, on a slightly different path, with an Enzyme
#       `TypeAnalysis` assertion and SIGABRT (~28 min). Disabling vertical advection is not a
#       workaround: SpeedyWeather has no no-op advection scheme, and switching off the dynamical
#       core would remove the atmospheric transport this experiment exists to measure.
#
#     Use Julia 1.10 and keep the resolution and step count small when retrying.
#
# ## Setup

import Pkg
Pkg.activate(@__DIR__)

# On Julia 1.10 (the version recommended for Enzyme here), Pkg ignores the `[sources]` entry
# that points `Terrarium` at this checkout, and will silently resolve a *registered* Terrarium
# instead, pinned to an older Oceananigans. Make sure the local package is the one in use:
#     Pkg.develop(path = joinpath(@__DIR__, "..", ".."))
# This is a one-time step per environment; Julia 1.11+ honors `[sources]` and does not need it.

using Terrarium

using Checkpointing
using Dates
using Enzyme
using Enzyme: Reverse, make_zero, set_runtime_activity
using Printf

using CairoMakie

import RingGrids
import SpeedyWeather as Speedy

# Enzyme runs the LLVM Attributor pass on Julia < 1.12. Stepping the clock inside an
# `@ad_checkpoint` loop sends the Attributor's `AAPotentialValues` analysis into unbounded
# recursion, which overflows the C++ stack and surfaces as a segfault. Disabling the pass
# avoids it. (Same workaround as SpeedyWeather's own sensitivity examples.)
Enzyme.Compiler.RunAttributor[] = false

arch = CPU()
NF = Float32

# Deliberately coarse. The adjoint tape of the coupled model grows with both the horizontal
# resolution and the number of soil layers, and the qualitative result (nonzero off-cell
# sensitivity) is already visible at low resolution.
truncation = 21
nlayers_atmos = 5
Nz = 4          # soil layers
Δz_min = 0.05   # (m) thickness of the topmost soil layer

# ## Building the coupled model
#
# We keep the land configuration minimal and data-free: a homogeneous soil column with
# energy, water, and carbon, no vegetation, and no external input datasets. That keeps the
# derivative interpretable, since every off-cell sensitivity then has to come from the
# atmosphere rather than from a shared input field.

speedy_arch = RingGrids.Architectures.architecture(arch)
spectral_grid = Speedy.SpectralGrid(; truncation, nlayers = nlayers_atmos, architecture = speedy_arch)
ring_grid = spectral_grid.grid

land_sea_mask = Speedy.EarthLandSeaMask(spectral_grid)
Speedy.load_mask!(land_sea_mask)

# The Terrarium mask must be a superset of the SpeedyWeather land, so build it from the same
# fractional mask the atmosphere uses.
land_grid = ColumnRingGrid(arch, NF, ExponentialSpacing(; N = Nz, Δz_min), ring_grid, land_sea_mask.land_fraction .> 0)

# Homogeneous soil (the default stratigraphy, made explicit): one texture and porosity for every
# layer and column, so no per-horizon input fields enter the differentiated state.
strat = HomogeneousSoilStratigraphy(eltype(land_grid))
soil = SoilEnergyWaterCarbon(eltype(land_grid); strat, hydrology = SoilHydrology(eltype(land_grid)))

# Soil freezing is the one physical process that can silently break this experiment, so it
# deserves a paragraph. Terrarium's only freeze curve wired into the energy closure is
# `FreeWater()`: all phase change happens at exactly 0 °C. A wet column sitting on that
# plateau has ``\partial T / \partial U = 0`` *identically*, since added energy melts ice
# rather than raising the temperature. If the target column is on the plateau, every
# sensitivity in this script, the local one included, is exactly zero, and remote columns on
# the plateau contribute exactly zero regardless of how strongly the atmosphere couples them.
# (Smooth `SFCC` curves are exported by Terrarium but not yet dispatched in the closure; that
# would be the principled fix, and it belongs in the source, not here.)
#
# Two things follow. First, the default initializer starts at `T₀ = 0` °C in wet soil, i.e.
# on the plateau; we start warmer. Second, and less obviously, a warm start is not enough on
# its own: SpeedyWeather's default initial atmosphere is a reference state, and the land
# coupling reads the *lowest model level* as the air temperature, so the first hours of a
# coupled run are a strong cold shock (several kelvin in the top 5 cm within half an hour)
# that can drive mid-latitude columns onto the plateau before the atmosphere equilibrates.
# The spin-up below has to be long enough to come out the other side of that transient, and
# the target column is chosen *after* spin-up from the columns that are actually unfrozen.
soil_initializer = SoilInitializer(eltype(land_grid); energy = QuasiThermalSteadyState(eltype(land_grid); T₀ = 10))

# No snow. This is a compile-time concession, not a physical one: `SingleLayerSnow` is the
# component that pushes Enzyme's analysis of a `LandModel` step past the hour mark (see the
# warning at the top). For a mid-latitude July target the snow-free assumption is mild, and it
# also removes one more place where a plateau (snow at 0 °C) could zero out sensitivities.
terrarium_model = Terrarium.LandModel(land_grid; soil, vegetation = nothing, snow = nothing, initializer = soil_initializer)

# Terrarium sub-steps five minutes inside each atmospheric step.
land = Speedy.LandModel(spectral_grid, terrarium_model; Δt = Minute(5))

time_stepping = Speedy.Leapfrog(spectral_grid, Δt_at_T32 = Minute(15))
model = Speedy.PrimitiveWetModel(
    spectral_grid;
    land,
    land_sea_mask,
    time_stepping,
    surface_heat_flux = Speedy.SurfaceHeatFlux(spectral_grid, land = Speedy.PrescribedLandHeatFlux()),
    surface_humidity_flux = Speedy.SurfaceHumidityFlux(spectral_grid, land = Speedy.PrescribedLandHumidityFlux()),
)

# ## Spin-up
#
# The adjoint of a model sitting in an unphysical initial state is not very informative, so
# we spin the coupled system up first with an ordinary (non-differentiated) run.

initial_date = DateTime(2024, 7, 1)
simulation = Speedy.initialize!(model; time = initial_date)
@info "Spinning up the coupled model"
@time Speedy.run!(simulation; period = Day(10))

(; variables) = simulation
state = variables.prognostic.land.terrarium
Terrarium.checkfinite!(state.prognostic)

# ## Choosing the target cell
#
# We want a land column in the mid-latitudes, where the atmospheric flow is vigorous enough
# that the coupling has somewhere to travel. Terrarium indexes its land columns `1:N_land`
# in ring-grid order, so we mask the ring-grid coordinates the same way to recover the
# coordinates of each column.

mask = Array(land_grid.mask.data)
londs_all, latds_all = RingGrids.get_londlatds(ring_grid)
londs_land = londs_all[mask]
latds_land = latds_all[mask]
N_land = length(londs_land)

"""Great-circle distance (km) between two points given in degrees."""
function great_circle_distance(lond1, latd1, lond2, latd2)
    φ1, φ2 = deg2rad(latd1), deg2rad(latd2)
    Δλ = deg2rad(lond2 - lond1)
    # spherical law of cosines, clamped against round-off outside [-1, 1]
    cosδ = clamp(sin(φ1) * sin(φ2) + cos(φ1) * cos(φ2) * cos(Δλ), -1, 1)
    return 6371.0 * acos(cosδ)
end

"""Index of the land column among `candidates` closest to (`lond`, `latd`) in great-circle distance."""
function nearest_land_column(lond, latd, candidates = 1:N_land)
    distances = [great_circle_distance(lond, latd, londs_land[j], latds_land[j]) for j in candidates]
    return candidates[argmin(distances)]
end

# Only columns that are unfrozen after spin-up are admissible targets (see the freezing
# discussion above). The half-kelvin margin keeps us clear of the plateau's edge.
T_surface_spun_up = Array(interior(state.temperature)[:, 1, end])
unfrozen_columns = findall(>(NF(0.5)), T_surface_spun_up)
@info "Land columns unfrozen after spin-up" unfrozen = length(unfrozen_columns) total = N_land
isempty(unfrozen_columns) && error("No unfrozen land column after spin-up; lengthen the spin-up or start warmer.")

# Central Europe, roughly Berlin, or the nearest unfrozen column to it.
const i_target = nearest_land_column(13.4, 52.5, unfrozen_columns)
@info "Target land column" i_target lond = londs_land[i_target] latd = latds_land[i_target] T_surface = T_surface_spun_up[i_target]

# Great-circle distance from the target to every other land column, reused below.
distance_from_target = [great_circle_distance(londs_land[i_target], latds_land[i_target], londs_land[j], latds_land[j]) for j in 1:N_land]

# ## The differentiated function
#
# We drive the time loop by hand rather than through `Speedy.run!`, because `run!` also does
# output, callbacks, and feedback, none of which Enzyme should see. `initialize!(simulation;
# steps)` re-applies the dynamical-core scaling and resets the clock for exactly `N_steps`
# steps.
#
# `@ad_checkpoint` stores the state only at the checkpoints chosen by the `Revolve` scheme
# and recomputes everything in between during the reverse pass. This bounds the tape memory
# and, just as importantly, keeps the compile time of the adjoint manageable.

N_steps = 96    # 96 × 15 min = 24 h

"""Surface soil temperature of column `i_target` after `N_steps` coupled time steps."""
function target_soil_temperature!(vars, model, N_steps, scheme, i_target)
    @ad_checkpoint scheme for _ in 1:N_steps
        Speedy.time_step!(vars, model.time_stepping, model)
        Speedy.time_step!(vars.prognostic.clock, model.time_stepping)
    end
    # `z`-index `end` is the topmost (surface) soil layer; the middle index is the degenerate
    # `y` dimension of the Oceananigans column field.
    return interior(vars.prognostic.land.terrarium.temperature)[i_target, 1, end]
end

"""Same integration without checkpointing, for the finite-difference check."""
function target_soil_temperature_plain!(vars, model, N_steps, i_target)
    for _ in 1:N_steps
        Speedy.time_step!(vars, model.time_stepping, model)
        Speedy.time_step!(vars.prognostic.clock, model.time_stepping)
    end
    return interior(vars.prognostic.land.terrarium.temperature)[i_target, 1, end]
end

Speedy.initialize!(simulation; steps = N_steps)
vars = simulation.variables

# Keep a copy of the initial state so the finite-difference check below can restart from it.
# `deepcopy` (rather than `materialize_views`) is deliberate: it preserves the view→fused-parent
# aliasing that the time stepper relies on.
vars_initial = deepcopy(vars)

scheme = Revolve(N_steps)

# ## Taking the adjoint
#
# `make_zero` allocates the shadow memory Enzyme accumulates the vector-Jacobian product
# into. Because the entire Terrarium state lives inside SpeedyWeather's `Variables` tree at
# `vars.prognostic.land.terrarium`, a single `Duplicated(vars, dvars)` covers both the soil
# and the atmospheric initial conditions: one adjoint pass gives us the sensitivity to every
# prognostic field of the coupled system.
#
# `dmodel` catches the parameter sensitivities. We do not use them here, but Enzyme needs
# somewhere to put them.

dvars = make_zero(vars)
dmodel = make_zero(model)

@info "Computing the adjoint (this will take a while to compile)"
@time autodiff(
    set_runtime_activity(Reverse), target_soil_temperature!, Active,
    Duplicated(vars, dvars), Duplicated(model, dmodel),
    Const(N_steps), Const(scheme), Const(i_target),
)

# Views into the fused `Variables` buffers have to be materialized before we can slice them
# freely.
dvars_materialized = Speedy.materialize_views(dvars)

# ## The sensitivity field
#
# A point that is easy to get wrong: in Terrarium the soil **prognostic** variable is
# `internal_energy` (J/m³), and `temperature` is *diagnostic*, recomputed from the internal
# energy through the energy closure on every step. The shadow slot of a diagnostic is not the
# sensitivity to an initial condition: perturbing `temperature` at ``t_0`` changes nothing,
# because the first `compute_auxiliary!` overwrites it. So the initial-condition gradient we
# want is the one accumulated in `internal_energy`:
#
# ```math
# \mathbf{g}(j) = \frac{\partial T_\text{soil}(i^*, t_N)}{\partial U(j, t_0)}
# \qquad \text{in K / (J m}^{-3}\text{)}
# ```
#
# To read it as a sensitivity to an initial *temperature* field instead, multiply through by
# the volumetric heat capacity of each column, ``\partial U/\partial T = C``. We keep the
# energy form here because it is exactly what Enzyme returns, with no extra assumptions.

∂T_∂U₀_surface = Array(interior(dvars_materialized.prognostic.land.terrarium.internal_energy)[:, 1, end])

# Scatter the land-column vector back onto the full ring grid (`NaN` over ocean) so it can be
# plotted geographically.
∂T_∂U₀_field = RingGrids.Field(∂T_∂U₀_surface, land_grid; fill_value = NaN)

sens_fig = heatmap(
    ∂T_∂U₀_field,
    title = "∂T_soil(target, 24 h) / ∂U(j, 0), surface layer",
    size = (900, 450),
)

# The self-sensitivity dominates by orders of magnitude, which flattens the colorbar. Plot it
# again on a symmetric log scale to bring the remote structure out. The floor is set relative
# to the self-sensitivity, since the absolute scale in K/(J m⁻³) is set by the inverse heat
# capacity (~10⁻⁷ to 10⁻⁶) and is not a number worth hard-coding.
self_sensitivity = abs(∂T_∂U₀_surface[i_target])
symlog(x, floor_value) = sign(x) * log10(1 + abs(x) / floor_value)
∂T_∂U₀_symlog = RingGrids.Field(symlog.(∂T_∂U₀_surface, 1.0e-4 * self_sensitivity), land_grid; fill_value = NaN)

sens_log_fig = heatmap(
    ∂T_∂U₀_symlog,
    title = "Same sensitivity, symmetric log scale",
    size = (900, 450),
)

# ## Quantifying the lateral coupling
#
# Two numbers summarize how much of the derivative is *not* local. The remote fraction is the
# share of the total absolute sensitivity that sits away from the target column.

total_sensitivity = sum(abs, ∂T_∂U₀_surface)
remote_fraction = (total_sensitivity - self_sensitivity) / total_sensitivity

@printf(
    "self sensitivity          : %12.6e\nremote sensitivity (Σ|·|) : %12.6e\nremote fraction           : %8.4f %%\n",
    self_sensitivity, total_sensitivity - self_sensitivity, 100 * remote_fraction,
)

# And the reach: how far from the target the sensitivity is still appreciable. We bin the
# absolute sensitivity by great-circle distance.

edges = [0.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 20000.0]
bin_centers = [(edges[b] + edges[b + 1]) / 2 for b in 1:(length(edges) - 1)]
binned = map(1:(length(edges) - 1)) do b
    in_bin = findall(j -> edges[b] <= distance_from_target[j] < edges[b + 1], 1:N_land)
    return isempty(in_bin) ? 0.0 : sum(abs, ∂T_∂U₀_surface[in_bin])
end

decay_fig = Figure()
Axis(
    decay_fig[1, 1],
    xlabel = "Great-circle distance from target (km)",
    ylabel = "Σ |∂T_target/∂U₀| in bin",
    yscale = log10,
)
barplot!(decay_fig[1, 1], bin_centers, max.(binned, 1.0e-16))
decay_fig

# ## The atmospheric pathway
#
# The same adjoint pass also gives the sensitivity to the atmospheric state. The lowest model
# level shows where in the air the signal that reaches the target column came from, which is
# the mechanism behind the off-cell land sensitivities above.
#
# The same prognostic/diagnostic caveat applies on the atmospheric side. SpeedyWeather's
# prognostics are spectral (`vars.prognostic.temperature`, a `LowerTriangularArray`); the
# gridded `vars.grid.temperature` is re-derived by `transform!` at the end of every step. Its
# ``t_0`` shadow is nonetheless meaningful, because the physics parameterizations and the land
# coupling read the gridded field *before* the first `transform!` overwrites it. So this map
# is the sensitivity to the lowest-level air temperature as seen by the surface physics in the
# first step, which is exactly the entry point of the land–atmosphere coupling. It is not the
# complete sensitivity to the atmospheric initial condition; for that, read the spectral
# shadow `dvars_materialized.prognostic.temperature` and transform it to grid space.

l = Speedy.which_prognostic_step(vars.grid.temperature, model.time_stepping, Speedy.DummyParameterization())
∂T_∂Tair₀ = RingGrids.field_view(dvars_materialized.grid.temperature, :, nlayers_atmos, l)

air_fig = heatmap(
    ∂T_∂Tair₀,
    title = "∂T_soil(target, 24 h) / ∂T_air(j, lowest level, 0)",
    size = (900, 450),
)

# ## Validating a remote sensitivity
#
# Nonzero adjoint values far from the target cell have two possible explanations, and they
# must be told apart before any of this is interpretable:
#
# 1. **Real lateral coupling.** A soil temperature anomaly changes the surface fluxes, the
#    atmosphere carries the anomaly, and it lands on the target column.
# 2. **Spectral ringing.** SpeedyWeather's dynamical core is spectral, so a grid-point
#    perturbation projects onto the whole truncated spherical-harmonic basis and reappears
#    globally, at very small amplitude, after a single time step. This is a property of the
#    discretization, not of the physics.
#
# The magnitudes distinguish them: ringing is uniformly tiny and has no coherent spatial
# structure, whereas real transport concentrates downwind of the target and grows with the
# integration length. A direct check is to perturb one remote column and re-run the forward
# model, comparing against the adjoint prediction.

"""
Finite-difference check of ``∂T_target/∂U₀(j_probe)`` by re-running the forward model.

The perturbation goes on `internal_energy`, the prognostic variable: perturbing the diagnostic
`temperature` would be overwritten on the first step and yield exactly zero. The default `ε`
of 2 × 10⁵ J/m³ corresponds to roughly 0.1 K for a typical soil volumetric heat capacity of
2–3 × 10⁶ J/(m³ K). In `Float32` the forward differences are close to round-off, so treat
this as an order-of-magnitude check rather than a precise validation.
"""
function finite_difference_sensitivity(j_probe, ε = NF(2.0e5))
    perturbed = deepcopy(vars_initial)
    interior(perturbed.prognostic.land.terrarium.internal_energy)[j_probe, 1, end] += ε
    reference = deepcopy(vars_initial)

    T_plus = target_soil_temperature_plain!(perturbed, model, N_steps, i_target)
    T_ref = target_soil_temperature_plain!(reference, model, N_steps, i_target)
    return (T_plus - T_ref) / ε
end

# Pick the strongest remote sensitivity as the probe: it has the best signal-to-noise ratio
# against the finite-difference truncation error.
remote_columns = filter(!=(i_target), 1:N_land)
j_probe = remote_columns[argmax(abs.(∂T_∂U₀_surface[remote_columns]))]

@info "Finite-difference probe" j_probe distance_km = distance_from_target[j_probe]
fd = finite_difference_sensitivity(j_probe)
ad = ∂T_∂U₀_surface[j_probe]
@printf("adjoint: %12.6e\nfinite difference: %12.6e\nrelative error: %8.4f %%\n", ad, fd, 100 * abs(fd - ad) / abs(fd))

# ## Where to take this
#
# * **Sweep the integration length.** Repeat for `N_steps` spanning a few hours to several
#   days and track `remote_fraction`. That curve is the growth rate of the atmospheric
#   teleconnection between land columns, and it is the quantity worth reporting.
# * **Other target variables.** `state.skin_temperature`, `state.saturation_water_ice`, and
#   the surface fluxes are all in the same `Variables` tree; swapping the objective is a
#   one-line change.
# * **Resolution dependence.** Increasing `truncation` should sharpen the spatial structure
#   while leaving the physically meaningful part of the signal intact. If the remote fraction
#   changes a lot with truncation, spectral ringing is contributing more than we would like.
