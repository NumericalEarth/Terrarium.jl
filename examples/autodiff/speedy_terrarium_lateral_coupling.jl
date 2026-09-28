# # Lateral coupling of land columns through the atmosphere
#
# Terrarium processes are currently all *column based*: every land grid cell integrates its own
# 1D soil column, and nothing in Terrarium itself moves heat or water laterally between columns.
# Coupled to SpeedyWeather, the columns stop being independent: each exchanges heat, moisture,
# and radiation with the atmosphere, which transports those fluctuations horizontally and
# deposits them on other land columns. The only path from one soil column to another runs
# through the air.
#
# This example measures that path directly. We take the adjoint (reverse-mode) derivative of
# the surface soil temperature of **one** land grid cell, at the end of a short coupled
# integration, with respect to the **full spatial field** of the initial soil state. The soil
# prognostic variable is the internal energy ``U`` (J/m³), from which temperature is
# diagnosed, so the derivative is taken with respect to ``U``:
#
# ```math
# \mathbf{g}(j) = \frac{\partial T_\text{soil}(i^*, t_N)}{\partial U(j, t_0)},
# \qquad j = 1, \dots, N_\text{land}
# ```
#
# A purely local column model gives ``\mathbf{g}(j) = 0`` for every ``j \neq i^*``. Anything
# nonzero off the target cell is lateral coupling mediated by SpeedyWeather, and the spatial
# structure of ``\mathbf{g}`` says how far and in which direction it reaches. Reverse mode is
# the natural choice: one scalar output, ``N_\text{land}`` inputs, one adjoint pass.
#
# !!! warning "Research script; the adjoint has not been seen to compile"
#     Everything up to the `autodiff` call (coupled model, spin-up, target selection, the
#     finite-difference check) runs as written. The `autodiff` call itself has not yet completed
#     on any configuration tried, down to T9 with one atmospheric and two soil layers. The last
#     measured attempt (2026-09-25; Julia 1.10.12, Enzyme 0.13.204, SpeedyWeather 0.22.1) got
#     past Terrarium and failed inside SpeedyWeather, either with an `EnzymeInternalError` in
#     `vertical_advection!` or with an Enzyme `TypeAnalysis` assertion. Since then this branch
#     has made `StateVariables` and the model types mutable (a large compile-time win on the
#     Terrarium side) and added a `tick!` reverse rule in `TerrariumEnzymeExt` that keeps the
#     clock's integer counters out of Enzyme's analysis, but the coupled step has not been
#     re-measured past that point. Details and the bisection numbers are in
#     `docs/dev/2026-09/2026-09-24_NOTE_enzyme_landmodel_compile_time.md`.
#
#     Use Julia 1.10, keep the resolution and step count small, and treat everything after the
#     `autodiff` call as the intended analysis rather than a result.
#
# ## Setup

import Pkg
Pkg.activate(@__DIR__)

# On Julia 1.10, Pkg ignores the `[sources]` entry pointing `Terrarium` at this checkout and
# silently resolves a *registered* Terrarium instead. Make sure the local package is in use:
#     Pkg.develop(path = joinpath(@__DIR__, "..", ".."))
using Terrarium

using Checkpointing
using Dates
using Enzyme
using Enzyme: Reverse, make_zero, set_runtime_activity
using Printf

using CairoMakie

import RingGrids
import SpeedyWeather as Speedy

# On Julia < 1.12, Enzyme's LLVM Attributor pass recurses without bound on the clock update
# inside an `@ad_checkpoint` loop and segfaults. Same workaround as SpeedyWeather's own
# sensitivity examples.
Enzyme.Compiler.RunAttributor[] = false

arch = CPU()
NF = Float32

# Deliberately coarse: the adjoint tape grows with both the horizontal resolution and the
# number of soil layers, and the qualitative result is visible at low resolution.
truncation = 21
nlayers_atmos = 5
Nz = 4          # soil layers
Δz_min = 0.05   # (m) thickness of the topmost soil layer

# ## Building the coupled model
#
# The land configuration is minimal and data-free: homogeneous soil, no vegetation, no snow,
# no external input datasets. Every off-cell sensitivity then has to come from the atmosphere
# rather than from a shared input field.

speedy_arch = RingGrids.Architectures.architecture(arch)
spectral_grid = Speedy.SpectralGrid(; truncation, nlayers = nlayers_atmos, architecture = speedy_arch)
ring_grid = spectral_grid.grid

land_sea_mask = Speedy.EarthLandSeaMask(spectral_grid)
Speedy.load_mask!(land_sea_mask)

# The Terrarium mask must be a superset of SpeedyWeather's land, so build it from the same
# fractional mask.
land_grid = ColumnRingGrid(arch, NF, ExponentialSpacing(; N = Nz, Δz_min), ring_grid, land_sea_mask.land_fraction .> 0)

# Homogeneous soil (the default, made explicit): one texture and porosity everywhere, so no
# per-horizon input fields enter the differentiated state.
strat = HomogeneousSoilStratigraphy(eltype(land_grid))
soil = SoilEnergyWaterCarbon(eltype(land_grid); strat, hydrology = SoilHydrology(eltype(land_grid)))

# Soil freezing can silently zero this experiment. The energy closure uses the `FreeWater`
# freeze curve, so all phase change happens at exactly 0 °C, and a wet column on that plateau
# has ``\partial T / \partial U = 0`` identically: added energy melts ice instead of raising the
# temperature. A target column on the plateau gives an all-zero sensitivity map, and remote
# columns on the plateau contribute nothing however strongly the atmosphere couples them.
# (Smooth `SFCC` curves exist but are not yet dispatched in the closure; that fix belongs in
# the source.)
#
# So we start warmer than the default `T₀ = 0` °C, and we also spin up: SpeedyWeather's initial
# atmosphere is a reference state and the land coupling reads the lowest model level, so the
# first hours of a coupled run are a cold shock of several kelvin in the top 5 cm that can push
# mid-latitude columns onto the plateau. The target column is chosen *after* spin-up, from
# the columns that are actually unfrozen.
soil_initializer = SoilInitializer(eltype(land_grid); energy = QuasiThermalSteadyState(eltype(land_grid); T₀ = 10))

# No snow. A compile-time concession, not a physical one: `SingleLayerSnow` is the single most
# expensive component for Enzyme's analysis of a `LandModel` step. For a mid-latitude July
# target the assumption is mild, and it removes one more 0 °C plateau.
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
# An ordinary (non-differentiated) run first, so the adjoint is taken about a physically
# settled state rather than the reference atmosphere.

initial_date = DateTime(2024, 7, 1)
simulation = Speedy.initialize!(model; time = initial_date)
@info "Spinning up the coupled model"
@time Speedy.run!(simulation; period = Day(10))

(; variables) = simulation
state = variables.prognostic.land.terrarium
Terrarium.checkfinite!(state.prognostic)

# ## Choosing the target cell
#
# A mid-latitude column, where the flow is vigorous enough for the coupling to travel.
# Terrarium indexes land columns `1:N_land` in ring-grid order, so masking the ring-grid
# coordinates the same way recovers each column's position.

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

# Only columns unfrozen after spin-up are admissible targets; the half-kelvin margin keeps
# clear of the plateau's edge.
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
# The time loop is driven by hand rather than through `Speedy.run!`, which also does output,
# callbacks, and feedback that Enzyme should not see. `initialize!(simulation; steps)`
# re-applies the dynamical-core scaling and resets the clock for exactly `N_steps` steps.
# `@ad_checkpoint` with `Revolve` bounds the tape memory and keeps the adjoint's compile time
# manageable.

N_steps = 96    # 96 × 15 min = 24 h

"""Surface soil temperature of column `i_target` after `N_steps` coupled time steps."""
function target_soil_temperature!(vars, model, N_steps, scheme, i_target)
    @ad_checkpoint scheme for _ in 1:N_steps
        Speedy.time_step!(vars, model.time_stepping, model)
        Speedy.time_step!(vars.prognostic.clock, model.time_stepping)
    end
    # `z`-index `end` is the surface layer; the middle index is the degenerate `y` dimension.
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

# Copy of the initial state for the finite-difference check. `deepcopy` (not
# `materialize_views`) preserves the view→fused-parent aliasing the time stepper relies on.
vars_initial = deepcopy(vars)

scheme = Revolve(N_steps)

# ## Taking the adjoint
#
# `make_zero` allocates the shadows Enzyme accumulates the vector-Jacobian product into. The
# Terrarium state lives inside SpeedyWeather's `Variables` tree at
# `vars.prognostic.land.terrarium`, so one `Duplicated(vars, dvars)` covers the soil and the
# atmospheric initial conditions alike. `dmodel` catches parameter sensitivities we do not use.

dvars = make_zero(vars)
dmodel = make_zero(model)

@info "Computing the adjoint (this will take a while to compile)"
@time autodiff(
    set_runtime_activity(Reverse), target_soil_temperature!, Active,
    Duplicated(vars, dvars), Duplicated(model, dmodel),
    Const(N_steps), Const(scheme), Const(i_target),
)

# Views into the fused `Variables` buffers must be materialized before slicing.
dvars_materialized = Speedy.materialize_views(dvars)

# ## The sensitivity field
#
# Easy to get wrong: the soil **prognostic** is `internal_energy`; `temperature` is
# *diagnostic*, recomputed from it every step, so its ``t_0`` shadow is not an initial-condition
# sensitivity (perturbing it changes nothing, the first `compute_auxiliary!` overwrites it). The
# gradient we want is the one accumulated in `internal_energy`, in K/(J m⁻³). Multiplying by
# the volumetric heat capacity ``C = \partial U/\partial T`` would turn it into a sensitivity to
# an initial *temperature* field; we keep the energy form, which is exactly what Enzyme returns.

∂T_∂U₀_surface = Array(interior(dvars_materialized.prognostic.land.terrarium.internal_energy)[:, 1, end])

# Scatter the land-column vector back onto the ring grid (`NaN` over ocean) for plotting.
∂T_∂U₀_field = RingGrids.Field(∂T_∂U₀_surface, land_grid; fill_value = NaN)

sens_fig = heatmap(
    ∂T_∂U₀_field,
    title = "∂T_soil(target, 24 h) / ∂U(j, 0), surface layer",
    size = (900, 450),
)

# The self-sensitivity dominates by orders of magnitude and flattens the colorbar, so plot
# again on a symmetric log scale. The floor is relative to the self-sensitivity, since the
# absolute scale (~10⁻⁷ to 10⁻⁶ K/(J m⁻³)) is just the inverse heat capacity.
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
# Two numbers summarize how much of the derivative is *not* local: the remote fraction (share
# of total absolute sensitivity away from the target) and the decay with distance.

total_sensitivity = sum(abs, ∂T_∂U₀_surface)
remote_fraction = (total_sensitivity - self_sensitivity) / total_sensitivity

@printf(
    "self sensitivity          : %12.6e\nremote sensitivity (Σ|·|) : %12.6e\nremote fraction           : %8.4f %%\n",
    self_sensitivity, total_sensitivity - self_sensitivity, 100 * remote_fraction,
)

# Absolute sensitivity binned by great-circle distance from the target.

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
# The same pass gives the sensitivity to the atmospheric state; the lowest model level shows
# where in the air the signal reaching the target came from. The prognostic/diagnostic caveat
# applies here too: SpeedyWeather's prognostics are spectral, and the gridded
# `vars.grid.temperature` is re-derived by `transform!` at the end of every step. Its ``t_0``
# shadow is still meaningful because the surface physics and land coupling read the gridded
# field *before* the first `transform!`, so this map is the sensitivity to the lowest-level air
# temperature as seen by the surface physics in the first step: the entry point of the
# coupling. The complete atmospheric initial-condition sensitivity is the spectral shadow
# `dvars_materialized.prognostic.temperature`, transformed to grid space.

l = Speedy.which_prognostic_step(vars.grid.temperature, model.time_stepping, Speedy.DummyParameterization())
∂T_∂Tair₀ = RingGrids.field_view(dvars_materialized.grid.temperature, :, nlayers_atmos, l)

air_fig = heatmap(
    ∂T_∂Tair₀,
    title = "∂T_soil(target, 24 h) / ∂T_air(j, lowest level, 0)",
    size = (900, 450),
)

# ## Validating a remote sensitivity
#
# Nonzero adjoint values far from the target have two possible sources: real lateral
# coupling (a soil anomaly changes the surface fluxes, the atmosphere carries it, it lands on
# the target), or spectral ringing (a grid-point perturbation projects onto the whole truncated
# spherical-harmonic basis and reappears globally at tiny amplitude after one step). Ringing is
# uniformly small with no coherent structure; real transport concentrates downwind and grows
# with integration length. The direct check is to perturb one remote column and re-run the
# forward model.

"""
Finite-difference check of ``∂T_target/∂U₀(j_probe)`` by re-running the forward model.

The perturbation goes on the prognostic `internal_energy`; the default `ε` of 2 × 10⁵ J/m³ is
roughly 0.1 K for a typical volumetric heat capacity of 2–3 × 10⁶ J/(m³ K). In `Float32` the
forward differences are near round-off, so this is an order-of-magnitude check.
"""
function finite_difference_sensitivity(j_probe, ε = NF(2.0e5))
    perturbed = deepcopy(vars_initial)
    interior(perturbed.prognostic.land.terrarium.internal_energy)[j_probe, 1, end] += ε
    reference = deepcopy(vars_initial)

    T_plus = target_soil_temperature_plain!(perturbed, model, N_steps, i_target)
    T_ref = target_soil_temperature_plain!(reference, model, N_steps, i_target)
    return (T_plus - T_ref) / ε
end

# Probe the strongest remote sensitivity, which has the best signal-to-noise ratio against
# the finite-difference error.
remote_columns = filter(!=(i_target), 1:N_land)
j_probe = remote_columns[argmax(abs.(∂T_∂U₀_surface[remote_columns]))]

@info "Finite-difference probe" j_probe distance_km = distance_from_target[j_probe]
fd = finite_difference_sensitivity(j_probe)
ad = ∂T_∂U₀_surface[j_probe]
@printf("adjoint: %12.6e\nfinite difference: %12.6e\nrelative error: %8.4f %%\n", ad, fd, 100 * abs(fd - ad) / abs(fd))

# ## Where to take this
#
# * **Sweep the integration length.** `remote_fraction` against `N_steps`, from hours to days,
#   is the growth rate of the atmospheric teleconnection between land columns.
# * **Other target variables.** `state.skin_temperature`, `state.saturation_water_ice`, and the
#   surface fluxes are in the same `Variables` tree; swapping the objective is a one-line change.
# * **Resolution dependence.** If the remote fraction changes a lot with `truncation`, spectral
#   ringing is contributing more than we would like.
