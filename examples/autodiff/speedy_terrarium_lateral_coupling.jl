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
# !!! note "Status and requirements"
#     This is a research script, not a doc-built example. As of 2026-09-29 the full T21 adjoint
#     below compiles in about 30 min (Julia 1.10.12) and its gradient has been validated against
#     finite differences on an 8-step integration: ∂T/∂U at the target column 5.268e-8 vs.
#     5.263e-8 (ε = 2e5 J/m³), and at a column 552 km away 3.62e-10 vs. 3.69e-10, with every
#     other column's sensitivity smaller than the target's. The lateral coupling is real and
#     measurable. Three things are required to get there:
#
#     * **Enzyme at or after PR 3709** (`EnzymeAD/Enzyme.jl#3709`, "Do not fold loads of
#       non-const globals on Julia 1.10"); the 0.13.205 release crashes in the reverse pass.
#     * **`make_zero!` on the Terrarium shadow** after `make_zero(vars)`: SpeedyWeather's
#       view-preserving `make_zero(::Variables)` copies the primal and zeroes only the leaf
#       types it knows, so without this the Terrarium state's shadow starts as a copy of the
#       state and the "gradient" is garbage of order 1e6 to 1e8. Done below; a SpeedyWeather-side
#       fix (zeroing foreign leaves with `Enzyme.make_zero!`) makes it redundant once merged.
#     * A `DateTime` clock, which the coupled model uses. A standalone Terrarium `LandModel`
#       with a floating-point clock on Oceananigans ≥ 0.113.2 additionally needs
#       `Adapt.adapt_structure(::CPU, clock::Clock) = clock`, because CPU kernel launches now
#       adapt the clock into a NamedTuple Enzyme has no shadow for.
#
#     Compile-time bisection and the history of these findings are in
#     `docs/dev/2026-09/2026-09-24_NOTE_enzyme_landmodel_compile_time.md`. `SingleLayerSnow`
#     remains the dominant Terrarium-side compile cost, so the script runs with `snow = nothing`.
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
using Enzyme: Reverse, make_zero, make_zero!, set_runtime_activity
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

# Figures and the raw sensitivity field are written here (git-ignored).
output_dir = mkpath(joinpath(@__DIR__, "outputs"))
target_marker!(fig::Figure) = scatter!(content(fig[1, 1]), [londs_land[i_target]], [latds_land[i_target]]; marker = :xcross, markersize = 18, color = :black, strokecolor = :white, strokewidth = 1)

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
# SpeedyWeather's `make_zero(::Variables)` builds the shadow as a `deepcopy` (to keep the fused-buffer
# views) and then zeroes only the SpeedyWeather leaf types it knows about. The Terrarium
# `StateVariables` nested at `prognostic.land.terrarium` is not one of them and would be left as a
# copy of the primal state, which Enzyme then accumulates the adjoint on top of. Zero it explicitly.
make_zero!(dvars.prognostic.land.terrarium)
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
# gradient Enzyme returns is the one accumulated in `internal_energy`, in K/(J m⁻³), for every
# soil layer of every column.

∂T_∂U₀ = Array(interior(dvars_materialized.prognostic.land.terrarium.internal_energy)[:, 1, :])   # (N_land, Nz)
∂T_∂U₀_surface = ∂T_∂U₀[:, end]

# A joule per cubic meter is a very small amount of energy (warming wet soil by 1 K takes
# ``C ≈ 2–4 × 10⁶`` J/m³), so the energy form is hard to read. Multiplying by a volumetric heat
# capacity ``C = ∂U/∂T`` gives the sensitivity to the initial *temperature* in K/K. The exact
# ``C`` varies by cell with composition and phase (and is undefined on the 0 °C plateau); the
# differences are a factor of two at most, so a single constant, the heat capacity of water, is
# used everywhere. Read the K/K numbers as "per kelvin of a water-like soil".
heat_capacity = SoilHeatCapacities(NF).water
∂T_∂T₀ = ∂T_∂U₀ .* heat_capacity                       # (N_land, Nz), K/K
∂T_∂T₀_surface = ∂T_∂T₀[:, end]
# Response to warming the *whole* initial column of `j` by 1 K: the column's memory, rather than
# that of its top 5 cm alone.
∂T_∂T₀_column = vec(sum(∂T_∂T₀; dims = 2))

# Write the raw fields first (one row per land column), so nothing downstream can lose them.
open(joinpath(output_dir, "sensitivity.csv"), "w") do io
    print(io, "column,lond,latd,distance_km")
    for k in 1:Nz; print(io, ",dT_dU0_layer$k"); end
    for k in 1:Nz; print(io, ",heat_capacity_layer$k"); end
    for k in 1:Nz; print(io, ",dT_dT0_layer$k"); end
    println(io, ",dT_dT0_column")
    for j in 1:N_land
        @printf(io, "%d,%.6f,%.6f,%.2f", j, londs_land[j], latds_land[j], distance_from_target[j])
        for k in 1:Nz; @printf(io, ",%.8e", ∂T_∂U₀[j, k]); end
        for k in 1:Nz; @printf(io, ",%.8e", heat_capacity); end
        for k in 1:Nz; @printf(io, ",%.8e", ∂T_∂T₀[j, k]); end
        @printf(io, ",%.8e\n", ∂T_∂T₀_column[j])
    end
end

# Scatter a land-column vector back onto the ring grid (`NaN` over ocean) for plotting.
to_map(v) = RingGrids.Field(v, land_grid; fill_value = NaN)
# Divergent colormap, symmetric about zero, so sign is readable at a glance.
symmetric_range(v) = (m = maximum(abs, filter(!isnan, v)); (-m, m))

sens_fig = heatmap(
    to_map(∂T_∂T₀_surface),
    colormap = Makie.Reverse(:RdBu), colorrange = symmetric_range(to_map(∂T_∂T₀_surface)),
    title = "∂T_soil(target, 24 h) / ∂T_soil(j, 0), surface layer  (K / K)",
    size = (900, 450),
)
target_marker!(sens_fig)
save(joinpath(output_dir, "sensitivity_surface_KperK.png"), sens_fig)
sens_fig

# The self-sensitivity dominates by orders of magnitude and flattens the colorbar, so plot
# again on a symmetric log scale, with the floor relative to the self-sensitivity.
self_sensitivity = abs(∂T_∂T₀_surface[i_target])
symlog(x, floor_value) = sign(x) * log10(1 + abs(x) / floor_value)

sens_log_fig = heatmap(
    to_map(symlog.(∂T_∂T₀_surface, 1.0e-4 * self_sensitivity)),
    colormap = Makie.Reverse(:RdBu), colorrange = symmetric_range(to_map(symlog.(∂T_∂T₀_surface, 1.0e-4 * self_sensitivity))),
    title = "Same, symmetric log₁₀ scale (floor 10⁻⁴ × self)",
    size = (900, 450),
)
target_marker!(sens_log_fig)
save(joinpath(output_dir, "sensitivity_surface_symlog.png"), sens_log_fig)
sens_log_fig

# The column-integrated version: what a uniform 1 K anomaly through the whole initial soil
# column of `j` does to the target's surface temperature a day later.
column_fig = heatmap(
    to_map(symlog.(∂T_∂T₀_column, 1.0e-4 * abs(∂T_∂T₀_column[i_target]))),
    colormap = Makie.Reverse(:RdBu), colorrange = symmetric_range(to_map(symlog.(∂T_∂T₀_column, 1.0e-4 * abs(∂T_∂T₀_column[i_target])))),
    title = "∂T_soil(target, 24 h) / ∂T_soil(j, 0), whole column, symmetric log₁₀ (K / K)",
    size = (900, 450),
)
target_marker!(column_fig)
save(joinpath(output_dir, "sensitivity_column_symlog.png"), column_fig)
column_fig

# ## Quantifying the lateral coupling
#
# Two numbers summarize how much of the derivative is *not* local: the remote fraction (share
# of total absolute sensitivity away from the target) and the decay with distance. Both use the
# K/K surface field.

valid = 1:N_land
total_sensitivity = sum(abs, ∂T_∂T₀_surface[valid])
remote_fraction = (total_sensitivity - self_sensitivity) / total_sensitivity

@printf(
    "self sensitivity (K/K)        : %12.6e\nremote sensitivity Σ|·| (K/K) : %12.6e\nremote fraction               : %8.4f %%\ncolumn-integrated self (K/K)  : %12.6e\n",
    self_sensitivity, total_sensitivity - self_sensitivity, 100 * remote_fraction, ∂T_∂T₀_column[i_target],
)

# Absolute sensitivity against great-circle distance from the target, every column as a point
# and the mean per distance bin on top. Bin *means* rather than sums, so the curve is not
# dominated by how many columns happen to fall in each bin.

edges = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 20000.0]
bin_centers = [sqrt(edges[b] * edges[b + 1]) for b in 1:(length(edges) - 1)]
bin_means = map(1:(length(edges) - 1)) do b
    in_bin = filter(j -> edges[b] <= distance_from_target[j] < edges[b + 1], valid)
    return isempty(in_bin) ? NaN : sum(abs, ∂T_∂T₀_surface[in_bin]) / length(in_bin)
end

remote_columns = filter(!=(i_target), valid)
decay_fig = Figure()
decay_ax = Axis(
    decay_fig[1, 1],
    xlabel = "Great-circle distance from target (km)",
    ylabel = "|∂T_target/∂T₀(j)|, surface layer  (K / K)",
    xscale = log10, yscale = log10,
)
scatter!(decay_ax, distance_from_target[remote_columns], max.(abs.(∂T_∂T₀_surface[remote_columns]), 1.0e-12); markersize = 5, color = (:steelblue, 0.6), label = "land columns")
scatterlines!(decay_ax, bin_centers, bin_means; color = :black, markersize = 12, label = "bin mean")
hlines!(decay_ax, [self_sensitivity]; color = :red, linestyle = :dash, label = "self sensitivity")
axislegend(decay_ax, position = :rt)
save(joinpath(output_dir, "sensitivity_decay.png"), decay_fig)
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
    colormap = Makie.Reverse(:RdBu), colorrange = symmetric_range(Array(∂T_∂Tair₀)),
    title = "∂T_soil(target, 24 h) / ∂T_air(j, lowest level, 0)  (K / K)",
    size = (900, 450),
)
target_marker!(air_fig)
save(joinpath(output_dir, "sensitivity_air_temperature.png"), air_fig)
air_fig

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
roughly 0.1 K for a typical volumetric heat capacity of 2–3 × 10⁶ J/(m³ K).

Over a short window this agrees with the adjoint to a few percent (8 steps: self 5.27e-8 vs.
5.26e-8, a column 552 km away 3.62e-10 vs. 3.69e-10). Over the full 24 h it does not: in
`Float32` the two forward runs diverge through the atmosphere by far more than the remote
signal (a 24 h probe gave 4.6e-8 against an adjoint of 2.0e-10, i.e. a 0.1 K remote
perturbation "moving" the target by more than its own initial condition does), so the
finite difference measures perturbation growth, not the derivative. Validate with a short
`N_steps`, or in `Float64`; the adjoint itself is the trustworthy number at 24 h.
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
