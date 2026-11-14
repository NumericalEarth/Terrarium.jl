# # Sensitivity of the land latent heat flux to the initial atmospheric humidity
#
# This script computes the adjoint (reverse-mode) sensitivity of the latent heat flux at a single
# land grid cell, at the end of a 24 hour coupled SpeedyWeather–Terrarium integration, to the
# **full three-dimensional field** of the specific humidity at the start of that integration:
#
# ```math
# g(j, k) = \frac{\partial \mathrm{LE}(i^*, t_0 + 24\,\mathrm{h})}{\partial q(j, k, t_0)},
# ```
#
# where ``i^*`` is the target land cell, ``j`` runs over all atmospheric grid points, and ``k``
# over the model levels. Reverse mode gives the whole field ``g`` in a single adjoint pass. The
# sensitivity combines the local pathway (the humidity of the air above the target cell sets the
# vapor pressure deficit that drives evaporation and transpiration) and the remote pathway (moisture
# that the atmosphere advects towards the target cell during the day, and its effect on clouds,
# radiation, and precipitation).
#
# SpeedyWeather's prognostic humidity is spectral. To obtain the sensitivity to the *gridded*
# initial humidity, the differentiated function adds a grid-space perturbation ``δq`` (zero) to the
# initial state, transforming it to spectral space within the function, and the gradient is taken
# with respect to ``δq``.
#
# The script follows `examples/autodiff/speedy_terrarium_lateral_coupling.jl`, which documents
# the Enzyme requirements and workarounds used below in more detail. It is a research script, not
# a doc-built example; the adjoint takes a long time to compile (tens of minutes) and is best run as
# a batch job. Vanilla Enzyme requires Julia 1.10 or 1.11, so run it with the 1.10 environment
# (`Manifest.toml`):
#
#     julia +lts --project=docs/joss docs/joss/run_coupled_sensitivity.jl
#
# ## Setup

using Terrarium

using Checkpointing
using Dates
using Enzyme
using Enzyme: Reverse, make_zero, make_zero!, set_runtime_activity
using Printf

import RingGrids
import SpeedyWeather as Speedy

# On Julia < 1.12, Enzyme's LLVM Attributor pass recurses without bound on the clock update inside
# an `@ad_checkpoint` loop and segfaults (same workaround as in SpeedyWeather's sensitivity examples).
Enzyme.Compiler.RunAttributor[] = false

arch = CPU()
NF = Float32

# Deliberately coarse: the adjoint tape and the compile time grow with the resolution.
truncation = 21
nlayers_atmos = 5
Nz = 10         # soil layers
Δz_min = 0.05   # (m) thickness of the topmost soil layer

# The raw sensitivity fields are written here; `figures.jl` plots them as Figure 4.
output_dir = mkpath(joinpath(@__DIR__, "outputs", "run_coupled_sensitivity"))

# ## Building the coupled model
#
# The land configuration needs no input data: homogeneous soil with the default soil hydrology
# (Richards' equation), no vegetation, and no snow, so the latent heat flux is the evaporation from
# the ground. Vegetation and snow are left out because they are among the most expensive land
# components for Enzyme to compile (the target is a mid-latitude cell in July), and the land is
# stepped explicitly with `ForwardEuler` rather than with the default implicit-explicit scheme for
# the same reason.

speedy_arch = RingGrids.Architectures.architecture(arch)
spectral_grid = Speedy.SpectralGrid(; truncation, nlayers = nlayers_atmos, architecture = speedy_arch)
ring_grid = spectral_grid.grid

land_sea_mask = Speedy.EarthLandSeaMask(spectral_grid)
Speedy.load_mask!(land_sea_mask)
land_grid = ColumnRingGrid(arch, NF, ExponentialSpacing(; N = Nz, Δz_min), ring_grid, land_sea_mask.land_fraction .> 0)

# The `Leapfrog` time step is adjusted when the model is initialized so that it divides the output
# interval, so compute that final value here.
output_interval = Hour(6)
time_stepping = Speedy.Leapfrog(spectral_grid, Δt_at_T32 = Minute(15))
Δt = Second(
    Speedy.get_Δt_millisec(
        time_stepping.Δt_at_T32, spectral_grid.truncation, Speedy.DEFAULT_RADIUS,
        time_stepping.adjust_with_output, output_interval,
    )
)
# SpeedyWeather advances the land by a whole number of land steps per atmospheric step, so the land
# time step must divide the atmospheric one: use the largest such step of at most 5 minutes.
land_Δt = Δt ÷ cld(Dates.value(Δt), 300)
@info "Atmosphere time step: $Δt, land time step: $land_Δt"

strat = HomogeneousSoilStratigraphy(NF)
soil = SoilEnergyWaterCarbon(NF; strat)
# warm, moist initial soil: 15 °C at the surface, 75 % saturation above a water table at 5 m depth
initializer = SoilInitializer(NF; energy = QuasiThermalSteadyState(NF; T₀ = 15), hydrology = SaturationWaterTable(NF))
timestepper = ForwardEuler(NF; Δt = Dates.value(land_Δt))
terrarium_model = Terrarium.LandModel(land_grid; soil, vegetation = nothing, snow = nothing, initializer, timestepper)
land = Speedy.LandModel(spectral_grid, terrarium_model; Δt = land_Δt)

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
# A 3 day forward run first, so that the adjoint is taken about an adjusted state rather than
# SpeedyWeather's reference atmosphere and the idealized initial soil state. The start time is
# chosen so that the 24 hour window after the spin-up ends at local midday in central Europe, when
# the latent heat flux there is large.

spinup = Day(3)
initial_date = DateTime(2024, 6, 28, 12)
simulation = Speedy.initialize!(model; time = initial_date)
@info "Spinning up the coupled model for $spinup"
@time Speedy.run!(simulation; period = spinup)

(; variables) = simulation
state = variables.prognostic.land.terrarium.state
Terrarium.checkfinite!(state.prognostic)

# ## Choosing the target cell
#
# Terrarium indexes land columns `1:N_land` in ring-grid order, so masking the ring-grid
# coordinates the same way recovers the position of each column.

mask = vec(Array(land_grid.mask.data))
ring_index_of_column = findall(mask)
londs_all, latds_all = RingGrids.get_londlatds(ring_grid)
londs_land = londs_all[mask]
latds_land = latds_all[mask]
N_land = length(londs_land)

"""Great-circle distance (km) between two points given in degrees."""
function great_circle_distance(lond1, latd1, lond2, latd2)
    φ1, φ2 = deg2rad(latd1), deg2rad(latd2)
    Δλ = deg2rad(lond2 - lond1)
    cosδ = clamp(sin(φ1) * sin(φ2) + cos(φ1) * cos(φ2) * cos(Δλ), -1, 1)
    return 6371.0 * acos(cosδ)
end

# Central Europe, roughly Berlin: the land column closest to it among those with a clearly positive
# latent heat flux after the spin-up.
latent_heat_flux_spun_up = Array(interior(state.latent_heat_flux)[:, 1, 1])
candidates = findall(>(NF(10)), latent_heat_flux_spun_up)
isempty(candidates) && error("No land column with a latent heat flux above 10 W/m² after the spin-up.")
i_target = candidates[argmin([great_circle_distance(13.4, 52.5, londs_land[j], latds_land[j]) for j in candidates])]
ij_target = ring_index_of_column[i_target]   # the same location on the atmospheric grid
@info "Target land column" i_target lond = londs_land[i_target] latd = latds_land[i_target] latent_heat_flux = latent_heat_flux_spun_up[i_target]

# ## The differentiated function
#
# `δq` (kg/kg) is a gridded perturbation of the specific humidity on all model levels. It is added
# to both leapfrog time levels of the spectral prognostic humidity, and the grid-space variables are
# then recomputed from the perturbed spectral state, so that the first time step sees it in both
# the dynamics and the physics. The time loop is driven by hand rather than with `Speedy.run!`,
# which also does output, callbacks, and feedback that Enzyme should not see; `@ad_checkpoint`
# with `Revolve` bounds the memory of the adjoint tape.

"""Latent heat flux (W/m²) of land column `i_target` after `N_steps` coupled time steps, with the
initial specific humidity perturbed by the gridded field `δq` (kg/kg)."""
function latent_heat_flux_after!(vars, δq, model, N_steps, scheme, i_target)
    S = model.spectral_transform
    for step in 1:2
        Speedy.set!(vars, model.geometry; humidity = δq, add = true, step, spectral_transform = S)
    end
    Speedy.transform!(vars, model)
    @ad_checkpoint scheme for _ in 1:N_steps
        Speedy.time_step!(vars, model.time_stepping, model)
        Speedy.time_step!(vars.prognostic.clock, model.time_stepping)
    end
    return interior(vars.prognostic.land.terrarium.state.latent_heat_flux)[i_target, 1, 1]
end

"""Adjoint of [`latent_heat_flux_after!`](@ref) starting from a copy of `vars_initial`; returns the
latent heat flux and its gradient with respect to the gridded initial humidity (W/m² per kg/kg)."""
function humidity_sensitivity(vars_initial, model, N_steps, i_target)
    vars = deepcopy(vars_initial)
    δq = zeros(NF, ring_grid, nlayers_atmos)
    dδq = zero(δq)
    dvars = make_zero(vars)
    # SpeedyWeather's `make_zero(::Variables)` zeroes only the leaf types it knows about and leaves the
    # Terrarium state nested at `prognostic.land.terrarium` as a copy of the primal, see the lateral
    # coupling example
    make_zero!(dvars.prognostic.land.terrarium.state)
    dmodel = make_zero(model)
    _, latent_heat_flux = autodiff(
        set_runtime_activity(ReverseWithPrimal), latent_heat_flux_after!, Active,
        Duplicated(vars, dvars), Duplicated(δq, dδq), Duplicated(model, dmodel),
        Const(N_steps), Const(Revolve(N_steps)), Const(i_target),
    )
    return latent_heat_flux, dδq
end

"""The same integration without differentiation, for the finite-difference check."""
function latent_heat_flux_forward(vars_initial, model, N_steps, i_target, δq)
    return latent_heat_flux_after!(deepcopy(vars_initial), δq, model, N_steps, Revolve(N_steps), i_target)
end

# Reset the clock for the differentiated window. `deepcopy` (rather than `materialize_views`)
# preserves the aliasing between the fused variable buffers and their views that the time stepper
# relies on.
N_steps = Int(Hour(24) ÷ Δt)
Speedy.initialize!(simulation; steps = N_steps)
vars_initial = deepcopy(simulation.variables)

# ## Checking the adjoint against finite differences
#
# Over the full 24 hours, finite differences in `Float32` measure the growth of the perturbation in
# the atmosphere rather than the derivative (see the lateral coupling example), so the adjoint is
# checked over a short window first, for a perturbation of the lowest-level humidity directly above
# the target cell. Changing `N_steps` does not recompile the adjoint.

N_check = 4
@info "Computing the adjoint over $N_check steps (this compiles the adjoint and takes a while)"
@time _, dδq_check = humidity_sensitivity(vars_initial, model, N_check, i_target)
ε = NF(1.0e-4)   # (kg/kg) = 0.1 g/kg
δq_probe = zeros(NF, ring_grid, nlayers_atmos)
δq_probe[ij_target, nlayers_atmos] = ε
fd = (latent_heat_flux_forward(vars_initial, model, N_check, i_target, δq_probe) - latent_heat_flux_forward(vars_initial, model, N_check, i_target, zero(δq_probe))) / ε
ad = dδq_check[ij_target, nlayers_atmos]
@printf("∂LE/∂q, lowest level above the target, after %d steps\n  adjoint:           %12.6e W/m² per kg/kg\n  finite difference: %12.6e W/m² per kg/kg\n  relative error:    %8.3f %%\n", N_check, ad, fd, 100 * abs(fd - ad) / abs(fd))

# ## The 24 hour sensitivity

@info "Computing the adjoint over $N_steps steps (24 h)"
@time latent_heat_flux, dδq = humidity_sensitivity(vars_initial, model, N_steps, i_target)
@info "Latent heat flux at the target after 24 h" latent_heat_flux

# Express the sensitivity per gram per kilogram of specific humidity, which is easier to read than
# per kilogram per kilogram. Level `k = nlayers_atmos` is the lowest model level.
sensitivity = Array(dδq.data) .* NF(1.0e-3)                  # (N_points, nlayers), W/m² per g/kg
sensitivity_lowest = sensitivity[:, nlayers_atmos]
sensitivity_column = vec(sum(sensitivity; dims = 2))        # response to a column-uniform perturbation

distance_from_target = [great_circle_distance(londs_land[i_target], latds_land[i_target], londs_all[j], latds_all[j]) for j in eachindex(londs_all)]
local_sensitivity = abs(sensitivity_column[ij_target])
remote_fraction = 1 - local_sensitivity / sum(abs, sensitivity_column)
@printf(
    "column sensitivity above the target: %12.6e W/m² per g/kg\nremote fraction of Σ|column sensitivity|: %8.3f %%\n",
    sensitivity_column[ij_target], 100 * remote_fraction,
)

# ## Mean winds over the 24 hour window
#
# For context in the figure, the winds that carry the humidity towards the target: the zonal and
# meridional wind averaged over the 24 hour window, from a plain forward integration of the same
# window (no differentiation). The σ levels are equally spaced, so the plain mean over the levels
# is the mass-weighted column mean, matching the column-uniform perturbation shown in Figure 4.

"""Mean zonal and meridional wind (m/s), averaged over `N_steps` time steps from a copy of
`vars_initial` and over the model levels, one value per atmospheric grid point."""
function mean_column_winds(vars_initial, model, N_steps)
    vars = deepcopy(vars_initial)
    Speedy.transform!(vars, model)
    u_sum = zeros(Float64, size(vars.grid.u))
    v_sum = zeros(Float64, size(vars.grid.v))
    for _ in 1:N_steps
        Speedy.time_step!(vars, model.time_stepping, model)
        Speedy.time_step!(vars.prognostic.clock, model.time_stepping)
        u_sum .+= Array(vars.grid.u.data)
        v_sum .+= Array(vars.grid.v.data)
    end
    nlayers = size(u_sum, 2)
    u_mean = vec(sum(u_sum; dims = 2)) ./ (N_steps * nlayers)
    v_mean = vec(sum(v_sum; dims = 2)) ./ (N_steps * nlayers)
    return u_mean, v_mean
end

u_mean, v_mean = mean_column_winds(vars_initial, model, N_steps)

# Write the raw fields (one row per atmospheric grid point).
open(joinpath(output_dir, "sensitivity.csv"), "w") do io
    print(io, "point,lond,latd,distance_km")
    for k in 1:nlayers_atmos
        print(io, ",dLE_dq_level$k")
    end
    println(io, ",u_mean,v_mean")
    for j in eachindex(londs_all)
        @printf(io, "%d,%.6f,%.6f,%.2f", j, londs_all[j], latds_all[j], distance_from_target[j])
        for k in 1:nlayers_atmos
            @printf(io, ",%.8e", sensitivity[j, k])
        end
        @printf(io, ",%.6e,%.6e\n", u_mean[j], v_mean[j])
    end
end

@info "Wrote the sensitivity fields to $output_dir"
