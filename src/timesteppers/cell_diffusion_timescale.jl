# Cell advection/diffusion timescale diagnostics for adaptive timestepping.
#
# These methods provide the `cell_diffusion_timescale` (and a trivial `cell_advection_timescale`)
# consumed by the Oceananigans `TimeStepWizard` callback so that a `Simulation` wrapping a
# `ModelIntegrator` can adapt its step size to the diffusive stability limit of the soil operators.

"""
    $TYPEDSIGNATURES

Return the minimum diffusive stability timescale ``τ = Δz² / D`` (seconds) over all grid cells of the
`integrator`, where `D` is the largest effective diffusivity among the model's diffusive processes.
This is the diagnostic consumed by the Oceananigans `TimeStepWizard` to enforce a diffusive
Courant–Friedrichs–Lewy (CFL) constraint of the form ``Δt ≈ \\mathrm{diffusive\\_cfl} · τ``.

The generic fallback (for models without an implemented diffusive timescale) returns `Inf`, i.e. no
diffusive restriction, mirroring the Oceananigans `infinite_diffusion_timescale` convention.
"""
cell_diffusion_timescale(integrator::ModelIntegrator) =
    cell_diffusion_timescale(integrator.state, integrator.model)

"""
    $TYPEDSIGNATURES

Return the advective stability timescale (seconds) of the `integrator`, consumed by the Oceananigans
`TimeStepWizard` to enforce an advective CFL constraint. For [`SoilModel`](@ref) and
[`LandModel`](@ref) this is the timescale of the explicit upwind advection of sensible heat by the soil
water flux (see [`advective_heat_flux`](@ref)), which is `Inf` without water flow. Models that
transport no quantity advectively return `Inf` (no advective restriction).
"""
cell_advection_timescale(integrator::ModelIntegrator) =
    cell_advection_timescale(integrator.state, integrator.model)

# Generic (no advective restriction) fallback.
cell_advection_timescale(state, ::AbstractModel{NF}) where {NF} = convert(NF, Inf)

cell_advection_timescale(state, model::SoilModel) = cell_advection_timescale(state, model.grid, model.soil)

cell_advection_timescale(state, model::LandModel) = cell_advection_timescale(state, get_grid(model), model.soil)

# Generic (no diffusive restriction) fallback.
cell_diffusion_timescale(state, ::AbstractModel{NF}) where {NF} = convert(NF, Inf)

# Soil and land models: the diffusive operators live in the soil component. Implicitly integrated
# heat conduction and Richards flow impose no diffusive restriction.
cell_diffusion_timescale(state, model::SoilModel) =
    cell_diffusion_timescale(state, model.grid, model.soil, model.constants, get_timestepper(model))

cell_diffusion_timescale(state, model::LandModel) =
    min(
    cell_diffusion_timescale(state, model.grid, model.soil, model.constants, get_timestepper(model)),
    cell_diffusion_timescale(state, model.grid, model.snow, model.constants)
)
