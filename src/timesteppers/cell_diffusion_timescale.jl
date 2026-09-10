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

Return the advective stability timescale of the `integrator`. Land models currently transport no quantity
advectively, so this is always `Inf` (no advective restriction); it exists only so that a plain
`TimeStepWizard` (which always evaluates an advective timescale) can be applied to a Terrarium
`Simulation` without providing a custom `cell_advection_timescale`.
"""
cell_advection_timescale(integrator::ModelIntegrator{NF}) where {NF} = convert(NF, Inf)

# Generic (no diffusive restriction) fallback.
cell_diffusion_timescale(state, ::AbstractModel{NF}) where {NF} = convert(NF, Inf)

# Soil and land models: the diffusive operators live in the soil component.
cell_diffusion_timescale(state, model::SoilModel) =
    cell_diffusion_timescale(state, model.grid, model.soil, model.constants)

cell_diffusion_timescale(state, model::LandModel) =
    min(
    cell_diffusion_timescale(state, model.grid, model.soil, model.constants),
    cell_diffusion_timescale(state, model.grid, model.snow, model.constants)
)
