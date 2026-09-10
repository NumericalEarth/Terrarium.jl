"""
    RichardsEq{PS} <: AbstractVerticalFlow

[`SoilHydrology`](@ref) flow operator implementing the mixed saturation-pressure form
of the Richardson-Richards equation.

State variables defined by the Richards' formulation of `SoilHydrology`:

- `saturation_water_ice`: saturation level of water and ice in the pore space.
- `hydraulic_conductivity`: hydraulic conductivity at cell centers (m/s).
- `water_table``: elevation of the water table (m).
- `liquid_water_fraction`: fraction of unfrozen liquid water in the pore space (dimensionless).

Excess water that reaches the soil surface is routed to the `surface_excess_water` pool defined by
[`AbstractSurfaceRunoff`](@ref) process (see [`adjust_saturation_profile!`](@ref)), if passed as a dependency.
For standalone soil hydrology with no surface runoff, surface excess water is discarded.

See also [`SoilSaturationPressureClosure`](@ref) and [`AbstractSoilHydraulics`](@ref) for details regarding the
closure relating saturation and pressure head.
"""
@kwdef struct RichardsEq <: AbstractVerticalFlow end

variables(hydrology::SoilHydrology{NF, RichardsEq}) where {NF} = (
    prognostic(:saturation_water_ice, XYZ(); closure = get_closure(hydrology), bounds = UnitInterval, desc = "Saturation level of water and ice in the pore space"),
    auxiliary(:hydraulic_conductivity, XYZ(z = Face()), units = u"m/s", desc = "Hydraulic conductivity of soil volumes in m/s"),
    auxiliary(:water_table, XY(), units = u"m", desc = "Elevation of the water table in meters"),
    input(:liquid_water_fraction, XYZ(), default = NF(1), bounds = UnitInterval, desc = "Fraction of unfrozen water in the pore space"),
)

# Top-level interface methods

""" $TYPEDSIGNATURES """
function initialize!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        constants::PhysicalConstants,
        args...
    ) where {NF}
    # Assume saturation is given as initial condition and apply closure!
    # TODO: Though rare, there may be cases where we want to set an initial
    # condition for pressure instead of saturation. This would need to be
    # refactored into another initialization method.
    closure!(state, grid, hydrology.closure, hydrology, soil)
    compute_auxiliary!(state, grid, hydrology, soil, constants)
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        args...
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    out = auxiliary_fields(state, hydrology)
    fields = get_fields(state, hydrology, strat, bgc; except = out)
    launch!(grid, XYZ, compute_hydraulics_kernel!, out, fields, hydrology, strat, bgc)
    return nothing
end

"""
    $TYPEDSIGNATURES

Fill the `pressure_head` halo and apply the top/bottom flux boundary conditions for
`saturation_water_ice`. The `fields` passed to those boundary conditions are built explicitly here
(rather than using `state.inputs`, which only carries declared `input` variables) so that a
discrete-form condition like [`saturation_infiltration_bc`](@ref) can read auxiliary fields
(`infiltration`) and per-horizon namespaces (via `strat`/`bgc`) that only exist once the full
`state` does. `infiltration` is only merged in when present, since standalone (non-`LandModel`)
configurations may not declare it at all.
"""
function compute_boundary_conditions!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    fill_halo_regions!(state.pressure_head, state)
    infiltration_field = hasproperty(state, :infiltration) ? (; infiltration = state.infiltration) : (;)
    fields = merge(get_fields(state, hydrology, strat, bgc), infiltration_field)
    compute_z_bcs!(state.tendencies.saturation_water_ice, state.saturation_water_ice, grid, state.clock, fields)
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_tendencies!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration} = nothing,
        args...
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    tendencies = tendency_fields(state, hydrology)
    fields = get_fields(state, hydrology, strat, bgc, evtr)
    clock = state.clock
    launch!(
        grid, XYZ, compute_tendencies_kernel!,
        tendencies, clock, fields, hydrology, strat, bgc, constants, evtr
    )
    return nothing
end

# Kernel functions

"""
    $SIGNATURES

Compute the volumetric water content (VWC) tendency at grid cell `i, j k` according to the
Richardson-Richards equation. Note that the VWC tendency is not scaled by the porosity and
is thus not the same as the saturation tendency.
"""
@propagate_inbounds function compute_volumetric_water_content_tendency(
        i, j, k, grid, clock, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        constants::PhysicalConstants,
        evapotranspiration::Optional{AbstractEvapotranspiration}
    ) where {NF}
    # Operators require the underlying Oceananigans grid
    field_grid = get_field_grid(grid)

    # Compute divergence of water fluxes
    # ∂θ∂t = ∇⋅K(θ)∇Ψ + S, where Ψ = ψₘ + ψₕ + ψz, and S is a forcing term for sources and sinks such as ET losses
    ∂θ∂t = (
        - ∂zᵃᵃᶜ(i, j, k, field_grid, darcy_flux, fields.pressure_head, fields.hydraulic_conductivity)  # Darcy flux
            + forcing(i, j, k, grid, clock, fields, evapotranspiration, hydrology, constants)          # ET source/sink
            + forcing(i, j, k, grid, clock, fields, hydrology.vwc_forcing, hydrology)                  # generic user-defined forcing
    )
    return ∂θ∂t
end

"""
    $SIGNATURES

Kernel function for computing the Darcy flux over layer faces from the pressure head `ψ` and hydraulic
conductivity `K`.
"""
@propagate_inbounds function darcy_flux(i, j, k, grid, ψ, K)
    # Darcy's law: q = -K ∂ψ/∂z
    # TODO: also account for effect of temperature on matric potential
    ∇ψ = ∂zᵃᵃᶠ(i, j, k, grid, ψ)
    ## Take minimum of hydraulic conductivities in the direction of flow
    Kₖ = (∇ψ < 0) * min(K[i, j, k - 1], K[i, j, k]) +
        (∇ψ >= 0) * min(K[i, j, k], K[i, j, k + 1])
    ## Note that elevation and hydrostatic pressure are assumed to be accounted
    ## for in the computation of ψ, so we don't need any additional terms.
    q = -Kₖ * ∇ψ
    return q
end

"""
    $SIGNATURES

Compute the hydraulic conductivity at the center of the grid cell `i, j, k`.
"""
@propagate_inbounds function hydraulic_conductivity(
        i, j, k, grid, fields,
        hydrology::SoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    soil = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    return hydraulic_conductivity(hydrology.hydraulic_properties, soil)
end

"""
    $TYPEDSIGNATURES

Computes the unsaturated hydraulic conductivity for `RichardsEq` configurations of `SoilHydrology`.
"""
@propagate_inbounds function compute_hydraulics!(
        out, i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    # Get underlying grid
    fgrid = get_field_grid(grid)
    # compute hydraulic conductivity
    Nz = fgrid.Nz
    k_below = ifelse(k >= Nz, Nz, max(k - 1, one(k)))
    k_above = min(k, Nz)
    K_face = min(
        hydraulic_conductivity(i, j, k_below, grid, fields, hydrology, strat, bgc),
        hydraulic_conductivity(i, j, k_above, grid, fields, hydrology, strat, bgc)
    )
    @inbounds out.hydraulic_conductivity[i, j, k] = K_face
    k_upper = ifelse(k >= Nz, Nz + one(Nz), k)
    @inbounds out.hydraulic_conductivity[i, j, k_upper] = K_face
    return nothing
end

# Kernels

@kernel inbounds = true function compute_tendencies_kernel!(
        tend, grid, clock, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration}
    ) where {NF}
    i, j, k = @index(Global, NTuple)
    compute_saturation_tendency!(tend.saturation_water_ice, i, j, k, grid, clock, fields, hydrology, strat, bgc, constants, evtr)
end
