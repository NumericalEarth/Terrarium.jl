"""
    RichardsEq{PS} <: AbstractVerticalFlow

[`SoilHydrology`](@ref) flow operator implementing the mixed saturation-pressure form
of the Richardson-Richards equation.

State variables defined by the Richards' formulation of `SoilHydrology`:

- `saturation_water_ice`: saturation level of water and ice in the pore space.
- `surface_excess_water`: excess water at the soil surface (m^3/m^2).
- `hydraulic_conductivity`: hydraulic conductivity at cell centers (m/s).
- `water_table``: elevation of the water table (m).
- `liquid_water_fraction`: fraction of unfrozen liquid water in the pore space (dimensionless).

See also [`SoilSaturationPressureClosure`](@ref) and [`AbstractSoilHydraulics`](@ref) for details regarding the
closure relating saturation and pressure head.
"""
@kwdef struct RichardsEq <: AbstractVerticalFlow end

variables(hydrology::SoilHydrology{NF, RichardsEq}) where {NF} = (
    prognostic(:saturation_water_ice, XYZ(); closure = get_closure(hydrology), domain = UnitInterval(), desc = "Saturation level of water and ice in the pore space"),
    prognostic(:surface_excess_water, XY(), units = u"m", desc = "Excess water at the soil surface in m³/m²"),
    auxiliary(:hydraulic_conductivity, XYZ(z = Face()), units = u"m/s", desc = "Hydraulic conductivity of soil volumes in m/s"),
    auxiliary(:water_table, XY(), units = u"m", desc = "Elevation of the water table in meters"),
    input(:liquid_water_fraction, XYZ(), default = NF(1), domain = UnitInterval(), desc = "Fraction of unfrozen water in the pore space"),
)

@propagate_inbounds surface_excess_water(i, j, grid, fields, ::SoilHydrology{NF, RichardsEq}) where {NF} = fields.surface_excess_water[i, j]

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
    fields = get_fields(state, hydrology, bgc; except = out)
    launch!(grid, XYZ, compute_hydraulics_kernel!, out, fields, hydrology, strat, bgc)
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_tendencies!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration} = nothing,
        runoff::Optional{AbstractSurfaceRunoff} = nothing,
        args...
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    tendencies = tendency_fields(state, hydrology)
    fields = get_fields(state, hydrology, bgc, evtr)
    clock = state.clock
    launch!(
        grid, XYZ, compute_tendencies_kernel!,
        tendencies, clock, fields, hydrology, strat, bgc, constants, evtr, runoff
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
    # ∂θ∂t = ∇⋅K(θ)∇Ψ + forcing, where Ψ = ψₘ + ψₕ + ψz, and "forcing" represents sources and sinks such as ET losses
    ∂θ∂t = (
        - ∂zᵃᵃᶜ(i, j, k, field_grid, darcy_flux, fields.pressure_head, fields.hydraulic_conductivity)
            + forcing(i, j, k, grid, clock, fields, evapotranspiration, hydrology, constants) # ET forcing
            + forcing(i, j, k, grid, clock, fields, hydrology.vwc_forcing, hydrology) # generic user-defined forcing
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
    soil = soil_volume(i, j, k, grid, fields, strat, hydrology, bgc)
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
    @inbounds if k <= 1
        out.hydraulic_conductivity[i, j, k] = hydraulic_conductivity(i, j, 1, fgrid, fields, hydrology, strat, bgc)
    elseif k >= fgrid.Nz
        out.hydraulic_conductivity[i, j, k] = hydraulic_conductivity(i, j, fgrid.Nz, fgrid, fields, hydrology, strat, bgc)
        out.hydraulic_conductivity[i, j, k + 1] = out.hydraulic_conductivity[i, j, k]
    else
        out.hydraulic_conductivity[i, j, k] = min_zᵃᵃᶠ(i, j, k, fgrid, hydraulic_conductivity, fields, hydrology, strat, bgc)
    end
    return nothing
end

# Kernels

@kernel inbounds = true function compute_tendencies_kernel!(
        tend, grid, clock, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration},
        runoff::Optional{AbstractSurfaceRunoff}
    ) where {NF}
    i, j, k = @index(Global, NTuple)
    compute_saturation_tendency!(tend.saturation_water_ice, i, j, k, grid, clock, fields, hydrology, strat, bgc, constants, evtr)
    compute_surface_excess_water_tendency!(tend.surface_excess_water, i, j, k, grid, clock, fields, hydrology, runoff)
end
