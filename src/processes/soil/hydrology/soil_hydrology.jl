"""
    $TYPEDEF

Base type for implementations of soil water flow dynamics.
"""
abstract type AbstractVerticalFlow end

"""
    $TYPEDEF

Represents a hydrology scheme where soil water is immobile.
"""
struct NoFlow <: AbstractVerticalFlow end

"""
    $TYPEDEF

Properties:
$TYPEDFIELDS
"""
struct SoilHydrology{
        NF,
        VerticalFlow <: AbstractVerticalFlow,
        SaturationClosure <: AbstractSoilWaterClosure,
        SoilHydraulics <: AbstractSoilHydraulics{NF},
        VWCForcing <: Union{Nothing, AbstractForcing},
    } <: AbstractSoilHydrology{NF}
    "Soil water vertical flow operator"
    vertical_flow::VerticalFlow

    "Closure relation for the soil hydrology state"
    closure::SaturationClosure

    "Soil hydraulic properties parameterization"
    hydraulic_properties::SoilHydraulics

    "Forcing for soil moisture (volumetric water content)"
    vwc_forcing::VWCForcing
end

function SoilHydrology(
        ::Type{NF};
        vertical_flow::AbstractVerticalFlow = NoFlow(),
        closure::AbstractSoilWaterClosure = SoilSaturationPressureClosure(),
        hydraulic_properties::AbstractSoilHydraulics = SoilHydraulicsSURFEX(NF),
        vwc_forcing::Union{Nothing, AbstractForcing} = nothing,
    ) where {NF}
    return SoilHydrology(vertical_flow, closure, hydraulic_properties, vwc_forcing)
end

function SoilHydrology(::Type{NF}, vertical_flow::AbstractVerticalFlow; kwargs...) where {NF}
    return SoilHydrology(NF; vertical_flow, kwargs...)
end

"""
    $TYPEDSIGNATURES

Return the soil water retention curve from the `hydraulic_properties` associated with
the given `SoilHydrology` configuration.
"""
@inline get_swrc(hydrology::SoilHydrology) = hydrology.hydraulic_properties.swrc

"""
    $TYPEDSIGNATURES

Return the soil hydraulic properties defined by the given `hydrology` process.
"""
@inline get_hydraulic_properties(hydrology::SoilHydrology) = hydrology.hydraulic_properties

"""
    $TYPEDSIGNATURES

Return the saturation-pressure closure defined by the given `hydrology` process, or `nothing`
if not defined for the given configuration.
"""
@inline get_closure(hydrology::SoilHydrology) = hydrology.closure

variables(hydrology::SoilHydrology{NF}) where {NF} = (
    auxiliary(:saturation_water_ice, XYZ(), bounds = UnitInterval, desc = "Saturation level of water and ice in the pore space"),
    auxiliary(:water_table, XY(), units = u"m", desc = "Elevation of the water table in meters"),
    auxiliary(:hydraulic_conductivity, XYZ(z = Face()), units = u"m/s", desc = "Hydraulic conductivity of soil volumes in m/s"),
    input(:liquid_water_fraction, XYZ(), default = 1, bounds = UnitInterval, desc = "Fraction of unfrozen water in the pore space"),
)

function compute_water_table!(state, grid, hydrology::SoilHydrology)
    launch!(
        grid, XY, compute_water_table_kernel!,
        state.water_table, state.saturation_water_ice, hydrology
    )
    return nothing
end

function adjust_saturation_profile!(state, grid, hydrology::SoilHydrology, runoff::Optional{AbstractSurfaceRunoff} = nothing)
    saturation_water_ice = state.saturation_water_ice
    # When a surface runoff process is present, excess water is routed into the runoff-owned
    # `surface_excess_water` pool; standalone (no runoff) the excess is discarded.
    out = if isnothing(runoff)
        (; saturation_water_ice)
    else
        (; saturation_water_ice, surface_excess_water = state.surface_excess_water)
    end
    launch!(grid, XY, adjust_saturation_profile_kernel!, out, hydrology, runoff)
    return nothing
end

function compute_hydraulics!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    out = (hydraulic_conductivity = state.hydraulic_conductivity,)
    fields = get_fields(state, hydrology, strat, bgc; except = out)
    launch!(grid, XYZ, compute_hydraulics_kernel!, out, fields, hydrology, strat, bgc)
    return nothing
end

# Immobile soil water (NoFlow)

""" $TYPEDSIGNATURES """
function initialize!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)
    compute_hydraulics!(state, grid, hydrology, soil)
    compute_water_table!(state, grid, hydrology)
    return nothing
end

""" $TYPEDSIGNATURES """
@inline function compute_auxiliary!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)
    compute_hydraulics!(state, grid, hydrology, soil, args...)
    return nothing
end

""" $TYPEDSIGNATURES """
@inline compute_tendencies!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...) = nothing

# Kernel functions

@propagate_inbounds saturation_water_ice(i, j, k, grid, fields, ::SoilHydrology) = fields.saturation_water_ice[i, j, k]

@propagate_inbounds hydraulic_conductivity(i, j, k, grid, fields, ::SoilHydrology) = fields.hydraulic_conductivity[i, j, k]

@propagate_inbounds liquid_water_fraction(i, j, k, grid, fields, ::SoilHydrology) = fields.liquid_water_fraction[i, j, k]

@propagate_inbounds water_table(i, j, grid, fields, ::SoilHydrology) = fields.water_table[i, j]

@propagate_inbounds surface_excess_water(i, j, grid, fields, ::SoilHydrology{NF}) where {NF} = zero(NF)

"""
    $TYPEDSIGNATURES

Discrete-form boundary-condition function (see [`InfiltrationFlux`](@ref)) computing `-infiltration
/ por` at column `i, j`, where `infiltration` is the physical infiltration flux (m/s of water depth,
positive downward) and `por` is the porosity of the topmost soil layer ([`porosity_top`](@ref)).
`saturation_water_ice` is the dimensionless saturation (VWC / porosity) so the flux crossing the top
boundary must be normalized by porosity to be dimensionally consistent with the interior Richards tendency,
which divides by porosity for the same reason (see `compute_saturation_tendency!`). Note that the hydrology
,odule computes infiltration as positive downward, so it is negated here since fluxes are by convention positive upward.
"""
@propagate_inbounds function saturation_infiltration_bc(i, j, grid, clock, fields, parameters)
    por = porosity_top(i, j, grid, fields, parameters.strat, parameters.bgc)
    return -fields.infiltration[i, j] / por
end

"""
    $TYPEDSIGNATURES

Kernel function that computes dynamic soil hydraulic properties.
"""
compute_hydraulics!(out, i, j, k, grid, fields, hydrology::SoilHydrology, args...) = nothing

"""
    $TYPEDSIGNATURES

Kernel function that diagnoses the water table at grid cell `i, j` given the current soil saturation profile.
"""
@propagate_inbounds function compute_water_table!(water_table, i, j, grid, sat, ::SoilHydrology{NF}) where {NF}
    zs = znodes(get_field_grid(grid), Center(), Center(), Face())
    # scan z axis starting from the bottom (index 1) to find first non-saturated grid cell
    water_table[i, j, 1] = findfirst_z(i, j, <(one(NF)), zs, sat)
    return nothing
end

"""
    $TYPEDSIGNATURES

Kernel function that adjusts saturation profiles to account for oversaturation and undersaturation
arising due to numerical error. This implementation scans over the saturation profiles at each lateral
grid cell and redistributes excess water upward layer-by-layer until reaching the topmost layer. The
remaining surface excess (as a water depth in m³/m²) is returned so that the caller can either route it
into a surface water pool or discard it.
"""
@propagate_inbounds function redistribute_saturation_profile!(sat, i, j, grid, hydrology::SoilHydrology{NF}) where {NF}
    props = get_hydraulic_properties(hydrology)
    sat_min = residual_saturation(props)
    field_grid = get_field_grid(grid)
    N = field_grid.Nz

    # First iterate over soil layers from bottom to top, transferring water from
    # overfilled layers to the layer above
    for k in 1:(N - 1)
        # calculate excess saturation
        excess_sat = max(sat[i, j, k] - one(NF), zero(NF))
        # subtract excess water and add to layer above;
        # note that we need to rescale by the cell thickness to properly conserve mass
        sat[i, j, k] -= excess_sat
        sat[i, j, k + 1] += excess_sat * Δzᵃᵃᶜ(i, j, k, field_grid) / Δzᵃᵃᶜ(i, j, k + 1, field_grid)
    end

    # then from top to bottom, extracting water for underfilled cells from layers below
    for k in N:-1:2
        # calculate saturation deficit from residual saturation level
        deficit_sat = max(-sat[i, j, k] + sat_min, zero(NF))
        # add back saturation deficit and subtract from layer below
        sat[i, j, k] += deficit_sat
        sat[i, j, k - 1] -= deficit_sat * Δzᵃᵃᶜ(i, j, k, field_grid) / Δzᵃᵃᶜ(i, j, k - 1, field_grid)
    end

    # If the uppermost (surface) layer is oversaturated, remove the excess and return it as a water depth
    excess_sat = max(sat[i, j, N] - one(NF), zero(NF))
    sat[i, j, N] -= excess_sat
    surface_excess = excess_sat * Δzᵃᵃᶜ(i, j, N, field_grid)

    # If the lowermost (bottom) layer has a deficit, just set to the residual saturation level.
    # This constitutes a mass balance violation but should not happen under realistic conditions.
    sat[i, j, 1] = max(sat[i, j, 1], sat_min)

    return surface_excess
end

"""
    $TYPEDSIGNATURES

Kernel function that adjusts the saturation profile at `i, j` and discards any excess water reaching the
surface. Used for standalone soil hydrology, where no surface runoff process owns a water pool.
"""
@propagate_inbounds function adjust_saturation_profile!(out, i, j, grid, hydrology::SoilHydrology{NF}, ::Nothing) where {NF}
    redistribute_saturation_profile!(out.saturation_water_ice, i, j, grid, hydrology)
    return nothing
end

"""
    $TYPEDSIGNATURES

Kernel function that adjusts the saturation profile at `i, j` and routes any excess water reaching the
surface into the `surface_excess_water` pool owned by the given surface `runoff` process.
"""
@propagate_inbounds function adjust_saturation_profile!(
        out, i, j, grid, hydrology::SoilHydrology{NF},
        runoff::AbstractSurfaceRunoff
    ) where {NF}
    surface_excess = redistribute_saturation_profile!(out.saturation_water_ice, i, j, grid, hydrology)
    out.surface_excess_water[i, j, 1] += surface_excess
    return nothing
end

""" $TYPEDSIGNATURES """
@propagate_inbounds function compute_saturation_tendency!(
        saturation_water_ice_tendency, i, j, k, grid, clock, fields,
        hydrology::SoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration}
    )
    # Compute volumetic water content tendency
    ∂θ∂t = compute_volumetric_water_content_tendency(i, j, k, grid, clock, fields, hydrology, constants, evtr)
    # Get porosity
    por = porosity(i, j, k, grid, fields, strat, bgc)
    # Rescale by porosity to get saturation tendency
    saturation_water_ice_tendency[i, j, k] += ∂θ∂t / por
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the volumetric water content (VWC) tendency at grid cell `i, j k` f. Note that the VWC tendency is not scaled by
the porosity and is thus not the same as the saturation tendency.
"""
@propagate_inbounds function compute_volumetric_water_content_tendency(
        i, j, k, grid, clock, fields,
        hydrology::SoilHydrology{NF},
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration}
    ) where {NF}
    # Compute divergence of water fluxes due to forcings only
    ET_loss = forcing(i, j, k, grid, clock, fields, evtr, hydrology, constants) # ET forcing
    F = forcing(i, j, k, grid, clock, fields, hydrology.vwc_forcing, hydrology) # generic user-defined forcing
    ∂θ∂t = ET_loss + F
    return ∂θ∂t
end

# Kernels

@kernel inbounds = true function compute_water_table_kernel!(water_table, grid, sat, hydrology::SoilHydrology{NF}) where {NF}
    i, j = @index(Global, NTuple)
    compute_water_table!(water_table, i, j, grid, sat, hydrology)
end

@kernel inbounds = true function adjust_saturation_profile_kernel!(
        out, grid, hydrology::SoilHydrology{NF}, runoff::Optional{AbstractSurfaceRunoff}
    ) where {NF}
    i, j = @index(Global, NTuple)
    adjust_saturation_profile!(out, i, j, grid, hydrology, runoff)
end

@kernel inbounds = true function compute_hydraulics_kernel!(out, grid, fields, hydrology::SoilHydrology, args...)
    i, j, k = @index(Global, NTuple)
    compute_hydraulics!(out, i, j, k, grid, fields, hydrology, args...)
end

# Default debug hooks
@inline debughook!(::typeof(compute_water_table_kernel!), out, args...) = checkfinite!(out)
@inline debughook!(::typeof(adjust_saturation_profile_kernel!), out, args...) = checkfinite!(out)
@inline debughook!(::typeof(compute_hydraulics_kernel!), out, args...) = checkfinite!(out)
