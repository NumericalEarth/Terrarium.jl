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
    auxiliary(:saturation_water_ice, Ground(XYZ()), bounds = UnitInterval, desc = "Saturation level of water and ice in the pore space"),
    auxiliary(:water_table, Ground(XY()), units = u"m", desc = "Elevation of the water table in meters"),
    auxiliary(:hydraulic_conductivity, Ground(XYZ(z = Face())), units = u"m/s", desc = "Hydraulic conductivity of soil volumes in m/s"),
    input(:liquid_water_fraction, Ground(XYZ()), default = 1, bounds = UnitInterval, desc = "Fraction of unfrozen water in the pore space"),
)

"""
    default_timestepping(hydrology::AbstractSoilHydrology)

Default [`Timestepping`](@ref) class of `saturation_water_ice` under an [`AbstractIMEX`](@ref)
timestepper: [`Implicit`](@ref) for the Richards equation (`SoilHydrology{NF, RichardsEq}`) and
[`Explicit`](@ref) otherwise.

```jldoctest
julia> Terrarium.default_timestepping(SoilHydrology(Float64, RichardsEq()))
Implicit()

julia> Terrarium.default_timestepping(SoilHydrology(Float64, NoFlow()))
Explicit()
```
"""
default_timestepping(::AbstractSoilHydrology) = Explicit()

function compute_water_table!(state, grid, hydrology::SoilHydrology)
    launch!(
        grid, XY, compute_water_table_kernel!,
        state.water_table, state.saturation_water_ice, hydrology
    )
    return nothing
end

"""
    adjust_saturation_profile!(state, grid, hydrology::SoilHydrology, runoff = nothing, strat = nothing, bgc = nothing, energy = nothing)

Redistribute over- and undersaturation in the soil columns of `state` (see
[`redistribute_saturation_profile!`](@ref)). With `runoff = nothing`, excess water reaching the surface
is discarded; otherwise it is routed into the `surface_excess_water` pool of `runoff` as a water depth
(m), using the porosity of the topmost layer from `strat` and `bgc`.

With an optional [`SoilThermodynamics`](@ref) `energy` process, the sensible heat of the water moves
with it ([`compute_redistributed_heat!`](@ref)), so the heat of the excess water leaves the column; this
requires `strat` and `bgc`.
"""
function adjust_saturation_profile!(
        state, grid,
        hydrology::SoilHydrology,
        runoff::Optional{AbstractSurfaceRunoff} = nothing,
        strat::Optional{AbstractStratigraphy} = nothing,
        bgc::Optional{AbstractSoilBiogeochemistry} = nothing,
        energy::Optional{AbstractSoilThermodynamics} = nothing
    )
    out = adjust_saturation_profile_fields(state, runoff, energy)
    fields = isnothing(strat) ? (;) : get_fields(state, strat, bgc)
    # the heat of the redistributed water is that of the cell it leaves
    fields = isnothing(energy) ? fields : merge(fields, (; temperature = state.temperature))
    launch!(grid, XY, adjust_saturation_profile_kernel!, out, fields, hydrology, runoff, energy, strat, bgc)
    return nothing
end

# Fields modified by `adjust_saturation_profile!`: saturation, the excess water pool of `runoff`, and the internal energy from `SoilThermodynamics`.
adjust_saturation_profile_fields(state, ::Nothing, ::Nothing) = (; state.saturation_water_ice)
adjust_saturation_profile_fields(state, ::AbstractSurfaceRunoff, ::Nothing) = (; state.saturation_water_ice, state.surface_excess_water)
adjust_saturation_profile_fields(state, ::Nothing, ::AbstractSoilThermodynamics) = (; state.saturation_water_ice, state.internal_energy)
adjust_saturation_profile_fields(state, ::AbstractSurfaceRunoff, ::AbstractSoilThermodynamics) =
    (; state.saturation_water_ice, state.surface_excess_water, state.internal_energy)

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
    zs = znodes(ground_domain(grid), Center(), Center(), Face())
    # scan z axis starting from the bottom (index 1) to find first non-saturated grid cell
    water_table[i, j, end] = findfirst_z(i, j, <(one(NF)), zs, sat)
    return nothing
end

"""
    $TYPEDSIGNATURES

Kernel function that adjusts saturation profiles to account for oversaturation and undersaturation
arising due to numerical error. This implementation scans over the saturation profiles at each lateral
grid cell and redistributes excess water upward layer-by-layer until reaching the topmost layer. The
remaining surface excess is returned as the excess saturation of the topmost layer times its thickness
(m); the caller multiplies by the topmost porosity to get a water depth (m³/m²) to route into a surface
water pool, or discards it. With an `energy` process, every saturation change also changes the internal
energy in `out` by the sensible heat of the water ([`compute_redistributed_heat!`](@ref)).
"""
@propagate_inbounds function redistribute_saturation_profile!(
        out, i, j, grid, fields,
        hydrology::SoilHydrology{NF},
        energy = nothing,
        strat = nothing,
        bgc = nothing
    ) where {NF}
    sat = out.saturation_water_ice
    props = get_hydraulic_properties(hydrology)
    sat_min = residual_saturation(props)
    ground_grid = ground_domain(grid)
    N = ground_grid.Nz

    # First iterate over soil layers from bottom to top, transferring water from
    # overfilled layers to the layer above
    for k in 1:(N - 1)
        # calculate excess saturation
        excess_sat = max(sat[i, j, k] - one(NF), zero(NF))
        # subtract excess water and add to layer above;
        # note that we need to rescale by the cell thickness to properly conserve mass
        Δs_above = excess_sat * Δzᵃᵃᶜ(i, j, k, ground_grid) / Δzᵃᵃᶜ(i, j, k + 1, ground_grid)
        sat[i, j, k] -= excess_sat
        sat[i, j, k + 1] += Δs_above
        # the water carries the temperature of cell k
        compute_redistributed_heat!(out, i, j, k, grid, fields, energy, strat, bgc, k, -excess_sat)
        compute_redistributed_heat!(out, i, j, k + 1, grid, fields, energy, strat, bgc, k, Δs_above)
    end

    # then from top to bottom, extracting water for underfilled cells from layers below
    for k in N:-1:2
        # calculate saturation deficit from residual saturation level
        deficit_sat = max(-sat[i, j, k] + sat_min, zero(NF))
        Δs_below = deficit_sat * Δzᵃᵃᶜ(i, j, k, ground_grid) / Δzᵃᵃᶜ(i, j, k - 1, ground_grid)
        # add back saturation deficit and subtract from layer below
        sat[i, j, k] += deficit_sat
        sat[i, j, k - 1] -= Δs_below
        # the water carries the temperature of cell k - 1
        compute_redistributed_heat!(out, i, j, k, grid, fields, energy, strat, bgc, k - 1, deficit_sat)
        compute_redistributed_heat!(out, i, j, k - 1, grid, fields, energy, strat, bgc, k - 1, -Δs_below)
    end

    # If the uppermost (surface) layer is oversaturated, remove the excess and return it as a water depth
    excess_sat = max(sat[i, j, N] - one(NF), zero(NF))
    sat[i, j, N] -= excess_sat
    compute_redistributed_heat!(out, i, j, N, grid, fields, energy, strat, bgc, N, -excess_sat)
    surface_excess = excess_sat * Δzᵃᵃᶜ(i, j, N, ground_grid)

    # If the lowermost (bottom) layer has a deficit, just set to the residual saturation level.
    # This constitutes a mass balance violation but should not happen under realistic conditions.
    deficit_bottom = max(sat_min - sat[i, j, 1], zero(NF))
    sat[i, j, 1] += deficit_bottom
    compute_redistributed_heat!(out, i, j, 1, grid, fields, energy, strat, bgc, 1, deficit_bottom)

    return surface_excess
end

# Heat moved with redistributed water; the `SoilThermodynamics` method is in `soil_energy.jl`.
@inline compute_redistributed_heat!(out, i, j, k, grid, fields, energy, args...) = nothing

"""
    $TYPEDSIGNATURES

Kernel function that adjusts the saturation profile at `i, j` and discards any excess water reaching the
surface. Used for standalone soil hydrology, where no surface runoff process owns a water pool.
"""
@propagate_inbounds function adjust_saturation_profile!(out, i, j, grid, fields, hydrology::SoilHydrology{NF}, ::Nothing, energy, strat, bgc) where {NF}
    redistribute_saturation_profile!(out, i, j, grid, fields, hydrology, energy, strat, bgc)
    return nothing
end

"""
    $TYPEDSIGNATURES

Kernel function that adjusts the saturation profile at `i, j` and routes any excess water reaching the
surface into the `surface_excess_water` pool owned by the given surface `runoff` process, as a water
depth (m): the excess saturation times the thickness and porosity ([`porosity_top`](@ref)) of the
topmost layer.
"""
@propagate_inbounds function adjust_saturation_profile!(
        out, i, j, grid, fields,
        hydrology::SoilHydrology{NF},
        runoff::AbstractSurfaceRunoff,
        energy,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    surface_excess = redistribute_saturation_profile!(out, i, j, grid, fields, hydrology, energy, strat, bgc)
    por = porosity_top(i, j, grid, fields, strat, bgc)
    out.surface_excess_water[i, j, end] += por * surface_excess
    return nothing
end

""" $TYPEDSIGNATURES """
@propagate_inbounds function compute_saturation_tendency!(
        saturation_water_ice_tendency, i, j, k, grid, clock, fields,
        hydrology::SoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration},
        timestepping::Timestepping = Explicit()
    )
    # Compute volumetic water content tendency
    ∂θ∂t = compute_volumetric_water_content_tendency(i, j, k, grid, clock, fields, hydrology, constants, evtr, timestepping)
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
        evtr::Optional{AbstractEvapotranspiration},
        ::Timestepping = Explicit()
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
        out, grid, fields, hydrology::SoilHydrology{NF}, runoff::Optional{AbstractSurfaceRunoff}, energy, strat, bgc
    ) where {NF}
    i, j = @index(Global, NTuple)
    adjust_saturation_profile!(out, i, j, grid, fields, hydrology, runoff, energy, strat, bgc)
end

@kernel inbounds = true function compute_hydraulics_kernel!(out, grid, fields, hydrology::SoilHydrology, args...)
    i, j, k = @index(Global, NTuple)
    compute_hydraulics!(out, i, j, k, grid, fields, hydrology, args...)
end

# Default debug hooks
@inline debughook!(::typeof(compute_water_table_kernel!), out, args...) = checkfinite!(out)
@inline debughook!(::typeof(adjust_saturation_profile_kernel!), out, args...) = checkfinite!(out)
@inline debughook!(::typeof(compute_hydraulics_kernel!), out, args...) = checkfinite!(out)
