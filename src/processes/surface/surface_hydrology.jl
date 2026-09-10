"""
    $TYPEDEF

Default representation of coupled surface hydrology processes including
canopy rain/snow interception, evapotranspiration, and surface runoff.

Properties:
$FIELDS
"""
struct SurfaceHydrology{
        NF,
        CanopyInterception <: AbstractCanopyInterception{NF},
        Evapotranspiration <: AbstractEvapotranspiration{NF},
        SurfaceRunoff <: AbstractSurfaceRunoff{NF},
    } <: AbstractSurfaceHydrology{NF}
    "Canopy hydrology scheme"
    canopy_interception::CanopyInterception

    "Canopy evapotranspiration scheme"
    evapotranspiration::Evapotranspiration

    "Surface runoff scheme"
    surface_runoff::SurfaceRunoff
end

function SurfaceHydrology(
        ::Type{NF};
        canopy_interception::CI = PALADYNCanopyInterception(NF),
        evapotranspiration::ET = PALADYNCanopyEvapotranspiration(NF),
        surface_runoff::SR = DirectSurfaceRunoff(NF)
    ) where {NF, CI, ET, SR}
    return SurfaceHydrology{NF, CI, ET, SR}(canopy_interception, evapotranspiration, surface_runoff)
end

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        hydrology::SurfaceHydrology,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::Optional{AbstractSoil} = nothing,
        vegetation::Optional{AbstractVegetation} = nothing,
        snow::Optional{AbstractSnow} = nothing,
        args...
    )
    compute_auxiliary!(state, grid, hydrology.canopy_interception, atmos)
    # `snow` lets the (bare-ground) evaporation scheme scale ground evaporation by the snow-free fraction
    compute_auxiliary!(state, grid, hydrology.evapotranspiration, hydrology.canopy_interception, constants, atmos, soil, vegetation, snow)
    # `snow` makes the surface runoff scheme's water input snow-aware (meltwater + bare-ground throughfall)
    compute_auxiliary!(state, grid, hydrology.surface_runoff, hydrology.canopy_interception, soil, snow)
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_tendencies!(
        state, grid,
        hydrology::SurfaceHydrology,
        args...,
    )
    # Compute tendencies for canopy interception
    compute_tendencies!(state, grid, hydrology.canopy_interception, hydrology.evapotranspiration)
    # Compute tendencies for the surface excess water pool owned by the runoff scheme
    compute_tendencies!(state, grid, hydrology.surface_runoff)
    return nothing
end
