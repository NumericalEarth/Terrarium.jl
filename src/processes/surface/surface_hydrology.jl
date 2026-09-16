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
    interception = hydrology.canopy_interception
    evapotranspiration = hydrology.evapotranspiration
    surface_runoff = hydrology.surface_runoff
    # The fused kernel writes the union of the three sub-processes' auxiliaries. Drop lazy
    # (`FunctionField`) auxiliaries — e.g. `NoCanopyInterception`'s `rainfall_ground` passthrough —
    # which no kernel writes.
    out = filter(v -> v isa Field, auxiliary_fields(state, interception, evapotranspiration, surface_runoff))
    # Full fields (no `except`): within a cell the kernel writes `out.foo` and a later sub-process
    # reads `fields.foo` — the same `Field` object, so the write is visible. This is what lets the
    # canopy → evapotranspiration → runoff dependency chain run in a single launch.
    fields = get_fields(state, interception, evapotranspiration, surface_runoff, atmos, soil, vegetation, snow)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, hydrology, constants, atmos, soil, vegetation, snow)
    return nothing
end

"""
    $TYPEDSIGNATURES

Fused auxiliary kernel for the coupled surface hydrology processes. Each sub-process's per-cell
mutating variant runs in dependency order — canopy interception, then evapotranspiration (which reads
the canopy saturation fraction), then surface runoff (which reads the ground rainfall) — so the chain
resolves within a single launch. A "prescribed"/no-op scheme contributes a no-op variant.
"""
@kernel inbounds = true function compute_auxiliary_kernel!(
        out, grid, fields,
        hydrology::SurfaceHydrology,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::Optional{AbstractSoil} = nothing,
        vegetation::Optional{AbstractVegetation} = nothing,
        snow::Optional{AbstractSnow} = nothing,
        args...
    )
    i, j = @index(Global, NTuple)
    compute_canopy_auxiliary!(out, i, j, grid, fields, hydrology.canopy_interception, atmos)
    # `snow` lets the (bare-ground) evaporation scheme scale ground evaporation by the snow-free fraction
    compute_evapotranspiration_auxiliary!(
        out, i, j, grid, fields, hydrology.evapotranspiration,
        hydrology.canopy_interception, constants, atmos, soil, vegetation, snow
    )
    # `snow` makes the surface runoff scheme's water input snow-aware (meltwater + bare-ground throughfall)
    compute_surface_runoff!(
        out, i, j, grid, fields, hydrology.surface_runoff,
        hydrology.canopy_interception, get_hydrology(soil), snow
    )
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
