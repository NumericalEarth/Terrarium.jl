"""
    $TYPEDEF

Simple surface runoff scheme that computes runoff as

```math
R = P + D - I
```
where `P` is precipitation reaching the ground, `D` is drainage from accumualted excess
water at the surface, and `I` is infiltration into the soil.

Properties:
$FIELDS
"""
@parameterized @kwdef struct DirectSurfaceRunoff{NF} <: AbstractSurfaceRunoff{NF}
    "Surface water removal timescale"
    @param τ_r::NF = 3600.0 (units = u"s", bounds = Positive)
end

DirectSurfaceRunoff(::Type{NF}; kwargs...) where {NF} = DirectSurfaceRunoff{NF}(; kwargs...)

"""
    $TYPEDSIGNATURES

Compute surface drainage flux from the current `surface_excess_water` resevoir state.
"""
@inline function compute_surface_drainage(runoff::DirectSurfaceRunoff{NF}, surface_excess_water) where {NF}
    let S = max(surface_excess_water, zero(NF))
        τ = runoff.τ_r
        D = S / τ
        return D
    end
end

"""
    $TYPEDSIGNATURES

Compute infiltration from the given `influx` (water available for infiltration), saturation of the uppermost
soil layer `sat_top`, and the maximum allowed infiltration `max_infil`.
"""
@inline function compute_infiltration(runoff::DirectSurfaceRunoff{NF}, influx, sat_top, max_infil) where {NF}
    let is_unsaturated = sat_top < one(NF)
        # Infiltration is min of
        infil = min(influx, max_infil) * is_unsaturated
        return infil
    end
end

"""
    $TYPEDSIGNATURES

Compute surface runoff as `precipitation + surface_drainage - infiltration`.
"""
@inline function compute_surface_runoff(runoff::DirectSurfaceRunoff, influx, surface_drainage, infil)
    let F = influx
        D = surface_drainage
        I = infil
        # Compute runoff as residual of precipitation + drainage - infiltration
        surface_runoff = F + D - I
        return surface_runoff
    end
end

# Top-level interface methods

variables(::DirectSurfaceRunoff) = (
    prognostic(:surface_excess_water, XY(), units = u"m", desc = "Excess water at the soil surface in m³/m²"),
    auxiliary(:surface_runoff, XY(), units = u"m/s", desc = "Total surface runoff"),
    auxiliary(:infiltration, XY(), units = u"m/s", desc = "Infiltration flux"),
)

@propagate_inbounds surface_excess_water(i, j, grid, fields, ::AbstractSurfaceRunoff) = fields.surface_excess_water[i, j]

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        runoff::DirectSurfaceRunoff,
        canopy_interception::AbstractCanopyInterception,
        soil::AbstractSoil,
        snow::Optional{AbstractSnow} = nothing,
        args...
    )
    soil_hydrology = get_hydrology(soil)
    out = auxiliary_fields(state, runoff)
    # merge the snow fields (cover fraction, liquid fraction) needed for the snow-aware surface water input
    fields = merge(get_fields(state, runoff, canopy_interception, soil_hydrology; except = out), get_fields(state, snow))
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, runoff, canopy_interception, soil_hydrology, snow)
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_tendencies!(
        state, grid,
        runoff::DirectSurfaceRunoff,
        args...
    )
    tendencies = tendency_fields(state, runoff)
    fields = get_fields(state, runoff)
    launch!(grid, XY, compute_tendencies_kernel!, tendencies, fields, runoff)
    return nothing
end

# Kernel function

"""
    $TYPEDSIGNATURES

Kernel function for computing the tendency of the prognostic `surface_excess_water` pool in the grid
column `i, j`. The pool is drained at the surface drainage rate `D = compute_surface_drainage`, capped
so that a single step cannot remove more water than is present (`min(D, S)`). The tendency is returned
as a *negative* rate (a removal) consistent with the positive-upward flux convention used by the other
surface hydrology tendencies, so that `surface_excess_water += ∂S∂t * Δt` draws the pool down.
"""
@propagate_inbounds function compute_surface_excess_water_tendency!(
        tendencies, i, j, grid, fields,
        runoff::DirectSurfaceRunoff{NF}
    ) where {NF}
    S = surface_excess_water(i, j, grid, fields, runoff)
    D = compute_surface_drainage(runoff, S)
    tendencies.surface_excess_water[i, j, 1] = -min(D, S)
    return tendencies
end

@propagate_inbounds function compute_surface_runoff!(
        out, i, j, grid, fields,
        runoff::DirectSurfaceRunoff{NF},
        canopy_interception::AbstractCanopyInterception,
        soil_hydrology::AbstractSoilHydrology,
        snow::Optional{AbstractSnow} = nothing
    ) where {NF}
    fgrid = get_field_grid(grid)

    # Get inputs. With snow, the surface water input is the snow-adjusted rainfall plus meltwater outflow
    # (the snow-covered fraction of rain is intercepted by the snowpack); without snow it is the ground rainfall.
    influx = soil_surface_water_flux(i, j, grid, fields, canopy_interception, snow)
    excess_water = surface_excess_water(i, j, grid, fields, runoff)
    k_unsat = hydraulic_conductivity(i, j, fgrid.Nz, grid, fields, soil_hydrology)
    sat_top = saturation_water_ice(i, j, fgrid.Nz, grid, fields, soil_hydrology)

    if excess_water > zero(NF)
        # Case 1: Excess water present at the surface -> precipitation adds to excess water
        # and we set the infiltration rate to the min of hydraulic conductivity and surface_excess_water
        # First, compute rate of excess water removal (surface drainage)
        surface_drainage = compute_surface_drainage(runoff, excess_water)
        # Calculate infiltration
        infil = out.infiltration[i, j, 1] = compute_infiltration(runoff, surface_drainage, sat_top, k_unsat)
    else
        # Case 2: No excess water -> rainfall is routed directly to infiltration
        surface_drainage = zero(NF)
        infil = out.infiltration[i, j, 1] = compute_infiltration(runoff, influx, sat_top, k_unsat)
    end

    # Compute surface runoff
    out.surface_runoff[i, j, 1] = compute_surface_runoff(runoff, influx, surface_drainage, infil)
    return out
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, runoff::AbstractSurfaceRunoff, args...)
    i, j = @index(Global, NTuple)
    compute_surface_runoff!(out, i, j, grid, fields, runoff, args...)
end

@kernel inbounds = true function compute_tendencies_kernel!(tendencies, grid, fields, runoff::AbstractSurfaceRunoff, args...)
    i, j = @index(Global, NTuple)
    compute_surface_excess_water_tendency!(tendencies, i, j, grid, fields, runoff, args...)
end
