"""
Base type for surface runoff processes.
"""
abstract type AbstractSurfaceRunoff{NF} <: AbstractProcess{NF} end

"""
    compute_surface_runoff!(
        out, i, j, grid, fields,
        runoff::DirectSurfaceRunoff{NF},
        canopy_interception::AbstractCanopyInterception,
        soil_hydrology::AbstractSoilHydrology
    ) where {NF}

Compute surface runoff in grid cell `i, j` and store the result in `out.surface_runoff`. Runoff is computed
based on the ground-reaching precipitation rate provided by `canopy_interception` and the soil saturation
state provided by `soil_hydrology`.
"""
function compute_surface_runoff! end

# Without a surface runoff scheme, there is no surface water pool
@propagate_inbounds surface_excess_water(i, j, grid, fields, ::Nothing) = zero(eltype(grid))
