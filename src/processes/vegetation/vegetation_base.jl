# Generic methods for all AbstractVegetation implementations

# TODO: Remove once dedicated vegetation surface parameterizations are added
@propagate_inbounds compute_albedo(i, j, grid, fields, veg::AbstractVegetation) = plant_traits(i, j, grid, fields, veg.traits).albedo

@propagate_inbounds compute_emissivity(i, j, grid, fields, veg::AbstractVegetation) = plant_traits(i, j, grid, fields, veg.traits).emissivity

# TODO: will need to change once PFTs are added
@inline vegetation_area_fraction(i, j, grid, fields, ::Union{Nothing, AbstractVegetation}) = zero(eltype(grid))

"""
    $TYPEDSIGNATURES

Return the leaf area index at grid point `i, j` relative to the maximum leaf area index of the given
plant `traits`, `LAI / LAI_max`, clamped to the unit interval.
"""
@propagate_inbounds function compute_relative_leaf_area(i, j, grid, fields, traits::AbstractPlantTraits{NF}) where {NF}
    local_traits = plant_traits(i, j, grid, fields, traits)
    LAI = fields.leaf_area_index[i, j, end]
    LAI_max = local_traits.maximum_leaf_area_index
    return clamp(LAI / LAI_max, zero(NF), one(NF))
end
