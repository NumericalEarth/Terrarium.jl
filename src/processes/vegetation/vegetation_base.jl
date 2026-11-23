# Generic methods for all AbstractVegetation implementations

# TODO: Remove once dedicated vegetation surface parameterizations are added
@propagate_inbounds compute_albedo(i, j, grid, fields, veg::AbstractVegetation) = plant_traits(i, j, grid, fields, veg.traits).albedo

@propagate_inbounds compute_emissivity(i, j, grid, fields, veg::AbstractVegetation) = plant_traits(i, j, grid, fields, veg.traits).emissivity

# TODO: will need to change once PFTs are added
@inline vegetation_area_fraction(i, j, grid, fields, ::Union{Nothing, AbstractVegetation}) = zero(eltype(grid))
