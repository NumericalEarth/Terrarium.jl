"""
    $TYPEDEF

Base type for representing the solid material composition of the soil.
"""
abstract type AbstractSoilMatrix{NF} end

"""
    organic_fraction(::AbstractSoilMatrix)

Return the fraction of the soil matrix that is organic material.
"""
function organic_fraction end

"""
    organic_fraction(::AbstractSoilMatrix)

Return the `SoilTexture` describing the mineral component of the soil matrix.
"""
function mineral_texture end

"""
    $TYPEDEF

Base type for parameterizations of soil porosity.
"""
abstract type AbstractSoilPorosity{NF} end

"""
    mineral_porosity(::AbstractSoilPorosity, texture::SoilTexture)

Compute or retrieve the natural porosity of the mineral soil constitutents, i.e.
excluding organic material.
"""
function mineral_porosity end

"""
    organic_porosity(::AbstractSoilPorosity, texture::SoilTexture)

Compute or retrieve the natural porosity of the organic soil constitutents, i.e.
excluding mineral material.
"""
function organic_porosity end

"""
    $TYPEDEF

Base type for soil stratigraphy parameterizations.
"""
abstract type AbstractStratigraphy{NF} end

# Kernel functions

"""
    soil_texture(i, j, k, grid, fields, ::AbstractStratigraphy, args...)

Return the texture of the soil at index `i, j, k` for the given stratigraphy parameterization.
"""
function soil_texture end

"""
    soil_matrix(i, j, k, grid, fields, ::AbstractStratigraphy, args...)

Return the solid matrix of the soil at index `i, j, k` for the given stratigraphy parameterization.
"""
function soil_matrix end

"""
    soil_volume(i, j, k, grid, fields, ::AbstractStratigraphy, args...)

Return a [`SoilVolume`](@ref) describing the full material composition of the soil volume at index
`i, j, k` for the given stratigraphy parameterization.
"""
function soil_volume end
