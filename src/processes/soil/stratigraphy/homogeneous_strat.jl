"""
    $TYPEDEF

Represents a soil stratigraphy of well mixed material with homogeneous soil texture.

Properties:
$TYPEDFIELDS
"""
struct HomogeneousStratigraphy{NF, SoilPorosity <: AbstractSoilPorosity{NF}} <: AbstractStratigraphy{NF}
    "Material composition of mineral soil component"
    texture::SoilTexture{NF}

    "Parameterization of soil porosity"
    porosity::SoilPorosity
end

function HomogeneousStratigraphy(
        ::Type{NF};
        texture::SoilTexture{NF} = SoilTexture(NF),
        porosity::AbstractSoilPorosity{NF} = ConstantSoilPorosity(NF)
    ) where {NF}
    return HomogeneousStratigraphy(texture, porosity)
end

# Kernel functions

soil_texture(i, j, k, grid, fields, strat::HomogeneousStratigraphy) = strat.texture

"""
    $TYPEDSIGNATURES

Compute the organic fraction of solid material in the soil volume at index `i, j, k`.
"""
@inline function organic_fraction(
        i, j, k, grid, fields,
        strat::HomogeneousStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    ρ_soc = density_soc(i, j, k, grid, fields, bgc)
    ρ_org = density_pure_soc(bgc)
    por = organic_porosity(i, j, k, grid, fields, strat.porosity, strat.texture)
    organic = ρ_soc / ((1 - por) * ρ_org)
    return organic
end

"""
    $TYPEDSIGNATURES

Compute the porosity of the soil volume at the given indices.
"""
@propagate_inbounds function porosity(
        i, j, k, grid, fields,
        strat::HomogeneousStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    organic = organic_fraction(i, j, k, grid, fields, strat, bgc)
    texture = strat.texture
    por_m = mineral_porosity(i, j, k, grid, fields, strat.porosity, texture)
    por_o = organic_porosity(i, j, k, grid, fields, strat.porosity, texture)
    return (1 - organic) * por_m + organic * por_o
end

"""
    $SIGNATURES

Construct a `SoilVolume` object summarizing the material composition of the soil volume
at the given indices `i, j, k` on `grid`.
"""
@propagate_inbounds function soil_volume(
        i, j, k, grid, fields,
        strat::AbstractStratigraphy,
        hydrology::AbstractSoilHydrology,
        bgc::AbstractSoilBiogeochemistry
    )
    # get current saturation and liquid fraction
    sat = saturation_water_ice(i, j, k, grid, fields, hydrology)
    liq = liquid_water_fraction(i, j, k, grid, fields, hydrology)
    # compute porosity and solid fractions
    por = porosity(i, j, k, grid, fields, strat, bgc)
    solid = soil_matrix(i, j, k, grid, fields, strat, bgc)
    return SoilVolume(por, sat, liq, solid)
end

"""
    $TYPEDSIGNATURES

Compute and return the soil solid matrix at index `i, j, k` on `grid`. The default
implementation assumes a simple `MineralOrganic` parameterization of the solid material.
"""
@propagate_inbounds function soil_matrix(
        i, j, k, grid, fields,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    organic = organic_fraction(i, j, k, grid, fields, strat, bgc)
    texture = soil_texture(i, j, k, grid, fields, strat)
    return MineralOrganic(texture, organic)
end
