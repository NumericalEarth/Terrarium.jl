"""
    $TYPEDEF

Represents the material composition of an elementary volume of soil.
The volume is decomposed into the key constitutents of water, ice, air,
and a mixture of organic and mineral solid material.

Properties:
$FIELDS
"""
@kwdef struct SoilVolume{NF, Solid <: AbstractSoilMatrix{NF}}
    "Natural porosity or void space of the soil"
    porosity::NF = 0.5

    "Fraction of the soil pores occupied by water or ice"
    saturation::NF = 1.0

    "Liquid (unfrozen) fraction of pore water"
    liquid::NF = 1.0

    "Parameterization of the solid phase (matrix) of the soil"
    solid::Solid = MineralOrganic(texture = SoilTexture(), organic = 0.0)

    # Scalar constructor
    function SoilVolume(porosity::NF, saturation::NF, liquid::NF, solid::AbstractSoilMatrix{NF}) where {NF <: Number}
        @assert zero(NF) <= porosity <= one(NF)
        @assert zero(NF) <= saturation <= one(NF)
        @assert zero(NF) <= liquid <= one(NF)
        return new{NF, typeof(solid)}(porosity, saturation, liquid, solid)
    end
end

porosity(soil::SoilVolume) = soil.porosity

saturation(soil::SoilVolume) = soil.saturation

liquid_fraction(soil::SoilVolume) = soil.liquid

water_ice(soil::SoilVolume) = soil.porosity * soil.saturation

organic_fraction(soil::SoilVolume) = organic_fraction(soil.solid)

mineral_texture(soil::SoilVolume) = mineral_texture(soil.solid)

"""
    $TYPEDSIGNATURES

Calculates the volumetric fractions of all constituents in the given soil volume
and returns them as a named tuple of the form `(; water, ice, air, solids...)`, where
`solids` correspodns to the volumetric fractions defined by the solid phase `soil.solid`.
"""
@inline function volumetric_fractions(soil::SoilVolume)
    # unpack relevant quantities
    let por = soil.porosity,
            sat = soil.saturation,
            liq = soil.liquid
        # calculate volumetric fractions
        water_ice = sat * por
        water = water_ice * liq
        ice = water_ice * (1 - liq)
        air = (1 - sat) * por
        solid_frac = 1 - por
        # get fractions of solid constituents
        solids = volumetric_fractions(soil.solid, solid_frac)
        return (; water, ice, air, solids...)
    end
end

"""
    $TYPEDEF

Soil matrix consisting of a simple, homogeneous mixture of mineral and organic material.

Properties:
$TYPEDFIELDS
"""
@kwdef struct MineralOrganic{NF} <: AbstractSoilMatrix{NF}
    "Mineral soil texture"
    texture::SoilTexture{NF} = SoilTexture()

    "Organic soil fraction"
    organic::NF = zero(eltype(texture))

    function MineralOrganic(texture::SoilTexture{NF}, organic::NF) where {NF}
        @assert zero(NF) <= organic <= one(NF) "organic content must be between zero and one"
        return new{NF}(texture, organic)
    end
end

"""
Alias for `SoilVolume{T, MineralOrganic{T}}`
"""
const MineralOrganicSoil{NF} = SoilVolume{NF, MineralOrganic{NF}}

@inline organic_fraction(solid::MineralOrganic) = solid.organic

"""
    $TYPEDSIGNATURES

Compute the volumetric fractions of the solid phase scaled by the overall solid fraction
of the soil `solid_frac`.
"""
@inline function volumetric_fractions(solid::MineralOrganic{NF}, solid_frac::NF) where {NF}
    organic = solid_frac * solid.organic
    mineral = solid_frac * (1 - solid.organic)
    return (; organic, mineral)
end
