"""
    $TYPEDEF

Base type for soil biogeochemistry process implementations.
"""
abstract type AbstractSoilBiogeochemistry{NF} <: AbstractProcess{NF} end

"""
    density_pure_soc(bgc::AbstractSoilBiogeochemistry)

Return the assumed constant density of pure organic material. The default implementation assumes
there to be a property `ρ_org` defined on the type of `bgc`.
"""
density_pure_soc(bgc::AbstractSoilBiogeochemistry) = bgc.ρ_org

"""
    density_soc(i, j, k, grid, fields, bgc::AbstractSoilBiogeochemistry{NF}) where {NF}

Calculate the organic solid fraction based on the prescribed SOC and natural porosity/density of
the organic material.
"""
@inline density_soc(i, j, k, grid, fields, bgc::AbstractSoilBiogeochemistry{NF}) where {NF} = zero(NF)

