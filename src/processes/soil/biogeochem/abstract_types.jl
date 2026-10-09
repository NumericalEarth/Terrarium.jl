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

Return the bulk soil organic carbon density ρ_soc (kg/m³) at index `i, j, k`, i.e. the mass of
organic material per unit volume of soil. The default implementation returns zero, corresponding to
a purely mineral soil. Implementations of `AbstractSoilBiogeochemistry` should override this
method; see `organic_fraction` for how ρ_soc is converted to the organic fraction of the soil matrix.
"""
@inline density_soc(i, j, k, grid, fields, bgc::AbstractSoilBiogeochemistry{NF}) where {NF} = zero(NF)
