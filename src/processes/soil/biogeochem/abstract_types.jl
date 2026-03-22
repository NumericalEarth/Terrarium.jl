"""
    $TYPEDEF

Base type for soil biogeochemistry process implementations.
"""
abstract type AbstractSoilBiogeochemistry{NF} <: AbstractProcess{NF} end

"""
    density_soc(i, j, k, grid, fields, bgc::AbstractSoilBiogeochemistry)

Compute or return the soil organic carbon density at grid cell index `i, j, k`.
"""
function density_soc end
