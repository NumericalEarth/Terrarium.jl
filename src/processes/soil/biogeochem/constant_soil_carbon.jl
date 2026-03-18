"""
    $TYPEDEF

Naive implementation of soil biogeochemistry that just assumes there to be a constant
organic content in all soil layers.

Properties:
$TYPEDFIELDS
"""
Base.@kwdef struct ConstantSoilCarbonDensity{NF} <: AbstractSoilBiogeochemistry{NF}
    "Soil organic carbon density [kg/m^3]"
    ρ_soc::NF = 0.0

    "Pure organic matter density [kg/m^3]"
    ρ_org::NF = 1300.0
end

ConstantSoilCarbonDensity(::Type{NF}; kwargs...) where {NF} = ConstantSoilCarbonDensity{NF}(; kwargs...)

variables(::ConstantSoilCarbonDensity) = ()

@inline density_soc(i, j, k, grid, fields, bgc::ConstantSoilCarbonDensity) = bgc.ρ_soc

@inline compute_auxiliary!(state, grid, bgc::ConstantSoilCarbonDensity, args...) = nothing

@inline compute_tendencies!(state, grid, bgc::ConstantSoilCarbonDensity, args...) = nothing
