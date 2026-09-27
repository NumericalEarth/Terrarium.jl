"""
    $TYPEDEF

Base type for snow hydraulic properties and parameterization schemes.
"""
abstract type AbstractSnowHydraulics{NF} end

number_format(::Type{<:AbstractSnowHydraulics{NF}}) where {NF} = NF

"""
    $TYPEDEF

Constant, spatially homogeneous snow hydraulic properties: a saturated hydraulic conductivity `K_sat`
and a capillary retention `L_c`, setting the Darcy-type meltwater outflow (see
[`compute_meltwater_outflow`](@ref)). Default values follow [tarbotonSpatiallyDistributedEnergy1994](@cite).

Properties:
$TYPEDFIELDS

# References

* [tarbotonSpatiallyDistributedEnergy1994](@cite) Tarboton, Chowdhury and Jackson (1994)
"""
@parameterized @kwdef struct ConstantSnowHydraulics{NF} <: AbstractSnowHydraulics{NF}
    "Hydraulic conductivity at saturation"
    @param saturated_conductivity::NF = 20.0 / 3600.0 (units = u"m/s", bounds = Positive, scale = 1.0e-3)

    "Capillary retention `liq_c`: liquid fraction held against gravity before meltwater drains"
    @param capillary_retention::NF = 0.05 (bounds = UnitInterval,)
end

ConstantSnowHydraulics(::Type{NF}; kwargs...) where {NF} = ConstantSnowHydraulics{NF}(; kwargs...)

"""
    $TYPEDSIGNATURES

Darcy-type meltwater outflow `M_r` (m/s, SWE). Liquid water in excess of the capillary retention `L_c`
drains from the snowpack with a cubic conductivity [tarbotonSpatiallyDistributedEnergy1994](@cite) (in excess-saturation
form): `M_r = K_sat · S*³` with `S* = max(θ_liq − L_c, 0) / (1 − L_c)`, where `θ_liq` is the liquid
fraction of the water substance. Outflow vanishes smoothly as `θ_liq → L_c` and saturates at `K_sat` as
`θ_liq → 1`.
# References

* [tarbotonSpatiallyDistributedEnergy1994](@cite) Tarboton et al., Report (1994)
"""
@inline function compute_meltwater_outflow(hydraulics::ConstantSnowHydraulics, θ_liq::NF) where {NF}
    liq_c = hydraulics.capillary_retention
    K_sat = hydraulics.saturated_conductivity
    Sstar = max(θ_liq - liq_c, zero(NF)) / (one(NF) - liq_c)
    return K_sat * Sstar^3
end
