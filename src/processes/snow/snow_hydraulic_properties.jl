"""
    $TYPEDEF

Base type for snow hydraulic properties and parameterization schemes.
"""
abstract type AbstractSnowHydraulics{NF} end

"""
    $TYPEDEF

Constant, spatially homogeneous snow hydraulic properties: a saturated hydraulic conductivity `K_sat`,
a capillary retention `L_c`, and a drainage timescale `τ_m`, setting the meltwater outflow (see
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

    "Drainage timescale `τ_m` of the liquid water in excess of the capillary retention, limiting the outflow to the water stored in the pack"
    @param drainage_timescale::NF = 3600.0 (units = u"s", bounds = Positive)
end

ConstantSnowHydraulics(::Type{NF}; kwargs...) where {NF} = ConstantSnowHydraulics{NF}(; kwargs...)

"""
    $TYPEDSIGNATURES

Meltwater outflow `M_r` (m/s, SWE) from a snowpack with snow water equivalent `W_snow` (m). Liquid water
in excess of the capillary retention `L_c` drains from the snowpack with a cubic conductivity
[tarbotonSpatiallyDistributedEnergy1994](@cite) (in excess-saturation form), `K_sat · S*³` with
`S* = max(θ_liq − L_c, 0) / (1 − L_c)`, where `θ_liq` is the liquid fraction of the water substance.
This Darcy rate does not depend on the amount of water in the pack, so it is additionally limited by the
excess liquid water depth `max(θ_liq − L_c, 0) · W_snow` drained over the timescale `τ_m`:

    M_r = min(K_sat · S*³, max(θ_liq − L_c, 0) · max(W_snow, 0) / τ_m)

Outflow vanishes smoothly as `θ_liq → L_c` or `W_snow → 0`, so a thin pack cannot drain more water than
it stores; forward Euler time steps should not exceed `τ_m`.
# References

* [tarbotonSpatiallyDistributedEnergy1994](@cite) Tarboton et al., Report (1994)
"""
@inline function compute_meltwater_outflow(hydraulics::ConstantSnowHydraulics, θ_liq::NF, W_snow::NF) where {NF}
    liq_c = hydraulics.capillary_retention
    K_sat = hydraulics.saturated_conductivity
    τ_m = hydraulics.drainage_timescale
    excess_liquid_fraction = max(θ_liq - liq_c, zero(NF))
    Sstar = excess_liquid_fraction / (one(NF) - liq_c)
    darcy_outflow = K_sat * Sstar^3
    # excess liquid water depth (m) drained over τ_m: the pack cannot release more water than it stores
    storage_limited_outflow = excess_liquid_fraction * max(W_snow, zero(NF)) / τ_m
    return min(darcy_outflow, storage_limited_outflow)
end
