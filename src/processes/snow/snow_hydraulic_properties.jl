"""
    $TYPEDEF

Base type for snow hydraulic properties and parameterization schemes.
"""
abstract type AbstractSnowHydraulics{NF} end

number_format(::Type{<:AbstractSnowHydraulics{NF}}) where {NF} = NF

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

    "Capillary retention `L_c`: liquid fraction held against gravity before meltwater drains"
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
it stores; explicit time steps should not exceed `τ_m`.

# References

* [tarbotonSpatiallyDistributedEnergy1994](@cite) Tarboton, Chowdhury and Jackson (1994)
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

"""
    $TYPEDSIGNATURES

Derivative `dM_r/dθ_liq` (m/s) of the meltwater outflow ([`compute_meltwater_outflow`](@ref)) with respect to
the liquid fraction `θ_liq` at a fixed ice content `W_ice` (m SWE), i.e. with the snow water equivalent
`W_snow = W_ice / (1 − θ_liq)`. It is the derivative of the active branch of the outflow:
`3 K_sat S*² / (1 − L_c)` where the Darcy rate is the smaller one, and `W_ice (1 − L_c) / (τ_m (1 − θ_liq)²)`
where the storage limit is; zero below the capillary retention `L_c`. Requires `θ_liq < 1`.
"""
@inline function compute_meltwater_outflow_derivative(hydraulics::ConstantSnowHydraulics, θ_liq::NF, W_ice::NF) where {NF}
    liq_c = hydraulics.capillary_retention
    K_sat = hydraulics.saturated_conductivity
    τ_m = hydraulics.drainage_timescale
    excess_liquid_fraction = max(θ_liq - liq_c, zero(NF))
    Sstar = excess_liquid_fraction / (one(NF) - liq_c)
    darcy_outflow = K_sat * Sstar^3
    storage_limited_outflow = excess_liquid_fraction * W_ice / ((one(NF) - θ_liq) * τ_m)
    ∂darcy∂θ = 3 * K_sat * Sstar^2 / (one(NF) - liq_c)
    ∂storage∂θ = ifelse(θ_liq > liq_c, W_ice * (one(NF) - liq_c) / (τ_m * (one(NF) - θ_liq)^2), zero(NF))
    return ifelse(darcy_outflow <= storage_limited_outflow, ∂darcy∂θ, ∂storage∂θ)
end
