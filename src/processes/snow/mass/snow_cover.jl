"""
    $TYPEDEF

Base type for snow areal-coverage parameterizations.
"""
abstract type AbstractSnowCover{NF} end

number_format(::Type{<:AbstractSnowCover{NF}}) where {NF} = NF

"""
    $TYPEDEF

Simple fractional snow cover parameterization `f_snow = W_snow/(W_snow + W_ref)` where `W_snow` is the
current snow water equivalent (SWE) within any given finite area and `W_ref` is the reference
SWE at which the area would be expected to be 50% covered. The function is smooth and differentiable,
with `f_snow → 0` as `W_snow → 0` and `f_snow → 1` as `W_snow → ∞`.

Default SWE level for `half_coverage` is set to 0.01 m following [douvilleNewSnowParameterization1995](@cite).

Properties:
$TYPEDFIELDS

# References

* [douvilleNewSnowParameterization1995](@cite) Douville et al., Climate Dynamics (1995)
"""
@parameterized @kwdef struct FractionalSnowCover{NF} <: AbstractSnowCover{NF}
    "Reference snow water equivalent level `W_ref`"
    @param half_coverage::NF = 0.01 (units = u"m", bounds = Positive)
end

FractionalSnowCover(::Type{NF}; kwargs...) where {NF} = FractionalSnowCover{NF}(; kwargs...)

"""
    $TYPEDSIGNATURES

Sub-grid snow-covered area fraction `f_snow = W_snow/(W_snow + W_ref)` ∈ [0,1) from the snow water equivalent `W_snow`
[m] and the reference level `W_ref` (`half_coverage`), clamping negative SWE (which can occur transiently
in the prognostic state) to zero cover.
"""
@inline function compute_snow_cover_fraction(cover::FractionalSnowCover{NF}, swe::NF) where {NF}
    # clamp negative SWE (which can occur transiently in the prognostic state) to zero cover
    W_snow = max(swe, zero(NF))
    W_ref = cover.half_coverage
    return W_snow / (W_snow + W_ref)
end
