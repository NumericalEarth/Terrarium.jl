# Interface

"""
    $TYPEDEF

Base type for unsaturated hydraulic conductivity parameterizations.
"""
abstract type AbstractUnsatK{NF} end

"""
    get_swrc(::AbstractUnsatK)

Return the soil water retention curve associated with the given unsaturated hydraulic conductivity scheme.
"""
function get_swrc end

"""
    $TYPEDEF

Base type for soil hydraulic properties and parameterization schemes.
"""
abstract type AbstractSoilHydraulics{NF, RC <: SWRC, UnsatK <: AbstractUnsatK} end

"""
    $SIGNATURES

Compute hydraulic conductivity at saturation.
"""
function saturated_hydraulic_conductivity end

"""
    $SIGNATURES

Compute (variably saturated) hydraulic conductivity based on the given hydraulic properties,
soil water retention curve (SWRC), and volumetric fractions.
"""
function hydraulic_conductivity end


"""
    $SIGNATURES

Compute the empirical wilting point of the soil.
"""
function wilting_point end

"""
    $SIGNATURES

Compute the empirical field capacity of the soil.
"""
function field_capacity end

"""
    $SIGNATURES

Compute the (numerical) residual saturation level of the soil.
"""
function residual_saturation end

"""
    minimum_moisture_capacity(hydraulics::AbstractSoilHydraulics)

Numerical lower bound (1/m) of the specific moisture capacity `∂θ/∂ψ` used on the diagonal of the
implicit Richards solve (see [`Picard`](@ref) and [`compute_implicit_capacity!`](@ref)).
It plays the role of a small specific storage: in saturated cells `∂θ/∂ψ = 0`, and a saturated cell
whose faces have zero hydraulic conductivity (e.g. a frozen saturated cell) would otherwise give a
singular row of the tridiagonal system.
"""
function minimum_moisture_capacity end

# Hydraulics parameterizations

"""
    $TYPEDEF

Represents a simple case where soil hydraulic properties are given as constant values.
This is mostly provided just for testing, although it may be useful in certain cases where direct
measurements of hydraulic properites are available.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct ConstantSoilHydraulics{NF, RC, UnsatK <: AbstractUnsatK{NF}} <: AbstractSoilHydraulics{NF, RC, UnsatK}
    "Soil water retention curve"
    @component swrc::RC

    "Unsaturated hydraulic conductivity formulation; defaults to `saturated_conductivity`"
    @component unsat_hydraulic_cond::UnsatK

    "Hydraulic conductivity at saturation"
    @param saturated_conductivity::NF = 1.0e-5 (units = u"m/s", bounds = Positive, scale = 1.0e-5)

    "Constant field capacity"
    @param field_capacity::NF = 0.25 (bounds = UnitInterval,)

    "Constant wilting point"
    @param wilting_point::NF = 0.05 (bounds = UnitInterval,)

    "Residual (minimum) saturation level"
    @param residual_saturation::NF = 0.01 (bounds = UnitInterval,)

    "Lower bound (1/m) of the specific moisture capacity ∂θ/∂ψ on the diagonal of the implicit Richards solve (implicit time stepping only)"
    minimum_moisture_capacity::NF = 1.0e-5
end

function ConstantSoilHydraulics(
        ::Type{NF};
        swrc = BrooksCorey(),
        unsat_hydraulic_cond = UnsatKLinear(NF),
        kwargs...
    ) where {NF}
    swrc = adapt(NumberFormatAdaptor{NF}(), ustrip(swrc))
    return ConstantSoilHydraulics{NF, typeof(swrc), typeof(unsat_hydraulic_cond)}(; swrc, unsat_hydraulic_cond, kwargs...)
end

@inline saturated_hydraulic_conductivity(hydraulics::ConstantSoilHydraulics, args...) = hydraulics.saturated_conductivity

@inline wilting_point(hydraulics::ConstantSoilHydraulics, args...) = hydraulics.wilting_point

@inline field_capacity(hydraulics::ConstantSoilHydraulics, args...) = hydraulics.field_capacity

@inline residual_saturation(hydraulics::ConstantSoilHydraulics, args...) = hydraulics.residual_saturation

@inline minimum_moisture_capacity(hydraulics::ConstantSoilHydraulics) = hydraulics.minimum_moisture_capacity

"""
    $TYPEDEF

Soil hydraulics parameterization that includes the SURFEX [noilhanISBA1996; Eq. (28-29)](@cite) formulation of field capacity
and wilting point as a function of soil texture.

Properties:
$TYPEDFIELDS

# References

* [noilhanISBA1996](@cite) Noilhan & Mahfouf, Global and Planetary Change (1996)
"""
@parameterized @kwdef struct SoilHydraulicsSURFEX{NF, RC, UnsatK <: AbstractUnsatK{NF}} <: AbstractSoilHydraulics{NF, RC, UnsatK}
    "Soil water retention curve"
    @component swrc::RC

    "Unsaturated hydraulic conductivity formulation; defaults to `saturated_conductivity`"
    @component unsat_hydraulic_cond::UnsatK

    "Hydraulic conductivity at saturation"
    @param saturated_conductivity::NF = 1.0e-5 (units = u"m/s", bounds = Positive, scale = 1.0e-5)

    "Linear coefficient of wilting point adjustment due to clay content"
    @param wilting_point_effect::NF = 37.13e-3 (bounds = Positive,)

    "Linear coefficient of field capacity adjustment due to clay content"
    @param field_capacity_effect::NF = 89.0e-3 (bounds = Positive,)

    "Exponent of field capacity adjustment due to clay content"
    @param field_capacity_exp::NF = 0.35 (bounds = Positive,)

    "Minimum field capacity at 0% clay content"
    @param field_capacity_min::NF = 0.01 (bounds = UnitInterval,)

    "Residual (minimum) saturation level"
    @param residual_saturation::NF = 0.01 (bounds = UnitInterval,)

    "Lower bound (1/m) of the specific moisture capacity ∂θ/∂ψ on the diagonal of the implicit Richards solve (implicit time stepping only)"
    minimum_moisture_capacity::NF = 1.0e-5
end

function SoilHydraulicsSURFEX(
        ::Type{NF};
        swrc = BrooksCorey(),
        unsat_hydraulic_cond = UnsatKLinear(NF),
        kwargs...
    ) where {NF}
    swrc = adapt(NumberFormatAdaptor{NF}(), ustrip(swrc))
    return SoilHydraulicsSURFEX{NF, typeof(swrc), typeof(unsat_hydraulic_cond)}(; swrc, unsat_hydraulic_cond, kwargs...)
end

# TODO: this is not quite correct, SURFEX uses a hydraulic conductivity function that decreases exponentially with depth
@inline saturated_hydraulic_conductivity(hydraulics::SoilHydraulicsSURFEX, args...) = hydraulics.saturated_conductivity

@inline residual_saturation(hydraulics::SoilHydraulicsSURFEX, args...) = hydraulics.residual_saturation

@inline minimum_moisture_capacity(hydraulics::SoilHydraulicsSURFEX) = hydraulics.minimum_moisture_capacity

@inline function wilting_point(hydraulics::SoilHydraulicsSURFEX, texture::SoilTexture)
    β_w = hydraulics.wilting_point_effect
    wp = β_w * sqrt(texture.clay * 100)
    return wp
end

@inline function field_capacity(hydraulics::SoilHydraulicsSURFEX, texture::SoilTexture)
    η = hydraulics.field_capacity_exp
    β_c = hydraulics.field_capacity_effect
    fc₀ = hydraulics.field_capacity_min
    fc = fc₀ + β_c * (texture.clay * 100)^η
    return fc
end

# Unsaturated hydraulic conductivity schemes

"""
    $TYPEDEF

Simple formulation of hydraulic conductivity as a linear function of the liquid water saturated fraction,
i.e. `soil.water / (soil.water + soil.ice + soil.air)`.
"""
struct UnsatKLinear{NF} <: AbstractUnsatK{NF} end

UnsatKLinear(::Type{NF}) where {NF} = UnsatKLinear{NF}()

function hydraulic_conductivity(
        hydraulics::AbstractSoilHydraulics{NF, RC, UnsatKLinear{NF}},
        soil::SoilComposition
    ) where {NF, RC}
    let fracs = volumetric_fractions(soil)
        θw = fracs.water # unfrozen water content
        θsat = fracs.water + fracs.ice + fracs.air # water + ice content at saturation (porosity)
        K_sat = saturated_hydraulic_conductivity(hydraulics, soil.solid.texture)
        K = K_sat * θw / θsat
        return K
    end
end

"""
    $TYPEDEF

Formulation of hydraulic conductivity as a function of saturated hydraulic conductivity `K_sat` and
volumetric fractions, assumed to include those of water, ice, and air, following the van Genuchten 
formulation [vangenuchtenHydraulicConductivity1980](@cite) extended with an ice impedance factor 
[westermannCryoGridCommunityModel2023](@cite).

# References

* [vangenuchtenHydraulicConductivity1980](@cite) Van Genuchten, Soil Science Society of America Journal (1980)
* [westermannCryoGridCommunityModel2023](@cite) Westermann et al., Geoscientific Model Development (2023)
"""
@parameterized @kwdef struct UnsatKVanGenuchten{NF} <: AbstractUnsatK{NF}
    "Exponential scaling factor for ice impedance"
    @param impedance::NF = 7 (bounds = Positive,)
end

UnsatKVanGenuchten(::Type{NF}; impedance::NF = NF(7)) where {NF} = UnsatKVanGenuchten(NF(impedance))

function hydraulic_conductivity(
        hydraulics::AbstractSoilHydraulics{NF, <:VanGenuchten, UnsatKVanGenuchten{NF}},
        soil::SoilComposition
    ) where {NF}
    # TODO: The SWRC parameters will need to also be spatially varying at some point
    let n = hydraulics.swrc.n # van Genuchten parameter `n`
        θw = water(soil) # volumetric content of unfrozen water
        θsat = porosity(soil) # porosity
        f = liquid_fraction(soil) # fraction of pore water that is unfrozen (equiv. θw / θwi)
        Ω = hydraulics.unsat_hydraulic_cond.impedance # scaling parameter for ice impedance
        I_ice = 10^(-Ω * (1 - f)) # ice impedance factor
        K_sat = saturated_hydraulic_conductivity(hydraulics, soil.solid.texture)
        # Effective saturation, clamped to the unit interval so that every fractional power below is
        # evaluated at a non-negative base. Illegal state values (θw > θsat or θw < 0), which may occur
        # when using adaptive time steppers.
        S = clamp(θw / θsat, zero(θw), one(θw))
        K = K_sat * I_ice * sqrt(S) * (one(S) - (one(S) - S^(n / (n + 1)))^((n - 1) / n))^2
        return K
    end
end
