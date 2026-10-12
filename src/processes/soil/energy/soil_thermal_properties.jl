"""
    $TYPEDEF

Properties:
$TYPEDFIELDS

Default values from [hillelIntroductionSoilPhysics1982](@cite).

# References

* [hillelIntroductionSoilPhysics1982](@cite) Hillel, Academic Press (1982)
"""
@parameterized @kwdef struct SoilThermalConductivities{NF}
    "Thermal conductivity of water"
    @param water::NF = 0.57 (units = u"W/m/K", bounds = Positive)

    "Thermal conductivity of ice"
    @param ice::NF = 2.2 (units = u"W/m/K", bounds = Positive)

    "Thermal conductivity of air"
    @param air::NF = 0.025 (units = u"W/m/K", bounds = Positive)

    "Thermal conductivity of quartz (sand) mineral grains"
    @param quartz::NF = 7.7 (units = u"W/m/K", bounds = Positive)

    "Thermal conductivity of non-quartz (silt/clay) mineral grains"
    @param mineral::NF = 2.0 (units = u"W/m/K", bounds = Positive)

    "Thermal conductivity of organic soil constituents"
    @param organic::NF = 0.25 (units = u"W/m/K", bounds = Positive)
end

SoilThermalConductivities(::Type{NF}; kwargs...) where {NF} = SoilThermalConductivities{NF}(; kwargs...)

Adapt.@adapt_structure SoilThermalConductivities

"""
    $TYPEDEF

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct SoilHeatCapacities{NF}
    "Volumetric heat capacity of water"
    @param water::NF = 4.2e6 (units = u"J/K/m^3", bounds = Positive, scale = 1.0e6)

    "Volumetric heat capacity of ice"
    @param ice::NF = 1.9e6 (units = u"J/K/m^3", bounds = Positive, scale = 1.0e6)

    "Volumetric heat capacity of air"
    @param air::NF = 0.00125e6 (units = u"J/K/m^3", bounds = Positive, scale = 1.0e6)

    "Volumetric heat capacity of mineral soil"
    @param mineral::NF = 2.0e6 (units = u"J/K/m^3", bounds = Positive, scale = 1.0e6)

    "Volumetric heat capacity of organic soil"
    @param organic::NF = 2.5e6 (units = u"J/K/m^3", bounds = Positive, scale = 1.0e6)
end

SoilHeatCapacities(::Type{NF}; kwargs...) where {NF} = SoilHeatCapacities{NF}(; kwargs...)

"""
    $TYPEDEF

Thermal properties of the soil: constituent thermal conductivities and heat capacities, the bulk
conductivity weighting, and the freeze curve of the energy–temperature closure.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct SoilThermalProperties{NF, FC, CondWeight, Cond}
    "Thermal conductivities for all constituents"
    @component conductivities::Cond

    "Method for computing bulk thermal conductivity from constituents"
    @component conductivity_weighting::CondWeight

    "Thermal conductivities for all constituents"
    @component heat_capacities::SoilHeatCapacities{NF}

    "Freezing characteristic curve needed for energy-temperature closure"
    @component freezecurve::FC
end

SoilThermalProperties(
    ::Type{NF};
    conductivities::SoilThermalConductivities = SoilThermalConductivities(NF),
    conductivity_weighting::AbstractBulkWeighting = InverseQuadratic(),
    heat_capacities::SoilHeatCapacities{NF} = SoilHeatCapacities(NF),
    freezecurve::FreezeCurve = FreeWater()
) where {NF} = SoilThermalProperties{NF, typeof(freezecurve), typeof(conductivity_weighting), typeof(conductivities)}(conductivities, conductivity_weighting, heat_capacities, freezecurve)

Adapt.@adapt_structure SoilThermalProperties

freezecurve(
    ::SoilThermalProperties{NF, FreeWater},
    ::AbstractSoilHydrology
) where {NF} = FreeWater()

"""
    $SIGNATURES

Compute the thermal conductivity of the mineral grains for the given `texture` as the
quartz-weighted geometric mean of the quartz and non-quartz mineral endpoints
([johansenThermalConductivitySoils1975](@cite); [petersLidardEffectSoilThermal1998](@cite)):

```math
\\lambda_{\\text{min}} = \\lambda_q^{\\,q} \\, \\lambda_o^{\\,1 - q}
```

where the quartz volume fraction ``q`` is assumed equal to the sand fraction.
"""
@inline function mineral_thermal_conductivity(conductivities::SoilThermalConductivities, texture::SoilTexture)
    q = texture.sand
    κ₁ = conductivities.quartz
    κ₂ = conductivities.mineral
    # Compute bulk mineral conductivity
    κₘ = κ₁^q * κ₂^(1 - q)
    return κₘ
end

"""
    $SIGNATURES

Compute the bulk thermal conductivity of the given soil volume.
"""
@inline function compute_thermal_conductivity(props::SoilThermalProperties, soil::SoilComposition)
    c = props.conductivities
    # the bulk mineral conductivity depends on soil texture; build the constituent conductivities
    # explicitly so the (texture-derived) `mineral` value enters the weighting and the auxiliary
    # `quartz` endpoint does not leak in as a spurious constituent
    κₘ = mineral_thermal_conductivity(c, mineral_texture(soil))
    κs = (; c.water, c.ice, c.air, c.organic, mineral = κₘ)
    fracs = volumetric_fractions(soil)
    # apply bulk conductivity weighting
    return props.conductivity_weighting(κs, fracs)
end

"""
    $SIGNATURES

Compute the bulk heat capacity of the given soil volume.
"""
@inline function compute_heat_capacity(props::SoilThermalProperties, soil::SoilComposition)
    cs = getproperties(props.heat_capacities)
    fracs = volumetric_fractions(soil)
    # for heat capacity, we just do a weighted average
    average = WeightedAverage()
    return average(cs, fracs)
end

"""
    $TYPEDEF

Simple weighted average formula for computing bulk quantities:

```math
\\bar{x} = \\sum_{i=1}^N \\theta_i x_i
```
"""
struct WeightedAverage <: AbstractBulkWeighting end

(f::WeightedAverage)(xs, weights) = sum(fastmap(*, xs, weights))

"""
    $TYPEDEF

The inverse quadratic (or "quadratic parallel") bulk weighting formula
for thermal conductivity ([cosenzaSimultaneousDeterminationThermal2003](@cite)):

```math
k = \\left[\\sum_{i=1}^N θᵢ\\sqrt{kᵢ}\\right]^2
```

# References
* [cosenzaSimultaneousDeterminationThermal2003](@cite) Cosenza et al., European Journal of Soil Science (2003)
"""
struct InverseQuadratic <: AbstractBulkWeighting end

# we use fastmap here so that the ordering of named tuples can be arbitrary.
# `√x` is written as the float power `x^(one(x)/2)` rather than `sqrt(x)` so the weighting stays
# device-kernel-safe when a constituent conductivity is a traced device scalar (`CuTracedRNumber`):
# those have no `sqrt` method (its absence lowers to a reachable `throw`, which is illegal in a
# kernel and breaks Reactant compilation), but they do support the float power, exactly as the
# mineral-grain mixing `κ^q` in `mineral_thermal_conductivity` already relies on. The exponent is
# `one(x)/2` (not the rational `1//2`) because `^(::CuTracedRNumber, ::Rational)` is likewise absent;
# `one(x)/2` matches `x`'s own number type (plain float or traced scalar) and stays type-generic.
(f::InverseQuadratic)(xs, weights) = sum(fastmap((x, w) -> x^(one(x) / 2) * w, xs, weights))^2
