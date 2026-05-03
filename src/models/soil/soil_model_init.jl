"""
    $TYPEDEF

Initializer for coupled soil energy/hydrology/biogeochemistry models.
"""
struct SoilInitializer{
        NF,
        EnergyInit <: AbstractInitializer{NF},
        HydrologyInit <: AbstractInitializer{NF},
        BGCInit <: AbstractInitializer{NF},
    } <: AbstractInitializer{NF}
    "Soil energy/temperature state initializer"
    energy::EnergyInit

    "Soil hydrology state initializer"
    hydrology::HydrologyInit

    "Soil biogeochemistry state initializer"
    biogeochem::BGCInit
end

function SoilInitializer(
        ::Type{NF};
        energy = QuasiThermalSteadyState(NF),
        hydrology = SaturationWaterTable(NF),
        biogeochem = DefaultInitializer(NF)
    ) where {NF}
    return SoilInitializer(energy, hydrology, biogeochem)
end

function initialize!(state, model::AbstractModel, init::SoilInitializer)
    initialize!(state, model, init.hydrology)
    initialize!(state, model, init.biogeochem)
    initialize!(state, model, init.energy)
    return nothing
end

# Soil energy initializers

"""
    $TYPEDEF

Initializer for soil/ground temperature that sets the temperature profile to a constant value.

Properties:
$TYPEDFIELDS
"""
@kwdef struct ConstantSoilTemperature{NF} <: AbstractInitializer{NF}
    T₀::NF = 0.0
end

"""
Creates a constant soil temperature initializer.
"""
ConstantSoilTemperature(::Type{NF}; kwargs...) where {NF} = ConstantSoilTemperature{NF}(; kwargs...)

initialize!(state, ::AbstractModel, init::ConstantSoilTemperature) = set!(state.temperature, init.T₀)

"""
    $TYPEDEF

Initializer that sets soil/ground temperature to a thermal quasi-steady state based on the given
surface temperature, geothermal heat flux, and bulk (constant) thermal conductivity. Note that this is
not a *true* thermal steady state, which would require iterative calculation of the thermal conductivity
from the soil properties and initial temperature profile.

Properties:
$TYPEDFIELDS
"""
@kwdef struct QuasiThermalSteadyState{NF} <: AbstractInitializer{NF}
    T₀::NF = 0.0
    Qgeo::NF = 0.02
    k_eff::NF = 1.0
end

QuasiThermalSteadyState(::Type{NF}; kwargs...) where {NF} = QuasiThermalSteadyState{NF}(; kwargs...)

function initialize!(state, ::AbstractModel, init::QuasiThermalSteadyState)
    set!(state.temperature, (x, z) -> init.T₀ - init.Qgeo / init.k_eff * z)
    return nothing
end

"""
    $TYPEDEF

Represents a piecewise linear temperature initializer specified from the given knots.

```julia
initializer = PiecewiseLinearInitialSoilTemperature(
    0.0u"m" => 5.0, # always in °C!
    0.5u"m" => 2.0,
    1.0u"m" => 1.0,
    10.0u"m" => 1.5,
    ...
)
```

Properties:
$TYPEDFIELDS
"""
struct PiecewiseLinearInitialSoilTemperature{NF, N}
    knots::NTuple{N, NF}
end

function PiecewiseLinearInitialSoilTemperature(knots::Pair{<:LengthQuantity, NF}...) where {NF}
    return PiecewiseLinearInitialSoilTemperature(knots)
end

function initialize!(state, ::AbstractModel, init::PiecewiseLinearInitialSoilTemperature)
    f = piecewise_linear(init.knots...)
    set!(state.temperature, (x, z) -> f(z))
    return nothing
end

# Soil hydrology initializers

"""
    $TYPEDEF

Initializer for soil water/ice sets the saturation profile to a constant value.

Properties:
$TYPEDFIELDS
"""
@kwdef struct ConstantSaturation{NF} <: AbstractInitializer{NF}
    sat::NF = 1.0
end

ConstantSaturation(::Type{NF}; kwargs...) where {NF} = ConstantSaturation{NF}(; kwargs...)

initialize!(state, ::AbstractModel, init::ConstantSaturation) = set!(state.saturation_water_ice, init.sat)

"""
    $TYPEDEF

Simple initialization scheme for soil/ground saturation that sets the initial water table at the
given depth and the saturation level in all layers in the vadose (unsaturated) to a constant value.

Properties:
$TYPEDFIELDS
"""
@kwdef struct SaturationWaterTable{NF} <: AbstractInitializer{NF}
    vadose_zone_saturation::NF = 0.75
    water_table_depth::NF = 5.0
end

SaturationWaterTable(::Type{NF}; kwargs...) where {NF} = SaturationWaterTable{NF}(; kwargs...)

function initialize!(state, ::AbstractModel, init::SaturationWaterTable{NF}) where {NF}
    set!(state.saturation_water_ice, (x, z) -> z <= -init.water_table_depth ? one(NF) : init.vadose_zone_saturation)
    return nothing
end
