"""
    $TYPEDEF

Initializer for coupled soil energy/hydrology/biogeochemistry models.
"""
@parameterized struct SoilInitializer{
        NF,
        EnergyInit <: AbstractInitializer{NF},
        HydrologyInit <: AbstractInitializer{NF},
        BGCInit <: AbstractInitializer{NF},
    } <: AbstractInitializer{NF}
    "Soil energy/temperature state initializer"
    @component energy::EnergyInit

    "Soil hydrology state initializer"
    @component hydrology::HydrologyInit

    "Soil biogeochemistry state initializer"
    @component biogeochem::BGCInit
end

function SoilInitializer(
        ::Type{NF};
        energy = QuasiThermalSteadyState(NF),
        hydrology = SaturationWaterTable(NF),
        biogeochem = DefaultInitializer(NF)
    ) where {NF}
    return SoilInitializer(energy, hydrology, biogeochem)
end

variables(init::SoilInitializer) = tuplejoin(variables(init.energy), variables(init.hydrology), variables(init.biogeochem))

function initialize!(state, model::AbstractModel, init::SoilInitializer)
    initialize!(state, model, init.hydrology)
    initialize!(state, model, init.biogeochem)
    initialize!(state, model, init.energy)
    return nothing
end

# Soil energy initializers

"""
    $TYPEDEF

Initializer for soil/ground temperature that sets the temperature profile of each column to the
value of the `initial_surface_temperature` input variable, whose default is `T₀`.

`T₀` may be a number, a function of the horizontal node coordinates, an array, a `Field`, or an
[`AbstractFieldInitializer`](@ref). It is re-applied as the default of the input variable at every
initialization and can be overridden by an [`InputSource`](@ref) named `initial_surface_temperature`.

Properties:
$TYPEDFIELDS
"""
@parameterized struct ConstantSoilTemperature{NF, T0} <: AbstractInitializer{NF}
    "Initial surface temperature (°C)"
    @param T₀::T0
end

"""
    $TYPEDSIGNATURES

Creates a constant soil temperature initializer with the given surface temperature `T₀` (°C).
"""
ConstantSoilTemperature(::Type{NF}; T₀ = zero(NF)) where {NF} = ConstantSoilTemperature{NF, typeof(T₀)}(T₀)
ConstantSoilTemperature(T₀::NF) where {NF <: AbstractFloat} = ConstantSoilTemperature{NF, NF}(T₀)

# Preserve `NF` when reconstructing from parameters (see `ParameterEditing.reconstruct`)
ConstructionBase.constructorof(::Type{<:ConstantSoilTemperature{NF}}) where {NF} = T₀ -> ConstantSoilTemperature{NF, typeof(T₀)}(T₀)

variables(init::ConstantSoilTemperature) = (
    input(:initial_surface_temperature, Ground(XY()), default = init.T₀, units = u"°C", desc = "Initial surface temperature of the soil column"),
)

function initialize!(state, model::AbstractModel, init::ConstantSoilTemperature)
    set!(state.temperature, kernel(compute_constant_temperature, init), state)
    return nothing
end

"""
    $TYPEDSIGNATURES

Initial temperature at cell `(i, j, k)` taken from the `initial_surface_temperature` (°C) field.
"""
@propagate_inbounds compute_constant_temperature(i, j, k, grid, fields, ::ConstantSoilTemperature) = fields.initial_surface_temperature[i, j, 1]

"""
    $TYPEDEF

Initializer that sets soil/ground temperature to a thermal quasi-steady state based on the given
surface temperature, geothermal heat flux, and bulk (constant) thermal conductivity:

    T(z) = T₀ - Qgeo / k_eff * z

with depth `z ≤ 0` (m). Note that this is not a *true* thermal steady state, which would require
iterative calculation of the thermal conductivity from the soil properties and initial temperature profile.

The surface temperature and the geothermal heat flux are declared as the input variables
`initial_surface_temperature` (°C) and `geothermal_heat_flux` (W/m²) with defaults `T₀` and `Qgeo`.
The latter is the same variable read by the [`GeothermalHeatFlux`](@ref) bottom boundary condition,
so that the initial profile and the boundary condition are consistent. Each default may be a number,
a function of the horizontal node coordinates, an array, a `Field`, or an [`AbstractFieldInitializer`](@ref).
Defaults are re-applied at every initialization, so that changes to parameters take effect, and can be
overridden by an [`InputSource`](@ref) of the same name (with matching units).

Properties:
$TYPEDFIELDS
"""
@parameterized struct QuasiThermalSteadyState{NF, T0, QG} <: AbstractInitializer{NF}
    "Initial surface temperature (°C)"
    @param T₀::T0

    "Geothermal heat flux (W/m²)"
    @param Qgeo::QG

    "Bulk thermal conductivity (W/m/K)"
    @param k_eff::NF bounds = Positive
end

"""
    $TYPEDSIGNATURES

Creates a quasi-steady-state soil temperature initializer with surface temperature `T₀` (°C),
geothermal heat flux `Qgeo` (W/m²), and bulk thermal conductivity `k_eff` (W/m/K).
"""
function QuasiThermalSteadyState(::Type{NF}; T₀ = zero(NF), Qgeo = convert(NF, 1 // 50), k_eff = one(NF)) where {NF}
    return QuasiThermalSteadyState{NF, typeof(T₀), typeof(Qgeo)}(T₀, Qgeo, convert(NF, k_eff))
end

ConstructionBase.constructorof(::Type{<:QuasiThermalSteadyState{NF}}) where {NF} = (T₀, Qgeo, k_eff) -> QuasiThermalSteadyState{NF, typeof(T₀), typeof(Qgeo)}(T₀, Qgeo, k_eff)

variables(init::QuasiThermalSteadyState) = (
    input(:initial_surface_temperature, Ground(XY()), default = init.T₀, units = u"°C", desc = "Initial surface temperature of the soil column"),
    input(:geothermal_heat_flux, Ground(XY()), default = init.Qgeo, units = u"W/m^2", desc = "Geothermal heat flux at the bottom of the soil column"),
)

function initialize!(state, model::AbstractModel, init::QuasiThermalSteadyState)
    set!(state.temperature, kernel(compute_quasi_steady_state_temperature, init), state)
    return nothing
end

"""
    $TYPEDSIGNATURES

Quasi-steady-state temperature `T₀ - Qgeo / k_eff * z` at cell `(i, j, k)` from the
`initial_surface_temperature` (°C) and `geothermal_heat_flux` (W/m²) fields.
"""
@propagate_inbounds function compute_quasi_steady_state_temperature(i, j, k, grid, fields, init::QuasiThermalSteadyState)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    T₀ = fields.initial_surface_temperature[i, j, 1]
    Qgeo = fields.geothermal_heat_flux[i, j, 1]
    return T₀ - Qgeo / init.k_eff * z
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

Initializer for soil water/ice that sets the saturation profile of each column to the value of the
`initial_saturation` input variable, whose default is `sat`. See [`QuasiThermalSteadyState`](@ref)
for the accepted forms of the default.

Properties:
$TYPEDFIELDS
"""
@parameterized struct ConstantSaturation{NF, S} <: AbstractInitializer{NF}
    "Initial water/ice saturation (-)"
    @param sat::S bounds = UnitInterval
end

ConstantSaturation(::Type{NF}; sat = one(NF)) where {NF} = ConstantSaturation{NF, typeof(sat)}(sat)

ConstructionBase.constructorof(::Type{<:ConstantSaturation{NF}}) where {NF} = sat -> ConstantSaturation{NF, typeof(sat)}(sat)

variables(init::ConstantSaturation) = (
    input(:initial_saturation, Ground(XY()), default = init.sat, bounds = UnitInterval, desc = "Initial water/ice saturation of the soil column"),
)

function initialize!(state, model::AbstractModel, init::ConstantSaturation)
    set!(state.saturation_water_ice, kernel(compute_constant_saturation, init), state)
    return nothing
end

"""
    $TYPEDSIGNATURES

Initial saturation at cell `(i, j, k)` taken from the `initial_saturation` field.
"""
@propagate_inbounds compute_constant_saturation(i, j, k, grid, fields, ::ConstantSaturation) = fields.initial_saturation[i, j, 1]

"""
    $TYPEDEF

Simple initialization scheme for soil/ground saturation that sets the initial water table at the
given depth and the saturation level in all layers in the vadose (unsaturated) zone to a constant value.
Both are declared as the input variables `water_table_depth` (m) and `vadose_zone_saturation` (-)
with defaults taken from the corresponding properties. See [`QuasiThermalSteadyState`](@ref) for the
accepted forms of the defaults.

Properties:
$TYPEDFIELDS
"""
@parameterized struct SaturationWaterTable{NF, S, D} <: AbstractInitializer{NF}
    "Saturation in the vadose zone above the water table (-)"
    @param vadose_zone_saturation::S bounds = UnitInterval

    "Depth of the water table below the surface (m)"
    @param water_table_depth::D bounds = Nonnegative
end

function SaturationWaterTable(::Type{NF}; vadose_zone_saturation = convert(NF, 3 // 4), water_table_depth = NF(5)) where {NF}
    return SaturationWaterTable{NF, typeof(vadose_zone_saturation), typeof(water_table_depth)}(vadose_zone_saturation, water_table_depth)
end

ConstructionBase.constructorof(::Type{<:SaturationWaterTable{NF}}) where {NF} = (sat, depth) -> SaturationWaterTable{NF, typeof(sat), typeof(depth)}(sat, depth)

variables(init::SaturationWaterTable) = (
    input(:vadose_zone_saturation, Ground(XY()), default = init.vadose_zone_saturation, bounds = UnitInterval, desc = "Initial saturation in the vadose zone above the water table"),
    input(:water_table_depth, Ground(XY()), default = init.water_table_depth, units = u"m", bounds = Nonnegative, desc = "Initial depth of the water table below the surface"),
)

function initialize!(state, model::AbstractModel, init::SaturationWaterTable)
    set!(state.saturation_water_ice, kernel(compute_water_table_saturation, init), state)
    return nothing
end

"""
    $TYPEDSIGNATURES

Initial saturation at cell `(i, j, k)`: unity at and below the `water_table_depth`, otherwise the
`vadose_zone_saturation`.
"""
@propagate_inbounds function compute_water_table_saturation(i, j, k, grid, fields, ::SaturationWaterTable)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    depth = fields.water_table_depth[i, j, 1]
    sat = fields.vadose_zone_saturation[i, j, 1]
    return ifelse(z <= -depth, one(sat), sat)
end
