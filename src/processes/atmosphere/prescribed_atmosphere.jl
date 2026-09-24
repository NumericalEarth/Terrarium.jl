"""
Generic type representing the concentration of a particular tracer gas in the atmosphere.
"""
@kwdef struct TracerGas{NF, name}
    "Default concentration of the gas"
    concentration::NF

    TracerGas(name::Symbol, concentration::NF) where {NF} = new{NF, name}(concentration)
end

Base.nameof(::TracerGas{NF, name}) where {NF, name} = name

variables(gas::TracerGas{NF, name}) where {NF, name} = (
    input(name, Atmosphere(XY()), default = gas.concentration, units = u"ppm", desc = "Ambient atmospheric $(name) concentration in ppm"),
)

"""
Creates a `TracerGas` for ambient CO2 with concentration prescribed by an input variable with
the given name.
"""
AmbientCO2(::Type{NF}, name = :CO2) where {NF} = TracerGas(name, NF(380))

"""
Creates a `NamedTuple` from the given tracer gas types.
"""
TracerGases(tracers::TracerGas...) = (; map(tracer -> nameof(tracer) => tracer, tracers)...)

"""
    $TYPEDEF

Represents prescribed atmospheric conditions given by the following input variables:
    - Air temperature
    - Humidity
    - Atmospheric pressure
    - Windspeed
    - Precipitation
    - Solar radiation
    - Zero or more tracer gases (defaults to CO2 only)

Precpitation and solar radiation are specified according to specialized subtypes which dictate
the form of the input data; for precipitation, this defaults to `RainSnow`, i.e.
rain- and snowfall given as separate inputs, while for solar radiation, the default is
`LongShortWaveRadiation` which partitions downwelling radiation into the common short- and long
wave lengths representing solar and thermal (infrared) radiation.
"""
struct PrescribedAtmosphere{
        NF,
        tracernames,
        Precip <: AbstractPrecipitation,
        IncomingRad <: AbstractIncomingRadiation,
        Humidity <: AbstractHumidity,
        Wind <: AbstractWind,
        Aerodynamics <: AbstractAerodynamics,
        Gases <: Tuple{Vararg{TracerGas{NF}}},
    } <: AbstractAtmosphere{NF, Precip, IncomingRad, Humidity, Wind, Aerodynamics}
    "Surface-relative altitude in meters at which the atmospheric forcings are assumed to be applied"
    altitude::NF

    "Minimum windspeed"
    min_windspeed::NF

    "Precipitation inputs"
    precip::Precip

    "Downwelling radiation inputs"
    radiation::IncomingRad

    "Specific or relative humidity"
    humidity::Humidity

    "Wind velocity formulation"
    wind::Wind

    "Aerodynamic resistances and drag coefficients"
    aerodynamics::Aerodynamics

    "Atmospheric tracer gases"
    tracers::NamedTuple{tracernames, Gases}
end

function PrescribedAtmosphere(
        ::Type{NF};
        altitude::NF = NF(10), # Default to 10 m
        min_windspeed::NF = NF(0.01), # Default to 0.01 m/s
        precip::AbstractPrecipitation = RainSnow(),
        radiation::AbstractIncomingRadiation = LongShortWaveRadiation(),
        humidity::AbstractHumidity = SpecificHumidity(),
        wind::AbstractWind = Windspeed(),
        aerodynamics::AbstractAerodynamics = ConstantAerodynamics(NF),
        tracers::NamedTuple = TracerGases(AmbientCO2(NF)),
    ) where {NF}
    return PrescribedAtmosphere(altitude, min_windspeed, precip, radiation, humidity, wind, aerodynamics, tracers)
end

ParameterEditing.parameters(::PrescribedAtmosphere) = (;)

minimum_windspeed(atmos::PrescribedAtmosphere) = atmos.min_windspeed

variables(atmos::PrescribedAtmosphere{NF}) where {NF} = (
    input(:air_temperature, Atmosphere(XY()), default = NF(10), units = u"°C", desc = "Near-surface air temperature in °C"),
    input(:air_pressure, Atmosphere(XY()), default = NF(101_325), units = u"Pa", desc = "Atmospheric pressure at the surface in Pa"),
    variables(atmos.wind)...,
    variables(atmos.humidity)...,
    variables(atmos.precip)...,
    variables(atmos.radiation)...,
    variables(atmos.aerodynamics)...,
    # splat all tracer variables into one tuple
    tuplejoin(map(variables, atmos.tracers)...)...,
)

@inline compute_auxiliary!(state, grid, atmos::PrescribedAtmosphere) = nothing

@inline compute_tendencies!(state, grid, atmos::PrescribedAtmosphere) = nothing

"""
    aerodynamic_resistance(i, j, grid, fields, atmos::PrescribedAtmosphere)

Compute the aerodynamic resistance (inverse conductance) at grid cell `i, j`.
"""
@inline function aerodynamic_resistance(i, j, grid, fields, atmos::PrescribedAtmosphere{NF}) where {NF}
    let C = drag_coefficient(i, j, grid, fields, atmos.aerodynamics)
        Vₐ = windspeed(i, j, grid, fields, atmos)
        rₐ = 1 / (C * Vₐ)
        return rₐ
    end
end

"""
    air_temperature(i, j, grid, fields, ::PrescribedAtmosphere)

Retrieve or compute the air temperature at the current time step.
"""
@propagate_inbounds air_temperature(i, j, grid, fields, ::PrescribedAtmosphere) = fields.air_temperature[i, j]

"""
    air_pressure(i, j, grid, fields, ::PrescribedAtmosphere)

Retrieve or compute the air pressure at the current time step.
"""
@propagate_inbounds air_pressure(i, j, grid, fields, ::PrescribedAtmosphere) = fields.air_pressure[i, j]

"""
    ambient_co2(i, j, grid, fields, ::PrescribedAtmosphere)

Return the current prescribed ambient CO2 concentration level.
"""
@propagate_inbounds ambient_co2(i, j, grid, fields, ::PrescribedAtmosphere) = fields.CO2[i, j, end]

"""
    air_density(i, j, grid, fields, atmos::AbstractAtmosphere, constants::PhysicalConstants)

Compute density (kg m⁻³) of a parcel of air under current atmospheric conditions.
"""
@propagate_inbounds function air_density(i, j, grid, fields, atmos::AbstractAtmosphere, constants::PhysicalConstants)
    Tₐ = air_temperature(i, j, grid, fields, atmos)
    pₐ = air_pressure(i, j, grid, fields, atmos)
    qₐ = specific_humidity(i, j, grid, fields, atmos)
    ρₐ = Thermodynamics.air_density(constants.thermodynamics, celsius_to_kelvin(constants.thermodynamics, Tₐ), pₐ, qₐ)
    return ρₐ
end

"""
    $TYPEDEF

Represents a windspeed as direct input/forcing variable.
"""
struct Windspeed <: AbstractWind end

variables(::Windspeed) = (
    input(:windspeed, Atmosphere(XY()), default = 0.1, units = u"m/s", desc = "Wind speed in m/s"),
)

"""
    $TYPEDSIGNATURES

Retrieve or compute the windspeed at the current time step.
"""
@propagate_inbounds windspeed(i, j, grid, fields, atmos::AbstractAtmosphere{NF, PR, IR, HD, Windspeed}) where {NF, PR, IR, HD} = max(fields.windspeed[i, j], minimum_windspeed(atmos))

"""
    $TYPEDEF

Represents a windspeed given as `u` (east-west) and `v` (south-north) velocity components.
"""
struct WindVelocity <: AbstractWind end

variables(::WindVelocity) = (
    input(:wind_u, Atmosphere(XY()), default = 0.1, units = u"m/s", desc = "Wind velocity u-component in m/s"),
    input(:wind_v, Atmosphere(XY()), default = 0.1, units = u"m/s", desc = "Wind velocity v-component in m/s"),
)

@propagate_inbounds windspeed(i, j, grid, fields, atmos::AbstractAtmosphere{NF, PR, IR, HD, WindVelocity}) where {NF, PR, IR, HD} = max(sqrt(fields.wind_u[i, j]^2 + fields.wind_v[i, j]^2), minimum_windspeed(atmos))


"""
    $TYPEDEF

Humidity parameterization in which the near-surface specific humidity [kg/kg] is
provided directly as an input field.
"""
struct SpecificHumidity <: AbstractHumidity end

variables(::SpecificHumidity) = (
    input(:specific_humidity, Atmosphere(XY()), default = 1.0e-3, units = u"kg/kg", desc = "Near-surface specific humidity in kg/kg"),
)

"""
    specific_humidity(i, j, grid, fields, ::PrescribedAtmosphere{PR, IR, <:SpecificHumidity})

Retrieve or compute the specific_humidity at the current time step.
"""
@propagate_inbounds specific_humidity(i, j, grid, fields, ::AbstractAtmosphere{NF, PR, IR, SpecificHumidity}) where {NF, PR, IR} = fields.specific_humidity[i, j]

"""
    $TYPEDSIGNATURES

Computes the vapor pressure deficit (VPD) [Pa] at atmospheric reference level given the current atmospheric fields
"""
@propagate_inbounds function compute_vapor_pressure_deficit(i, j, grid, fields, atmos::AbstractAtmosphere, c::PhysicalConstants)
    T_air = air_temperature(i, j, grid, fields, atmos)
    q_air = specific_humidity(i, j, grid, fields, atmos)
    p = air_pressure(i, j, grid, fields, atmos)
    vpd = vapor_pressure_deficit(c.thermodynamics, T_air, p, q_air)
    return vpd
end

"""
    $TYPEDEF

Precipitation parameterization in which liquid rainfall [m/s] and frozen snowfall [m/s]
are provided as separate input fields.
"""
struct RainSnow <: AbstractPrecipitation end

variables(::RainSnow) = (
    input(:rainfall, Atmosphere(XY()), units = u"m/s", desc = "Liquid precipitation (rainfall) rate"),
    input(:snowfall, Atmosphere(XY()), units = u"m/s", desc = "Frozen precipitation (snowfall) rate"),
)

"""
    rainfall(i, j, grid, fields, ::AbstractAtmosphere{NF, <:RainSnow})

Retrieve or compute the liquid precipitation (rainfall) at the current time step.
"""
@inline rainfall(i, j, grid, fields, ::AbstractAtmosphere{NF, RainSnow}) where {NF} = fields.rainfall[i, j]

"""
    snowfall(i, j, grid, fields, ::AbstractAtmosphere{NF, <:RainSnow})

Retrieve or compute the frozen precipitation (snowfall) at the current time step.
"""
@inline snowfall(i, j, grid, fields, ::AbstractAtmosphere{NF, RainSnow}) where {NF} = fields.snowfall[i, j]

"""
    $TYPEDEF

Incoming radiation parameterization in which downwelling shortwave [W/m²] and
longwave [W/m²] radiation are provided as separate input fields, along with daytime
length [hr].
"""
struct LongShortWaveRadiation <: AbstractIncomingRadiation end

variables(::LongShortWaveRadiation) = (
    input(:surface_shortwave_down, Atmosphere(XY()), default = 341, units = u"W/m^2", desc = "Incoming (downwelling) shortwave solar radiation"),
    input(:surface_longwave_down, Atmosphere(XY()), default = 333, units = u"W/m^2", desc = "Incoming (downwelling) longwave thermal radiation"),
    input(:daytime_length, Atmosphere(XY()), default = 12, units = u"hr", desc = "Number of daytime hours varying with the season and orbital parameters"),
)

"""
    shortwave_down(i, j, grid, fields, ::AbstractAtmosphere{NF, PR, <:LongShortWaveRadiation})

Retrieve or compute the incoming/downwelling shortwave radiation at the current time step.
"""
shortwave_down(i, j, grid, fields, ::AbstractAtmosphere{NF, PR, <:LongShortWaveRadiation}) where {NF, PR} = fields.surface_shortwave_down[i, j]

"""
    longwave_down(i, j, grid, fields, ::AbstractAtmosphere{NF, PR, <:LongShortWaveRadiation})

Retrieve or compute the incoming/downwelling longwave radiation at the current time step.
"""
longwave_down(i, j, grid, fields, ::AbstractAtmosphere{NF, PR, <:LongShortWaveRadiation}) where {NF, PR} = fields.surface_longwave_down[i, j]

"""
    daytime_length(i, j, grid, fields, ::AbstractAtmosphere{PR, <:LongShortWaveRadiation})

Retrieve the length of the day (in hours) at grid cell `i, j`. Defaults to a constant 12 hours if no input is provided.
"""
daytime_length(i, j, grid, fields, ::AbstractAtmosphere{NF, PR, <:LongShortWaveRadiation}) where {NF, PR} = fields.daytime_length[i, j]
