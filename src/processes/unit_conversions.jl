"""
    $SIGNATURES

Return the number of seconds per day in the given number format.
"""
seconds_per_day(::Type{NF}) where {NF} = ustrip(u"s", NF(1)u"d")

"""
    $SIGNATURES

Return the number of seconds per hour in the given number format.
"""
seconds_per_hour(::Type{NF}) where {NF} = ustrip(u"s", NF(1)u"hr")

"""
    $SIGNATURES

Convert a concentration `c` in parts-per-million (PPM) to mole fraction.
"""
ppm_to_mole_fraction(c::NF) where {NF} = c * NF(1.0e-6)

"""
    $SIGNATURES

Convert a pressure `p` from pascals (Pa) to kilopascals (kPa).
"""
pa_to_kpa(p::NF) where {NF} = p * NF(1.0e-3)

"""
    $SIGNATURES

Compute partial pressure of O₂ (Pa) from surface pressure.
"""
@inline function partial_pressure_O2(pres::NF) where {NF}
    # TODO Shouldn't this be in physical constants?
    pres_O2 = NF(0.209) * pres
    return pres_O2
end

"""
    $SIGNATURES

Compute partial pressure of CO₂ (Pa) from surface pressure and CO₂ concentration.
"""
@inline function partial_pressure_CO2(pres::NF, conc_co2::NF) where {NF}
    pres_co2 = conc_co2 * NF(1.0e-6) * pres
    return pres_co2
end

"""
    celsius_to_kelvin(c::ThermodynamicConstants, T)

Convert the given temperature in °C to Kelvin based on the constant `temperature_water_freeze`.
"""
@inline celsius_to_kelvin(c::ThermodynamicConstants, T) = T + c.temperature_water_freeze

"""
    kelvin_to_celsius(c::ThermodynamicConstants, T)

Convert the given temperature in °C to Kelvin based on the constant `temperature_water_freeze`.
"""
@inline kelvin_to_celsius(c::ThermodynamicConstants, T) = T - c.temperature_water_freeze

"""
    $SIGNATURES

Compute near-surface specific humidity [kg/kg] from the dewpoint temperature `T_dew` [°C] and air
pressure `p` [Pa]. By definition, the actual vapor pressure of the air equals the saturation vapor
pressure at the dewpoint, so this delegates to [`saturation_vapor_pressure`](@ref) — dispatching over
ice for `T_dew <= 0°C` and over liquid water otherwise — and then to
[`vapor_pressure_to_specific_humidity`](@ref).
"""
@inline function dewpoint_specific_humidity(c::ThermodynamicConstants, T_dew, p)
    vapor_pressure = saturation_vapor_pressure(c, T_dew)
    return vapor_pressure_to_specific_humidity(c, vapor_pressure, p)
end
