import Thermodynamics:
    partial_pressure_vapor,
    saturation_vapor_pressure,
    q_vap_from_RH,
    q_vap_saturation,
    Ice,
    Liquid

"""
Return the number of seconds per day in the given number format.
"""
seconds_per_day(::Type{NF}) where {NF} = ustrip(u"s", NF(1)u"d")

"""
Return the number of seconds per hour in the given number format.
"""
seconds_per_hour(::Type{NF}) where {NF} = ustrip(u"s", NF(1)u"hr")

"""
    $SIGNATURES

Compute partial pressure of oxygen from surface pressure in Pa.
"""
@inline function partial_pressure_O2(pres::NF) where {NF}
    # TODO Shouldn't this be in physical constants?
    pres_O2 = NF(0.209) * pres
    return pres_O2
end

"""
    $SIGNATURES

Compute partial pressure of CO2 from surface pressure and CO2 concentration in Pa.
"""
@inline function partial_pressure_CO2(pres::NF, conc_co2::NF) where {NF}
    pres_co2 = conc_co2 * NF(1.0e-6) * pres
    return pres_co2
end

"""
    relative_to_specific_humidity(r_h, pr, T, c::PhysicalConstants)

Derives specific humidity from measured relative humidity `r_h` [%], air pressure `pr` [Pa],
air temperature `T` [°C], and physical constants `c`. Assumes saturation over ice for
`T <= 0°C` and over liquid water otherwise.
"""
@inline function relative_to_specific_humidity(r_h, pr, T, c::PhysicalConstants)
    T_K = celsius_to_kelvin(c, T)
    phase = T <= zero(T) ? Ice() : Liquid()
    return q_vap_from_RH(c, pr, T_K, r_h / 100, phase)
end

"""
    saturation_vapor_pressure(T)

Saturation vapor pressure of an air parcel at the given temperature `T` in °C. By default, the saturation vapor
pressure is computed over ice for `T <= 0°C` and over water for `T > 0°C`
Coefficients of August-Roche-Magnus equation taken from [alduchovImprovedMagnusForm1996](@cite).

# References
* [alduchovImprovedMagnusForm1996](@cite) Alduchov and Eskridge, Journal of Applied Meteorology and Climatology (1996)
"""
@inline function saturation_vapor_pressure(c::PhysicalConstants, T::NF) where {NF}
    T_K = celsius_to_kelvin(c, T)
    return if T <= zero(T)
        saturation_vapor_pressure(c, T_K, Ice())
    else
        saturation_vapor_pressure(c, T_K, Liquid())
    end
end

"""
    q_vap_saturation(c::PhysicalConstants, T, ρ)

Saturation specific humidity at temperature `T` [°C] and density `ρ` [kg/m³]. Dispatches
over ice for `T <= 0°C` and over liquid water otherwise.
"""
@inline function q_vap_saturation(c::PhysicalConstants, T, ρ)
    T_K = celsius_to_kelvin(c, T)
    return if T <= zero(T)
        q_vap_saturation(c, T_K, ρ, Ice())
    else
        q_vap_saturation(c, T_K, ρ, Liquid())
    end
end
