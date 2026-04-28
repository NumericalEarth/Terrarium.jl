import Thermodynamics:
    partial_pressure_vapor,
    saturation_vapor_pressure,
    q_vap_from_RH,
    q_vap_saturation

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
    vapor_pressure_to_specific_humidity(e, p, ε)

Convert the vapor pressure `e` to specific humidity at the given pressure `p` based on the
molecular weight ratio ε.
"""
@inline vapor_pressure_to_specific_humidity(e, p, ε) = ε * e / p
# TODO: Replace by q_vap_from_p_vap

"""
    specific_humidity_to_vapor_pressure(q, p, ε)

Convert the specific humidity `q` to vapor pressure at the given pressure `p` based on the
molecular weight ratio ε.
"""
@inline function specific_humidity_to_vapor_pressure(q, p, ε)
    e = q * p / (ε + (1 - ε) * q)
    return e
end
# TODO: replace by partial_pressure_vapor
"""
    relative_to_specific_humidity(r_h, pr, T, ε)

Derives specific humidity from measured relative humidity, air pressure, air temperature, and molecular weight ratio.
"""
@inline relative_to_specific_humidity(r_h, pr, T, ε) = ε * (r_h / 100) * saturation_vapor_pressure(T) / pr

# saturation vapor pressure
"""
    saturation_vapor_pressure(T, a₁, a₂, a₃)

August-Roche-Magnus equation for saturation vapor pressure at temperature `T` with empirical
coefficients a₁, a₂, and a₃.
"""
@inline saturation_vapor_pressure(T, a₁, a₂, a₃) = a₁ * exp(a₂ * T / (T + a₃))

"""
    saturation_vapor_pressure(T)

Saturation vapor pressure of an air parcel at the given temperature `T`. By default, the saturation vapor
pressure is computed over ice for `T <= 0°C` and over water for `T > 0°C`
Coefficients of August-Roche-Magnus equation taken from [alduchovImprovedMagnusForm1996](@cite).

# References
* [alduchovImprovedMagnusForm1996](@cite) Alduchov and Eskridge, Journal of Applied Meteorology and Climatology (1996)
"""
@inline function saturation_vapor_pressure(c::PhysicalConstants, T::NF) where {NF}
    return if T <= zero(T)
        saturation_vapor_pressure(c, T, Ice())
    else
        saturation_vapor_pressure(c, T, Liquid())
    end
end
