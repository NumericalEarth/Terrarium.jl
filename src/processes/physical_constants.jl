"""
    $TYPEDEF

A collection of general physical constants that do not (usually) need to be varied in parameter calibration.

Properties:
$FIELDS
"""
@kwdef struct PhysicalConstants{NF}
    "Density of water in kg/m^3"
    ρw::NF = 1000.0

    "Density of ice in kg/m^3"
    ρi::NF = 916.2

    "Density of air at standard pressure and 0°C in kg/m^3"
    ρₐ::NF = 1.293

    "Specific heat capacity of dry air at standard pressure and 0°C in J/(m^3*K)"
    cₐ::NF = 1005.7

    "Sepcific latent heat of fusion of water in J/kg"
    Lsl::NF = 3.34e5

    "Specific latent heat of vaporization of water in J/kg"
    Llg::NF = 2.257e6

    "Specific latent heat of sublimation of water in J/kg"
    Lsg::NF = 2.834e6

    "Gravitational constant in m/s^2"
    g::NF = 9.80665

    "Reference temperature (0°C in Kelvin)"
    Tref::NF = 273.15

    "Stefan-Boltzmann constant in J/(s*m^2*K^4)"
    σ::NF = 5.6704e-8

    "von Kármán constant"
    κ::NF = 0.4

    "Ratio of molecular weight of water vapor to dry air"
    ε::NF = 0.622

    "Specific gas constant of air in J/(kg*K)"
    Rₐ::NF = 287.058

    "Atomic mass of carbon [gC/mol]"
    C_mass::NF = 12.0
end

PhysicalConstants(::Type{NF}; kwargs...) where {NF} = PhysicalConstants{NF}(; kwargs...)

"""
    celsius_to_kelvin(c::PhysicalConstants, T)

Convert the given temperature in °C to Kelvin based on the constant `Tref`.
"""
@inline celsius_to_kelvin(c::PhysicalConstants, T) = T + c.Tref

"""
    stefan_boltzmann(c::PhysicalConstants, T, ϵ)

Stefan-Boltzmann law ``M = \\epsilon \\sigma T^4`` where T is the surface temperature in Kelvin
and ϵ is the emissivity.
"""
@inline stefan_boltzmann(c::PhysicalConstants, T, ϵ) = ϵ * c.σ * T^4

"""
    psychrometric_constant(c::PhysicalConstants, p)

Calcualte the psychrometric constant at the given atmospheric pressure `p`.
"""
@inline psychrometric_constant(c::PhysicalConstants, p) = c.cₐ * p / (c.Llg * c.ε)
