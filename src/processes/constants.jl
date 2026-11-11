# Thermodynamic constants

"""
    $TYPEDEF

Thermodynamic and atmospheric constants used in surface energy, turbulent flux,
and vegetation process implementations. Subtypes `AbstractThermodynamicsParameters`
so that it integrates directly with [Thermodynamics.jl](https://github.com/CliMA/Thermodynamics.jl).

```jldoctest
julia> show(ThermodynamicConstants(Float64))
ThermodynamicConstants{Float64}(1004.5, 2070.0, 4181.0, 1859.0, 333550.0, 2.5008e6, 2.83435e6, 273.16, 273.15, 273.16, 611.657, 287.05, 461.52)
```

Properties:
$FIELDS
"""
@kwdef struct ThermodynamicConstants{NF} <: Thermodynamics.Parameters.AbstractThermodynamicsParameters{NF}
    "Isobaric specific heat capacity of dry air at standard pressure and 0°C in J/(m^3*K)"
    specific_heat_capacity_dry_air::NF = 1004.5

    "Isobaric specific heat capacity of ice at standard pressure and 0°C in J/(m^3*K)"
    specific_heat_capacity_ice::NF = 2070.0

    "Isobaric specific heat capacity of liquid water at standard pressure and 0°C in J/(m^3*K)"
    specific_heat_capacity_liquid_water::NF = 4181.0

    "Isobaric specific heat capacity of water vapor at standard pressure and 0°C in J/(m^3*K)"
    specific_heat_capacity_water_vapor::NF = 1859.0

    "Specific latent heat of fusion of water in J/kg at 0°C"
    latent_heat_fusion::NF = 3.3355e5

    "Specific latent heat of vaporization of water in J/kg at 0°C"
    latent_heat_vaporization::NF = 2.5008e6

    "Specific latent heat of sublimation of water in J/kg at 0°C"
    latent_heat_sublimation::NF = latent_heat_fusion + latent_heat_vaporization

    "Reference temperature (0°C in Kelvin)"
    temperature_reference::NF = 273.16

    "Freezing temperature of water in Kelvin"
    temperature_water_freeze::NF = 273.15

    "Triple point temperature of water in Kelvin"
    temperature_water_triple_point::NF = 273.16

    "Triple point pressure of water in Pa"
    pressure_water_triple_point::NF = 611.657

    "Specific gas constant of dry air in J/(kg*K)"
    gas_constant_dry_air::NF = 287.05

    "Specific gas constant of water vapor in J/(kg*K)"
    gas_constant_water_vapor::NF = 461.52
end

ThermodynamicConstants(::Type{NF}; kwargs...) where {NF} = ThermodynamicConstants{NF}(; kwargs...)

Base.eltype(::ThermodynamicConstants{NF}) where {NF} = NF

# Material constants

"""
    $TYPEDEF

Material constants for water, ice, and carbon used in soil energy, hydrology,
and vegetation process implementations.

```jldoctest
julia> show(MaterialConstants(Float64))
MaterialConstants{Float64}(1000.0, 916.7, 0.012, 28.9647)
```

Properties:
$FIELDS
"""
@kwdef struct MaterialConstants{NF}
    "Density of water in kg/m^3"
    density_water::NF = 1000.0

    "Density of ice in kg/m^3"
    density_ice::NF = 916.7

    "Atomic mass of carbon in kgC/mol"
    atomic_weight_carbon::NF = 0.012

    "Molecular weight of dry air in g/mol"
    molecular_weight_dry_air::NF = 28.9647
end

MaterialConstants(::Type{NF}; kwargs...) where {NF} = MaterialConstants{NF}(; kwargs...)

Base.eltype(::MaterialConstants{NF}) where {NF} = NF

# Universal constants

"""
    $TYPEDEF

Universal physical constants used in surface energy and turbulent flux process
implementations.

```jldoctest
julia> show(UniversalConstants(Float64))
UniversalConstants{Float64}(9.80665, 5.6704e-8, 0.4)
```

Properties:
$FIELDS
"""
@kwdef struct UniversalConstants{NF}
    "Gravitational constant in m/s^2"
    gravitational_acceleration::NF = 9.80665

    "Stefan-Boltzmann constant in J/(s*m^2*K^4)"
    stefan_boltzmann_constant::NF = 5.6704e-8

    "von Kármán constant"
    von_karman_constant::NF = 0.4
end

UniversalConstants(::Type{NF}; kwargs...) where {NF} = UniversalConstants{NF}(; kwargs...)

Base.eltype(::UniversalConstants{NF}) where {NF} = NF

# Container type

"""
    $TYPEDEF

Top-level container for all physical constants used in Terrarium. Groups three
sub-structs by category:

- [`ThermodynamicConstants`](@ref) — thermodynamic and atmospheric constants
- [`MaterialConstants`](@ref) — material properties of water, ice, and carbon
- [`UniversalConstants`](@ref) — universal constants (gravity, Stefan-Boltzmann, von Kármán)

## Construction

```jldoctest
julia> show(PhysicalConstants())
PhysicalConstants{Float64}(ThermodynamicConstants{Float64}(1004.5, 2070.0, 4181.0, 1859.0, 333550.0, 2.5008e6, 2.83435e6, 273.16, 273.15, 273.16, 611.657, 287.05, 461.52), MaterialConstants{Float64}(1000.0, 916.7, 0.012, 28.9647), UniversalConstants{Float64}(9.80665, 5.6704e-8, 0.4))
```

To override individual constants, pass a customised sub-struct:

```jldoctest
julia> tc = ThermodynamicConstants(Float64; temperature_reference = 273.15);

julia> c = PhysicalConstants(Float64; thermodynamics = tc);

julia> c.thermodynamics.temperature_reference
273.15
```

Properties:
$FIELDS
"""
struct PhysicalConstants{NF}
    thermodynamics::ThermodynamicConstants{NF}
    material::MaterialConstants{NF}
    universal::UniversalConstants{NF}
end

PhysicalConstants() = PhysicalConstants(Float64)

function PhysicalConstants(
        ::Type{NF};
        thermodynamics::ThermodynamicConstants{NF} = ThermodynamicConstants(NF),
        material::MaterialConstants{NF} = MaterialConstants(NF),
        universal::UniversalConstants{NF} = UniversalConstants(NF),
    ) where {NF}
    return PhysicalConstants{NF}(thermodynamics, material, universal)
end

Base.eltype(::PhysicalConstants{NF}) where {NF} = NF
