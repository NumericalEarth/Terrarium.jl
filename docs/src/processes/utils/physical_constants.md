```@meta
CurrentModule = Terrarium
```

```@setup consts
using Terrarium
using InteractiveUtils
```

# Physical constants

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Physical constants are organised into three sub-structs grouped by category and
collected into the top-level [`PhysicalConstants`](@ref) container. All constants
are passed explicitly through the call graph — avoiding global state and keeping
the code fully differentiable with Enzyme.jl. Every struct is parametrically
typed so that constants are automatically promoted to the model's numeric
precision `NF`.

Functions that require only a subset of constants take the most specific
sub-struct as input. Functions that span multiple categories, or kernel-launching
and kernel functions, take the full [`PhysicalConstants`](@ref).

```@docs; canonical = false
PhysicalConstants
```

## Thermodynamic constants

[`ThermodynamicConstants`](@ref) holds thermodynamic and atmospheric constants.
It subtypes `AbstractThermodynamicsParameters` to integrate directly with
[Thermodynamics.jl](https://github.com/CliMA/Thermodynamics.jl).

```@docs; canonical = false
ThermodynamicConstants
```

| Field | Symbol | Default | Units | Description |
|---|---|---|---|---|
| `specific_heat_capacity_dry_air` | $c_{p,d}$ | 1004.5 | J/(kg·K) | Isobaric specific heat capacity of dry air at 0°C |
| `specific_heat_capacity_ice` | $c_{p,i}$ | 2070.0 | J/(kg·K) | Isobaric specific heat capacity of ice at 0°C |
| `specific_heat_capacity_liquid_water` | $c_{p,l}$ | 4186.0 | J/(kg·K) | Isobaric specific heat capacity of liquid water at 0°C |
| `specific_heat_capacity_water_vapor` | $c_{p,v}$ | 1846.0 | J/(kg·K) | Isobaric specific heat capacity of water vapor at 0°C |
| `latent_heat_fusion` | $L_{sl}$ | 3.3355×10⁵ | J/kg | Specific latent heat of fusion at 0°C |
| `latent_heat_vaporization` | $L_{lv}$ | 2.5008×10⁶ | J/kg | Specific latent heat of vaporization at 0°C |
| `latent_heat_sublimation` | $L_{sg}$ | 2.83435×10⁶ | J/kg | Specific latent heat of sublimation at 0°C, derived as $L_{sl} + L_{lv}$ for thermodynamic consistency |
| `temperature_reference` | $T_{\text{ref}}$ | 273.16 | K | Reference temperature (0°C in Kelvin) |
| `temperature_water_freeze` | $T_{\text{freeze}}$ | 273.16 | K | Freezing temperature of water |
| `temperature_water_triple_point` | $T_{\text{triple}}$ | 273.16 | K | Triple point temperature of water |
| `pressure_water_triple_point` | $p_{\text{triple}}$ | 611.657 | Pa | Triple point pressure of water |
| `gas_constant_dry_air` | $R_d$ | 287.058 | J/(kg·K) | Specific gas constant of dry air |
| `gas_constant_water_vapor` | $R_v$ | 461.5 | J/(kg·K) | Specific gas constant of water vapor |

## Material constants

[`MaterialConstants`](@ref) holds material properties. 

```@docs; canonical = false
MaterialConstants
```

| Field | Symbol | Default | Units | Description |
|---|---|---|---|---|
| `density_water` | $\rho_w$ | 1000.0 | kg/m³ | Density of liquid water |
| `density_ice` | $\rho_i$ | 916.7 | kg/m³ | Density of ice |
| `atomic_weight_carbon` | $M_C$ | 0.012 | kgC/mol | Atomic mass of carbon |

## Universal constants

[`UniversalConstants`](@ref) holds universal constants.

```@docs; canonical = false
UniversalConstants
```

| Field | Symbol | Default | Units | Description |
|---|---|---|---|---|
| `gravitational_acceleration` | $g$ | 9.80665 | m/s² | Gravitational acceleration |
| `stefan_boltzmann_constant` | $\sigma$ | 5.6704×10⁻⁸ | W/(m²·K⁴) | Stefan-Boltzmann constant |
| `von_karman_constant` | $\kappa$ | 0.4 | — | von Kármán constant |

## Construction

All four structs default to `Float64` and can be constructed without arguments:

```@example consts
PhysicalConstants()
```

To use a different numeric precision, pass the type as the first positional argument:

```@example consts
PhysicalConstants(Float32)
```

To override individual constants, construct the relevant sub-struct explicitly and
pass it as a keyword argument:

```@example consts
tc = ThermodynamicConstants(Float64; temperature_reference = 273.15)
PhysicalConstants(Float64; thermodynamics = tc)
```

Sub-structs can also be constructed and used independently, for example:

```@example consts
ThermodynamicConstants(Float64)
```

## Methods

```@docs; canonical = false
celsius_to_kelvin
```

```@docs; canonical = false
specific_heat_capacity_moist_air
```

```@docs; canonical = false
stefan_boltzmann
```

```@docs; canonical = false
psychrometric_constant
```
