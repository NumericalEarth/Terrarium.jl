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

[`PhysicalConstants`](@ref) collects fundamental physical constants used throughout Terrarium's process implementations. All constants are stored as fields of a single struct so that they are passed explicitly through the call graph — avoiding global state and keeping the code fully differentiable with Enzyme.jl. The struct is parametrically typed so that constants are automatically promoted to the model's numeric precision `NF`. It also subtypes `AbstractThermodynamicsParameters` to integrate directly with `Thermodynamics.jl`.

```@docs; canonical = false
PhysicalConstants
```

Default values follow standard references. Individual constants can be overridden at
construction to support unit-testing or sensitivity studies.

| Field | Symbol | Default | Units | Description |
|---|---|---|---|---|
| `ρw` | $\rho_w$ | 1000.0 | kg/m³ | Density of liquid water |
| `ρi` | $\rho_i$ | 916.7 | kg/m³ | Density of ice |
| `cp_d` | $c_{p,d}$ | 1004.5 | J/(kg·K) | Isobaric specific heat capacity of dry air at 0°C |
| `cp_i` | $c_{p,i}$ | 2070.0 | J/(kg·K) | Isobaric specific heat capacity of ice at 0°C |
| `cp_l` | $c_{p,l}$ | 4186.0 | J/(kg·K) | Isobaric specific heat capacity of liquid water at 0°C |
| `cp_v` | $c_{p,v}$ | 1846.0 | J/(kg·K) | Isobaric specific heat capacity of water vapor at 0°C|
| `Lsl` | $L_{sl}$ | 3.34×10⁵ | J/kg | Latent heat of fusion at 0°C |
| `Llg` | $L_{lv}$ | 2.257×10⁶ | J/kg | Latent heat of vaporization at 0°C |
| `Lsg` | $L_{sg}$ | 2.834×10⁶ | J/kg | Latent heat of sublimation at 0°C |
| `g` | $g$ | 9.80665 | m/s² | Gravitational acceleration |
| `T_ref` | $T_{\text{ref}}$ | 273.16 | K | Reference temperature |
| `T_freeze` | $T_{\text{freeze}}$ | 273.16 | K | Freezing temperature of water |
| `T_triple` | $T_{\text{triple}}$ | 273.16 | K | Triple point temperature of water |
| `press_triple` | $p_{\text{triple}}$ | 611.657 | Pa | Triple point pressure of water |
| `σ` | $\sigma$ | 5.6704×10⁻⁸ | W/(m²·K⁴) | Stefan-Boltzmann constant |
| `κ` | $\kappa$ | 0.4 | — | von Kármán constant |
| `R_d` | $R_d$ | 287.058 | J/(kg·K) | Specific gas constant of dry air |
| `R_v` | $R_v$ | 461.5 | J/(kg·K) | Specific gas constant of water vapor |
| `C_mass` | — | 12.0 | gC/mol | Molar mass of carbon |

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
