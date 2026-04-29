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

[`PhysicalConstants`](@ref) collects fundamental physical constants used throughout Terrarium's process implementations. All constants are stored as fields of a single struct so that they are passed explicitly through the call graph — avoiding global state and keeping the code fully differentiable with Enzyme.jl. The struct is parametrically typed so that constants are automatically promoted to the model's numeric precision `NF`.

```@docs; canonical = false
PhysicalConstants
```

Default values follow standard references. Individual constants can be overridden at
construction to support unit-testing or sensitivity studies.

| Field | Symbol | Default | Units | Description |
|---|---|---|---|---|
| `ρw` | $\rho_w$ | 1000.0 | kg/m³ | Density of liquid water |
| `ρi` | $\rho_i$ | 916.2 | kg/m³ | Density of ice |
| `ρₐ` | $\rho_a$ | 1.293 | kg/m³ | Density of dry air at 0°C, 1 atm |
| `cₐ` | $c_a$ | 1005.7 | J/(kg·K) | Specific heat of dry air |
| `Lsl` | $L_{sl}$ | 3.34×10⁵ | J/kg | Latent heat of fusion |
| `Llg` | $L_{lv}$ | 2.257×10⁶ | J/kg | Latent heat of vaporization |
| `Lsg` | $L_{sg}$ | 2.834×10⁶ | J/kg | Latent heat of sublimation |
| `g` | $g$ | 9.80665 | m/s² | Gravitational acceleration |
| `Tref` | $T_{\text{ref}}$ | 273.15 | K | Reference temperature (0°C) |
| `σ` | $\sigma$ | 5.6704×10⁻⁸ | W/(m²·K⁴) | Stefan-Boltzmann constant |
| `κ` | $\kappa$ | 0.4 | — | von Kármán constant |
| `ε` | $\varepsilon$ | 0.622 | — | Ratio of molecular weights $M_v / M_d$ |
| `Rₐ` | $R_a$ | 287.058 | J/(kg·K) | Specific gas constant of dry air |
| `C_mass` | — | 12.0 | gC/mol | Molar mass of carbon |

## Methods

```@docs; canonical = false
celsius_to_kelvin
```

```@docs; canonical = false
stefan_boltzmann
```

```@docs; canonical = false
psychrometric_constant
```
