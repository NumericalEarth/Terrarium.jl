```@meta
CurrentModule = Terrarium
```

```@setup physics_utils
using Terrarium
using InteractiveUtils
```

# Physics utilities

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

This module provides small, self-contained thermodynamic and atmospheric utility functions that are shared across multiple process implementations. All functions are `@inline`d and scalar-valued; they are intended to be called from within kernel functions. Note that these functions are mostly intended for internal use within the `Terrarium.jl` codebase, and are therefore not exported as part of the public API. However, once can still access them via `Terrarium.function_name` if needed.

Some of the functionality here builds upon [Thermodynamics.jl](@extref `Thermodynamics.jl`), which provides the basic thermodynamic building blocks, based on Rankine-Kirchhoff approximations. For a mathematical overview of these approximations, see [here](@extref Thermodynamics `Formulation`).

### Saturation vapor pressure

The saturation vapor pressure $e_{\text{sat}}$ is computed using a wrapper around [`saturation_vapor_pressure`](@extref Thermodynamics.saturation_vapor_pressure) from `Thermodynamics.jl` and is based on the integration of the Clausius-Clapeyron relation (see [here](@extref Thermodynamics 9.-Saturation-Vapor-Pressure) for details). 

### Vapor pressure and humidity conversions

Specific humidity $q$ and vapor pressure $e$ conversions (including related variables like relative humidity) are handled by `Thermodynamics.jl` functions (or wrappers around these functions). Specifically:
- Transform $q$ to $e$ via [`partial_pressure_vapor`](@extref Thermodynamics.partial_pressure_vapor).
- Transform $e$ to $q$ via [`vapor_pressure_to_specific_humidity`](@ref). With $\varepsilon = R_d / R_v$, the conversion is given by [shuttleworthTerrestrialHydrometWaterVapor2012; Eq. (2.8)](@cite) using the total air pressure $p$:
```math
\begin{equation}
q = \frac{\varepsilon e}{p - e (1 - \varepsilon)} .
\end{equation}
```


### Partial pressures of trace gases

The partial pressures of O₂ and CO₂ are computed from total surface pressure and, for CO₂, the volumetric concentration in ppm:

```math
\begin{align}
p_{\text{O}_2} &= 0.209 \, p \\
p_{\text{CO}_2} &= C_{\text{CO}_2} \times 10^{-6} \times p
\end{align}
```

## Methods

```@docs; canonical = false
seconds_per_day
```

```@docs; canonical = false
seconds_per_hour
```

```@docs; canonical = false
saturation_vapor_pressure
```

```@docs; canonical = false
saturation_specific_humidity_vapor
```

```@docs; canonical = false
vapor_pressure_to_specific_humidity
```

```@docs; canonical = false
relative_to_specific_humidity
```

```@docs; canonical = false
partial_pressure_O2
```

```@docs; canonical = false
partial_pressure_CO2
```

## References

```@bibliography
Pages = ["physics_utils.md"]
Canonical = false
```