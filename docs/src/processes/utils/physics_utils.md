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

This module provides small, self-contained thermodynamic and atmospheric utility functions that are shared across multiple process implementations. All functions are `@inline`d and scalar-valued; they are intended to be called from within kernel functions.

### Saturation vapor pressure

The saturation vapor pressure $e_{\text{sat}}$ is computed using the August-Roche-Magnus empirical formula [alduchovImprovedMagnusForm1996](@cite):

```math
\begin{equation}
e_{\text{sat}}(T) = a_1 \exp\!\left(\frac{a_2 T}{T + a_3}\right)
\end{equation}
```

where $T$ is temperature in °C and the coefficients differ for liquid water ($T \geq 0$°C) and ice ($T < 0$°C):

| Phase | $a_1$ (Pa) | $a_2$ | $a_3$ (°C) |
|---|---|---|---|
| Liquid water ($T \geq 0$°C) | 611.0 | 17.62 | 243.12 |
| Ice ($T < 0$°C) | 611.0 | 22.46 | 272.62 |

### Vapor pressure and humidity conversions

Specific humidity $q$ and vapor pressure $e$ are related through the molecular weight ratio $\varepsilon = M_v / M_d \approx 0.622$ and the total atmospheric pressure $p$:

```math
\begin{equation}
q = \frac{\varepsilon \, e}{p}
\end{equation}
```

The inverse conversion (vapor pressure from specific humidity) accounts for the partial pressure of dry air:

```math
\begin{equation}
e = \frac{q \, p}{\varepsilon + (1 - \varepsilon) q}
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
vapor_pressure_deficit
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