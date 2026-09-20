# Evapotranspiration

```@meta
CurrentModule = Terrarium
```

```@setup ET
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Evapotranspiration ($\text{ET}$) (m/s) is the combined process of water evaporation from soil and open water surfaces, evaporation of water intercepted by the canopy, and transpiration through leaf stomata. These processes remove water from the surface, driving latent heat flux and competing with sensible heat and ground heat fluxes in the surface energy balance.

All evapotranspiration pathways are primarily driven by the **vapor pressure difference** or **specific humidity difference** at the surface $\Delta q = q_{\text{sat}}(T_s) - q_a$, where $q_{\text{sat}}(T_s)$ is the saturation specific humidity at surface temperature $T_s$ (kg/kg) and $q_a$ is the atmospheric specific humidity (kg/kg) at a particular reference height.

Each pathway is also modulated by **aerodynamic resistance**(s) $r_a$ (s/m) (between surface and atmosphere) and possibly **stomatal resistance** $r_s$ (s/m) (in the case of transpiration).

```@docs; canonical = false
AbstractEvapotranspiration
```

```@example ET
subtypes(Terrarium.AbstractEvapotranspiration)
```

## Flux conventions: kinematic humidity vs. liquid water mass

Evapotranspiration computations involve (at least) three different physical domains, each with its own natural units:
the **atmosphere** (humidity fluxes), the **surface energy balance** (energy fluxes), and **soil hydrology** (water mass fluxes). To maintain clarity and prevent unit errors, Terrarium distinguishes between **kinematic humidity fluxes** $Q_h$ used in the calculation of the energy budget and **liquid water mass fluxes** $E$ that drive the hydrological state of the land surface.

### Kinematic humidity flux $Q_h$

The **kinematic humidity flux** $Q_h$ [m/s] represents the transport of water vapor through the air, scaled by air density:

```math
\begin{equation}
Q_h = \Delta q \cdot g \quad \text{[m/s]},
\end{equation}
```

where $\Delta q$ (kg/kg) is the specific humidity difference and $g$ (m/s) is the vapor conductance.

### Liquid water mass flux $E$

The **liquid water mass flux** $E$ (m/s) represents the equivalent flux of liquid water that would produce the same mass transfer:

```math
\begin{equation}
E = Q_h \cdot \frac{\rho_a}{\rho_w} \quad \text{[m/s]},
\end{equation}
```

where $\rho_a$ (kg/m³) is air density and $\rho_w$ (kg/m³) is liquid water density.

### Why the distinction matters

The ratio $\rho_a / \rho_w \approx 1.2 / 1000 \approx 1.2 \times 10^{-3}$ means that a kinematic humidity flux of $Q_h = 10^{-5}$ m/s corresponds to a liquid water flux of only $E \approx 1.2 \times 10^{-8}$ m/s. This scaling is critical for:

1. **Energy balance consistency**: The SEB uses $Q_h$ to compute latent heat ($H_l = \rho_a L Q_h$)
2. **Hydrology consistency**: Soil moisture budgets use $E$ as a water loss term
3. **Mass conservation**: The same physical process must remove the same mass from both atmosphere and soil

### Summary

| Flux type | Symbol | Units | Used in | Computed by |
|-----------|--------|-------|---------|-------------|
| Kinematic humidity | $Q_h$ | m/s | Surface energy balance | `humidity_flux`, `compute_surface_humidity_flux` |
| Liquid water mass | $E$ | m/s | Soil hydrology | `compute_evapotranspiration_fluxes!` (rescales $Q_h$) |
| Latent heat | $H_l$ | W/m² | Surface energy balance | `compute_latent_heat_flux` (uses $Q_h$) |

---

## Bare ground evaporation

```@docs; canonical = false
BareGroundEvaporation
```

```@example ET
variables(BareGroundEvaporation(Float32))
```

## Vegetated land evapotranspiration

```@docs; canonical = false
PALADYNCanopyEvapotranspiration
```

```@example ET
variables(PALADYNCanopyEvapotranspiration(Float32))
```

### Evaporation from the canopy

Canopy evaporation of intercepted water $E_{\text{can}}$ (liquid water flux, m/s) depends on the saturation state of the canopy (fraction of leaves wet). The underlying **kinematic humidity flux** $Q_{h,\text{can}}$ is:

```math
\begin{equation}
Q_{h,\text{can}} = f_{\text{can}} \frac{\Delta q}{r_a} \quad \text{[m/s]},
\end{equation}
```

where $f_{\text{can}}$ is the canopy saturation fraction (0 = dry, 1 = saturated) (-) and $\Delta q$ is the specific humidity difference (kg vapor/kg air).

When $f_{\text{can}} = 0$ (completely dry canopy), $Q_{h,\text{can}} = 0$. When $f_{\text{can}} = 1$ (wet canopy), evaporation proceeds at the potential rate.

The canopy evaporation vapor conductance is computed as
```math
\begin{equation}
g_{\text{can}} = \frac{f_{\text{can}}}{r_a} \quad \text{[m/s]},
\end{equation}
```
so that $Q_{h,\text{can}} = g_{\text{can}} \cdot \Delta q$.

The **liquid water flux** $E_{\text{can}}$ is then obtained by rescaling:
```math
\begin{equation}
E_{\text{can}} = Q_{h,\text{can}} \cdot \frac{\rho_a}{\rho_w} \quad \text{[m/s]}.
\end{equation}
```

### Ground evaporation

Evaporation from exposed soil or under-canopy surfaces $E_{\text{ground}}$ (liquid water flux, m/s) is limited by soil water availability. The underlying **kinematic humidity flux** $Q_{h,\text{ground}}$ is:

```math
\begin{equation}
Q_{h,\text{ground}} = \beta \frac{\Delta q}{r_a + r_e} \quad \text{[m/s]},
\end{equation}
```

where $\beta$ is the ground evaporation resistance factor (0 to 1) (-), $r_a$ is aerodynamic resistance (s/m), and $r_e$ is the aerodynamic resistance between ground and canopy (s/m).

The ground evaporation vapor conductance is computed as
```math
\begin{equation}
g_{\text{ground}} = \frac{\beta}{r_a + r_e} \quad \text{[m/s]},
\end{equation}
```
so that $Q_{h,\text{ground}} = g_{\text{ground}} \cdot \Delta q$.

The **liquid water flux** $E_{\text{ground}}$ is then:
```math
\begin{equation}
E_{\text{ground}} = Q_{h,\text{ground}} \cdot \frac{\rho_a}{\rho_w} \quad \text{[m/s]}.
\end{equation}
```

The resistance factor $\beta$ is computed from soil moisture in the upper layer: $\beta = 1$ when soil is wet (at field capacity) and $\beta \to 0$ as soil dries.

### Transpiration

Plant transpiration occurs through stomata and is controlled by stomatal conductance. The kinematic humidity flux for transpiration $Q_{h,\text{trp}}$ is:

```math
\begin{equation}
Q_{h,\text{trp}} = \frac{\Delta q}{r_a + r_s} \quad \text{[m/s]},
\end{equation}
```

where $r_s = 1 / g_w$ is the stomatal resistance (s/m) and $g_w$ is the stomatal conductance (m/s) (computed from photosynthesis; see [Stomatal conductance](@ref)).

The transpiration vapor conductance is computed as
```math
\begin{equation}
g_{\text{trp}} = \frac{1}{r_a + r_s} \quad \text{[m/s]},
\end{equation}
```
so that $Q_{h,\text{trp}} = g_{\text{trp}} \cdot \Delta q$.

The **liquid water flux** $E_{\text{trp}}$ is then:
```math
\begin{equation}
E_{\text{trp}} = Q_{h,\text{trp}} \cdot \frac{\rho_a}{\rho_w} \quad \text{[m/s]}.
\end{equation}
```

High stomatal conductance (when photosynthetically active) leads to low stomatal resistance and high transpiration. This creates a strong coupling between carbon uptake (photosynthesis) and water loss (transpiration).

### Total evapotranspiration

The PALADYN approach combines all three pathways in parallel. The **total kinematic humidity flux** is:

```math
\begin{equation}
Q_{h,\text{total}} = Q_{h,\text{can}} + Q_{h,\text{ground}} + Q_{h,\text{trp}} \quad \text{[m/s]},
\end{equation}
```

which is converted to **latent heat flux** for the surface energy balance:
```math
\begin{equation}
H_l = \rho_a \, L_v \, Q_{h,\text{total}} \quad \text{[W/m²]},
\end{equation}
```

where $L_v$ is the latent heat of vaporization (J/kg).

The **total liquid water flux** (sum of component liquid water fluxes) is used as a source/sink in soil hydrology:
```math
\begin{equation}
E_{\text{total}} = E_{\text{can}} + E_{\text{ground}} + E_{\text{trp}} \quad \text{[m/s]}.
\end{equation}
```

## Evaporation flux computation

All evapotranspiration pathways share a unified functional form for the **kinematic humidity flux**:

```math
\begin{equation}
Q_h = \Delta q \cdot g \quad \text{[m/s]},
\end{equation}
```

where $\Delta q$ [kg/kg] is the specific humidity difference and $g$ [m/s] is the vapor conductance specific to each pathway. The unified function `humidity_flux` handles all three pathways:

```@docs; canonical = false
humidity_flux
```

**Note**: The function returns $Q_h$ (kinematic humidity flux). To obtain the liquid water mass flux $E$ for soil hydrology, multiply by the density ratio $\rho_a/\rho_w$ as shown in the equations above.

## Conductance functions

The vapor conductances for each pathway are computed separately and stored as auxiliary fields during the `compute_auxiliary!` pass. These conductances are skin-temperature-independent (held fixed during the surface energy balance solve).

### Transpiration conductance

```@docs; canonical = false
transpiration_conductance
```

### Canopy evaporation conductance

```@docs; canonical = false
canopy_evaporation_conductance
```

### Ground evaporation conductance

```@docs; canonical = false
ground_evaporation_conductance
```

## Ground resistance parameterizations

The ground evaporation resistance factor $\beta$ is computed from soil moisture using parameterizations of `AbstractGroundEvaporationResistanceFactor`:

```@docs; canonical = false
ConstantEvaporationResistanceFactor
```

```@docs; canonical = false
SoilMoistureResistanceFactor
```

```@docs; canonical = false
ground_evaporation_resistance_factor
```

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, ::BareGroundEvaporation, ::NoCanopyInterception, ::PhysicalConstants, ::AbstractAtmosphere, ::Optional{AbstractSoil}, ::Optional{AbstractSnow})
```

```@docs; canonical = false
compute_auxiliary!(state, grid, ::PALADYNCanopyEvapotranspiration, ::AbstractCanopyInterception, ::PhysicalConstants, ::AbstractAtmosphere, ::AbstractSoil, ::AbstractVegetation, args...)
```

## Coupling to soil hydrology

Subtypes of `AbstractEvapotranpsiration` automatically inherit a default implementation of the [`forcing`](@ref) interface for [`SoilHydrology`](@ref), which computes the contribution of ET to the soil moisture tendency of the uppermost soil layer using [`ground_evapotranspiration_flux`](@ref).

```@docs; canonical = false
forcing(i, j, k, grid, clock, fields, evapotranspiration::AbstractEvapotranspiration, ::AbstractSoilHydrology, args...)
```

## Kernel functions

```@docs; canonical = false
ground_evapotranspiration_flux
```

```@docs; canonical = false
compute_surface_humidity_flux
```

```@docs; canonical = false
compute_surface_humidity_fluxes
```

```@docs; canonical = false
compute_evapotranspiration_conductances!
```

```@docs; canonical = false
compute_evapotranspiration_fluxes!
```
