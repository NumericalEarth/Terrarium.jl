# Skin temperature and ground heat flux

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Skin temperature definition

The skin temperature (or surface temperature) $T_0$ is the effective radiative temperature of the land surface. It represents the temperature at which the surface emits longwave radiation and is the temperature that directly controls evaporation and sensible heat flux in the surface energy balance.

Skin temperature differs from the air temperature at height $z$ by the effects of turbulent mixing and surface properties. It is not necessarily the same as subsurface soil temperature, $T_1$ (the temperature of the top soil layer), due to the thermal resistance between surface and soil.

### Implicit skin temperature

The implicit approach solves for skin temperature that simultaneously satisfies the surface energy balance:

```math
\begin{equation}
R_{\text{net}}(T_0) = H_s(T_0) + H_l(T_0) + G(T_0)
\end{equation}
```

where all fluxes are **positive upward** (away from the surface):
- $R_{\text{net}}$ is the net radiation budget [W/m²], positive upward
- $H_s$ is the sensible heat flux [W/m²], positive when surface is warmer than air
- $H_l$ is the latent heat flux [W/m²], positive during evaporation
- $G$ is the ground heat flux [W/m²], computed as the residual of the energy balance (negative when heat flows into soil)

This requires solving a nonlinear equation at each grid point and time step. The implicit approach is more physically consistent but computationally more intensive.

### Ground heat flux

The ground heat flux $G$ represents heat flow between the surface and the soil below. With the upward-positive convention, $G$ is computed from the energy balance residual: $$G = R_{\text{net}} - H_s - H_l$$

where:
- $G > 0$ means net energy is directed away from surface (unusual; would require energy input that exceeds radiation)
- $G < 0$ means energy flows INTO the soil (typical during daytime heating; ground absorbs surplus surface energy after turbulent fluxes are accounted for)

The magnitude of $G$ is determined by:
1. **Thermal properties**: Soil thermal conductivity $\kappa_s$ and heat capacity
2. **Temperature gradient**: Difference between surface and subsurface
3. **Residual energy**: After radiation and turbulent fluxes are accounted for via $G = R_{\text{net}} - H_s - H_l$

### Thermal properties

The surface thermal conductivity $\kappa_s$ represents the effective heat transfer between the skin and the first subsurface layer. This may be:
- A constant parameter (simplest)
- Derived from soil type and moisture (more realistic)
- Include vegetation insulation effects (complex)

## Abstract types

```@docs; canonical = false
AbstractSkinTemperature
```

## Concrete types

```@docs; canonical = false
PrescribedSkinTemperature
```

```@docs; canonical = false
ImplicitSkinTemperature
```

## Methods

```@docs; canonical = false
compute_skin_temperature
```

```@docs; canonical = false
compute_ground_heat_flux
```
