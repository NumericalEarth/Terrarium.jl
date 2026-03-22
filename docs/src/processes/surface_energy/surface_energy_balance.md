# Surface energy balance

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Energy balance fundamentals

The surface energy balance describes how solar radiation, thermal radiation, and heat fluxes interact at the land-atmosphere interface. The balance states that net radiation must be partitioned among sensible heat, latent heat (evaporation), and ground heat conduction:

```math
\begin{equation}
R_{\text{net}} = H_s + H_l + G
\end{equation}
```

where all fluxes are **positive upward** (away from surface, toward atmosphere):
- $R_{\text{net}}$ is the net radiation (W/m², positive upward)
- $H_s$ is the sensible heat flux (W/m², positive upward)
- $H_l$ is the latent heat flux (W/m², positive upward)
- $G$ is the ground heat flux (W/m², positive downward into soil, negative in upward-positive convention)

The surface energy balance couples radiative processes (shortwave and longwave), atmospheric turbulence (aerodynamic exchange), and surface properties (albedo, emissivity, thermal conductivity).

### Radiative energy budget

The net radiation is determined by shortwave and longwave components. With the upward-positive convention, net radiation is the sum of upwelling and downwelling fluxes:

```math
\begin{equation}
R_{\text{net}} = S_{\uparrow} - S_{\downarrow} + L_{\uparrow} - L_{\downarrow}
\end{equation}
```

where:
- $S_{\uparrow} = \alpha S_{\downarrow}$ is reflected (upwelling) shortwave radiation
- $S_{\downarrow}$ is incident (downwelling) shortwave radiation (W/m²)
- $L_{\uparrow} = \epsilon \sigma T_0^4 + (1-\epsilon) L_{\downarrow}$ is total upwelling longwave radiation
- $L_{\downarrow}$ is incident (downwelling) longwave radiation (W/m²)
- $\alpha$ is surface albedo (fraction of shortwave radiation reflected)
- $\epsilon$ is surface emissivity (fraction of thermal radiation emitted)
- $\sigma$ is the Stefan-Boltzmann constant
- $T_0$ is the skin temperature

This can be equivalently written as:

```math
\begin{equation}
R_{\text{net}} = -(1 - \alpha) S_{\downarrow} + \epsilon \sigma T_0^4 - \epsilon L_{\downarrow}
\end{equation}
```

### Turbulent fluxes

Sensible and latent heat fluxes are driven by temperature and humidity gradients between the surface and atmosphere, quantified through bulk aerodynamic approaches. These fluxes depend on wind speed, atmospheric stability, surface roughness, and the availability of soil moisture.

### Skin temperature determination

The skin temperature $T_0$ is the effective radiative temperature of the land surface. For an implicit approach, $T_0$ self-consistently satisfies the energy balance at each time step. For a prescribed approach, $T_0$ is given as input. The ground heat flux at the surface is derived either directly or as a residual from the energy balance.

## Abstract types

```@docs; canonical = false
AbstractSurfaceEnergyBalance
```

## Concrete types

### Surface Energy Balance

```@docs; canonical = false
SurfaceEnergyBalance
```

### Supporting components

- **Skin temperature**: See [Skin Temperature and Ground Heat](skin_temperature.md)
- **Radiative fluxes**: See [Radiative Fluxes](radiative_fluxes.md)
- **Turbulent fluxes**: See [Turbulent Fluxes](turbulent_fluxes.md)
- **Albedo**: See [Albedo and Emissivity](albedo.md)

## Methods

```@docs; canonical = false
compute_surface_energy_fluxes!
```
