# [Surface energy balance](@id surface_energy_balance_docs)

```@meta
CurrentModule = Terrarium
```

```@setup seb
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

The surface energy balance (SEB) describes how solar radiation, thermal radiation, and heat fluxes interact at the interface between the land and the atmosphere. The SEB broadly consists of four key components: the net radiation budget $R_\text{net}$ (W/m²), sensible heat flux $H_s$ (W/m²), latent heat flux $H_l$ (W/m²), and ground heat flux $G$ (W/m²). The energy balance can be expressed as a simple sum of each flux term:

```math
\begin{equation}
R_{\text{net}} + H_s + H_l - G = 0\,.
\end{equation}
```
Following the standard convention of Terrarium and Oceananigans, all surface energy fluxes are defined **positive upward**. The negative sign in front of $G$ reflects that heat flows *towards* the surface from the uppermost ground layer.

The [`SurfaceEnergyBalance`](@ref) process is responsible for computing all of the above flux terms and thus closing the energy balance between the atmosphere and land surface. Implementations of [`AbstractSurfaceEnergyBalance`](@ref) should generally include, at minimum, representations of each of the four SEB components:
- An implementation of [`AbstractSkinTemperature`](@ref) that defines and updates both the skin temperature $T_s$ and the ground heat flux $G$. The skin temperature $T_s$ is the effective radiative temperature of the land surface. For an implicit approach, $T_s$ self-consistently satisfies the energy balance at each time step. For a prescribed approach, $T_s$ is given as input. The ground heat flux at the surface is derived either directly or as a residual from the energy balance. See [Skin temperature and ground heat flux](@ref) for further details.
- An implementation of [`AbstractRadiativeFluxes`](@ref) that compute the partitioning of the radiation budget. The radiative energy budge $R_{\text{net}}$ is the sum of all incoming and outgoing radiative fluxes at the interface between the atmosphere and the land surface. This typically consists of both  *shortwave* and *longwave* radiation bands. See [Radiative fluxes](@ref) for further details.
- An implementation of [`AbstractTurbulentFluxes`](@ref) that compute the turbulent (sensible and latent) heat fluxes. Sensible and latent heat fluxes are driven by temperature and humidity gradients between the surface and atmosphere, quantified through bulk aerodynamic approaches. These fluxes depend on wind speed, atmospheric stability, surface roughness, and the availability of soil moisture. See [Turbulent fluxes](@ref) for further details.
- A scheme for representing the [albedo](@ref "Albedo and emissivity") in the [Radiative energy budget](@ref "Radiative fluxes").

```@docs; canonical = false
SurfaceEnergyBalance
```

```@example seb
variables(SurfaceEnergyBalance(Float32))
```

!!! warning "Prescribed energy fluxes"
    `SurfaceEnergyBalance` allows you to mix and match which terms in the SEB are diagnosed vs. prescribed depending on the choice of implementation. While this has the potential to be convenient in cases where data on skin temperature or turbulent heat fluxes is available, it should be noted that this may result in surface energy fluxes that are inconsistent and do not fully satisfy the SEB equation.

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, seb::SurfaceEnergyBalance, constants::PhysicalConstants, atmos::AbstractAtmosphere, hydrology::Optional{AbstractSurfaceHydrology}, args...)
```

## Methods

```@docs; canonical = false
compute_surface_energy_fluxes!
```
