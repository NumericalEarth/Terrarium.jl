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
- An implementation of [`AbstractSkinTemperature`](@ref) that defines and updates the skin temperature $T_s$, the effective radiative temperature of the land surface. For an implicit approach, $T_s$ self-consistently satisfies the energy balance at each time step. For a prescribed approach, $T_s$ is given as input. See [Skin temperature](@ref "Skin temperature") for further details.
- An implementation of [`AbstractGroundHeatFlux`](@ref) that determines the ground heat flux $G$, either as the residual that closes the energy balance or as an externally supplied input. See [Ground heat flux](@ref ground_heat_flux_docs) for further details.
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

## Supported configurations

Every flux group is an exchangeable sub-process, so the coupling designs are *configurations* rather than distinct types. The combinations of skin temperature and ground heat flux differ in their solve sequencing, and not every pairing is well posed:

| Skin temperature | Ground heat flux | Meaning |
|---|---|---|
| [`ImplicitSkinTemperature`](@ref) | [`DiagnosedGroundHeatFlux`](@ref) | Standalone Terrarium (the default). |
| [`PrescribedSkinTemperature`](@ref) | [`DiagnosedGroundHeatFlux`](@ref) | Coupled hybrid: $G$ as the residual evaluated at the coupler's $T_s$. |
| [`PrescribedSkinTemperature`](@ref) | [`PrescribedGroundHeatFlux`](@ref) | Fully prescribed; see [`PrescribedSurfaceEnergyBalance`](@ref) below. |
| [`ImplicitSkinTemperature`](@ref) | [`PrescribedGroundHeatFlux`](@ref) | Back out $T_s$ from a supplied $G$ by inverting the conduction relation. |

```@docs; canonical = false
PrescribedSurfaceEnergyBalance
PrescribedSurfaceEnergyBalance(::Type{NF}) where {NF}
```

```@example seb
variables(PrescribedSurfaceEnergyBalance(Float32))
```

!!! note "Coupling: deferring the SEB to an external model"
    A [`PrescribedSurfaceEnergyBalance`](@ref) lets an external coupler (e.g. an atmosphere model computing turbulent fluxes via Monin-Obukhov similarity theory) own the entire atmosphere-land interface: the skin temperature, the radiative and turbulent fluxes, *and* the ground heat flux are all supplied as inputs, and Terrarium does not re-close the surface energy budget. The coupler-assembled $G$ routes straight to the soil column's Neumann top boundary condition. A partially prescribed configuration ([`PrescribedSkinTemperature`](@ref) + [`PrescribedTurbulentFluxes`](@ref) with [`DiagnosedRadiativeFluxes`](@ref) and [`DiagnosedGroundHeatFlux`](@ref)) instead leaves Terrarium to close $G = R_\text{net} + H_s + H_l$ itself.

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, seb::SurfaceEnergyBalance, vegetation::Optional{AbstractVegetation}, snow::Optional{AbstractSnow}, args...)
```

## Methods

The `SurfaceEnergyBalance` process provides two primary method interfaces:

```@docs; canonical = false
solve_surface_energy_balance!(state, grid, seb::SurfaceEnergyBalance{NF}, constants::PhysicalConstants, atmos::AbstractAtmosphere, hydrology::Optional{AbstractSurfaceHydrology}, snow::Optional{AbstractSnow}, args...) where {NF}
```

```@docs; canonical = false
compute_surface_energy_fluxes!(state, grid, seb::SurfaceEnergyBalance, constants::PhysicalConstants, atmos::AbstractAtmosphere, hydrology::Optional{AbstractSurfaceHydrology}, args...)
```

The fused kernel evaluates each flux group through a per-process *mutating* variant, so a diagnosed process computes and stores its fluxes while a prescribed one (whose fluxes are supplied as input fields) is a no-op: [`compute_radiative_fluxes!`](@ref), [`compute_turbulent_fluxes!`](@ref), and [`compute_ground_heat_flux!`](@ref). The ground heat flux is always evaluated last, since it depends on the other three flux terms.
