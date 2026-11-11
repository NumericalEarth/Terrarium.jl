# Snow model

```@meta
CurrentModule = Terrarium
```

```@setup snowmodel
using Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

[`SnowModel`](@ref) is a minimal standalone model of a dynamic snowpack spatially distributed over the given `grid`. It couples an [`AbstractSnow`](@ref) process (by default [`SingleLayerSnow`](@ref)) with a prescribed atmosphere that supplies precipitation and air temperature. It is intended primarily for unit testing of the snow physics in isolation.

Because the model is standalone, the fluxes that a full land model would obtain from the surface energy
balance and the snow→soil conduction are instead **prescribed input fields**: the net surface heat flux
`snow_surface_heat_flux`, the basal conductive heat flux `snow_basal_heat_flux`, and the `sublimation` rate (see
[Prescribed forcing](@ref snowmodel.forcing) below).

```@example snowmodel
arch = CPU()
grid = ColumnGrid(arch, Float32, ExponentialSpacing(N = 3))
model = SnowModel(grid)
```

```@docs; canonical = false
SnowModel
```

The prognostic and diagnostic state variables of the assembled model are

```@example snowmodel
variables(model)
```

The two prognostic variables are the depth-integrated (column) internal energy ``\bar{U}_\text{snow}``
(J/m², `snow_energy`) and the snow water equivalent ``W_\text{snow}`` (m, `snow_water_equivalent`); their
mass and energy balances are documented on the [Snow mass balance](@ref) and [Snow energy balance](@ref)
pages.

## Components

| Field | Type | Scope | Process page |
|-------|------|-------|---------------|
| `snow` | [`AbstractSnow`](@ref) | Single-layer snowpack mass and energy balance | [Snow](@ref snow_process_docs) |
| `atmosphere` | [`AbstractAtmosphere`](@ref) | Prescribed near-surface forcing (precipitation, air temperature) | [Atmospheric inputs](@ref atmosphere_docs) |

In addition to the physics components, `SnowModel` carries the standard model machinery: `constants`
([`PhysicalConstants`](@ref)), an `initializer`, and a `timestepper`.

### Snow

The `snow` component is the single-layer snowpack process. The default is [`SingleLayerSnow`](@ref), a
lumped layer of constant bulk density loosely based on the Utah Energy Balance model [tarbotonSpatiallyDistributedEnergy1994](@cite),
composed of independently swappable sub-parameterizations for areal cover, density, thermal conductivity, and
hydraulic properties. See the [Snow](@ref snow_process_docs) overview and [Snow parameterizations](@ref) for details.

### Atmosphere

The `atmosphere` component provides the near-surface forcing consumed by the snow tendencies —
snowfall, rainfall, and air temperature — together with the humidity and wind quantities used to
diagnose sublimation. The default is [`PrescribedAtmosphere`](@ref).

## [Prescribed forcing](@id snowmodel.forcing)

Unlike the coupled land model, the standalone `SnowModel` does not solve a surface energy balance or a
snow→soil conduction. The corresponding fluxes are therefore declared as **input** fields and must be
supplied (they default to zero):

| Field | Units | Meaning | Supplied in the coupled model by |
|-------|-------|---------|----------------------------------|
| `snow_surface_heat_flux` | W/m² | Net heat flux at the snow surface (positive upward) | Surface energy balance closure |
| `snow_basal_heat_flux` | W/m² | Conductive heat flux at the snow base (positive upward, soil → snow) | Snow→soil conduction |
| `sublimation` | m/s | Sublimation/evaporation rate from the snow surface (SWE) | Surface energy balance latent flux |

In the [Land model](@ref "Land models") these three inputs become diagnosed couplings: `snow_surface_heat_flux`
and `sublimation` are set by the [surface energy balance](@ref surface_energy_balance_docs), and
`snow_basal_heat_flux` by the snow→soil conductive interface.

## Model interface

`SnowModel` implements the standard model interface: `initialize!` runs the model/field initializers and
then the snow process initializer (the inverse closure ``(T_\text{snow}, W_\text{snow}) \to \bar{U}_\text{snow}``), `compute_auxiliary!` diagnoses the geometric and thermal properties of the snowpack as well as the and energy-temperature closure, and `compute_tendencies!` accumulates the mass and energy tendencies from the prescribed fluxes and atmospheric forcing. `closure!` first enforces the constraints on the snow state (non-negative snow water equivalent, and no energy without snow) and then applies the energy-temperature closure:

```@docs; canonical = false
enforce_snow_constraints!
```
