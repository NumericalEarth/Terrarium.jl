# Soil biogeochemistry

```@meta
CurrentModule = Terrarium
```

```@setup soilbgc
using Terrarium
using InteractiveUtils
```

!!! warning "Work in progress"
    Soil biogeochemistry in Terrarium is still under active development. The one-pool soil carbon
    scheme is experimental and its interface may change.

## Overview

Soils store more organic carbon than vegetation and the atmosphere combined. The size of this store is set by the balance between carbon inputs, mainly litterfall from vegetation, and losses through microbial decomposition, which releases carbon to the atmosphere as heterotrophic respiration. Both decomposition and vertical redistribution of carbon depend on the physical state of the soil, in particular its temperature and moisture.

In Terrarium, soil biogeochemistry processes describe the evolution of the soil organic carbon (SOC) density $\rho_{\text{soc}}$ (kg/m³), defined as the mass of organic material per unit bulk volume of soil. The main inputs are the litter carbon flux $I$ (kg/m²/s) at the soil surface and the soil temperature $T$ (°C) from the [Soil energy balance](@ref). The main outputs are the SOC density and the respiration rate $R$ (kg/m³/s).

In one vertical dimension, conservation of SOC can be written as
```math
\begin{equation}
\label{eq:socconservation}
\frac{\partial \rho_{\text{soc}}}{\partial t} = -\frac{\partial}{\partial z}\left(q_{\text{d}} + q_{\text{a}}\right) - R(\rho_{\text{soc}}, T)\,,
\end{equation}
```
where $q_{\text{d}}$ is the diffusive flux (kg/m²/s) representing mixing by soil fauna (bioturbation) and cryoturbation, $q_{\text{a}}$ is an advective flux representing burial or upward transport, and $R$ is the decomposition rate. Litter inputs enter through the flux boundary condition at the soil surface, $q(z_{\text{top}}) = -I$, following the convention that fluxes are positive upward.

### Coupling to soil physics

Soil organic carbon also changes the physical properties of the soil. Organic material has a much higher natural porosity and a lower thermal conductivity than mineral material. Terrarium therefore computes the organic fraction of the soil matrix from the SOC density,
```math
\begin{equation}
f_{\text{org}} = \frac{\rho_{\text{soc}}}{(1 - \phi_{\text{org}})\,\rho_{\text{org}}}\,,
\end{equation}
```
where $\rho_{\text{org}}$ (kg/m³) is the density of pure organic material and $\phi_{\text{org}}$ is the natural porosity of organic soil. The denominator is the bulk density of a soil made entirely of organic material, so $f_{\text{org}}$ is dimensionless. The bulk porosity is then a mixture of the mineral and organic porosities,
```math
\begin{equation}
\phi = (1 - f_{\text{org}})\,\phi_{\text{min}} + f_{\text{org}}\,\phi_{\text{org}}\,,
\end{equation}
```
which in turn determines the heat capacity, thermal conductivity, and hydraulic properties of the soil (see [Soil stratigraphy](@ref)). This relation is only physically meaningful for $0 \leq f_{\text{org}} \leq 1$, i.e. for $\rho_{\text{soc}} \leq (1 - \phi_{\text{org}})\,\rho_{\text{org}}$, which is about 130 kg/m³ with the default parameters.

!!! note "Carbon versus organic matter"
    $\rho_{\text{soc}}$ is used as the density of organic *material* when computing $f_{\text{org}}$. Organic matter is only about 50–58% carbon by mass, so if $\rho_{\text{soc}}$ is interpreted as a carbon density, the organic fraction is underestimated by roughly a factor of two.

All implementations should extend [`AbstractSoilBiogeochemistry`](@ref) and provide a method for [`density_soc`](@ref), which the other soil processes use to compute the soil composition.

```@docs; canonical = false
AbstractSoilBiogeochemistry
```

```@example soilbgc
subtypes(Terrarium.AbstractSoilBiogeochemistry)
```

## Implementations

### Constant soil carbon density

The simplest implementation prescribes a constant SOC density in all soil layers. It has no state variables and no tendencies and only affects the soil through the organic fraction described above. This is the default biogeochemistry scheme of [`SoilEnergyWaterCarbon`](@ref), with $\rho_{\text{soc}} = 0$ (purely mineral soil).

```@docs; canonical = false
ConstantSoilCarbonDensity
```

### One-pool soil carbon

The one-pool scheme treats all soil organic carbon as a single pool with prognostic density $\rho_{\text{soc}}$ in each soil layer, which evolves according to ``\eqref{eq:socconservation}``.

**Decomposition.** Decomposition follows first-order kinetics with a Q10 temperature sensitivity,
```math
\begin{equation}
R = k_{\text{ref}}\, Q_{10}^{(T - T_{\text{ref}})/10}\, \rho_{\text{soc}}\,,
\end{equation}
```
where $k_{\text{ref}}$ (1/s) is the decomposition rate at the reference temperature $T_{\text{ref}}$ (°C) and $Q_{10}$ is the factor by which the rate increases for a 10 °C warming. The defaults are $k_{\text{ref}} = 0.1$ per year, $Q_{10} = 2$, and $T_{\text{ref}} = 10$ °C. The respiration rate is stored as the auxiliary variable `respiration_rate`. Moisture limitation of decomposition is not yet considered.

```@docs; canonical = false
SoilCarbonRespiration
```

**Transport.** Vertical transport consists of Fickian diffusion with constant diffusivity $D_b$ (m²/s) and advection with constant velocity $\omega$ (m/s),
```math
\begin{equation}
q_{\text{d}} = -D_b \frac{\partial \rho_{\text{soc}}}{\partial z}\,, \qquad q_{\text{a}} = \omega\, \rho_{\text{soc}}\,.
\end{equation}
```
The defaults are $D_b = 1$ cm²/yr and $\omega = 0.002$ mm/yr.

```@docs; canonical = false
SoilCarbonTransport
```

**Litter input.** Litterfall enters the uppermost soil layer through a flux boundary condition on `density_soc`, which reads the `litter` input variable (kg/m²/s). The bottom boundary has no flux. This boundary condition must be passed explicitly when initializing the model; see [`LitterfallFlux`](@ref) below.

```@docs; canonical = false
SinglePoolSoilCarbon
```

The following example sets up a soil model with one-pool soil carbon and a constant litter input of 0.5 kg/m²/yr:

```@example soilbgc
grid = ColumnGrid(CPU(), Float64, UniformSpacing(N = 10))
biogeochem = SinglePoolSoilCarbon(eltype(grid))
soil = SoilEnergyWaterCarbon(eltype(grid); biogeochem)
model = SoilModel(grid; soil)

seconds_per_year = 365 * 24 * 3600.0
litter = FieldTimeSeries(grid, XY(), [0.0, 10 * seconds_per_year])
litter.data .= 0.5 / seconds_per_year
inputs = InputSources(InputSource(grid, litter; units = u"kg/m^2/s", name = :litter))

initializers = (; density_soc = 10.0, temperature = 5.0, saturation_water_ice = 1.0)
boundary_conditions = Terrarium.LitterfallFlux(biogeochem)
integrator = initialize(model; inputs, initializers, boundary_conditions)
timestep!(integrator)
integrator.state.density_soc
```

## [Process interface](@id soilbgc.dispatches)

Dispatches for `SinglePoolSoilCarbon`:
```@docs; canonical = false
compute_auxiliary!(state, grid, soc::SinglePoolSoilCarbon, soil::AbstractSoil, args...)

compute_boundary_conditions!(state, grid, ::SinglePoolSoilCarbon)

compute_tendencies!(state, grid, soc::SinglePoolSoilCarbon, soil::AbstractSoil, args...)
```

## Boundary conditions

```@docs; canonical = false
LitterfallFlux
```

## Methods

```@docs; canonical = false
density_pure_soc
```

```@docs; canonical = false
compute_respiration_rate
```

## Kernel functions

```@docs; canonical = false
density_soc
```

```@docs; canonical = false
compute_respiration!
```

```@docs; canonical = false
compute_soc_tendency!
```

```@docs; canonical = false
compute_soc_tendency
```

```@docs; canonical = false
compute_soc_diffusive_flux
```

```@docs; canonical = false
compute_soc_advective_flux
```

```@docs; canonical = false
litterfall_bc
```
