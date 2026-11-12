# Soil hydrology

```@meta
CurrentModule = Terrarium
```

```@setup soilwater
using Terrarium
using InteractiveUtils
```

## Overview

Soil hydrology processes characterize the dynamics of ground water in both saturated and unsaturated soil. It defines parameters and methods needed to compute water fluxes between layers and grid cells within the soil domain. Implementations should extend [`AbstractSoilHydrology`](@ref) and should generally consist of at least four components:
- A scheme for computing vertical water fluxes between soil layers
- A closure parameterization linking soil saturation and pressure head
- A parameterization for [`soil hydraulic properties`](@ref "Hydraulic properties")
- A forcing term representing user-defined, internal sources/sinks in the soil domain (not including evapotranspiration)

```@docs; canonical = false
AbstractSoilHydrology
```

```@example soilwater
subtypes(Terrarium.AbstractSoilHydrology)
```

Terrarium currently provides a single general implementation of `SoilHydrology` following the above interface:

```@docs; canonical = false
SoilHydrology
```

### [Process interface](@id soilhydrology.interface)

Dispatches for `NoFlow`:
```@docs; canonical = false
initialize!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)

compute_auxiliary!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)

compute_tendencies!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)
```

Dispatches for `RichardsEq`:
```@docs; canonical = false
initialize!(
    state, grid,
    hydrology::SoilHydrology{NF, RichardsEq},
    soil::AbstractSoil,
    constants::PhysicalConstants,
    args...
) where {NF}

compute_auxiliary!(
    state, grid,
    hydrology::SoilHydrology{NF, RichardsEq},
    soil::AbstractSoil,
    args...
) where {NF}

compute_tendencies!(
    state, grid,
    hydrology::SoilHydrology{NF, RichardsEq},
    soil::AbstractSoil,
    constants::PhysicalConstants,
    evtr::Optional{AbstractEvapotranspiration},
    timestepping::Timestepping
) where {NF}
```

## Vertical flow

### Static soil hydrology ("No Flow")

The simplest possible soil hydrology scheme is one in which the soil saturation state remains constant over time. This can be appropriate for simulations in regions where the soil is normally waterlogged or where changes in the hydrological state can be otherwise assumed to play a negligible role. 

```@docs; canonical = false
NoFlow
```

```@example soilwater
variables(SoilHydrology(Float32, NoFlow()))
```

### Richardson-Richards equation for variably saturated flow

The vertical flow of water in porous media, such as soils, can be formulated as following the conservation law
```math
    \phi\frac{\partial \xi(\psi)}{\partial t} = - \boldsymbol{\nabla} \cdot \textbf{j}_{\text{w}} + F_{\text{w}}(z,t),
```
where (as defined in [Soil stratigraphy](@ref)) $\phi$ is the natural porosity (or saturated water content) of the soil volume and $\xi \in [0,1]$ the saturation of pore water/ice. $F_{\text{w}}(z,t)$ is an inhomogeneous source/sink (forcing) term (1/s) and $\textbf{j}_{\text{w}}$ the water flux vector (m/s). 

Vertical fluxes in the soil column can be represented by combining gravity-driven advection with Darcy's law
```math
\begin{equation}
\textbf{j}_{\text{w}}^{\mathrm{v}} =\textbf{j}_{\text{w}} \cdot \mathbf{n}_z = -\kappa_{\text{w}}\frac{\partial \left(\psi + z\right)}{\partial z},
\end{equation}
```
where $\psi$ is the matric potential (m), $\mathbf{n}_z$ the normal vector perpendicular to the surface and $\kappa_{\text{w}}$ the hydraulic conductivity (m/s). Given the positive upwards convention (see [Numerical core](@ref)), the $z-$axis is thus defined positive away from the surface. Substituting this equation into the aforementioned conservation law yields the widely known Richardson-Richards equation for variably saturated flow in porous media [richardsCapillaryConductionLiquids1931](@cite). When the divergence of the water flux vector ($\boldsymbol{\nabla} \cdot \textbf{j}_{\text{w}}$) is simplified to only consider vertical water fluxes, this gives:
```math
\begin{equation}
\phi\frac{\partial \xi(\psi)}{\partial t} = \frac{\partial}{\partial z}\left[\kappa_w \frac{\partial \left(\psi + z\right)}{\partial z}\right] + F_{\text{w}}(z,t)
\end{equation}
```
which is called the mixed-form equation, as the rate of change on the left-hand side is in $\xi$ but the vertical gradient on the right-hand side is in $\psi$ [bonanClimateChangeTEMSoilMoisture2019](@cite). 

```@docs; canonical = false
RichardsEq
```

```@example soilwater
variables(SoilHydrology(Float32, RichardsEq()))
```

### [Implicit Richards equation](@id soil-hydrology-implicit-richards)

The `Timestepping` trait argument of [`compute_tendencies!`](@ref soilhydrology.interface) for `RichardsEq` is the resolved [`timestepping`](@ref) class of `saturation_water_ice` (see [Implicit-explicit time stepping](@ref)). It is [`Explicit`](@ref) unless the model is run with an [`IMEX`](@ref) timestepper, in which case [`SoilModel`](@ref) and [`LandModel`](@ref) route the saturation of a Richards-equation hydrology to the implicit sub-stepper:

```@docs; canonical = false
default_timestepping
```

- Under `Explicit()`, the tendency contains the full divergence of the [`darcy_flux`](@ref), the evapotranspiration sink, the user-defined forcing, and the flux boundary conditions on `saturation_water_ice` (e.g. [`InfiltrationFlux`](@ref)).
- Under `Implicit()`, the Darcy fluxes through the *interior* cell faces are omitted, since they are integrated by the implicit sub-stepper. Only the Darcy fluxes through the two boundary faces of the column remain, via [`compute_boundary_darcy_flux_divergence`](@ref). These vanish for the default (no-flux) conditions on `pressure_head` and carry the prescribed flux for a gradient condition on `pressure_head` (e.g. [`FreeDrainage`](@ref)).

In both cases, the Darcy fluxes at ``t^n`` through all cell faces are stored in the `water_flux` auxiliary variable (m/s, positive upward, on cell faces; see [`compute_water_flux`](@ref)). Under `Implicit()`, the implicit step overwrites the fluxes through the interior faces with those of its conservative update ([`compute_implicit_water_flux`](@ref)). After a step, `water_flux` therefore holds the flux that moved the water, with which the soil energy advects heat (see [Soil energy balance](@ref)).

The hydrology closure redistributes over- and undersaturation with [`adjust_saturation_profile!`](@ref). It moves the sensible heat ``C_\text{w} T`` of the water it shifts between cells in the `internal_energy`, at the temperature of the cell that the water leaves, and removes that of the excess water it discards (standalone [`SoilModel`](@ref)) or routes to the `surface_excess_water` pool ([`LandModel`](@ref)), which stores no heat. The energy closure, which runs after the hydrology closure, then recomputes the temperature from the adjusted energy.

The implicit step solves the mixed form of the Richards equation (see [Richardson-Richards equation for variably saturated flow](@ref)) with the mass-conservative modified Picard iteration of [CeliaEtAl1990](@cite), using the lagged face conductivity of the Darcy flux and the lagged specific moisture capacity given by [`compute_implicit_conductivity!`](@ref) and [`compute_implicit_capacity!`](@ref). With porosity ``\phi``, saturation ``\xi`` (so that the volumetric water content is ``\theta = \phi \xi``), and total hydraulic head ``\Psi = \psi_m(\theta) + \psi_z + \psi_h`` (matric, elevation, and hydrostatic head), the backward Euler step reads
```math
\phi\, \frac{\xi^{n+1} - \xi^n}{\Delta t} = \frac{\partial}{\partial z}\left[K \frac{\partial \Psi^{n+1}}{\partial z}\right] + \phi\, G_\text{exp}\,,
```
where ``G_\text{exp}`` (1/s) is the explicit tendency of the saturation ``\xi``; it is multiplied by ``\phi`` because the equation is written for the volumetric water content ``\theta = \phi \xi``, the quantity whose flux divergence is the right-hand side. ``G_\text{exp}`` contains the flux boundary conditions on the saturation (e.g. [`InfiltrationFlux`](@ref)), the Darcy fluxes through boundary faces implied by gradient conditions on the pressure head, evapotranspiration, and the user-defined forcing. The elevation head and the hydrostatic head below the water table are held fixed within the step, so that ``\partial\theta/\partial\Psi = C``, the specific moisture capacity. Linearizing the water content about the latest iterate ``m``, ``\phi \xi^{m+1} \approx \phi \xi^m + C^m (\Psi^{m+1} - \Psi^m)``, with the hydraulic conductivity ``K^m`` of the Darcy flux lagged at the same iterate, gives the tridiagonal system
```math
C^m\, \Psi^{m+1} - \Delta t\, \frac{\partial}{\partial z}\left[K^m \frac{\partial \Psi^{m+1}}{\partial z}\right] = C^m\, \Psi^m - \phi\,(\xi^m - \xi^+)\,, \qquad \xi^+ = \xi^n + \Delta t\, G_\text{exp}\,,
```
for the total head, starting from ``\xi^0 = \xi^n``. As for heat conduction (see [Implicit vertical heat conduction](@ref)), the saturation is then updated conservatively from the implicit fluxes through the interior faces,
```math
\xi^{m+1} = \xi^+ + \frac{\Delta t}{\phi}\, \frac{\partial}{\partial z}\left[K^m \frac{\partial \Psi^{m+1}}{\partial z}\right],
```
and the hydrology closure is applied after every iteration: it redistributes over- and undersaturation (see [`adjust_saturation_profile!`](@ref)), updates the water table, and recomputes ``\Psi^{m+1}`` from ``\xi^{m+1}``. Excess water that reaches a saturated surface is routed into the `surface_excess_water` pool of a [`LandModel`](@ref)'s surface runoff only by the closure of the last iteration (and is discarded by a standalone [`SoilModel`](@ref)), since every iteration restarts from ``\xi^+``. For the same reason, only the closure of the last iteration moves the heat of the redistributed water (see [Coupling of water and energy](@ref imex-water-energy-coupling)), so the internal energy is adjusted exactly once per step. The column water content therefore changes by exactly ``\Delta t`` times the boundary fluxes and sources in ``G_\text{exp}``, minus the excess routed to the surface pool, for any number of iterations.

In saturated cells ``C = 0``, and the diagonal of the system is carried by the conductivity terms ``\Delta t K / \Delta z^2``. Where the conductivity vanishes as well (frozen saturated cells), the system would be singular, so ``C`` is bounded from below by the small `minimum_moisture_capacity` (1/m, default ``10^{-5}``) of the hydraulic properties (see [`ConstantSoilHydraulics`](@ref)), which acts like a small specific storage within the iteration but does not affect mass conservation. [CeliaEtAl1990](@cite) recommend at least two iterations, which is the default of [`Picard`](@ref).

Flux boundary conditions on `saturation_water_ice` and gradient boundary conditions on `pressure_head` are supported; value boundary conditions on `pressure_head` are not yet supported.

The terms that stay explicit can remove more water from a cell in one step than it holds, e.g. gravity drainage through the bottom face of a draining sandy column. [`water_outflow_timescale`](@ref) reports the time in which they would drain a cell to its residual saturation, which bounds the step of a `TimeStepWizard` through `cell_advection_timescale` (see [Adaptive time stepping with IMEX](@ref)). It evaluates the explicit terms selected by the resolved class of `saturation_water_ice`, so under `Explicit()` it includes the interior Darcy fluxes as well.

The implicit operator of `saturation_water_ice` is a [`TridiagonalOperator`](@ref) in the total hydraulic head `pressure_head`, the variable of the [`SoilSaturationPressureClosure`](@ref) (see its [`varname`](@ref) method below), returned by [`implicit_operator`](@ref) (which also validates the configuration), and each iteration of the implicit solver calls the process hooks of `SoilHydrology` with `RichardsEq` below:

```@docs; canonical = false
implicit_operator(hydrology::SoilHydrology{NF, RichardsEq}, state, grid) where {NF}
linearize!(operator::TridiagonalOperator, rhs, state, grid, hydrology::SoilHydrology{NF, RichardsEq}, soil::AbstractSoil, constants::PhysicalConstants, explicit_saturation, Δt) where {NF}
implicit_update!(state, grid, hydrology::SoilHydrology{NF, RichardsEq}, soil::AbstractSoil, operator::TridiagonalOperator, explicit_saturation, Δt) where {NF}
implicit_closure!(state, grid, hydrology::SoilHydrology{NF, RichardsEq}, soil::AbstractSoil, constants::PhysicalConstants, runoff::Optional{AbstractSurfaceRunoff}, ::Iteration{name, target, true}) where {NF, name, target}
implicit_closure!(state, grid, hydrology::SoilHydrology{NF, RichardsEq}, soil::AbstractSoil, constants::PhysicalConstants, runoff::Optional{AbstractSurfaceRunoff}, ::Iteration{name, target, false}) where {NF, name, target}
compute_saturation_closure!
```

## Hydraulic properties

!!! warning "Match the conductivity to the retention curve"
    The unsaturated hydraulic conductivity and the soil water retention curve should come from the same family.
    By default, [`ConstantSoilHydraulics`](@ref) and [`SoilHydraulicsSURFEX`](@ref) select the conductivity that matches the retention curve (see [`Terrarium.default_unsat_hydraulic_cond`](@ref)).
    In particular, [`UnsatKLinear`](@ref) is not suitable for dry soil with a Brooks–Corey retention curve: the matric potential grows as a power law of the inverse saturation while the conductivity falls only linearly, so that the suction fluxes in dry soil become unphysically large.

Besides the static soil characteristics mentioned in [Soil stratigraphy](@ref) such as porosity ($\phi$), a key time-varying hydraulic property is the (non-saturated) hydraulic conductivity $\kappa_{\text{w}}$. Different ways of modelling $\kappa_{\text{w}}$ are defined as subyptes of [`AbstractUnsatK`](@ref). The simplest formulation, called [`UnsatKLinear`](@ref), models $\kappa_{\text{w}}$ as:
```math
\begin{equation}
\kappa_{\text{w}} = \frac{\theta_{\text{liq}}}{\theta + \theta_{\text{air}}} \kappa_{\text{w,sat}}\,
\end{equation}
```
with $\kappa_{\text{w,sat}}$ the saturated hydraulic conductivity (m/s).  A more complex formulation taking ice into account, called [`UnsatKVanGenuchten`](@ref),follows [vangenuchtenHydraulicConductivity1980](@cite), here presented in the form of [westermannCryoGridCommunityModel2023; Eq. (26)](@cite):
```math
\begin{equation}
\kappa_{\text{w}} = I_\text{ice}\left(\frac{\theta_{\text{liq}}}{\phi}\right)^{0.5} \left[1 - \left(1 - \left(\frac{\theta_{\text{liq}}}{\phi}\right)^{n/(n+1)}\right)^{(n-1)/n}\right]^2 \kappa_{\text{w,sat}}\,,
\end{equation}
```
which adds an ice impedance factor $I_\text{ice} = 10^{-\Omega\theta_{\text{ice}}/\theta}$ to the classic van Genuchten formulation to account for the blocking of water-filled pores by ice. 

```@docs; canonical = false
UnsatKLinear
```

```@docs; canonical = false
UnsatKVanGenuchten
```

The conductivity of the Brooks–Corey model, called [`UnsatKBrooksCorey`](@ref), is a power law of the liquid water saturation, with the same ice impedance factor:
```math
\begin{equation}
\kappa_{\text{w}} = I_\text{ice}\left(\frac{\theta_{\text{liq}}}{\phi}\right)^{3 + 2/\lambda} \kappa_{\text{w,sat}}\,,
\end{equation}
```
where $\lambda$ is the pore size distribution index of the retention curve.
[brooksHydraulicPropertiesPorous1964](@cite) derived the exponent from their retention curve using the pore size distribution model of Burdine; [campbellSimpleMethodDetermining1974](@cite) obtained the same law, written with $b = 1/\lambda$ as $(\theta_{\text{liq}}/\phi)^{2b + 3}$, and [clappEmpiricalEquationsSoil1978](@cite) tabulated $b$ by soil texture.
In this form, it is the conductivity used in SURFEX/ISBA [booneInfluenceInclusionSoil2000](@cite).

```@docs; canonical = false
UnsatKBrooksCorey
```

```@docs; canonical = false
Terrarium.default_unsat_hydraulic_cond
```

Values for $\kappa_{\text{w,sat}}$, wilting point ($\theta_{wp}$), and field capacity   ($\theta_{fc}$) can be set as constants (see [`ConstantSoilHydraulics`](@ref)) or can be calculated from soil texture using pedotransfer functions (PTFs). Currently, PTFs based on [noilhanISBA1996](@cite) for $\theta_{wp}$ and $\theta_{fc}$ are available in [`SoilHydraulicsSURFEX`](@ref).

```@docs; canonical = false
ConstantSoilHydraulics
```

```@docs; canonical = false
SoilHydraulicsSURFEX
```

## Closures

Critical to the modelling of water transport in the soil, is the relationship between soil water/ice saturation ($\xi$) and the matric potential ($\Psi_m$), which is called the soil-water retention curve (SWRC). Recall from [the documentation on the core interfaces](@ref core_interfaces_hydrology_closure) that the total hydraulic head $\Psi$  is defined as:
```math
\begin{equation}
\Psi = \Psi_m(\xi) + \Psi_z + \Psi_h
\end{equation}
```
with $\Psi_z$ the elevation head and $\Psi_h$ the hydrostatic head contributed by free water above the water table. Currently, two implementations of the SWRC are available in Terrarium via [FreezeCurves.jl](https://github.com/CryoGrid/FreezeCurves.jl): the Brooks-Corey ([`BrooksCorey`](@extref FreezeCurves.BrooksCorey)) and van Genuchten ([`VanGenuchten`](@extref FreezeCurves.VanGenuchten)) formulations. 

```@docs; canonical = false
SoilSaturationPressureClosure
varname(::SoilSaturationPressureClosure)
```

## Methods

```@docs; canonical = false
get_hydraulic_properties
```

```@docs; canonical = false
get_swrc
```

```@docs; canonical = false
get_closure
```

```@docs; canonical = false
compute_water_table!
```

```@docs; canonical = false
adjust_saturation_profile!
```

```@docs; canonical = false
compute_hydraulics!
```

```@docs; canonical = false
minimum_moisture_capacity
```

```@docs; canonical = false
water_outflow_timescale
```

## Kernel functions

```@docs; canonical = false
saturation_water_ice
```

```@docs; canonical = false
hydraulic_conductivity
```

```@docs; canonical = false
Terrarium.ice_impedance_factor
```

```@docs; canonical = false
liquid_water_fraction
```

```@docs; canonical = false
water_table
```

```@docs; canonical = false
surface_excess_water
```

```@docs; canonical = false
compute_volumetric_water_content_tendency
compute_darcy_flux_divergence
compute_boundary_darcy_flux_divergence
darcy_flux
darcy_conductivity
compute_water_flux
redistribute_saturation_profile!
compute_redistributed_heat!
compute_matric_potential
compute_specific_moisture_capacity
specific_moisture_capacity
compute_water_outflow_timescale
compute_saturation_boundary_flux
```

The following kernel functions implement the implicit Richards step (see [Implicit Richards equation](@ref soil-hydrology-implicit-richards)):

```@docs; canonical = false
compute_implicit_conductivity!(face_conductivity, i, j, k, grid, fields, hydrology::SoilHydrology{NF, RichardsEq}) where {NF}
compute_implicit_capacity!(diagonal_coefficient, i, j, k, grid, fields, hydrology::SoilHydrology{NF, RichardsEq}, strat::AbstractStratigraphy, bgc::AbstractSoilBiogeochemistry) where {NF}
compute_implicit_water_flow_coefficients!
compute_implicit_water_flow_rhs
compute_implicit_saturation_update
compute_implicit_water_flux
```

## [References](@id "soilhydrology.refs")

```@bibliography
Pages = ["soil_hydrology.md"]
Canonical = false
```
