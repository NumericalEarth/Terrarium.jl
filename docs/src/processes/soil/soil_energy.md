# Soil energy balance

```@meta
CurrentModule = Terrarium
```

```@setup soilenergy
using Terrarium
using InteractiveUtils
```

## Overview

Heat transfer in the subsurface can be represented according to the heat equation, with the upper boundary set to surface temperature and the lower boundary set to a constant positive heat flux representing heat produced by the inner earth [jaegerApplicationTheoryHeat1965,lachenbruchChangingClimateGeothermal1986](@cite). If both the upper and lower boundaries are assumed to be constant over time, the steady-state temperature profile takes the form of a continuous piecewise linear function increasing over depth with the slope determined by the thermal properties of the ground material. The instantaneous temperature field can then be generally represented as
```math
\begin{equation}
T(z,t) = T_0 + \frac{Q_{\text{geo}}}{\kappa_{\text{h}}(z)}z + \Delta T(z,t)
\end{equation}
```
where $T(z,t)$ is the temperature field (K) over depth $z$ (m) and time $t$ (s), $T_0$ is the mean annual GST (K), $Q_{\text{geo}}$ is the geothermal heat flux (W/m²), and $\kappa_{\text{h}}(z)$ is the thermal conductivity which may vary with depth depending on the material (W/m K). The last term $\Delta T(z,t)$ represents transient disturbances to the steady state temperature profile due to both seasonal and long-term fluctuations in the upper and lower boundary conditions of the vertical domain. Simulating the impacts of these transient changes is one of the primary objectives of most numerical permafrost and land surface models.

Diffusive heat flow in a solid medium is governed by Fourier's law,
```math
\begin{equation}
     \mathbf{j}_\text{h}^{\mathrm{v}} = \mathbf{j}_\text{h} \cdot \mathbf{n}_z = -\kappa_{\text{h}}\frac{\partial T}{\partial z}\,,
\end{equation}
```
where $\mathbf{j}_\text{h}$ is the diffusive heat flux vector (W/m²) and $\mathbf{n}_z$ is the upward facing normal vector along the vertical $z$ axis.


### Phase change of pore water/ice

Since ground materials are often porous, i.e., there exists void space between the solid particles, it is necessary to consider the potential presence of water and/or ice in this void space, which is hereafter referred to as pore space, or simply, soil pores. The thermal effects of water and ice can be accounted for by considering not only the temperature of the material but rather the total internal energy of the elementary volume. Combining the diffusive flux with a potential advective heat flux $\mathbf{j}_{\text{h}}^{\text{w}}$ due to water flow yields the energy conservation law,
```math
\begin{equation}
    \label{eq:energyconservation}
\frac{\partial U(T,\theta)}{\partial t} = - \boldsymbol{\nabla} \cdot \left(\mathbf{j}_\text{h} + \mathbf{j}_h^{\text{w}}\right) + F_h(z,t),
\end{equation}
```
where $U(T,\theta)$ (J/m³) is the volumetric internal energy as a function of temperature and total water/ice content $\theta$ (m³/m³), and $F_h(z,t)$ is an inhomogeneous heat source/sink (forcing) term.

The advective heat flux $\mathbf{j}_{\text{h}}^{\text{w}}$ can be represented as,
```math
\begin{equation}
\mathbf{j}_{\text{h}}^{\text{w}} = \left( c_{\text{w}} T + L_{\text{sl}} \right) \mathbf{j}_{\text{w}} \rho_{\text{w}}
\end{equation}
```
where $L_{\text{sl}}$ (J/kg) and $c_{\text{w}}$ (J/(kg K)) represent the specific latent heat of fusion (solid → liquid) and heat capacity of liquid water respectively. Note that $\mathbf{j}_{\mathrm{w}}$ is the water flux vector as defined in [Soil hydrology](@ref). This flux term accounts for the energy transferred by the movement of water within the soil matrix. In model configurations that neglect subsurface water flow, this flux term is implicitly assumed to be zero.

```@docs; canonical = false
SoilThermodynamics
```

As for [Soil hydrology](@ref), only the vertical fluxes are currently considered. This simplifies the notation of ``\eqref{eq:energyconservation}`` to
```math
\begin{equation}
\frac{\partial U(T,\theta)}{\partial t} = \frac{\partial}{\partial z}\left[\kappa_{\text{h}} \frac{\partial T}{\partial z}\right] + F_h(z,t)\,.
\end{equation}
```

## [Process interface](@id soilenergy.dispatches)

```@docs; canonical = false
initialize!(state, grid, energy::SoilThermodynamics, soil::AbstractSoil, constants::PhysicalConstants, args...)

compute_auxiliary!(state, grid, energy::SoilThermodynamics, soil::AbstractSoil, args...)

compute_tendencies!(state, grid, energy::SoilThermodynamics, soil::AbstractSoil, constants::Optional{PhysicalConstants}, timestepping::Timestepping)
```

## Advection of heat by soil water flow

When the soil water flows (a [`RichardsEq`](@ref) hydrology), it carries its heat along, as described by the advective heat flux ``\mathbf{j}_{\text{h}}^{\text{w}}`` above. Since ``U`` is measured relative to liquid water at 0 °C, moving liquid water carries no latent heat, and ice is immobile, so only the sensible heat of the liquid water is advected: the vertical advective heat flux is ``C_{\text{w}}\, T\, j_{\text{w}}``, where ``j_{\text{w}} = \mathbf{j}_{\text{w}} \cdot \mathbf{n}_z`` is the vertical water flux (m/s, positive upward; see [Soil hydrology](@ref)) and ``C_{\text{w}} = \rho_{\text{w}} c_{\text{w}}`` the volumetric heat capacity of liquid water (the `water` heat capacity of [`SoilThermalProperties`](@ref), the same value with which water enters the bulk heat capacity). The energy equation then reads
```math
\frac{\partial U}{\partial t} = \frac{\partial}{\partial z}\left[\kappa_{\text{h}} \frac{\partial T}{\partial z}\right] - \frac{\partial}{\partial z}\left(C_{\text{w}}\, T\, j_{\text{w}}\right) + F_h(z,t)\,.
```
In the discretization, the water flux is evaluated at the cell faces and the temperature of the advective flux at a face is that of the upwind cell (first-order upwind; see [`advective_heat_flux`](@ref)).

The flux ``j_{\text{w}}`` is the `water_flux` auxiliary variable of the hydrology, i.e. the Darcy flux that moved the water in the step: the flux at ``t^n`` under explicit time stepping of the saturation, and the flux of the conservative update of the implicit Richards step otherwise (see [Soil hydrology](@ref)). Heat thus always moves with the water that moves. Under explicit time stepping of `internal_energy`, the advective term is part of the explicit tendency and subject to an advective CFL condition, reported by `cell_advection_timescale` (see [Adaptive time stepping](@ref)). Under implicit time stepping, it is part of the implicit heat operator (see [Implicit vertical heat conduction](@ref)) and has no CFL limit.

Water that crosses the column boundaries through flux boundary conditions on `saturation_water_ice`, such as the infiltration flux, carries the temperature of the cell it enters or leaves, i.e. infiltrating water enters at the temperature of the top soil cell; this heat is added by [`compute_boundary_advected_heat!`](@ref) from exactly the same boundary flux as the water. Water that the hydrology closure redistributes between cells, or removes from an oversaturated surface, carries its sensible heat as well (see [`adjust_saturation_profile!`](@ref) and [`compute_redistributed_heat!`](@ref)); the heat of excess water routed to the surface excess pool leaves the system. Without water flow ([`NoFlow`](@ref)) the advective flux is zero.

## Implicit vertical heat conduction

The `Timestepping` trait argument of [`compute_tendencies!`](@ref soilenergy.dispatches) is the resolved [`timestepping`](@ref) class of `internal_energy` (see [Implicit-explicit time stepping](@ref)). It is [`Explicit`](@ref) unless the model is run with an [`IMEX`](@ref) timestepper, in which case it is [`Implicit`](@ref).

- Under `Explicit()`, the tendency contains the full divergence of the diffusive and advective heat fluxes, [`compute_energy_tendency`](@ref), plus the flux boundary conditions.
- Under `Implicit()`, the diffusive and advective heat fluxes through the *interior* cell faces are omitted, since they are integrated by the implicit sub-stepper. Only the diffusive fluxes through the two boundary faces of the column, via [`compute_boundary_energy_tendency`](@ref), and the heat of the water entering the column through them with the boundary Darcy flux, via [`compute_boundary_advective_energy_tendency`](@ref), remain. For flux boundary conditions the diffusive boundary fluxes vanish because the halo cells are filled with a zero gradient, and the boundary fluxes themselves enter through the flux boundary conditions on `internal_energy`. For a gradient boundary condition on `temperature`, they carry the prescribed flux ``-\kappa\, \partial T / \partial z`` through the boundary face. The heat of the water leaving the column through the boundary faces (e.g. by gravity drainage) is part of the implicit operator, at the temperature of the boundary cell at ``t^{n+1}``, so that an explicit drainage flux that overdraws the water of the boundary cell cannot remove more heat than the cell holds.

For the energy equation with advection (see [Advection of heat by soil water flow](@ref)),
```math
\frac{\partial U}{\partial t} = \frac{\partial}{\partial z}\left[\kappa \frac{\partial T}{\partial z}\right] - \frac{\partial}{\partial z}\left(C_\text{w}\, T\, j_\text{w}\right) + G_\text{exp}\,,
```
where the second term is the advection of sensible heat by the vertical soil water flux ``j_\text{w}`` (zero without water flow), and ``G_\text{exp}`` contains the flux boundary conditions, the heat fluxes through the boundary faces, and any other explicit sources, the implicit step is a backward Euler step in temperature, solved with the [`Picard`](@ref) solver (see [Implicit Euler with a Picard solver](@ref)). Writing ``C_\text{app} = \partial U / \partial T`` for the apparent heat capacity and ``T^m``, ``U^m`` for the latest iterate of ``T^{n+1}``, ``U^{n+1}``, the energy increment is linearized as ``U^{n+1} - U^m \approx C_\text{app}\,(T^{n+1} - T^m)``, giving
```math
C_\text{app}\, T^{n+1} - \Delta t\, \frac{\partial}{\partial z}\left[\kappa \frac{\partial T^{n+1}}{\partial z}\right] + \Delta t\, \frac{\partial}{\partial z}\left(C_\text{w}\, T^{n+1} j_\text{w}\right) = C_\text{app}\, T^m - U^m + U^+\,, \qquad U^+ = U^n + \Delta t\, G_\text{exp}\,.
```
Discretized in space, with the temperature of the advective flux at each face taken from the upwind cell (first-order upwind; see [Tridiagonal operator](@ref)), this system is tridiagonal in each column and is solved with a batched tridiagonal solver. The coefficients ``\kappa`` (at cell faces) and ``C_\text{app}`` (at cell centers) are *lagged*, i.e. evaluated at the latest iterate rather than at ``T^{n+1}``. The water flux ``j_\text{w}`` is the flux with which the water was moved in the same step (see [Coupling of water and energy](@ref imex-water-energy-coupling)); for a fixed ``j_\text{w}``, the upwind terms keep the system an M-matrix (see [Tridiagonal operator](@ref)), so the solution has no spurious oscillations and the advection needs no CFL limit. The first iterate is the internal energy at the beginning of the step, ``U^n``, stored by [`update_cache!`](@ref), with the temperature re-diagnosed from it at the current saturation.

The `FreeWater` freeze curve has an apparent heat capacity that is a Dirac delta at the melting point, which cannot be used directly in a linear system. Terrarium therefore regularizes it: for cells undergoing phase change (liquid water fraction strictly between zero and one), the latent heat term is replaced by a large constant capacity,
```math
C_\text{app} = C + C_\text{pc}\,, \qquad C_\text{pc} = 1.5 \times 10^8~\text{J/m}^3\text{/K}\,,
```
with bulk heat capacity ``C``. Outside of phase change, ``C_\text{app} = C``. For typical water contents, ``C_\text{pc}`` corresponds to spreading the latent heat over an interval of roughly 1 K. Since the energy is updated conservatively (see below), ``C_\text{pc}`` only affects how quickly the temperature of a freezing or thawing cell may change between iterates, not the energy balance.

Once the temperature has been solved for, the internal energy is updated *conservatively* from the implicit diffusive and advective fluxes through the interior faces,
```math
U^{n+1} = U^+ + \Delta t\, \frac{\partial}{\partial z}\left[\kappa \frac{\partial T^{n+1}}{\partial z}\right] - \Delta t\, \frac{\partial}{\partial z}\left(C_\text{w}\, T^{n+1} j_\text{w}\right)\,,
```
rather than from the linearization. The temperature and liquid water fraction are then diagnosed from the closure, after every iteration including the last, so that they are consistent with ``U^{n+1}`` when the implicit step returns. Since the interior fluxes cancel pairwise between adjacent cells, the change in the column-integrated energy over a step equals ``\Delta t`` times the net boundary flux contained in ``G_\text{exp}``, independent of the number of iterations and the regularization.

The lagged apparent heat capacity and face conductivity are given by [`compute_implicit_capacity!`](@ref) and [`compute_implicit_conductivity!`](@ref), and, with water flow, the advective face velocity ``C_\text{w} j_\text{w}`` by [`compute_implicit_advective_velocity!`](@ref). The implicit step is supported for the `FreeWater` freeze curve, flux boundary conditions on `internal_energy`, and gradient boundary conditions on `temperature`; value boundary conditions on `temperature` are not yet supported.

The implicit operator of `internal_energy` is a [`TridiagonalOperator`](@ref) in temperature, with a face velocity for the advection if the water flows, returned by [`implicit_operator`](@ref) (which also validates the configuration), and each iteration of the implicit solver calls the process hooks of `SoilThermodynamics` below. The linear system is solved for the `temperature`, the variable of the [`SoilEnergyTemperatureClosure`](@ref) (see its [`varname`](@ref) method below). If `saturation_water_ice` is integrated implicitly as well, the Richards step is solved first, so that the heat step is evaluated at the saturation ``\xi^{n+1}`` at the end of the step and advects heat with the water flux of the step (see [Coupling of water and energy](@ref imex-water-energy-coupling)); [`linearize!`](@ref) therefore re-evaluates the energy closure at the current saturation before the coefficients.

```@docs; canonical = false
implicit_operator(energy::SoilThermodynamics, hydrology::AbstractSoilHydrology, state, grid)
linearize!(operator::TridiagonalOperator, rhs, state, grid, energy::SoilThermodynamics, soil::AbstractSoil, constants::PhysicalConstants, explicit_energy, Δt)
implicit_update!(state, grid, energy::SoilThermodynamics, soil::AbstractSoil, operator::TridiagonalOperator, explicit_energy, Δt)
```

After each iteration, the temperature and liquid water fraction are re-diagnosed with the regular energy closure, which has no once-per-step effects.

## [Coupling of water and energy](@id imex-water-energy-coupling)

With a Richards-equation hydrology, the water flux advects sensible heat: the soil energy equation carries the advective term ``-\partial_z (C_\text{w} T\, j_\text{w})``, discretized with first-order upwinding (see [Advection of heat by soil water flow](@ref)). For the scheme to keep a uniform temperature uniform, heat must move with exactly the water that moves. Terrarium ensures this as follows.

- **Water before energy.** The [`SplitOperator`](@ref) of [`SoilModel`](@ref) and [`LandModel`](@ref) solves `saturation_water_ice` before `internal_energy`. The heat step therefore sees the saturation at the end of the step and the water flux of the step, while the Richards step sees the temperature and liquid water fraction at the beginning of the step: the effect of frozen ground on the hydraulic conductivity is lagged by one step, as in most sequentially split land models. The Stefan problem test (`test/soil/soil_freeze_thaw_tests.jl`) quantifies this against a fully explicit reference.
- **The water flux of the step.** The `water_flux` auxiliary variable holds the Darcy flux through the cell faces that moved the water: the flux at ``t^n``, computed with the tendencies, through the boundary faces and, under explicit time stepping of the saturation, through the interior faces; under implicit time stepping, the flux of the conservative update of the last Picard iteration through the interior faces. The advective heat fluxes use this flux, explicitly when `internal_energy` is explicit, and implicitly (upwind, in the heat operator) when it is implicit. In the implicit case, water that enters the column through a boundary face carries the temperature of the boundary cell at ``t^n`` (explicitly), while water that leaves it, e.g. by gravity drainage, carries the temperature of the boundary cell at ``t^{n+1}`` (implicitly), so that a large explicit drainage flux cannot remove more heat from the bottom cell than it holds. The heat carried by flux boundary conditions on the saturation (e.g. infiltration, see [`compute_boundary_advected_heat!`](@ref)) is explicit in both cases, since the water flux is.
- **Heat moved by the hydrology closure.** [`adjust_saturation_profile!`](@ref) moves the sensible heat ``C_\text{w} T`` of the water it shifts between cells, at the temperature of the cell that the water leaves, and removes the heat of the excess water that it discards or routes to the surface excess pool. The hydrology closure runs before the energy closure, so the temperature is recomputed from the adjusted energy. When both variables are implicit, the internal energy then holds ``U^n`` plus this adjustment, and since the [`Picard`](@ref) solver forms the explicit stage of the heat step from the current internal energy and the explicit increment, the adjustment is kept, as in a sequential split.

The water heat capacity ``C_\text{w}`` is the one with which water enters the bulk heat capacity. Since the bulk heat capacity also includes air, water that replaces air in the pores changes the temperature slightly (by a fraction of the order of ``C_\text{air} / C_\text{w} \approx 3 \times 10^{-4}`` of the change of the water content), as the heat of the displaced air is not tracked.

!!! warning "Implicit water with explicit energy"
    Integrating `saturation_water_ice` implicitly while `internal_energy` is explicit (by specializing [`timestepping`](@ref)) is allowed, but not recommended with water flow: the explicit advection of heat is then computed at ``t^n``, before the water is moved by the implicit flux of the step, so heat and water move with different fluxes. During wetting fronts, this can warm or cool the soil substantially and become unstable for large time steps. The default routing of [`SoilModel`](@ref) and [`LandModel`](@ref) integrates both implicitly.

Under explicit time stepping of `internal_energy`, the stability limit of the explicit advection is reported by `cell_advection_timescale` (see [Adaptive time stepping](@ref)); under implicit time stepping, it is `Inf`.

## Closures

The constitutive relationship between energy and temperature plays a critical role in simulating the thermal state of porous media. This relation can be defined in integral form as
```math
\begin{equation}
    U(T,\theta) = \int_{T_{\text{ref}}}^T \tilde{C}(x,\theta) \, \mathrm{d}x 
\end{equation}
```
where $\tilde{C}$ is referred to as the *effective* or *apparent* heat capacity and $T_{\text{ref}}$ is a reference temperature. The internal energy $U$ is defined relative to a reference temperature $T_{\text{ref}}$ and a reference physical state if phase change is considered. For Terrarium, we choose $T_{\text{ref}} = 273.15$ K. We assume water to be the only constituent undergoing phase change; as such, the total internal energy of the soil volume is computed as,
```math
\begin{equation}
U(T,\theta) = C(\theta_{\text{w}},\theta) (T - T_{\text{ref}}) - \rho_{\text{w}} L_{\text{sl}} (\theta - \theta_{\text{w}}(\theta, T)),
\end{equation}
```
where $\theta_{\text{w}}(T,\theta)$ is the volumetric unfrozen water content as a function of temperature $T$ and total water/ice content $\theta$. Note that, under this formulation, zero internal energy corresponds to zero ice content (fully thawed) at 0°C. $C(\theta_{\text{w}},\theta)$  (J/(K m³)) is the bulk volumetric material heat capacity of the total soil volume as a function of the unfrozen and total ($\theta$) water contents, calculated as $\sum_x \rho_x c_x$. The apparent heat capacity is then defined as the derivative of the energy-temperature relation,

```math
\begin{equation}
\tilde{C}(T,\theta) := \frac{\partial U}{\partial T} =
\overbrace{C(\theta_{\text{w}},\theta) + T \frac{\partial C}{\partial \theta_{\text{w}}}\frac{\partial \theta_{\text{w}}}{\partial T}}^{\text{Sensible}} \, + \,
\overbrace{\rho_{\text{w}} L_{\text{sl}} \frac{\partial\theta_{\text{w}}}{\partial T}}^{\text{Latent}}\,,
\end{equation}
```
where the chain-rule is applied on $C(\theta_{\text{w}},\theta) (T - T_{\text{ref}})$ as both $C$ and $\theta_{\text{w}}$ are functions of $T$. The grouping of terms on the right-hand side show the partitioning of energy change into **sensible** and **latent** heat. The sensible component represents the energy necessary to heat a volume of the material to a particular temperature, whereas the latent component corresponds to the energy required for the phase change of water in the volume from solid (frozen) to liquid (thawed).

In the simplest case where we neglect the effect of capillary action in the soil, the energy-temperature relation can be derived according to that of "free" water (i.e. unbound by the soil matrix),
```math
\begin{equation}
    \theta_{\text{w}}(U) =
        \begin{cases}
            0                   & U < -\rho_{\text{w}}L_{\text{sl}}\theta \\
            \theta \left( 1 + \frac{U}{\rho_{\text{w}}L_{\text{sl}}\theta} \right) & -\rho_{\text{w}}L_{\text{sl}}\theta \leq U < 0 \\
            \theta              & U \geq 0\,,
        \end{cases}
\end{equation}
```
with temperature then determined by
```math
\begin{equation}
    T - T_{\text{ref}} = U^{-1}(U(T,\theta)) =
    \begin{cases}
    \frac{U + \rho_{\text{w}}L_{\text{sl}}\theta}{C} & U < -\rho_{\text{w}}L_{\text{sl}}\theta \\
    0 & -\rho_{\text{w}}L_{\text{sl}}\theta \leq U < 0 \\
    \frac{U}{C} & U \geq 0\,.
    \end{cases}
\end{equation}
```

By expressing $T$ in °C, $T_{ref}$ can be dropped as it equals 0°C.
For more information on more complex formulations of these so-called soil freezing characteristic curves, see [FreezeCurves.jl](https://github.com/CryoGrid/FreezeCurves.jl/tree/main?tab=readme-ov-file). 

```@docs; canonical = false
SoilEnergyTemperatureClosure
varname(::SoilEnergyTemperatureClosure)
```

## Methods

```@docs; canonical = false
get_thermal_properties
```

```@docs; canonical = false
compute_boundary_advected_heat!
```

## Kernel functions

```@docs; canonical = false
compute_energy_tendency
compute_boundary_energy_tendency
compute_advective_energy_tendency
advective_heat_flux
compute_advective_water_flux
compute_boundary_advective_energy_tendency
compute_boundary_advected_energy_tendency
compute_heat_advection_timescale
```

```@docs; canonical = false
compute_energy_tendencies!
```

```@docs; canonical = false
compute_thermal_conductivity
```

The following kernel functions implement the implicit heat conduction and advection step (see [Implicit vertical heat conduction](@ref)):

```@docs; canonical = false
compute_implicit_conductivity!
compute_implicit_capacity!
compute_implicit_advective_velocity!
compute_implicit_heat_coefficients!
compute_implicit_energy_rhs
compute_implicit_energy_update
compute_implicit_diffusion_tendency
compute_implicit_advection_tendency
```

## [References](@id "soilenergy.refs")

```@bibliography
Pages = ["soil_energy.md"]
Canonical = false
```
