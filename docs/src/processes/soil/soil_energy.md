# Soil energy balance

```@meta
CurrentModule = Terrarium
```

```@setup default
using Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Heat conduction

Heat transport along the vertical axis perpendicular to the land surface can be represented according to the heat equation, with the upper boundary set to surface temperature and the lower boundary set to a constant positive heat flux representing heat produced by the inner earth (Lachenbruch 1986, Jaeger 1965). If both the upper and lower boundaries are assumed to be constant over time, the steady-state temperature profile takes the form of a continuous piecewise linear function increasing over depth with the slope determined by the thermal properties of the ground material. The instantaneous temperature field can then be generally represented as
```math
\begin{equation}
T(z,t) = T_0 + \frac{Q_{\text{geo}}}{\kappa_{\text{h}}(z)}z + \Delta T(z,t)
\end{equation}
```
where $T(z,t)$ is the temperature field (K) over depth $z$ (m) and time $t$ (s), $T_0$ is the mean annual GST (K), $Q_{\text{geo}}$ is the geothermal heat flux (W/m²), and $\kappa_{\text{h}}(z)$ (W/m K) is the thermal conductivity which may vary with depth depending on the material. The last term $\Delta T(z,t)$ represents transient disturbances to the steady state temperature profile due to both seasonal and long-term fluctuations in the upper and lower boundary conditions of the vertical domain. Simulating the impacts of these transient changes is one of the primary objectives of most numerical permafrost and land surface models.

Diffusive heat flow in a solid medium is governed by Fourier's law,
```math
\begin{equation}
    \mathbf{j}_\text{h} \cdot \mathbf{n}_z = -\kappa_{\text{h}}\frac{\partial T}{\partial z}\,,
\end{equation}
```
where $\mathbf{j}_\text{h}$ (W/m²) is the diffusive heat flux vector and $\mathbf{n}_z$ is the upward facing normal vector along the vertical $z$ axis.

Since ground materials are often porous, i.e., there exists void space between the solid particles, it is necessary to consider the potential presence of water and/or ice in this void space, which is hereafter referred to as pore space, or simply, soil pores. The thermal effects of water and ice can be accounted for by considering not only the temperature of the material but rather the total internal energy of the elementary volume. Combining the diffusive flux with a potential advective heat flux $j_z^{\text{w}}$ due to water flow yields the energy conservation law,
```math
\begin{equation}
\frac{\partial U(T,\theta)}{\partial t} - \boldsymbol{\nabla} \cdot \left(\mathbf{j}_\text{h} + \mathbf{j}_h^{\text{w}}\right) - F_h(z,t) = 0\,,
\end{equation}
```
where $U(T,\theta)$ (J/m³) is the volumetric internal energy as a function of temperature and total water/ice content $\theta$ (m³/m³), and $F_h(z,t)$ is an inhomogeneous heat source/sink (forcing) term.

The advective heat flux $j_{\text{h}}^{\text{w}}$ can be represented as,
```math
\begin{equation}
\mathbf{j}_{\text{h}}^{\text{w}} = \left( c_{\text{w}} T + L_{\text{sl}} \right) \mathbf{j}_{\text{w}} \rho_{\text{w}}
\end{equation}
```
where $L_{\text{sl}}$ and $c_{\text{w}}$ (J/kg) represent the specific latent heat of fusion and heat capacity of liquid water respectively. This flux term accounts for the energy transferred by the movement of water within the soil matrix. In model configurations that neglect subsurface water flow, this flux term is implicitly assumed to be zero.

### Energy-temperature closure

The constitutive relationship between energy and temperature plays a crticial role in characterizing the subsurface energy balance. This relation can be defined in integral form as
```math
\begin{equation}
    U(T,\theta) = \int_{T_{\text{ref}}}^T \tilde{C}(x,\theta) \, \mathrm{d}x\,,
    %= \overbrace{\HC(\thetaw,\thetai)\left[T-T_{\text{ref}}\right]}^{\text{Sensible}} + \overbrace{\densityw \LHF\thetaw(T,\thetawi)}^{\text{Latent}},
\end{equation}
```
where $\tilde{C}$ is referred to as the *effective* or *apparent* heat capacity and $T_{\text{ref}}$ is a reference temperature. The apparent heat capacity is then defined as the derivative of the energ-temperature relation,
```math
\begin{equation}
\tilde{C}(T,\theta) := \frac{\partial U}{\partial T} =
\overbrace{C(\theta_{\text{w}},\theta) + T \frac{\partial C}{\partial \theta_{\text{w}}}\frac{\partial \theta_{\text{w}}}{\partial T}}^{\text{Sensible}} \,+\,
\overbrace{\rho_{\text{w}} L_{\text{sl}} \frac{\partial\theta_{\text{w}}}{\partial T}}^{\text{Latent}}\,,
\end{equation}
```
where $\theta_{\text{w}}(T,\theta)$ is the volumetric unfrozen water content as a function of temperature and total water/ice content is the bulk volumetric material heat capacity of the volume as a function of the unfrozen and total water contents;  $\rho_{\text{w}}$ (kg/m³) and $L_{\text{sl}}$ (J/kg) correspond to the density and specific latent heat of fusion of water, respectively. The grouping of terms on the right-hand side show the partitioning of energy change into **sensible** and **latent** heat. The sensible component represents the energy necessary to heat a volume of the material to a particular temperature, whereas the latent component corresponds to the energy required for the phase change of water in the volume from solid (frozen) to liquid (thawed).

In the simplest case where we neglect the effect of capillary action in the soil, the energy-temperature relation can be derived according to that of "free" water (i.e. unbound by the soil matrix),
```math
\begin{equation}
    \theta_{\text{w}}(U) =
        \begin{cases}
            0                   & U < -\rho_{\text{w}}L_{\text{sl}}\theta \\
            \frac{U}{L} & -\rho_{\text{w}}L_{\text{sl}}\theta \leq U < 0 \\
            \theta              & U \geq 0\,,
        \end{cases}
\end{equation}
```
with temperature then determined by
```math
\begin{equation}
    U^{-1}(U(T,\theta)) =
    \begin{cases}
    \frac{U(T,\theta) - \rho_{\text{w}}L_{\text{sl}}\theta}{C} & U(T,\theta) < -\rho_{\text{w}}L_{\text{sl}}\theta \\
    0 & 0 \leq U(T,\theta) \leq \rho_{\text{w}}L_{\text{sl}}\theta \\
    \frac{U(T,\theta)}{C} &   U(T,\theta) \geq 0\,,
    \end{cases}
\end{equation}
```
where $C = C(\theta_{\text{w}},\theta)$ is the volumetric heat capacity (J/K/m³) as a function of the unfrozen and total water content.

## Abstract types

```@docs; canonical = false
AbstractSoilEnergyBalance
```

```@docs; canonical = false
AbstractHeatOperator
```

```@docs; canonical = false
AbstractSoilEnergyClosure
```

## Concrete types

```@docs; canonical = false
SoilEnergyBalance
```

### [State variables](@id soilenergy.vars)
```@example default
variables(SoilEnergyBalance(Float32))
```

### [Process method dispatches](@id soilenergy.dispatches)

```@docs; canonical = false
initialize!(state, grid, energy::SoilEnergyBalance, soil::AbstractSoil, constants::PhysicalConstants, args...)
```

```@docs; canonical = false
compute_auxiliary!(state, grid, energy::SoilEnergyBalance, soil::AbstractSoil, args...)
```

```@docs; canonical = false
compute_tendencies!(state, grid, energy::SoilEnergyBalance, soil::AbstractSoil, args...)
```

## Closures

```@docs; canonical = false
SoilEnergyTemperatureClosure
```

## Methods

```@docs; canonical = false
get_thermal_properties
```

## Kernel functions

```@docs; canonical = false
compute_energy_tendency
```

```@docs; canonical = false
compute_energy_tendencies!
```

```@docs; canonical = false
compute_thermal_conductivity
```
