# Vegetation dynamics and competition

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

Plant functional types (PFTs) compete for resources and space on a landscape. Vegetation dynamics describes how the fractional area occupied by a PFT changes over time in response to:
- **Productivity**: High net primary productivity (NPP) allows a PFT to expand
- **Disturbance**: Fire, disease, and other mortality mechanisms limit vegetation extent
- **Seed availability**: A minimum seed fraction ensures PFTs can reestablish

Terrarium implements vegetation dynamics following the Lotka–Volterra approach of PALADYN (Willeit 2016), which balances growth and disturbance to determine PFT area fraction dynamics.

### Lotka-Volterra vegetation competition

The vegetation area fraction ($\nu$) for a single PFT evolves according to:
```math
\begin{equation}
\frac{d\nu}{dt} = \left( \lambda_{\text{NPP}} \frac{C_{\text{veg}}}{\nu^*} \right) (1 - \nu) - \gamma_v \nu^*
\end{equation}
```

where:
- $\lambda_{\text{NPP}}$ is an NPP-dependent rate parameter (from carbon dynamics)
- $C_{\text{veg}}$ is the accumulated vegetation carbon
- $\nu^* = \max(\nu, \nu_{\text{seed}})$ is the effective PFT fraction (always at least the seed fraction)
- $\gamma_v$ is the disturbance rate (mortality, fire, disease) [1/time]
- $(1 - \nu)$ is the available space for expansion

**Growth term**: $(λ_{\text{NPP}} C_{\text{veg}} / \nu^*) (1 - \nu)$ represents logistic growth where productive PFTs expand into unoccupied space.

**Disturbance term**: $\gamma_v \nu^*$ represents losses due to disturbance, proportional to PFT extent.

### Seed fraction and PFT establishment

The seed fraction $\nu_{\text{seed}}$ ensures that a PFT at very low abundance can still persist and reestablish after disturbance:
```math
\begin{equation}
\nu^* = \max(\nu, \nu_{\text{seed}})
\end{equation}
```

Typical values are $\nu_{\text{seed}} \approx 0.001$ (0.1%) to allow slow recovery from local extinction.

### Disturbance rates

The disturbance rate $\gamma_v$ represents losses from fire, pest outbreaks, disease, and senescence. A minimum baseline disturbance rate is specified:
```math
\begin{equation}
\gamma_v \geq \gamma_v^{\text{min}}
\end{equation}
```

In the current implementation, $\gamma_v = \gamma_v^{\text{min}}$ (constant). Future implementations may tie disturbance to soil moisture, fire danger, or vegetation state variables.

### Coupling to productivity

Vegetation expansion is controlled by vegetation carbon ($C_{\text{veg}}$) and the NPP-productivity parameter ($\lambda_{\text{NPP}}$) computed by the carbon dynamics model. Higher NPP accelerates vegetation expansion; lower NPP leads to vegetation retreat when disturbance exceeds growth.

## Abstract types

```@docs; canonical = false
AbstractVegetationDynamics
```

## Concrete types

```@docs; canonical = false
PALADYNVegetationDynamics
```

## Methods

```@docs; canonical = false
compute_γv
```

```@docs; canonical = false
compute_ν_star
```

```@docs; canonical = false
compute_ν_tendency
```
