# Surface runoff

```@meta
CurrentModule = Terrarium
```

```@setup runoff
using Terrarium
using InteractiveUtils
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Overview

Surface runoff occurs when the rate of water reaching the surface exceeds the rate of infiltration into the soil. Surface runoff (overland flow) is often distinguished from other sources of subsurface runoff such as interflow in the vadose zone and baseflow in the saturated zone. Surface runoff is a key output of both hydrological and global land surface models since it serves as a input flux to lakes and rivers. Runoff that is routed into rivers can then serve as a key input to river routing schemes that produce freshwater fluxes for ocean components of Earth system models.

```@docs; canonical = false
AbstractSurfaceRunoff
```

```@example runoff
subtypes(Terrarium.AbstractSurfaceRunoff)
```

## Direct runoff

The simplest runoff scheme provided by Terrarium is [`DirectSurfaceRunoff`](@ref) which first routes all rainwater into the ground as infiltration and then simply routes the residual water flux as runoff.

Water in excess of instantaneous infiltration capacity accumulates as surface water and slowly drains back into the soil,
```math
\begin{equation}
D = \frac{S}{\tau_r}
\end{equation}
```
where $S$ is the accumulated surface water depth (m) and $\tau_r$ is the surface water drainage timescale (s).

```@docs; canonical = false
DirectSurfaceRunoff
```

```@example runoff
variables(DirectSurfaceRunoff(Float32))
```

### Infiltration capacity and soil saturation

Infiltration is limited by the infiltration capacity of the soil and by the saturation of the uppermost soil layer,
```math
\begin{equation}
I = \begin{cases}
\min(P_{\text{ground}}, I_{\max}) & \text{if } \xi_{\text{top}} < 1, \\
0 & \text{otherwise,}
\end{cases}
\end{equation}
```
where $P_{\text{ground}}$ is the water reaching the ground (m/s), $\xi_{\text{top}}$ is the saturation of the uppermost soil layer (-), and $I_{\max}$ is the infiltration capacity (m/s) provided by the soil hydrology (see [`Terrarium.infiltration_capacity`](@ref)).

For the Richards equation, the infiltration capacity is the saturated hydraulic conductivity of the uppermost soil layer, reduced by the ice impedance factor of frozen soil,
```math
\begin{equation}
I_{\max} = I_{\text{ice}}(f) \, K_{\text{sat}},
\end{equation}
```
where $f$ is the unfrozen fraction of the pore water, as in CLM5 [lawrenceTechnicalDescriptionVersion2018; Eq. (2.7.34)](@cite).
The infiltration capacity of a soil is at least its saturated conductivity: the matric suction gradient at a wetting front raises it above $K_{\text{sat}}$ in dry soil, and it decays toward $K_{\text{sat}}$ as the soil wets up [greenStudiesSoilPhysics1911, philipTheoryInfiltration1957](@cite).
$K_{\text{sat}}$ is thus a conservative bound, and rain only runs off as infiltration excess (Hortonian runoff) when its rate exceeds $K_{\text{sat}}$.
Immobile soil water ([`NoFlow`](@ref)) has an infiltration capacity of zero.

When the uppermost soil layer becomes saturated ($S_{\text{top}} = 1$), infiltration drops to zero and all precipitation is routed to either surface runoff or surface water storage.

### Excess water at the surface

The `DirectSurfaceRunoff` scheme tracks excess water accumulated at the surface. When excess water is present:
- Precipitation adds to the excess water reservoir
- Infiltration is computed from the drainage flux (not directly from precipitation)
- The infiltration rate is limited by hydraulic conductivity and soil saturation state

When no excess water is present, precipitation is routed directly to infiltration (subject to the same limits).

### Water table and drainage processes

This simplified surface runoff scheme does not explicitly model subsurface drainage or return flows from the water table. It is appropriate for upland areas where:
- Water table is deep relative to the surface
- Saturation excess runoff is minimal
- Hortonian overland flow (infiltration excess) is dominant

In riparian areas or regions with high water tables, more sophisticated approaches (e.g., variable source areas) would be necessary to capture saturation-excess runoff.

## Process interface

```@docs; canonical = false
compute_auxiliary!(state, grid, runoff::DirectSurfaceRunoff, canopy_interception::AbstractCanopyInterception, soil::AbstractSoil, snow::Optional{AbstractSnow}, args...)
```

## Kernel functions

```@docs; canonical = false
compute_surface_drainage
```

```@docs; canonical = false
compute_infiltration
```

```@docs; canonical = false
compute_infiltration_capacity
```

```@docs; canonical = false
compute_surface_runoff
```
