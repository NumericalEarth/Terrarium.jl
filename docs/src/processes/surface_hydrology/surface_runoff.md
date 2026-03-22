# Surface runoff

```@meta
CurrentModule = Terrarium
```

!!! warning
    This page is a work in progress. If you have any questions or notice any errors, please [raise an issue](https://github.com/NumericalEarth/Terrarium.jl/issues).

## Theory

### Surface water balance

At the land surface, precipitation (after canopy interception) must partition into infiltration into the soil or runoff toward streams and rivers. The surface water balance determines this critical partitioning:

```math
\begin{equation}
Q = P_{\text{ground}} + D - I
\end{equation}
```

where:
- $Q$ is surface runoff [m/s]
- $P_{\text{ground}}$ is precipitation reaching the ground (after canopy interception) [m/s]
- $D$ is surface drainage from accumulated excess water [m/s]
- $I$ is infiltration into the upper soil layer [m/s]

Negative runoff (which would represent condensation) cannot occur, so $Q = \max(0, P_{\text{ground}} + D - I)$.

### Surface water reservoir

Precipitation and infiltration create a surface water reservoir at the soil surface. Water in excess of instantaneous infiltration capacity accumulates as surface water and slowly drains back into the soil:

```math
\begin{equation}
D = \frac{S}{\tau_r}
\end{equation}
```

where:
- $S$ is the accumulated surface water depth [m]
- $\tau_r$ is the surface water drainage timescale [s]

This represents water pooling on the surface due to topographic depressions or soil sealing, which gradually drains downward as the pressure head increases.

### Infiltration capacity and soil saturation

Infiltration into the soil is limited by:
1. **Capillary supply**: The maximum rate soil can accept water from the surface
2. **Soil saturation state**: Saturated soil cannot accept additional water

The effective infiltration is:

```math
\begin{equation}
I = \min(P_{\text{ground}}, I_{\max}) \times (1 - f_{\text{sat}})
\end{equation}
```

where:
- $I_{\max}$ is the maximum infiltration capacity determined by soil hydraulic conductivity [m/s]
- $f_{\text{sat}}$ is the saturation fraction of the upper soil layer (0 = dry, 1 = saturated) [dimensionless]

When soil becomes saturated ($f_{\text{sat}} = 1$), infiltration drops to zero and all precipitation becomes either surface runoff or surface water storage.

### Water table and drainage processes

This simplified surface runoff scheme does not explicitly model subsurface drainage or return flows from the water table. It is appropriate for upland areas where:
- Water table is deep relative to the surface
- Saturation excess runoff is minimal
- Hortonian overland flow (infiltration excess) dominates

In riparian areas or regions with high water tables, more sophisticated approaches (e.g., variable source areas) would be necessary to capture saturation-excess runoff.

## Abstract types

```@docs; canonical = false
AbstractSurfaceRunoff
```

## Concrete types

```@docs; canonical = false
DirectSurfaceRunoff
```

## Methods

There are no high-level user-facing methods for surface runoff; runoff is computed internally as part of the surface hydrology auxiliary computation.

## Kernel functions

```@docs; canonical = false
compute_surface_drainage
```

```@docs; canonical = false
compute_infiltration
```

```@docs; canonical = false
compute_surface_runoff
```
