# Land surface refactor: unify the surface energy balance and evapotranspiration into `AbstractLandSurface`

> Status: **planned**. Design drafted and awaiting human review; no implementation has started.

Date of initial draft: 2026-10-07

Author: Claude Fable 5.1 (claude-fable-5-1), Claude Code (VS Code extension)

Reviewer:

Base revision: `10f3e07a14c183a6d39eabcfcefee434bc558682` (tip of `bg/refactor-surface`)

## Originating prompt

> I am considering a major refactoring of the surface processes. [...]
>
> Currently most of the surface processes are defined by `SurfaceEnergyBalance` and `SurfaceHydrology`,
> with evapotranspiration (ET) belonging to the latter. This has resulted in a somewhat awkward code
> structure where the core surface flux calculations are split between the `SurfaceEnergyBalance` and
> `SurfaceHydrology` processes, even though the SEB and ET are extremely tightly coupled with each other.
> Furthermore, canopy interception and (even more so) surface runoff do not pair naturally with ET in
> `SurfaceHydrology`. This also caused problems when trying to optimize kernel launches in PR#192 where
> kernel fusion for `SurfaceHydrology` was ultimately abandoned. Furthermore, the current setup makes it
> difficult to represent the overall surface flux network design in the dispatch patterns since there is
> no single type which represents the land surface flux resistance network. Thanks to @olivierbonte for
> pointing this out.
>
> Following several offline discussions, we have settled on a new design for the land surface processes.
> `SurfaceEnergyBalance` and `SurfaceHydrology` will be (partially) merged into a single
> `AbstractLandSurface` type hierarchy, where each subtype represents a fixed configuration of the surface
> flux resistance network. There will initially be two implementations corresponding to the two current
> ET variants:
>
> - `BareGround <: AbstractLandSurface` corresponding to the single-source evaporation with aerodynamic
>   resistance network currently implemented by `BareGroundEvaporation`
> - `CanopyGround <: AbstractLandSurface` corresponding to all resistance network configurations with a
>   canopy and ground surface but no canopy air space. This includes the
>   `PALDAYNCanopyEvapotranspiration` as a special case.
>
> Each `AbstractLandSurface` implementation will have the following components:
> - A `fluxes` parameterization which itself holds the `radiative_fluxes`, `turbulent_fluxes`, and
>   `ground_heat_flux` schemes, each of which can be optionally prescribed by the user, as currently in
>   the `SurfaceEnergyBalance`.
> - A resistance topology marker `Series` or `Parallel`
> - A `roughness` parameterization
> - A solution_method which is either `Linearized` (e.g. Penman-Monteith) or `Implicit` for an iterative
>   skin temperature solver (current)
> - 1 or more resistance parameterizations representing each resistance term in the network.

Subsequent decisions from the same discussion:

> Interception should stay in `CanopyGround` for now and runoff would indeed move to `SoilHydrology`.
> Aerodynamic resistance would be removed from the atmosphere type entirely and would instead be defined
> as one of the resistance parameterizations, also depending on roughness. We would start with only one
> implementation of constant surface roughness.

> `CanopyGround` specifies the *sources* but not the network configuration, so I think we still need a
> `topology` type to determine how they are connected, but maybe we need a more complex set of options
> than `Series` and `Parallel`.

> Please draft an implementation plan based on the discussion so far. Defer the implementation of
> `Linearized` (Penman-Monteith) and canopy air space for a future PR.

> ConstantDragResistance should be kept as an alternative implementation. I think the surface roughness
> change is also out of scope for this refactoring since it is substantial new physics. Keep the outline
> but move it to the deferred section.

> roughness can just be set to nothing for now

> We should also consider how to handle skin humidity. Currently we assume saturation of the skin but this could be relaxed. The question is whether there should be both skin_temperature and skin_humidity or whether they should be somehow combined into a single formulation of the skin. We should also consider the ambiguity of how canopy/leaf temperature and later air space temperature is represented under the current design.

> I agree, but drop the source node marker for now, this seems a bit excessively complex. I suppose
> future implementations of `AbstractSkin` could consider such a change.

## Revision log

- **Rev 1 (2026-10-07, Claude Fable 5.1):** Initial draft.
- **Rev 2 (2026-10-07, Claude Fable 5.1):** Surface roughness and the log-law aerodynamic resistance
  are removed from scope as substantial new physics. The aerodynamic resistance still moves from the
  atmosphere to the surface, but as `ConstantDragResistance` carrying the current drag coefficient, so
  the refactor is bit-for-bit reproducible. The network surface types keep a `roughness` slot that is
  `nothing` for now; the roughness design outline is kept under Future work. Flat
  `radiative_fluxes`/`ground_heat_flux` fields confirmed over a nested `fluxes` struct.
- **Rev 3 (2026-10-07, Claude Fable 5.1):** The skin temperature component is generalized to a
  `skin::AbstractSkin` component responsible for the state (temperature and humidity) of the
  atmosphere-facing node. The existing `ImplicitSkinTemperature` and `PrescribedSkinTemperature` are
  its first implementations and imply saturation at the skin temperature. `skin_humidity` is added as a
  diagnosed auxiliary. A per-path source node marker was considered and rejected as excessive for now;
  the choice of node temperature per path remains hard-coded in each surface type. The joint
  temperature/humidity skin solve and a soil vapor diffusion resistance are recorded under Future work.

## Problem description

The land surface flux calculation is currently spread over two coupled-process types that are
siblings in `LandModel`:

- `SurfaceEnergyBalance` owns the skin temperature solve, the radiative fluxes, the turbulent fluxes,
  the ground heat flux, and the albedo.
- `SurfaceHydrology` owns canopy interception, evapotranspiration (ET), and surface runoff.

The two are tightly coupled through the latent heat flux. `compute_latent_heat_flux` in
`turbulent_fluxes.jl` reaches into the hydrology object with `get_evapotranspiration` to obtain the
humidity flux, and the fused SEB kernel in `surface_energy_balance.jl` recomputes the ET component
fluxes from the converged skin temperature by calling `compute_evapotranspiration_fluxes!` on the
hydrology's ET scheme. The ET conductances are computed in a separate `SurfaceHydrology` kernel launch
beforehand because the ET scheme does not know about the skin temperature solve. This split caused the
kernel fusion for `SurfaceHydrology` in PR#192 to be abandoned.

There is no type that represents the land surface resistance network as a whole. The network is implied
by the pairing of an `AbstractEvapotranspiration` scheme with the `DiagnosedTurbulentFluxes` scheme, and
the aerodynamic resistance is a method of `PrescribedAtmosphere`, even though surface roughness is a
land property. Vegetation-free configurations need a compatibility dispatch in
`evapotranspiration_base.jl` that is marked with a `TODO` about redesigning the call interface.

Finally, `SurfaceHydrology` groups processes with weak mutual coupling. Surface runoff depends only on
throughfall, snowmelt, and the soil infiltration capacity, and canopy interception is coupled to the
canopy evaporation term, not to runoff.

## Background

### Current call structure

`LandModel.compute_auxiliary!` runs, in order: atmosphere, soil, snow, vegetation,
`SurfaceHydrology` (interception, ET conductances and fluxes, runoff), `SurfaceEnergyBalance`
(albedo), then `solve_surface_energy_balance!` (one fused XY kernel: skin temperature solve, ET
component fluxes at the converged skin temperature, radiative, turbulent, and ground heat fluxes), then
`compute_snow_interface_fluxes!`. `compute_tendencies!` passes the hydrology into the soil tendencies
so the ET sink and the infiltration flux can be applied.

### Current resistance networks

`BareGroundEvaporation` computes a single ground conductance `g = β / rₐ`, where `β` is a ground
evaporation resistance factor and `rₐ` is the aerodynamic resistance from the atmosphere.

`PALADYNCanopyEvapotranspiration` (Willeit and Ganopolski, 2016, Eq. 5) computes three conductances:

- ground evaporation `g_gnd = β / (rₐ + rₐ_can)`, evaluated at the ground temperature,
- wet canopy evaporation `g_can = f_can / rₐ`, evaluated at the skin temperature,
- transpiration `g_trp = g_stm / (1 + g_stm rₐ)`, evaluated at the skin temperature, clamped nonnegative.

Here `rₐ_can = (1 − exp(−LAI − SAI)) / (C_can Vₐ)` is the ground-to-canopy resistance, `f_can` is the
canopy water saturation fraction from the interception scheme, and `g_stm` is the stomatal conductance
from the vegetation scheme. The sensible heat flux uses `rₐ` from the skin temperature.

Both networks share one structural property: the atmosphere reference height is the only shared node.
Each vapor source connects to it through its own chain of series resistances, and the chains sum in
parallel. There is no internal node with its own state. A canopy air space would introduce one and is
explicitly out of scope.

### Snow

Snow is handled as an area fraction `f_snow` on the ground node. The implicit skin temperature inverts
the area-weighted sum of the ground and snow-top conduction relations, the latent heat flux blends
ground/canopy evaporation with snow sublimation by `f_snow`, and the ET component fluxes are scaled by
`1 − f_snow`. Sublimation (`compute_snow_sublimation_flux` in `snow_mass.jl`) uses the atmosphere's
aerodynamic resistance directly.

### Prescription

`SurfaceEnergyBalance` has `Prescribed`/`Diagnosed` pairs for the radiative fluxes, the turbulent
fluxes, and the ground heat flux, plus `PrescribedSkinTemperature`/`ImplicitSkinTemperature`.
`PrescribedSurfaceEnergyBalance` is a type alias for the all-prescribed combination (see the plan of
2026-09-01). When the turbulent fluxes are prescribed, every ET parameter is inert.

## Summary of changes

### 1. New type hierarchy

```julia
abstract type AbstractLandSurface{NF} <: AbstractCoupledProcesses{NF} end
```

Three concrete implementations:

| Type | Vapor sources | Resistance paths | Components |
|---|---|---|---|
| `BareGround` | ground (at skin temperature) | `ground` | `radiative_fluxes`, `ground_heat_flux`, `albedo`, `skin`, `roughness` (`nothing`) |
| `CanopyGround` | ground (at ground temperature), wet canopy, dry canopy (both at skin temperature) | `ground`, `canopy_evaporation`, `transpiration` | as above, plus `canopy_interception` |
| `PrescribedLandSurface` | none | none | `radiative_fluxes`, `ground_heat_flux`, `albedo`; skin temperature, sensible, and latent heat fluxes are inputs |

The `fluxes` container from the originating prompt is flattened to two fields, `radiative_fluxes` and
`ground_heat_flux`, since the turbulent flux scheme no longer has a `Prescribed` variant on the network
types (see change 6). Grouping two fields in a nested struct adds a level of indirection for no gain.
`BareGround` and `CanopyGround` carry a `roughness::Optional{AbstractRoughness{NF}}` field whose only
admissible value in this PR is `nothing`. The abstract type is declared with no implementations, so the
slot and its type parameter are in place for the deferred roughness work (see change 3 and Future work)
without any physics being added now.

`skin` is the *solution method* of the originating prompt, generalized: it determines the state of the
atmosphere-facing node, i.e. both the skin temperature and the skin humidity. See change 1a.
`Linearized` (Penman-Monteith) is deferred (see Future work), so no new solution method is added in
this PR.

### 1a. The skin component

```julia
abstract type AbstractSkin{NF} <: AbstractProcess{NF} end
```

replaces `AbstractSkinTemperature`. `ImplicitSkinTemperature` and `PrescribedSkinTemperature` become
its first implementations and are otherwise unchanged. Both imply a saturated skin: the humidity of the
skin node is the saturation specific humidity at the skin temperature, and any sub-saturation is
expressed by the conductance factors and series resistances of the paths, never by the skin itself.

Rationale: in a network without internal nodes the intermediate humidities are eliminated analytically, so there is nothing to solve for.

A new auxiliary `skin_humidity` (kg/kg) is diagnosed on the network surface types after the solve as the
humidity of the atmosphere-facing node recovered from the total humidity flux and the aerodynamic
resistance, `q_skin = q_air + Q_h rₐ`. For a saturated single-source skin this equals the saturation
humidity at the skin temperature; for `CanopyGround` it is the effective humidity the atmosphere sees
across the three paths. It is output only and is not read by any process in this PR.

**Node temperatures remain hard-coded per surface type.** `BareGround` evaluates its one path at the
skin temperature. `CanopyGround` evaluates the canopy paths at the skin temperature and the ground path
at the ground temperature, exactly as PALADYN does today. The docstring of `CanopyGround` states that
the skin node *is* the canopy node (longwave emission temperature, sensible heat source, and canopy
vapor sources all share it) and that the ground does not emit longwave independently. A per-path
`source` node marker was considered to make this explicit in data and rejected for now as excessive;
a future `AbstractSkin` implementation (for example one with a canopy air space) may reintroduce it.

### 2. Resistance paths replace the topology marker

The network of a surface type is described by data rather than by a tag. Each vapor source has a path:

```julia
struct ResistancePath{NF, Rs <: Tuple{Vararg{AbstractResistance{NF}}}, F <: AbstractConductanceFactor{NF}}
    "Series chain of resistances from the source to the atmosphere"
    resistances::Rs
    "Dimensionless multiplier on the path conductance (e.g. β or f_can)"
    factor::F
end
```

The path conductance is `factor / sum(resistances)`. Series is "within a path", parallel is "across
paths", which is sufficient for any network without internal nodes. The `factor` slot is needed because
`β` and `f_can` are conductance multipliers, not resistances, and should not be shoehorned into the
resistance abstraction. A `UnitFactor` is the default.

The PALADYN configuration becomes a constructor over `CanopyGround`, not a separate type:

```julia
CanopyGround(NF;
    ground = ResistancePath((GroundCanopyResistance(NF), ConstantDragResistance(NF)), SoilMoistureFactor(NF)),
    canopy_evaporation = ResistancePath((ConstantDragResistance(NF),), WetCanopyFraction(NF)),
    transpiration = ResistancePath((ConstantDragResistance(NF), StomatalResistance(NF))),
)
PALADYNCanopyGround(NF; kwargs...)  # alias constructor with the defaults above
```

`BareGround(NF)` has a single `ground = ResistancePath((ConstantDragResistance(NF),), SoilMoistureFactor(NF))`.

Resistance and factor types introduced:

| Type | Replaces | Inputs |
|---|---|---|
| `ConstantDragResistance <: AbstractAerodynamicResistance <: AbstractResistance` | `aerodynamic_resistance(…, atmos)`, `ConstantAerodynamics` | drag coefficient `Cₕ`, windspeed |
| `GroundCanopyResistance <: AbstractResistance` | `aerodynamic_resistance(…, atmos, evtr, vegetation)`, `C_can` | LAI, SAI, windspeed |
| `StomatalResistance <: AbstractResistance` | inline `g_stm` read | `canopy_water_conductance` from vegetation |
| `SoilMoistureFactor`, `ConstantFactor`, `UnitFactor <: AbstractConductanceFactor` | `SoilMoistureResistanceFactor`, `ConstantEvaporationResistanceFactor` | soil saturation |
| `WetCanopyFraction <: AbstractConductanceFactor` | inline `saturation_canopy_water` read | canopy water from interception |

Kernel function signature for all resistances:

```julia
compute_resistance(i, j, grid, fields, r::AbstractResistance, surface::AbstractLandSurface, atmos::AbstractAtmosphere)
```

The `surface` argument lets `StomatalResistance` and `WetCanopyFraction` make their dependence on
vegetation and interception explicit, and gives a future roughness-based resistance access to
`surface.roughness`. `ConstantDragResistance` ignores it. Shared resistances
(the aerodynamic resistance appears in every path) are evaluated once per cell into a named tuple keyed
by resistance type, then each path sums its entries. Because the tuples are concretely typed, this
unrolls at compile time to the current arithmetic with no runtime branching.

Note: the `ResistancePath` type parameters above are in the same order as the fields, following the
repository convention; `@parameterized` support for `Tuple` fields must be checked during
implementation. If `ParameterEditing` cannot traverse the tuple, the paths will be declared with
`@component` on each resistance in a small fixed-arity wrapper instead.

### 3. The aerodynamic resistance moves from the atmosphere to the surface

`ConstantAerodynamics`, `AbstractAerodynamics`, `drag_coefficient`, and the `aerodynamics` field of
`PrescribedAtmosphere` are removed, as is `aerodynamic_resistance(i, j, grid, fields, atmos)`.

The aerodynamic resistance becomes a resistance parameterization on the surface:

```julia
abstract type AbstractAerodynamicResistance{NF} <: AbstractResistance{NF} end

@parameterized @kwdef struct ConstantDragResistance{NF} <: AbstractAerodynamicResistance{NF}
    "Bulk transfer coefficient for heat and moisture"
    @param Cₕ::NF (bounds = Positive,)
end
```

with `rₐ = 1 / (Cₕ Vₐ)`, exactly the current formula with the current default `Cₕ`. The abstract
intermediate type exists so that the sensible heat flux and the snow sublimation flux, which need
"the" aerodynamic resistance of the surface rather than any resistance, can dispatch on it, and so a
roughness-based implementation can be added later without touching the surface types (see Future
work). The `roughness` slot on the surface types is `nothing` until then. The sensible heat path is implicit: every network type has exactly one
`AbstractAerodynamicResistance` among its paths, and that one is used for sensible heat. A host-side
check in the keyword constructors enforces this.

`compute_snow_sublimation_flux` currently takes `atmos` and calls `aerodynamic_resistance` on it. It
will instead take the surface's aerodynamic resistance and `atmos` (for wind and humidity).

Because the formula and default are unchanged, this change is bit-for-bit reproducible.

### 4. Snow as a ground-node modifier

The current `f_snow` blending is kept unchanged in physics and moves wholesale into the surface types:

- `ImplicitSkinTemperature` keeps its area-weighted conduction inversion.
- The latent heat flux blends the network humidity flux (scaled by `1 − f_snow`) with the sublimation
  flux (scaled by `f_snow`), as now.
- The per-path ET fluxes are scaled by `1 − f_snow`, as now.

No separate snow source is added to the resistance network. This is documented on the surface types so
the two representations (area-fraction modifier vs. explicit network node) are not confused later.

### 5. Fused kernel

Each network surface type has one `solve_land_surface!` launch replacing `solve_surface_energy_balance!`
and the ET part of `SurfaceHydrology.compute_auxiliary!`. Per cell, the kernel:

1. computes all distinct resistances and factors, and from them the path conductances, into the
   auxiliary fields (`ground_evaporation_conductance`, `canopy_evaporation_conductance`,
   `transpiration_conductance`, kept with their current names);
2. runs the skin temperature solve (`ImplicitSkinTemperature`) or reads the input
   (`PrescribedSkinTemperature`), where the residual evaluates the radiative, sensible, latent, and
   ground heat fluxes from the stored conductances exactly as now;
3. stores the final per-path ET fluxes (`evaporation_ground`, `evaporation_canopy`, `transpiration`),
   the four energy fluxes, and the diagnosed `skin_humidity` at the converged skin temperature.

The separate `compute_auxiliary!` for the surface keeps only what does not depend on the skin
temperature and is needed by other processes before the solve: interception auxiliaries and albedo.
This resolves the `TODO` in both ET `compute_auxiliary_kernel!`s about merging conductances back into
`fields`, since conductances are now computed and consumed in the same kernel.

`compute_surface_energy_fluxes!` (the non-solving variant) is kept for `PrescribedLandSurface` and
for diagnostics.

### 6. Prescription

- `radiative_fluxes`, `ground_heat_flux`, and `albedo` keep their `Prescribed`/`Diagnosed` variants on
  all three surface types, unchanged.
- `PrescribedSkinTemperature` remains available as the `skin` of `BareGround` and `CanopyGround`; the
  paths are still evaluated, only the root solve is skipped.
- `PrescribedTurbulentFluxes` is removed from the network types. `PrescribedLandSurface` replaces the
  `PrescribedSurfaceEnergyBalance` alias and reads `skin_temperature`, `sensible_heat_flux`, and
  `latent_heat_flux` as inputs. `DiagnosedTurbulentFluxes` becomes the only turbulent flux scheme and
  is no longer a user-visible component; its kernel functions become methods on the surface types.
- `PrescribedLandSurface` must still provide `ground_evapotranspiration_flux` for the soil water sink.
  Default: derive the water flux from the prescribed latent heat flux as `E = H_l / (ρ_w L_v)`, which
  keeps the energy and water budgets mutually consistent. A prescribed `evapotranspiration` input is
  not added in this PR.

### 7. Surface runoff moves to `SoilHydrology`

`SoilHydrology` gains a `surface_runoff::AbstractSurfaceRunoff` field (default `DirectSurfaceRunoff`),
and the `surface_excess_water` prognostic owned by the runoff scheme moves with it. The runoff files move
from `src/processes/surface/runoff/` to `src/processes/soil/hydrology/runoff/`. `soil_coupled.jl`
no longer needs `get_surface_runoff(surface_hydrology)`.

The water reaching the ground crosses the surface/soil interface through one new interface method on
`AbstractLandSurface`:

```julia
ground_water_influx(i, j, grid, fields, surface::AbstractLandSurface, atmos, snow)
```

returning rainfall for `BareGround`, throughfall (rainfall minus interception) for `CanopyGround`, and
rainfall for `PrescribedLandSurface`. Snow-aware adjustments (meltwater, snowfall routed to the
snowpack) stay where they are now in the runoff scheme.

`ground_evapotranspiration_flux(i, j, grid, fields, surface)` is kept as the soil water sink interface,
with the ET scheme argument replaced by the surface.

### 8. Interception stays with the canopy surface

`CanopyGround` holds `canopy_interception::AbstractCanopyInterception` (default
`PALADYNCanopyInterception`). `NoCanopyInterception` becomes unreachable, since the no-canopy case is
`BareGround`, and is deleted along with the vegetation-free compatibility dispatch in
`evapotranspiration_base.jl`. The interception tendency continues to read the canopy evaporation flux
from the surface auxiliaries.

### 9. Model-level changes

- `LandModel`: the `surface_energy_balance` and `surface_hydrology` fields are replaced by one
  `surface::AbstractLandSurface`. Defaults: `BareGround(NF)` when `vegetation === nothing`, else
  `PALADYNCanopyGround(NF)`. The `compute_auxiliary!`, `compute_tendencies!`, `closure!`, and
  `invclosure!` methods pass `surface` where they currently pass `surface_hydrology` or
  `surface_energy_balance`.
- `SurfaceEnergyModel` and `SurfaceHydrologyModel` are merged into one `LandSurfaceModel` wrapping an
  `AbstractLandSurface` with a prescribed atmosphere and optional prescribed ground inputs.
- `compute_snow_interface_fluxes!` and `snow_interfaces.jl` take the surface instead of the SEB.
- `diagnostics/surface_fluxes.jl` dispatches on the surface instead of `SurfaceEnergyBalance` and
  the ET scheme.

### 10. Removed and renamed public names

Removed: `SurfaceEnergyBalance`, `PrescribedSurfaceEnergyBalance`, `SurfaceHydrology`,
`BareGroundEvaporation`, `PALADYNCanopyEvapotranspiration`, `NoCanopyInterception`,
`PrescribedTurbulentFluxes`, `DiagnosedTurbulentFluxes` (export only; type kept internal),
`ConstantAerodynamics`, `SoilMoistureResistanceFactor`, `ConstantEvaporationResistanceFactor`,
`SurfaceEnergyModel`, `SurfaceHydrologyModel`.

Removed (abstract): `AbstractSkinTemperature`, `AbstractSurfaceEnergyBalance`,
`AbstractSurfaceHydrology`, `AbstractEvapotranspiration`, `AbstractAerodynamics`.

Added: `AbstractLandSurface`, `AbstractSkin`, `AbstractRoughness` (no implementations yet), `BareGround`,
`CanopyGround`, `PALADYNCanopyGround`,
`PrescribedLandSurface`, `ResistancePath`, `ConstantDragResistance`, `GroundCanopyResistance`,
`StomatalResistance`, `SoilMoistureFactor`, `ConstantFactor`, `UnitFactor`, `WetCanopyFraction`,
`LandSurfaceModel`.

The package is pre-1.0 and no deprecation shims are added. Downstream users known to be affected:
`examples/simulations/speedy_wet_land.jl`, `test/reactant/setup.jl`,
`benchmark/model_configurations.jl`, and the NumericalEarth.jl coupling (handled in that repository).

### 11. Variable ownership after the refactor

| Variable | Owner | Kind |
|---|---|---|
| `skin_temperature` | `ImplicitSkinTemperature` / `PrescribedSkinTemperature` / `PrescribedLandSurface` | prognostic / input / input |
| `skin_humidity` | `BareGround`, `CanopyGround` | auxiliary |
| `ground_temperature` | skin scheme, `CanopyGround` | input |
| `*_conductance`, `evaporation_*`, `transpiration` | `BareGround`, `CanopyGround` | auxiliary |
| `sensible_heat_flux`, `latent_heat_flux` | network types / `PrescribedLandSurface` | auxiliary / input |
| radiative and ground heat flux variables | unchanged | unchanged |
| `canopy_water`, interception auxiliaries | `CanopyGround.canopy_interception` | unchanged |
| `surface_excess_water`, `surface_runoff`, `infiltration` | `SoilHydrology.surface_runoff` | unchanged kinds |

### 12. Suggested commit sequence

The change is large enough that it should be landed as a sequence of commits on one branch, each
leaving the test suite green:

1. Move surface runoff into `SoilHydrology` and add `ground_water_influx`. No behavior change.
2. Introduce `AbstractResistance`, `AbstractAerodynamicResistance`, `AbstractConductanceFactor`, and
   the concrete types; migrate `aerodynamic_resistance` callers; remove `ConstantAerodynamics`. No
   behavior change.
3. Introduce `AbstractLandSurface`, `BareGround`, `CanopyGround`, and the fused kernel, side by side
   with the old types. Add equivalence tests.
4. Switch `LandModel`, the standalone models, snow interfaces, and diagnostics to the surface type.
5. Add `PrescribedLandSurface`, delete the old types, update exports, docs, examples, and benchmarks.

Alternatively, steps 1 and 2 can be separate PRs since they are independently reviewable.

## Testing and verification

- **Equivalence tests** (step 3, then kept): `BareGround` reproduces
  `SurfaceEnergyBalance + BareGroundEvaporation` and `PALADYNCanopyGround` reproduces
  `SurfaceEnergyBalance + PALADYNCanopyEvapotranspiration` on the existing `LandModel` test
  configurations, for skin temperature, all four energy fluxes, and the three ET component fluxes.
  Since the arithmetic is unchanged, the target is bit-for-bit equality, with roundoff tolerance
  accepted only where the fused kernel reorders a floating-point sum. These tests are written against stored
  reference values from the base revision so they survive deletion of the old types.
- **Unit tests** per resistance and factor type, including the compile-time deduplication of shared
  resistances (one evaluation per cell, checked by a counting dummy resistance).
- **Prescribed tests**: `PrescribedLandSurface` with and without snow, including the derived
  evapotranspiration flux; `PrescribedSkinTemperature` on both network types.
- **Skin humidity**: on `BareGround` with a unit conductance factor, the diagnosed `skin_humidity`
  equals the saturation specific humidity at the converged skin temperature; with β < 1 it lies between
  the air humidity and saturation.
- **Snow tests**: existing snow/SEB partitioning tests (`test/surface/skin_temperature.jl`,
  `test/surface/turbulent_fluxes.jl`) ported to the new types unchanged in their assertions.
- **Runoff tests**: `test/surface/hydrology/surface_runoff_tests.jl` moves under `test/soil/` and
  gains a test for `ground_water_influx` on both network types.
- **Conservation**: the existing coupled water and energy budget checks in
  `test/coupled_models/land_model_tests.jl` must still pass.
- **Differentiability**: `test/differentiability/prescribed_seb_diff.jl` ported to
  `PrescribedLandSurface`; a new Enzyme test differentiating the fused `solve_land_surface!` of
  `PALADYNCanopyGround` with respect to `Cₕ` and `C_can`.
- **Reactant**: `test/reactant/setup.jl` configurations ported; the fused kernel must raise to
  StableHLO, which means the path evaluation must contain no reachable throw path.
- **Type stability and allocations**: `@inferred` and zero-allocation checks on the fused kernel
  function for both network types, following the existing pattern in `test/surface/seb_tests.jl`.
- Full suite via `Pkg.test()`, Enzyme suite via `Pkg.test(; test_args=["enzyme"])`, draft doc build.

## Documentation changes

- New section `docs/src/processes/land_surface/` replacing both `surface_energy/` and
  `surface_hydrology/`: overview (network concept, nodes and paths, snow as a ground-node modifier),
  `BareGround`, `CanopyGround`, `PrescribedLandSurface`, resistances and factors, skin (temperature
  and humidity, with the node-temperature conventions of each surface type stated explicitly),
  radiative fluxes, ground heat flux, albedo, canopy interception.
- `surface_runoff.md` moves under the soil hydrology section.
- `docs/src/processes/atmosphere/atmosphere.md` loses the aerodynamics section.
- `docs/src/models/land_model.md`, `docs/src/running/configuring.md`,
  `docs/src/extending/core_interfaces.md`, and `docs/src/extending/coupling_processes.md` updated
  for the new field names and interfaces.
- `docs/make.jl` page tree updated accordingly.
- Docstrings for every new type, `initialize!`/`compute_auxiliary!`/`compute_tendencies!` dispatch,
  and kernel function, per the repository rules. The PALADYN citation moves to `PALADYNCanopyGround`
  and `GroundCanopyResistance`.

## Known limitations

- The path representation covers only networks whose single shared node is the atmosphere. A canopy
  air space (Shuttleworth and Wallace, 1985) needs an internal node with its own temperature and
  humidity balance and cannot be expressed by adding paths. It will require a separate network
  abstraction, not an extension of `ResistancePath`.
- The aerodynamic resistance keeps the constant bulk drag coefficient. Surface roughness and stability
  corrections are deferred (see Future work).
- The skin is always saturated at its temperature. Sub-saturation is representable only through path
  factors and series resistances. `skin_humidity` is a diagnostic and feeds back into nothing.
- Which temperature each path is evaluated at is fixed by the surface type, not configurable.
- `PrescribedLandSurface` derives the evapotranspiration water flux from the latent heat flux and
  cannot represent a prescribed partitioning into ground evaporation, canopy evaporation, and
  transpiration.
- No deprecation path; downstream code must be updated in lockstep.

## Future work

- `LinearizedSkinTemperature` (Penman-Monteith). Linearizing the saturation humidity around the air
  temperature changes the humidity difference of every path, and with several source temperatures the
  closed form is a Shuttleworth-Wallace style derivation. The path structure in this plan does not
  assume one path, so it should accommodate this without redesign.
- Canopy air space surface type with an internal node. This is also the natural point to revisit a
  per-path source node marker and to separate the leaf, air space, and ground temperatures, which the
  present `skin` conflates for `CanopyGround`.
- **Joint skin state solve.** A new `AbstractSkin` implementation that solves skin temperature and skin
  humidity together as one fixed point, required once a stability-corrected aerodynamic resistance makes
  the transfer coefficients depend on the skin humidity.
- **Surface roughness** (deferred from Rev 1 as new physics). Outline: a first `AbstractRoughness{NF}`
  implementation, `ConstantRoughness`, holding the momentum and heat roughness lengths `z₀ₘ`, `z₀ₕ`,
  filling the `roughness` slot that this PR leaves as `nothing`; a second `AbstractAerodynamicResistance`
  implementation, `LogLawDragResistance`, computing the neutral bulk resistance
  `rₐ = ln(z_ref / z₀ₘ) ln(z_ref / z₀ₕ) / (κ² Vₐ)` from the roughness and the atmosphere's reference
  height (`altitude` on `PrescribedAtmosphere`). `ConstantDragResistance` stays as the alternative.
  Since the log law cannot reproduce a constant `Cₕ` at every reference height, that PR should decide
  the public default and regenerate reference values if it changes. `compute_resistance` already
  receives the surface, so `surface.roughness` can be reached without a signature change. A host-side
  constructor check should require a non-`nothing` roughness when `LogLawDragResistance` appears in any
  path. Follow-ups on the
  same track: stability-corrected (Monin-Obukhov) resistance, vegetation-dependent roughness.
- Blended canopy/ground albedo on `CanopyGround`.
- Explicit snow source in the resistance network, should the area-fraction modifier prove limiting.
