# PR-B: ground/soil heat-flux disambiguation

> Status: **completed**. Preliminary refactoring landed before the single-layer snow scheme: the
> surface energy balance closure flux keeps the standard name `ground_heat_flux` while the soil-top
> boundary condition is renamed `soil_heat_flux` (alias `SoilHeatFlux`).

Date of initial draft: 2026-07-10

Base revision: 462c6422b22829b88e491457894264419704d691

## Originating prompt

This PR was motivated by problem (1) in the snow design doc's revision-2 follow-up prompt (verbatim):

> Currently, `ground_heat_flux` is directly wired to the soil temperature state variable and is also
> used in the SEB. This becomes problematic here because our snow surface heat flux would be distinct
> from the conductive heat flux at the ground-snow interface. Investigate this ambiguity and suggest a
> workaround or design change to the current BC handling.

The drafting of this standalone document was requested in:

> Draft separate design documents for PR-A and PR-B (see design doc for single-layer snow scheme in
> `scratch/design/`). [...]

## Revision log

- The initial plan split the dual-role field into `surface_heat_flux` (SEB closure flux) and
  `ground_heat_flux` (soil-top BC). On the author's direction the naming was swapped: `ground_heat_flux`
  is retained for the SEB closure flux (standard terminology) and the *soil-top BC* is renamed
  `soil_heat_flux` (alias `SoilHeatFlux` → `SoilHeatFlux`). Net effect: the SEB source is unchanged;
  only the soil BC alias, its export, and the `LandModel` wiring changed.

## The ambiguity

`ground_heat_flux` is a single `XY` auxiliary `Field` that currently serves **two distinct roles**:

1. **SEB-internal closure flux.** The implicit skin-temperature solve writes it as
   `G = R_net + H_s + H_l` ([`skin_temperature.jl:104-107`](../../src/processes/surface/skin_temperature.jl#L104-L107),
   [`skin_temperature.jl:259-264`](../../src/processes/surface/skin_temperature.jl#L259-L264)),
   and reads it back to derive `Ts = Tg − G·Δz/(2κₛ)`
   ([`skin_temperature.jl:165-177`](../../src/processes/surface/skin_temperature.jl#L165-L177)).
   It is also written directly by the SEB driver
   ([`surface_energy_balance.jl:130`](../../src/processes/surface/surface_energy_balance.jl#L130)).
   This role is the *conductive flux from the skin into the medium immediately below it*.
2. **Soil-top boundary condition.** `initialize(model::LandModel)` wraps the same field as a
   `FluxBoundaryCondition` on soil `internal_energy` via `SoilHeatFlux(...)`
   ([`land_model.jl:58-67`](../../src/models/coupled/land_model.jl#L58-L67),
   [`soil_model_bcs.jl:6`](../../src/models/soil/soil_model_bcs.jl#L6)); the soil energy tendency
   consumes it through the top-boundary handling.

Without snow these two roles describe the **same interface** — the skin sits directly on the soil top —
so one field is consistent. **With snow they are two different fluxes:** the SEB closes the balance
against the *snow surface*, so `G` is the flux into the *top of the snowpack*, whereas the soil sees
conduction across the *snow base*, `Q_base(T_snow, T_soil_top)`. The two differ by snowpack energy
storage `dE/dt` plus advected meltwater heat. Feeding the SEB's `G` into the soil BC under snow would
inject the surface flux directly into the soil, bypass the snowpack, and break energy conservation.

## Design change: split the field by semantics

Decouple the two roles into two named `XY` fluxes with unambiguous meaning:

- **`ground_heat_flux`** — the SEB skin-temperature closure flux `G`: conduction from the skin into the
  medium directly beneath it (snow top, or ground top when no snow). Owned and written by the SEB. This
  keeps the standard surface-energy-balance terminology (ground heat flux `G`).
- **`soil_heat_flux`** — the conductive flux into the top of the *soil* column; feeds the soil energy
  BC (alias `SoilHeatFlux`). The name is unambiguously soil-specific: with snow the "ground" surface is
  the snow top, so the flux into the *soil* under the snow is a distinct quantity.

Coupling rule (dispatched on the uppermost medium):

- **No snow** (`NoSnow`): `soil_heat_flux ≡ ground_heat_flux`.
- **Snow present** (added by the snow PR, *not* this PR):
  `soil_heat_flux = f_snow·Q_base + (1 − f_snow)·ground_heat_flux`, where `Q_base` is the snow→soil
  conduction. `Q_base` depends only on `T_snow` and `T_soil_top` (both available after the soil and snow
  auxiliary passes, before the SEB), so there is no circular dependency with the skin-temperature solve.

PR-B implements only the **no-snow** rule and the renaming/wiring scaffold. It is therefore
snow-independent and a strict no-op on current behavior.

## Wiring strategy: static by model type (strategy A)

The snow component type is known at `initialize(model::LandModel)` time, so the soil-top energy BC can
be wired statically:

- Wire the soil-top `internal_energy` BC (`SoilHeatFlux`) directly to the SEB **`ground_heat_flux`**
  field when the snow component is `NoSnow` (today's only case). No separate `soil_heat_flux` state
  field is created — the soil BC reads the SEB closure flux exactly as before the split.
- The snow PR will instead create a distinct `soil_heat_flux` field (the blended flux) and wire the BC
  to it when snow is present.

No dual-writer field, no runtime branch, and the no-snow path is byte-for-byte identical to today (only
the field feeding the BC is renamed). The alternative — a single `soil_heat_flux` writer populated by
an extra coupling pass dispatched on the snow type (strategy B) — is more uniform but adds a pass and a
dual-writer; it is **not** adopted.

## File-level changes (as implemented)

The surface energy balance is left **untouched**: it keeps writing and reading its `ground_heat_flux`
closure flux (the standard SEB `G`). The disambiguation is achieved by renaming the *soil-top boundary
condition* only, so the diff is small.

- [`soil_model_bcs.jl`](../../src/models/soil/soil_model_bcs.jl): the soil-top energy BC alias
  `SoilHeatFlux` → **`SoilHeatFlux`**, its default variable `:ground_heat_flux` → `:soil_heat_flux`,
  and docstring updated ("net heat flux into the top of the soil column").
- [`models.jl`](../../src/models/models.jl): export `SoilHeatFlux` → `SoilHeatFlux`.
- [`land_model.jl:58-68`](../../src/models/coupled/land_model.jl#L58-L68): wire the soil-top BC with
  `SoilHeatFlux(ground_heat_flux)` (strategy A, single field). The SEB `ground_heat_flux` field feeds the
  soil BC directly in the no-snow case, so results are bit-for-bit unchanged. The snow PR will create a
  distinct `soil_heat_flux` field (the blended flux) and wire the BC to it.
- Docs: [`docs/src/models/soil_model.md`](../../docs/src/models/soil_model.md) `@docs` reference
  `SoilHeatFlux` → `SoilHeatFlux`.

The SEB source (`skin_temperature.jl`, `surface_energy_balance.jl`, `surface/abstract_types.jl`) and the
surface / skin-temperature / coupled tests are **unchanged**: `ground_heat_flux` remains the SEB closure
flux and `state.ground_heat_flux` is still the field the soil BC reads.

## On the `infiltration` water-flux analogue

The snow design doc proposed handling `infiltration` "in the same preliminary PR for consistency."
On inspection, the water flux does **not** have the same dual-role structure in the no-snow case:
`infiltration` is produced by the surface hydrology pass and consumed by the soil top BC (single
producer, single consumer), whereas `ground_heat_flux` is *written by the SEB and re-read both by the
SEB and by the soil*. There is consequently nothing to disambiguate for water until snow exists.

**Recommendation:** keep PR-B scoped to the energy-flux split only. The `infiltration` blend
(`infiltration = f_snow·M_r + (1 − f_snow)·(rainfall + throughfall) + …`) requires `f_snow` and `M_r`
and so naturally belongs to the snow PR, not to this snow-independent preliminary. This is a minor,
deliberate deviation from the snow doc's grouping; it keeps PR-B a verifiable no-op.

## Testing

- **No-snow regression (headline):** a `LandModel`/`SoilModel` run reproduces current results
  bit-for-bit. The soil energy tendency sees the identical flux it does today; only the BC alias name
  changes.
- **BC wiring:** the soil-top `internal_energy` BC condition is the SEB closure flux field
  `ground_heat_flux` in the no-snow configuration (`SoilHeatFlux(ground_heat_flux)`).
- **Skin-temperature unit tests:** unchanged; still exercise `ground_heat_flux`.
- **Doctests:** the `soil_model.md` `@docs SoilHeatFlux` reference resolves after the export rename.

## Dependencies and order

- Snow-independent; verifiable on its own by the no-snow regression.
- Suggested order **PR-A then PR-B**, though the two are largely independent and can be reordered. The
  snow PR depends on both: it adds the `f_snow`-blended `soil_heat_flux` field (wired to the soil BC in
  the snow branch of strategy A) computed from the SEB `ground_heat_flux` and the snow-base conduction.
