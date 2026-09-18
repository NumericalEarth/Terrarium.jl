# `PrescribedSurfaceEnergyBalance`: coupler-owned surface fluxes

> Status: **planned**. Draft design for a surface energy balance implementation that computes nothing
> internally and instead consumes externally-prescribed surface fluxes, enabling single-owner (Option A)
> coupling with NumericalEarth's `InterfaceComputations`.

Date of initial draft: 2026-09-01

Base revision: cc6b24090 (Terrarium.jl; the coupling work targets `bg/kernel-fusing`, tip `1bb5824ae` —
confirm the branch before implementing)

## Originating prompt

> [...] let's try to make Option A work with the `BulkTemperature` scheme and `apply_air_land_radiative_fluxes`
> using the `land_surface_energy_flux` method I added, which can be overridden in the extension. Let's write
> the ground heat flux directly into Terrarium, ignoring its side for now. We'll update it in a second step to
> have a `PrescribedSurfaceEnergyBalance` scheme that accepts all fluxes as prescribed, nothing calculated
> internally. Will that work?

This document is **step 2** of that plan. Step 1 (NumericalEarth extension plumbing) is already implemented:
the coupler now assembles the net surface energy flux (`𝒬ᵀ + 𝒬ᵛ` in Phase 3 via `update_net_fluxes!`, plus
the radiative part in Phase 4 via `apply_air_land_radiative_fluxes!`) directly into Terrarium's
`ground_heat_flux` field, which — without snow — *is* the soil-top heat-flux boundary condition
(`soil_heat_flux === ground_heat_flux`, [land_model.jl](../../../src/models/coupled/land_model.jl)).

## Revision log

- **Rev 1 (2026-09-01):** Initial draft. Energy-only prescribed SEB; vapor/moisture prescription deferred
  to a follow-up sub-step. *Awaiting human approval before implementation.*

## Problem description

In the coupled `EarthSystemModel`, NumericalEarth's `InterfaceComputations` is the single owner of the
atmosphere–land interface fluxes. With the current `SurfaceEnergyBalance` (SEB), Terrarium *also* computes a
surface energy balance internally (skin temperature, radiation, ground heat flux) inside its `time_step!`.
This double-computation is:

1. **Redundant** — the same physics runs in two places.
2. **Non-conservative / inconsistent** — the two SEBs generally disagree, and the empirical ERA5 example
   exposed a cold-column runaway traced to `G` closing against a stale, cross-stage-inconsistent `R_net`
   (see `numerical_earth/scratch/terrarium_ne_interface_design_discussion.md`).
3. **Self-defeating under coupling** — because Terrarium's `compute_auxiliary!` recomputes `ground_heat_flux`
   from its own SEB immediately before the soil tendency reads it, the coupler's write to `ground_heat_flux`
   (step 1) does not cleanly survive: the internal SEB overwrites it every step.

**Goal:** a surface energy balance implementation that computes *nothing* internally and treats all surface
fluxes as prescribed inputs, so the coupler-assembled net flux in `ground_heat_flux` is the authoritative
soil-top boundary condition. This realizes **Option A** ("NE owns the SEB, hands back net `G` as a pure BC").

## Background

### The current SEB structure

`SurfaceEnergyBalance{NF, SkinTemperature, TurbulentFluxes, RadiativeFluxes, Albedo}`
([surface_energy_balance.jl](../../../src/processes/surface/surface_energy_balance.jl)) composes four
sub-processes and, via `solve_surface_energy_balance!` and `compute_surface_energy_fluxes!`, solves the skin
temperature and closes `G = R_net + H_s + H_l` (all fluxes **positive upward**,
[skin_temperature.jl:137](../../../src/processes/surface/skin_temperature.jl#L137)).

The sub-processes already include *prescribed* variants (`PrescribedTurbulentFluxes`,
`PrescribedRadiativeFluxes`, `PrescribedSkinTemperature`) that read input fields rather than computing. The
current coupled example composes these — but the SEB *driver* still runs `solve_surface_energy_balance!`,
which recomputes `ground_heat_flux` as the residual, defeating the coupler's write. The gap is a driver that
does no solve at all.

### Sign convention (verified)

Terrarium `ground_heat_flux = R_net + H_s + H_l`, positive upward, equals NumericalEarth's
`surface_energy_flux` term-for-term (`𝒬ᵀ + 𝒬ᵛ − ΣQ_rad`, also positive upward). **No sign flip** is needed
when the coupler writes into `ground_heat_flux`.

### Field wiring

Without snow, `soil_heat_flux === ground_heat_flux` (same `Field` object), and the soil energy BC is
`SoilHeatFlux(soil_heat_flux)`. So making `ground_heat_flux` a *prescribed input* (not a computed auxiliary)
routes the coupler's value straight to the soil column's Neumann top BC.

## Summary of changes

### 1. New type `PrescribedSurfaceEnergyBalance <: AbstractSurfaceEnergyBalance{NF}`

A minimal SEB whose interfaces are no-ops (or pass-throughs):

- `solve_surface_energy_balance!(state, grid, ::PrescribedSurfaceEnergyBalance, ...)` → **no-op**. It must
  *not* write `ground_heat_flux`, `skin_temperature`, or any flux — those are inputs.
- `compute_surface_energy_fluxes!(...)` → **no-op** for the same reason.
- `compute_auxiliary!(...)` → **no-op** (or, at most, diagnostic copies; see open items on albedo/LW_up).
- `initialize!(...)` → initialize the prescribed input fields to sensible defaults (e.g. `ground_heat_flux = 0`).

### 2. Variable classification

`variables(::PrescribedSurfaceEnergyBalance)` declares the surface fluxes as **`input`** (externally set),
not `auxiliary` (internally computed). Minimally:

- `input(:ground_heat_flux, XY(), units = u"W/m^2")` — the coupler-assembled net surface energy flux
  (positive upward); consumed directly as the soil BC.
- `input(:skin_temperature, XY(), units = u"°C")` — prescribed for diagnostics and any downstream use
  (under Option A, NE computes `LW_up` from its own interface temperature, so Terrarium's `skin_temperature`
  is diagnostic only).

This is the crux: reclassifying `ground_heat_flux` from `auxiliary` (as `ImplicitSkinTemperature` declares it)
to `input` is what stops it from being overwritten.

### 3. Model wiring

`SoilModel` / `LandModel` must accept `PrescribedSurfaceEnergyBalance` in the `surface_energy_balance` slot,
and the `ground_heat_flux`/`soil_heat_flux` BC wiring in
[land_model.jl:65-84](../../../src/models/coupled/land_model.jl#L65) must continue to route the (now input)
`ground_heat_flux` field to `SoilHeatFlux`. Verify the snow path (`soil_heat_flux !== ground_heat_flux`)
either errors clearly or is explicitly out of scope for the prescribed SEB in this step.

### 4. NumericalEarth extension (already partly done)

Step 1 already assembles the net flux into `ground_heat_flux` and overrides `land_surface_energy_flux`.
Once the prescribed SEB lands, the extension's `update_net_fluxes!` pushes of `skin_temperature`,
`sensible_heat_flux`, and `latent_heat_flux` into now-unused internal SEB inputs become dead and should be
trimmed to just what the prescribed SEB and hydrology consume.

## Testing and verification

1. **Unit: no overwrite.** Construct a `LandModel` with `PrescribedSurfaceEnergyBalance`, set
   `ground_heat_flux` to a known field, run one `timestep!`, and assert the soil tendency saw exactly that
   flux (i.e. `ground_heat_flux` is unchanged by `compute_auxiliary!`).
2. **Unit: energy conservation.** With a prescribed constant `G`, verify the soil column's integrated
   internal-energy change over N steps equals `G · area · Δt · N` (within tolerance).
3. **Type stability / allocations.** `@code_warntype` / `@allocated` on `compute_auxiliary!` and the timestep
   for the prescribed SEB (must remain allocation-free and type-stable, per AGENTS.md kernel rules).
4. **Differentiability.** Enzyme adjoint through a `timestep!` w.r.t. the prescribed `ground_heat_flux`
   (the whole point of Option A is a clean differentiable BC seam).
5. **Coupled: runaway gone.** Re-run `examples/era5_forced_terrarium_land.jl` (NumericalEarth) with the
   prescribed SEB; the cold-column min should track the delivered forcing instead of collapsing, and the
   stored `ground_heat_flux` should equal `R_net + H + LE` by construction.
6. **Draft doc build** after implementation (`julia --project=docs docs/make.jl --local --draft`).

## Documentation changes

- New docstrings for `PrescribedSurfaceEnergyBalance` and its `initialize!`/`compute_auxiliary!` no-op
  dispatches (with `# References` / equation notes stating the positive-upward convention and the
  `G = surface_energy_flux` identity).
- Add an "Implementations" subsection under the surface energy balance doc page, with a `canonical = false`
  docstring and a warning that this scheme performs no internal balance (all fluxes prescribed).

## Known limitations

- **Energy-only in this step.** Moisture (evaporation/vapor flux) is *not* yet prescribed. If NE owns `LE`
  while Terrarium's hydrology still computes its own evapotranspiration, moisture is double-counted the same
  way energy was. This step deliberately scopes energy only; the moisture seam is the immediate follow-up
  (a `PrescribedSurfaceHydrology`-style vapor-flux input, mirroring this design).
- **Option A-lite (BulkTemperature).** Paired with the coupler's `BulkTemperature`, the coupling is still
  explicit-lag (fluxes at last step's soil-top `T`); stability relies on the top soil cell's heat capacity,
  which is ample at `Δt = 5 min`, `Δz = 0.1 m`, but not a substitute for a true implicit skin solve at large
  `Δt`. `SkinTemperature` (true Option A) remains the more robust target and is compatible with this scheme.
- **Snow.** With snow, `soil_heat_flux !== ground_heat_flux` and a blended conductive flux intervenes; the
  prescribed SEB's interaction with the snow energy balance is out of scope here and must be designed
  separately.

## Future work

- Prescribed vapor/moisture flux (`PrescribedSurfaceHydrology` or equivalent) to close the moisture budget
  under NE ownership.
- True implicit Option A via NE's `SkinTemperature(internal_flux)` with a Terrarium ground-conductance law
  (see the `TerrariumGroundFlux` sketch in the NE design-discussion counter-proposal), removing the residual
  explicit lag.
- The delegated `surface_interface_inputs` seam (NE calls Terrarium's differentiable surface-constitutive
  kernels) — the broader architectural direction this step is a concrete first move toward.
