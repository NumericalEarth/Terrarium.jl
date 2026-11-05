# Simple single-layer snow scheme

> Status: **completed**. Phases 1–5 (process, closure, mass/energy tendencies, standalone `SnowModel`,
> SEB coupling, and `LandModel` integration) are implemented and tested; the process parameters have been
> decomposed into composable sub-parameterizations (see Revision 5); the snow→soil water coupling,
> sublimation, and differentiability/Reactant/conservation tests have landed (see Revision 6); the surface
> latent flux is partitioned between sublimation and evapotranspiration (see Revision 8); and the
> sublimation energy-reference correction plus latent-heat-constant consistency have landed (see
> Revision 9), closing the item flagged in Revision 7. The snow documentation pages are written. Two
> non-blocking follow-ups are tracked under [Open questions / deferred work](#open-questions--deferred-work):
> the full-timestep Enzyme autodiff LLVM failure (items 7) and a real Reactant `:snow_column` CI run (item 8).

Date of initial draft: 2026-07-01

Base revision: 462c6422b22829b88e491457894264419704d691

## Originating prompt

> Please review the existing Terrarium processes and prepare a plan for the implementation of a
> simple, single-layer snow scheme. The scheme should be loosely based on the Utah energy balance
> model (see attachment) and should consist of a single layer with homogeneous density and thermal
> properties. The surface temperature should be handled by the surface energy balance whose
> interface will need to be modified to optionally except an AbstractSnow type. Write your initial
> plan into the `scratch/design` folder. Include this prompt text at the top. Feel free to ask any
> clarifying questions that arise.

## Revision log

Follow-up prompt (revision 2), verbatim:

> Make sure to append my prompts to the design document to keep track of this process. Regarding the
> open question, I agree that this is a problem and would update the plan to make the depth-integrated
> energy content the prognostic variable while exposing the volumetric energy content to the closure.
> This should be done via a method and without allocating an auxiliary variable.
>
> A couple of additional problems need to be addressed, however, before starting the implementation:
> 1. Currently, `ground_heat_flux` is directly wired to the soil temperature state variable and is
>    also used in the SEB. This becomes problematic here because our snow surface heat flux would be
>    distinct from the conductive heat flux at the ground-snow interface. Investigate this ambiguity
>    and suggest a workaround or design change to the current BC handling.
> 2. The existing soil energy and hydrology code is set up and named to be for soil. However, much of
>    the physics is transferrable as noted in the design plan. We should consider an initial
>    refactoring where the existing "soil" module is renamed to "ground" with soil-specific
>    parameterizations where appropriate. This should be done as a separate, preliminary work item
>    (and associated pull request) to avoid too many changes in one go.
> Please provide your feedback regarding these points and update the plan accordingly.

Resolutions adopted in this revision: (i) the prognostic snow energy is the **depth-integrated**
(column) energy content (J/m²); the volumetric energy seen by the closure is computed by a **method**
with no auxiliary field (see "Physical formulation"). (ii) The `ground_heat_flux` dual-role ambiguity
is addressed by a heat-flux split (see "Boundary-condition disambiguation"). (iii) The soil/ground
refactor is adopted as a scoped, abstraction-first preliminary work item (see "Preliminary refactoring
work items").

Revision 3 (2026-07-11) — **plan reconciliation** after the preliminary PRs landed. PR-A and PR-B are
now **complete**; this plan is reconciled to their outcomes (physics unchanged):

- The generic energy machinery lives in a top-level `thermodynamics/` module: `AbstractThermodynamics`,
  the `ExplicitTwoPhaseHeatConduction` operator, and the medium-agnostic `FreeWater` enthalpy maps
  (`liquid_water_fraction` / `energy_to_temperature`, in `thermodynamics/enthalpy.jl`). The snow closure
  reuses these directly.
- The soil energy process is `SoilThermodynamics` (was `SoilEnergyBalance`) under `soil/energy/`; the
  `soil/` tree was **kept** (not renamed to `ground/`).
- The heat-flux split (PR-B) keeps `ground_heat_flux` as the SEB closure flux `G` and introduces
  `soil_heat_flux` (alias `SoilHeatFlux`) as the soil-top BC. **The snow blend therefore writes
  `soil_heat_flux`, not `ground_heat_flux`.**
- Surface processes now live under a single `surface/` tree (was `surface_energy/` + `surface_hydrology/`).

Revision 4 (2026-07-12) — **implementation record** (phases 1–5 complete and tested). Delivered:

- Phases 1–4 as planned: `SingleLayerSnow`/`NoSnow` process, `ConstantSnowDensity` scheme, the
  depth-integrated enthalpy closure (`SnowEnergyTemperatureClosure`, temperature clipped at 0°C with the
  excess derived on demand from `U_v > 0`), the mass/energy tendencies (`compute_snow_tendencies!` with
  Darcy meltwater outflow), the standalone `SnowModel`, and the snow-aware SEB path (`DiagnosticAlbedo`
  blending albedo/emissivity by `f_snow`, and the snow-weighted `ground_thermal_interface`).
- Phase 5 `LandModel` integration in two steps. **5a** added `@component snow = NoSnow` and threaded the
  snow through `initialize!`/`compute_auxiliary!`/`compute_tendencies!`/`closure!`/`invclosure!`; the
  no-snow model is byte-for-byte unchanged. **5b** added the snow↔surface/soil **energy** coupling in
  `src/models/coupled/snow_coupling.jl`:
  - `Q_base` uses the **snow-resistance-only** closure `Q_base = 2·κ_snow·(T_soil − T_snow)/d_snow`
    (author's decision; the snowpack dominates the series resistance, so the soil-side conductivity need
    not be threaded into the snow coupling). Written to `basal_heat_flux` before the SEB.
  - After the SEB: `surface_heat_flux ← ground_heat_flux` (the snow's top loss `G`) and the soil-top BC
    flux `soil_heat_flux = f_snow·Q_base + (1 − f_snow)·G`. `soil_heat_flux` is a declared coupling
    auxiliary (via `snow_coupling_variables`) so it is accessible in the state and backs the soil BC.
  - Static **strategy A** BC wiring: `NoSnow` wires the soil-top BC to `ground_heat_flux` directly;
    `SingleLayerSnow` wires it to the blended `soil_heat_flux` (`snow_soil_bc_flux`).
  - The SEB field-gather now merges the snow thermal auxiliaries (`seb_conduction_fields`) so the
    snow-aware `ground_thermal_interface` can be evaluated.
- Verification: coupled land tests 30/30 (incl. a new snow-coupled testset), standalone snow tests 51/51,
  skin-temperature tests 30/30. No regressions.

**Deferred to a follow-up** (not part of this energy-coupling milestone): (i) the **water** coupling —
routing snow meltwater outflow into soil infiltration and partitioning rain-on-snow vs. bare-ground
throughfall; (ii) **sublimation** driven by the SEB latent-heat flux (currently the `sublimation` input
defaults to zero in the coupled model); (iii) an **Enzyme** differentiability test for the snow
tendencies/closure.

Revision 5 (2026-07-13) — **parameterization decomposition.** The `SingleLayerSnow` process was refactored
from a flat set of scalar parameters into composable sub-parameterization components, mirroring the soil
hydraulics/thermal-properties pattern and consistent with Terrarium's modularity principle (each component
is independently swappable):

- `SingleLayerSnow{NF, Cover, Density, Conductivity, Hydraulics, Closure}` now holds five `@component`
  fields: `cover`, `density`, `thermal_conductivity`, `hydraulic_properties`, and `closure`. The former
  scalar parameters moved into the corresponding schemes.
- **Cover** ([snow_cover.jl](../../src/processes/snow/snow_cover.jl)): `AbstractSnowCover{NF}` with
  `FractionalSnowCover` (`f_snow = W/(W + W_ref)`, parameter `half_coverage`, negative-SWE clamped to zero).
  `compute_snow_cover_fraction` now dispatches on the cover scheme.
- **Density** ([snow_density.jl](../../src/processes/snow/snow_density.jl)): `AbstractSnowDensity{NF}` with
  `ConstantSnowDensity` (parameter `density`).
- **Thermal conductivity** ([snow_thermal_conductivity.jl](../../src/processes/snow/snow_thermal_conductivity.jl)):
  `AbstractSnowThermalConductivity{NF}` with `PowerLawSnowThermalConductivity` (default),
  `LogarithmicSnowThermalConductivity`, and `QuadraticSnowThermalConductivity`. All expose the shared
  (exported) generic `compute_thermal_conductivity(cond, constants::MaterialConstants, ρ_snow)` — the snow
  conductivity is recovered lazily at the call sites (SEB conduction target, basal flux) rather than stored
  as an auxiliary field.
- **Hydraulic properties** ([snow_hydraulic_properties.jl](../../src/processes/snow/snow_hydraulic_properties.jl)):
  `AbstractSnowHydraulics{NF}` with `ConstantSnowHydraulics` (parameters `saturated_conductivity`,
  `capillary_retention`). `compute_meltwater_outflow` now dispatches on the hydraulics scheme.
- **Closure**: `SnowEnergyTemperatureClosure` (unchanged), retrieved via `get_closure(snow) = snow.closure`.

The snow `@kernel`s that wrap the shared `energy_to_temperature!`/`temperature_to_energy!` kernel functions
were given snow-specific names (`snow_energy_to_temperature_kernel!`, `snow_temperature_to_energy_kernel!`)
because KernelAbstractions `@kernel` does not support multiple dispatch on one kernel name — the soil energy
closures define 3D (XYZ) kernels under the base names.

Revision 6 (2026-07-21) — **water/sublimation coupling and test coverage** (previously deferred in
Revision 4). Delivered:

- **Water coupling** (author's choice: route through the existing infiltration/runoff partition). The
  surface runoff scheme's water input is now snow-aware: `influx = (1 − f_snow)·rainfall_ground + M_r`,
  where the snow-covered fraction intercepts rain into the snowpack and `M_r` is the meltwater outflow draining
  from the snowpack base (`soil_surface_water_flux` in `snow_mass.jl`). `snow` is threaded (optional arg) through
  `SurfaceHydrology`→`DirectSurfaceRunoff`, mirroring the SEB. Meltwater is therefore subject to the
  infiltration capacity limit, with the remainder becoming surface runoff.
- **Sublimation** driven by the SEB latent-heat flux: the post-SEB coupling (renamed
  `compute_snow_interface_fluxes!`) now sets `sublimation = f_snow·H_l/(ρ_w·L_s)` alongside the blended
  `soil_heat_flux`. Known limitation: the latent flux is not yet partitioned between snow sublimation and
  soil/canopy evapotranspiration, so the two over-count when both are active.
- The snow-field gather helper `seb_conduction_fields` was removed
- **Tests**: standalone melt energy↔mass conservation (`snow_model_tests.jl`); coupled water-conservation
  (`infiltration + surface_runoff ≈ M_r`, `land_model_tests.jl`); function-level Enzyme adjoints of the
  snow-specific physics — meltwater outflow, cover fraction, depth, thermal conductivity, basal flux
  (`differentiability/snow_model_diff.jl`, registered under the Enzyme suite); a CPU-vs-Reactant
  `:snow_column` correctness case (`test/reactant/setup.jl` + `runtests.jl`).

Known AD limitation: reverse-mode autodiff of the *full* `SnowModel` timestep currently fails to compile
under Enzyme with an internal `LLVM error: Canonicalization failed` (in `EnzymeCreateAugmentedPrimal`).
The snow closure reuses the `FreeWater` maps (adjoints covered in `soil_energy_diff.jl`) and the
snow-specific physics is differentiable in isolation, so the failure is isolated to the full-model
reverse pass; it is documented in `snow_model_diff.jl` for separate investigation.

Still deferred: partitioning the surface latent flux between snow sublimation and soil/canopy ET; a snow
process documentation page; resolving the full-timestep Enzyme limitation; a real Reactant CI run of the
`:snow_column` case (the `test/reactant` environment was not instantiated in the dev session).

Revision 7 (2026-07-21) — **enthalpy-reference correction of the advective flux terms.** The `FreeWater`
closure references internal energy to liquid water at 0 °C (`U = 0`; the phase-change band is
`U ∈ [-Lθ, 0]` with ice at 0 °C = `-Lθ`), consistent with the soil energy. The snow mass/energy
tendencies, however, had computed the advective (mass-carrying) heat terms against an *ice* reference,
an inconsistency of `ρ_w·L_f` per unit mass. Corrected in `snow_mass.jl` (`compute_snow_precip_heat_flux`,
renamed from `compute_rain_heat_flux`):

- **Meltwater** drains as liquid water at 0 °C (`U = 0`), so it carries no enthalpy — the previous
  `Q_melt = ρ_w·L_f·M_r` term is dropped. The energy tendency is now `dE/dt = Q_base − Q_top + Q_precip`
  (no meltwater term). The old term spuriously refroze draining meltwater (`dW_ice/dt = +M_r`).
- **Rain-on-snow** carries only sensible heat `c_w·max(T_air,0)` above the liquid reference; the previous
  `+L_f` double-counted the freezing latent heat (which the enthalpy closure already captures implicitly).
- **Fresh snowfall** is ice, `L_f` below the liquid reference, so it carries `c_i·min(T_air,0) − L_f`; the
  previous term omitted `−L_f`, which caused fresh cold snow to be diagnosed near 0 °C instead of `T_air`.

`compute_snow_tendencies!` was also split into `compute_snow_water_tendency` and
`compute_snow_energy_tendency` helpers (equations no longer buried in the kernel function). New/updated
tests: draining meltwater conserves energy (`dE ≈ 0`); fresh snowfall accretes at the snowpack's specific
enthalpy (`dE/dW = E/W`, catching the missing `−L_f`); the standalone snowfall/melt assertions were
updated to the liquid reference.

Open follow-up (not yet resolved): the **sublimation** energy term is subject to the same reference
subtlety. Sublimated ice mass carries `−L_f` enthalpy in the liquid reference, so under this reference the
energy budget implies an additional `+ρ_w·L_f·E_subl` term beyond the SEB latent flux already in `Q_top`;
this is entangled with the SEB's latent-heat convention (`L_v` vs `L_s`) and the still-deferred
latent-flux partitioning, and is left for that work item.

Revision 8 (2026-07-21) — **latent-flux partitioning between sublimation and evapotranspiration.** The
surface latent flux is now split by snow-covered area fraction `f_snow`, threading `snow` through the
surface energy balance as an optional argument (author's directive):

- **SEB latent heat flux** (`compute_latent_heat_flux`, turbulent_fluxes.jl): `H_l = (1 − f_snow)·L_v·ρₐ·Q_h_ground
  + ρ_w·L_s·E_subl`, i.e. ground/canopy evaporation (latent heat of vaporization) over the snow-free
  fraction plus snow sublimation (latent heat of sublimation) over the snow-covered fraction. Reduces to
  the original flux without snow (`f_snow = 0`).
- **Snow sublimation** (`compute_snow_sublimation_flux`, shared helper): the snow-fraction, saturated
  bulk-aerodynamic vapor flux `f_snow·ρₐ·Δq(T_skin)/rₐ/ρ_w`, with `Δq` taken **over ice** for the
  sub-freezing surface (the existing `saturation_specific_humidity_vapor` already dispatches over ice for
  `T ≤ 0`). This one flux drives both the `H_l` snow term and the snow's `sublimation` mass sink, so the
  energy and mass are consistent. It replaces the previous `f_snow·H_l/(ρ_w·L_s)` estimate.
- **Ground evaporation** (`bare_ground_evaporation.jl`): `evaporation_ground` is scaled by `(1 − f_snow)`
  at its source, so both the SEB latent flux ground term and the soil-water ET sink (`forcing`) see the
  reduced flux — no water/energy double-count over the snow-covered fraction. Canopy transpiration is left
  unchanged over snow (a documented simplification; only bare-ground evaporation is scaled).
- `snow` is threaded through `SurfaceHydrology`/`DirectSurfaceRunoff`/`BareGroundEvaporation` and the SEB
  solve chain (`solve_skin_temperature!`/residual → `compute_surface_energy_fluxes!`). The snow coupling
  (`compute_snow_interface_fluxes!`) now takes `atmos` to evaluate the sublimation flux at the
  converged skin temperature.

Test: `LandModel: latent flux partitioned` — under near-full snow cover, ground evaporation is suppressed
to `< 10%` of the snow-free value while sublimation carries the flux. This resolves the double-counting
noted in Revision 4/6. The sublimation energy-reference subtlety flagged in Revision 7 remains open.

Revision 9 (2026-07-23) — **sublimation energy-reference correction** (closes the open item from
Revision 7) and **thermodynamic consistency of the latent-heat constants**.

- **Constants** (constants.jl): the tabulated latent heats were internally inconsistent (`L_v + L_f`
  exceeded the tabulated `L_s` by ~400 J/kg). `latent_heat_fusion` is now the base value `3.3355e5` and
  `latent_heat_sublimation` is derived as `L_f + L_v`, enforcing `L_s = L_f + L_v` exactly. The `show`
  doctests and the constants documentation table were updated (fusion-first field order; `L_s = 2.83435e6`).
- **Snow energy tendency** (`compute_snow_energy_tendency`, snow_mass.jl): added the advective sublimation
  correction `Q_subl = ρ_w·L_f·E_subl`. `Q_top` (the SEB residual) removes the full sublimation enthalpy
  `ρ_w·L_s·E_subl` via the surface latent flux, but the mass leaving the snowpack departs as ice (enthalpy
  `−L_f` relative to the liquid-water-at-0 °C reference). Adding back `ρ_w·L_f·E_subl` leaves a net pack
  loss of exactly `ρ_w·L_v·E_subl` (the vaporization enthalpy of the departing vapor), the ice→vapor
  analogue of the meltwater term. With the now-consistent constants, `L_f` and `(L_s − L_v)` are identical,
  so `L_f` is used directly. The SEB latent flux and skin-temperature solve are unchanged (they correctly
  see `L_s`).

Test: `conservation: sublimation nets the vaporization enthalpy` (snow_model_tests.jl) — with `Q_top`
set to the SEB's `ρ_w·L_s·E_subl` contribution, the snow energy tendency equals `−ρ_w·L_v·E_subl`. Docs
updated (snow.md, snow_energy.md energy-balance equations).

## Resolved design decisions

The following forks were resolved with the model author before drafting:

1. **Thermal state representation — depth-integrated energy prognostic, volumetric closure
   (revised).** The prognostic snow energy is the **depth-integrated (column) energy content** `E`
   (J/m²), which is the conserved quantity and keeps the energy tendency free of the moving-boundary
   `1/d_snow` factor that the original volumetric prognostic introduced. The **volumetric** energy
   `U_v = E/d_snow` is exposed to the shared `FreeWater` enthalpy maps
   ([`thermodynamics/enthalpy.jl`](../../src/processes/thermodynamics/enthalpy.jl)) via a
   **method**, with no auxiliary field allocated for `U_v`. Snow layer depth `d_snow` is a diagnostic
   derived from SWE and a homogeneous bulk density. This realizes the original intent (reuse the
   volumetric phase-change closure) while fixing the conservation/regularization concern raised in
   review.
2. **Scope — process first, then couple.** Deliver an `AbstractSnow` process plus a minimal
   standalone `SnowModel` that can be unit- and Enzyme-tested in isolation, *then* wire it into
   `LandModel` as an optional component (default `NoSnow`).
3. **Meltwater — continuous Darcy-type outflow.** Liquid fraction is diagnosed from energy content;
   meltwater drains to the soil surface as a continuous outflow flux (UEB eqns 23–24), preserving
   continuous-time, differentiable dynamics per `AGENTS.md`.
4. **Surface transition — fractional snow cover.** A diagnosed sub-grid snow-covered area fraction
   `f_snow ∈ [0,1]` blends snow-surface and bare-soil/soil contributions in the surface energy
   balance and in the soil boundary fluxes.

## Background: the Utah Energy Balance (UEB) model

Tarboton, Chowdhury & Jackson (1994) describe a lumped, depth-averaged, single-layer snowpack with
two prognostic state variables: water equivalence `W` (m) and energy content `U` (kJ/m²) relative to
ice at 0 °C. The governing ODEs (their eqns 1–2) are

```
dU/dt = Q_sn + Q_li + Q_p + Q_g − Q_le + Q_h + Q_e − Q_m
dW/dt = P_r + P_s − M_r − E
```

with net shortwave `Q_sn`, incoming/outgoing longwave `Q_li`/`Q_le`, advected precipitation heat
`Q_p`, ground heat flux `Q_g`, sensible/latent fluxes `Q_h`/`Q_e`, advected meltwater heat `Q_m`,
rain/snow inputs `P_r`/`P_s`, meltwater outflow `M_r`, and sublimation `E`. Depth-averaged snow
temperature follows from `(U, W)` by phase partitioning (eqns 6–8); the **snow surface temperature**
`T_s` is solved separately from an equilibrium surface energy balance (eqns 20–22) because snow is a
strong insulator and `T_s ≠ T_avg`. Meltwater outflow uses a Darcy relation on the liquid fraction
in excess of capillary retention (eqns 23–24).

### Adaptation to Terrarium

- UEB's predictor–corrector time integration is **not** carried over. Terrarium supplies the
  integrator; the snow scheme contributes only well-formed ODE tendencies via `compute_tendencies!`.
- UEB's effective soil interaction depth `D_e` (which lumps a 0.4 m soil slab into `U`) is **dropped**.
  Terrarium already resolves an explicit multi-layer soil column, so the snow energy is snow-only and
  snow↔soil exchange is an explicit conductive flux at the snow base. This is a deliberate departure.
- UEB's equilibrium surface-temperature parameterization is realized by Terrarium's existing
  **skin-temperature solve** in the surface energy balance, which already solves the same equilibrium
  balance via a Newton iteration ([`skin_temperature.jl`](../../src/processes/surface/skin_temperature.jl),
  [`surface_energy_balance.jl`](../../src/processes/surface/surface_energy_balance.jl)). The
  snow scheme supplies the *sub-surface* side of that balance (temperature, conductance, layer
  thickness) instead of the soil top layer.
- Precipitation phase partitioning is already available upstream: `RainSnow`
  ([`prescribed_atmosphere.jl`](../../src/processes/atmosphere/prescribed_atmosphere.jl)) provides
  `rainfall` and `snowfall` inputs. The snow scheme consumes these directly rather than re-deriving
  the temperature-threshold partition (UEB eqn 12).

## How the relevant existing machinery works

These are the integration points the scheme attaches to (verified against current source):

- **Skin temperature / SEB.** `SurfaceEnergyBalance{NF, SkinTemperature, TurbulentFluxes,
  RadiativeFluxes, Albedo}` ([`surface/surface_energy_balance.jl`](../../src/processes/surface/surface_energy_balance.jl)).
  The `ImplicitSkinTemperature` solve closes `R_net(T_s) + H_s(T_s) + H_l(T_s) − G(T_s, T_g) = 0`
  (positive-upward convention). The conduction term uses
  `compute_skin_temperature(skinT, Tg, G, Δz) = Tg − G·Δz/(2κₛ)`, where `Tg` is the `ground_temperature`
  input and `Δz` is the **topmost soil cell** thickness
  ([`surface/skin_temperature.jl`](../../src/processes/surface/skin_temperature.jl)). An in-code TODO in
  `compute_skin_temperature` already anticipates pulling `κₛ`/thickness from snow/canopy.
- **Ground temperature source.** `SoilThermodynamics` registers `ground_temperature` as a `view` of the
  uppermost soil layer, with the TODO "Revisit this if/when we extend the vertical layers to include
  snow and canopy" ([`soil/energy/soil_energy.jl`](../../src/processes/soil/energy/soil_energy.jl)).
  The snow scheme is the resolution of that TODO.
- **Soil top boundary fluxes.** `StateVariables(model::LandModel)` wires the `soil_heat_flux` BC (alias
  `SoilHeatFlux`, fed by the SEB `ground_heat_flux` field in the no-snow case) as a
  `FluxBoundaryCondition` on soil `internal_energy`, and `infiltration` (negated) as a flux on
  `saturation_water_ice` ([`land_model.jl`](../../src/models/coupled/land_model.jl),
  [`soil_model_bcs.jl`](../../src/models/soil/soil_model_bcs.jl)). The snow scheme changes *how the
  `soil_heat_flux` and `infiltration` BC values are computed* when snow is present (the blend below).
- **Coupling order.** `LandModel.compute_auxiliary!` runs atmos → soil → vegetation →
  surface_hydrology → SEB; `compute_tendencies!` runs surface_hydrology → soil → vegetation
  ([`land_model.jl`](../../src/models/coupled/land_model.jl)).
- **Enthalpy closure to reuse.** The medium-agnostic `FreeWater` maps `liquid_water_fraction(fc, U, Lθ, sat)`
  and `energy_to_temperature(fc, U, Lθ, C)` operate on volumetric energy `U`, latent-heat content `Lθ`,
  and heat capacity `C` ([`thermodynamics/enthalpy.jl`](../../src/processes/thermodynamics/enthalpy.jl)).
  The snow closure is the same enthalpy map with snow-specific `Lθ` and `C` (this shared home is the
  payoff of PR-A).
- **Constants.** `PhysicalConstants` exposes thermodynamic constants via `constants.thermodynamics`
  (`latent_heat_fusion`, `latent_heat_sublimation`, `specific_heat_capacity_ice`,
  `specific_heat_capacity_liquid_water`) and material constants via `constants.material`
  (`density_water`, `density_ice`); the Stefan–Boltzmann constant is under `constants.universal`
  ([`constants.jl`](../../src/processes/constants.jl)).
- **Process template.** `BareGroundEvaporation`
  ([`surface/evapotranspiration/bare_ground_evaporation.jl`](../../src/processes/surface/evapotranspiration/bare_ground_evaporation.jl))
  and the degree-day example ([`examples/extending/simple_snow_ddm.jl`](../../examples/extending/simple_snow_ddm.jl))
  are the structural patterns for a new `XY()`-fielded process and its kernels. `NoCanopyInterception`
  ([`surface/canopy_interception/canopy_interception.jl`](../../src/processes/surface/canopy_interception/canopy_interception.jl))
  is the template for the inert `NoSnow` default, and the SEB already threads an
  `Optional{AbstractSurfaceHydrology} = nothing` component argument — the exact pattern the optional
  `snow` argument will mirror.

## Physical formulation

All snow fields are `XY()` (single layer, one value per column, indexed `[i, j, 1]`). Bulk snow
density `ρ_snow` is a homogeneous, constant parameter (the "simple" assumption); homogeneous thermal
properties follow from it.

### State variables (prognostic)

- `snow_energy` `E` (J/m²) — depth-integrated (column) energy content relative to ice at 0 °C. This
  is the conserved quantity; surface and basal fluxes add to it directly.
- `snow_water_equivalent` `W` (m) — total water substance (ice + retained liquid), per UEB `W`.

### Diagnostics (auxiliary / closure)

- `snow_depth` `d_snow = W·ρ_w/ρ_snow` (m).
- `snow_temperature` `T_snow` (°C) and `snow_liquid_fraction` `θ_liq ∈ [0,1]` — from the enthalpy
  closure.
- **Volumetric energy `U_v = E/d_snow` (J/m³) is *not* an auxiliary field.** It is returned by a method
  (e.g. `compute_snow_volumetric_energy(i, j, grid, fields, snow)`) evaluated inside the closure kernel, using
  `safediv` to remain finite as `W → 0` (the indeterminate limit is masked downstream by `f_snow → 0`).
- `snow_cover_fraction` `f_snow ∈ [0,1]` — smooth function of `W`, e.g. `f_snow = W/(W + W₀)` or
  `tanh(W/W_ref)`; differentiable, → 0 as `W → 0`.
- `snow_thermal_conductivity` `κ_snow` (W/m/K) — from `ρ_snow` (e.g. Sturm/Yen power law `κ = a·ρ_snow^b`),
  constant for fixed `ρ_snow`.
- `ground_heat_flux` (SEB closure flux `G`, reused), `soil_heat_flux` (soil-top BC, written by the
  blend), `infiltration` (reused) — see soil coupling below.

### Enthalpy closure (reusing `FreeWater`)

Treat the bulk snow as a water-substance medium of volumetric mass `ρ_snow` (kg/m³), fraction `θ_liq`
liquid and `1−θ_liq` ice. Then

```
U_v  = E / d_snow                        # volumetric energy via method (not stored)
Lθ_v = ρ_snow · L_f                      # volumetric latent heat for complete melt (J/m³)
C_v  = ρ_snow · (θ_liq·c_w + (1−θ_liq)·c_i)   # volumetric heat capacity (J/m³/K)
θ_liq   = liquid_water_fraction(FreeWater(), U_v, Lθ_v, 1)
T_snow  = energy_to_temperature(FreeWater(), U_v, Lθ_v, C_v)
```

This is the identical enthalpy map used for soil with `sat·por → 1` (snow is "fully saturated" with
water substance in the lumped sense). It is implemented as a `closure!`/`invclosure!` pair so that
`T_snow`/`θ_liq` are recovered from `U_v` exactly as soil temperature is recovered from soil energy,
reusing the shared `energy_to_temperature`/`temperature_to_energy` methods (dispatched on the snow
closure type). The only division by `d_snow` lives here, inside the closure, and is regularized with a
finite `eps` offset (`U_v = E/(d_snow+eps)`) rather than `safediv`: `safediv` returns `Inf` at exactly
`d_snow = 0` (even for `E = 0`), which would give `NaN` when multiplied by `f_snow = 0` in the SEB blend.
`invclosure!` (initialization from a prescribed `T_snow ≤ 0`) sets `E = (T·C_v − Lθ_v·(1−θ_liq))·d_snow`,
so `W` must be initialized before `snow_energy`.

**Temperature clip.** Snow temperature cannot exceed 0 °C, so `T_snow = min(T_freewater, 0)`. The
free-water map only returns `T > 0` when `U_v > 0` (the snowpack is fully melted at 0 °C plus sensible
excess); this positive part is *not* stored — the excess energy available to drive melt and sublimation
is derived on demand from `U_v > 0` in the mass/energy tendencies. `snow_energy` itself is unaffected.

### Mass balance (continuous)

```
dW/dt = f_snow·0 + P_s + R_on_snow − M_r − E_subl
```

- `P_s` snowfall (mass source; can initiate a pack from `W = 0`).
- `R_on_snow = f_snow·rainfall` — rainfall intercepted by the snow-covered fraction (adds to `W`,
  advects sensible/latent heat into `U`). Rainfall on the bare fraction bypasses the snowpack (→ soil).
- `M_r` meltwater outflow (Darcy, below).
- `E_subl` sublimation/evaporation from the snow surface, supplied by the latent-heat flux of the SEB
  over the snow-covered fraction (couples to the existing latent-flux machinery).

### Meltwater outflow (UEB eqns 23–24, continuous)

```
S* = clamp((θ_liq/(1−θ_liq) − L_c) / (ρ_w/ρ_snow − ρ_w/ρ_i − L_c), 0, 1)
M_r = K_sat · S*^3
```

with capillary retention `L_c` and saturated snow conductivity `K_sat` as parameters. `S*` is the
relative saturation in excess of capillary retention; the cubic Darcy form gives continuous outflow
that persists while liquid is present (even under a negative energy balance), as in UEB. Output is
added to soil infiltration (below). `clamp` is expressed with `min`/`max` (kernel-safe).

### Energy balance (continuous, depth-integrated)

With `E` (J/m²) prognostic, the column energy tendency is a direct flux sum — no `1/d_snow` factor and no
moving-boundary term:

```
dE/dt = Q_top − Q_base + Q_precip − Q_melt
```

- `Q_top` — conductive flux delivered to the snow surface, equal to the SEB `ground_heat_flux` (`G`)
  over the snow-covered fraction (sign set so energy leaving the snow top reduces `E`); see
  "Boundary-condition disambiguation".
- `Q_base` — snow→soil basal conduction (below); this is the (snow-covered part of the) soil-top
  `soil_heat_flux`.
- `Q_precip` — advected heat of `P_s` (at `min(T_air,0)`) and `R_on_snow` (latent + sensible relative
  to 0 °C ice reference), UEB eqn 13.
- `Q_melt = ρ_w·L_f·M_r` — heat advected out with meltwater at 0 °C (UEB eqn 25).

The depth dependence now appears only inside the enthalpy closure (`U_v = E/d_snow`), where it is guarded
and masked by `f_snow → 0` in the thin-snow limit.

## Surface-energy-balance interface change

The SEB must "optionally accept an `AbstractSnow`". Plan:

1. **Thread an optional `snow` argument** through the SEB entry points, mirroring the existing
   `hydrology::Optional{AbstractSurfaceHydrology} = nothing` pattern:
   - `compute_auxiliary!(state, grid, seb, constants, atmos, hydrology, snow=nothing)`
   - `solve_surface_energy_balance!`, `compute_surface_energy_fluxes!`, and the fused kernels gain a
     trailing `snow::Optional{AbstractSnow} = nothing`.
2. **Snow-aware sub-surface conduction.** Replace the direct reads in
   `compute_skin_temperature(i, j, grid, fields, skinT)` with accessor functions dispatched on the
   presence of snow, returning the fraction-weighted conduction target and conductance:
   ```
   Tg_eff = f_snow·T_snow + (1−f_snow)·T_soil_top
   κ_eff  = f_snow·κ_snow + (1−f_snow)·κₛ
   Δz_eff = f_snow·d_snow     + (1−f_snow)·Δz_soil_top
   ```
   The default method (`snow === nothing`) returns today's soil-only values, so non-snow behavior is
   byte-for-byte unchanged. This also resolves the `κₛ`/thickness TODO in `compute_skin_temperature`.
   The SEB writes its closure flux into `ground_heat_flux` (unchanged by PR-B); the snow energy balance
   reads it as `Q_top`.
3. **Snow-aware albedo/emissivity.** Add a `SnowAlbedo <: AbstractAlbedo` (or make the SEB blend),
   returning `f_snow`-weighted snow vs underlying albedo/emissivity. A first pass uses constant snow
   albedo/emissivity (UEB Table 2: `A_bg = 0.25`, snow `ε_s = 0.99`); snow-age decay (UEB / BATS) is
   deferred.
4. **Single composite skin temperature.** Fractional cover is realized as a *composite blended
   surface* (one skin temperature per column, fluxes weighted by `f_snow`) rather than separate
   energy-balance tiles. This is the pragmatic single-layer realization; independent snow/bare tiles
   are noted as future work.

## Boundary-condition disambiguation (ground vs soil heat flux)

> **Resolved by PR-B** ([`2026-07-10_PLAN_01B_surface_ground_heat_flux_split.md`](2026-07-10_PLAN_01B_surface_ground_heat_flux_split.md)).
> The original single `ground_heat_flux` field served two roles — the SEB skin-temperature closure flux
> `G` (conduction from the skin into the medium below) *and* the soil-top `internal_energy` BC. Without
> snow these coincide; **with snow they are distinct** (the SEB closes against the *snow surface*, while
> the soil sees conduction across the *snow base* `Q_base(T_snow, T_soil_top)`, differing by snowpack
> storage and advected meltwater heat). PR-B split them: `ground_heat_flux` stays the SEB closure flux,
> and `soil_heat_flux` (alias `SoilHeatFlux`) is the soil-top BC. In the no-snow case the soil BC is
> wired directly to `ground_heat_flux`.

What remains **for the snow PR** is the snow-present blend and its wiring (static strategy A):

- **No snow** (`NoSnow`): `soil_heat_flux ≡ ground_heat_flux` (already wired).
- **Snow present**: `soil_heat_flux = f_snow·Q_base + (1−f_snow)·ground_heat_flux`, where `Q_base` is the
  snow→soil conduction (Fourier between `T_snow` and `T_soil_top` across series snow+soil conductances).
  `Q_base` depends only on `T_snow` and `T_soil_top` (both available after the soil and snow auxiliary
  passes, before the SEB), so there is no circular dependency with the skin-temperature solve.

Wiring (strategy A): the snow component type is known at `StateVariables(model::LandModel)` time, so the
soil-top BC is wired to a distinct `soil_heat_flux` field (holding the blend) when snow is present, and
directly to `ground_heat_flux` when `NoSnow` (today's behavior). No dual-writer field, no runtime branch.

### Water-flux analogue (`infiltration`)

`infiltration` (feeds the soil `saturation_water_ice` top BC) has no dual-role structure until snow
exists (single producer = surface hydrology, single consumer = soil BC), so PR-B left it untouched. The
snow PR introduces the blend: `infiltration = f_snow·M_r + (1−f_snow)·(rainfall + throughfall)` plus the
existing surface-hydrology contributions, so meltwater and rain-through reach the soil as a continuous
flux.

## Preliminary refactoring work items (completed)

Two structural changes landed as their own PRs **before** the snow scheme, to keep each change
reviewable (cf. `AGENTS.md` "scope creep in PRs"). Both are now **complete**.

### PR-A: energy/thermodynamics abstraction — **done**

> [`2026-07-10_PLAN_01A_ground_abstraction.md`](2026-07-10_PLAN_01A_ground_abstraction.md)

The medium-agnostic energy machinery was lifted into a top-level `thermodynamics/` module so the snow
scheme can reuse it without depending on soil:

- `AbstractThermodynamics{NF}` (the internal-energy balance supertype), the `ExplicitTwoPhaseHeatConduction`
  operator, and the `FreeWater` enthalpy maps `liquid_water_fraction` / `energy_to_temperature`
  ([`thermodynamics/`](../../src/processes/thermodynamics/)).
- The soil energy process is `SoilThermodynamics <: AbstractThermodynamics` under `soil/energy/`; soil
  stratigraphy (`SoilComposition`, texture, porosity), Richards hydrology, thermal *constituent*
  properties, and biogeochemistry stay soil-specific. The `soil/` tree was **kept** (the originally
  proposed `soil → ground` rename was dropped as unnecessary once the generic physics moved to
  `thermodynamics/`). The speculative `AbstractGround` coupled-process supertype was not introduced.

Snow reuse: the snow closure calls the `FreeWater` scalar maps directly; `SingleLayerSnow` is its own
coupled process (`AbstractSnow <: AbstractCoupledProcesses`), **not** an `AbstractThermodynamics` (its
depth-integrated energy gives a flux-sum tendency, not a conduction divergence).

### PR-B: ground/soil heat-flux split — **done**

> [`2026-07-10_PLAN_01B_surface_ground_heat_flux_split.md`](2026-07-10_PLAN_01B_surface_ground_heat_flux_split.md)

The dual-role flux was split: `ground_heat_flux` remains the SEB closure flux `G`; the soil-top BC is
`soil_heat_flux` (alias `SoilHeatFlux`). No-snow wiring feeds the soil BC directly from `ground_heat_flux`
(byte-for-byte unchanged). The `infiltration` analogue was deferred to this snow PR (no dual role until
snow exists). Verified by a no-snow regression.

## Module structure (new and modified files)

```
src/processes/snow/
├── abstract_types.jl     # AbstractSnow{NF}, NoSnow, interface accessors
│                         #   (snow_temperature, snow_depth, snow_cover_fraction,
│                         #    snow_thermal_conductivity, ...)
├── snow_properties.jl    # ρ_snow, depth d_snow(W), κ_snow(ρ_snow), f_snow(W), compute_snow_volumetric_energy method,
│                         #   albedo helpers
├── snow_energy.jl        # enthalpy closure (reuse FreeWater via U_v = E/d_snow), column energy tendency
├── snow_mass.jl          # SWE mass balance, Darcy meltwater outflow, sublimation coupling
└── snow.jl               # SingleLayerSnow concrete process + variables/initialize!/
                          #   compute_auxiliary!/compute_tendencies!/closure!

src/models/snow/
└── snow_model.jl         # minimal standalone SnowModel for unit + AD testing

modified (snow PR — builds on the completed PR-A and PR-B):
src/processes/surface/skin_temperature.jl            # snow-aware conduction accessors
src/processes/surface/surface_energy_balance.jl      # optional snow arg threading
src/processes/surface/albedo.jl                      # SnowAlbedo (or blend)
src/models/coupled/land_model.jl                     # @component snow (default NoSnow), ordering,
                                                     #   pass snow to SEB, blend soil_heat_flux +
                                                     #   infiltration
src/Terrarium.jl                                     # includes + exports
```

Reuse (no changes, imported by the snow module): `thermodynamics/enthalpy.jl` (`FreeWater` maps),
`constants.jl` (thermodynamic/material constants).

Naming: concrete type `SingleLayerSnow` (UEB-inspired); `NoSnow` is the inert default analogous to
`NoCanopyInterception`. `AbstractSnow{NF} <: AbstractCoupledProcesses{NF}`.

## Implementation phases

**Preliminary (separate PRs, before snow) — both complete:** PR-A (`thermodynamics/` abstraction) and
PR-B (`ground_heat_flux` / `soil_heat_flux` split). See "Preliminary refactoring work items".

**Snow scheme:**

1. **Process skeleton + properties.** `AbstractSnow`/`NoSnow`/`SingleLayerSnow`, `variables`,
   `snow_properties.jl` (depth, density, conductivity, cover fraction, `compute_snow_volumetric_energy` method).
   No coupling yet.
2. **Energy closure.** `closure!`/`invclosure!` reusing the `FreeWater` maps from `thermodynamics/` via
   `U_v = E/d_snow` (guarded); verify `T_snow`/`θ_liq` recovery and round-trip `invclosure!`→`closure!`.
3. **Tendencies.** Mass balance + Darcy outflow + depth-integrated energy tendency `dE/dt`, driven by
   prescribed top/base fluxes (standalone `SnowModel`).
4. **SEB interface change.** Thread optional `snow`; snow-aware conduction accessors; `SnowAlbedo`;
   confirm non-snow path unchanged.
5. **LandModel integration.** Add `@component snow = NoSnow(...)`; insert snow auxiliary (closure +
   `Q_base`) before SEB and snow tendencies after surface hydrology; write the blended `soil_heat_flux`
   and `infiltration`, and wire the soil-top BC to `soil_heat_flux` in the snow branch of strategy A.
6. **Docs.** A `docs/` page following the AGENTS.md process-page template (Overview / Implementations /
   Methods / Kernel functions), with a "scheme under development" warning. Add an implementation-plan
   entry per the `docs/dev` workflow.

## Testing

- **Unit:** `f_snow`, `d_snow`, `κ_snow` monotonicity and limits; closure recovers `T_snow`/`θ_liq` and
  round-trips with `invclosure!`; `M_r` zero below capillary retention and increasing in `θ_liq`.
- **Conservation:** integrate the standalone `SnowModel` with zero surface/base flux and check column
  energy `E = U_v·d_snow` and water `W` conservation under accumulation, melt, and complete ablation;
  verify no energy/water created as `W → 0`.
- **Coupling regression:** `LandModel` with `NoSnow` reproduces current results bit-for-bit (the
  default conduction accessors return soil-only values).
- **Differentiability:** Enzyme adjoints through `compute_tendencies!` and the closure, following
  `test/differentiability`; the `1/d_snow` regularization must keep gradients finite as `W → 0`.
- **GPU/type stability:** `SingleLayerSnow` kernels are `Float32`-clean and allocation-free; no literal
  `0.0`/`1.0`; `ifelse`/`min`/`max` instead of branches.

## Open questions / deferred work

1. **(Resolved)** ~~`1/d_snow` regularization near `W → 0`.~~ Resolved in revision 2: the prognostic is the
   depth-integrated energy `E` (J/m²), so the tendency carries no `1/d_snow` factor. The single remaining
   division is inside the closure (`U_v = E/d_snow`), guarded with `safediv` and masked by `f_snow → 0`.
2. **Composite surface vs separate tiles.** The single blended skin temperature is an approximation of
   fractional cover. Separate snow/bare energy-balance tiles (two skin temperatures) are more rigorous
   but heavier; deferred.
3. **Snow albedo dynamics.** Constant snow albedo first; age-/temperature-dependent decay (UEB/BATS,
   Dickinson et al. 1993) and the shallow-snow albedo interpolation (UEB `r = (1−z/h)e^{−z/2h}`)
   deferred.
4. **Snow density evolution.** `ρ_snow` is constant ("simple" scheme). Compaction/densification (and the
   resulting time-varying `κ_snow`, `d_snow`) is a clear future extension and would relax the homogeneity
   assumption.
5. **Aerodynamic roughness over snow.** UEB reduces turbulent transfer with a snow roughness; the first
   pass keeps the existing aerodynamic resistance. Worth revisiting if turbulent fluxes over snow are
   biased.
6. **Stability corrections.** Like UEB, neutral turbulent transfer is assumed; Richardson/Monin–Obukhov
   corrections are out of scope.
7. **Full-timestep Enzyme autodiff.** Reverse-mode autodiff of the *full* `SnowModel` timestep currently
   fails to compile under Enzyme with an internal `LLVM error: Canonicalization failed` (in
   `EnzymeCreateAugmentedPrimal`). The snow closure reuses the `FreeWater` maps (adjoints covered in
   `soil_energy_diff.jl`) and the snow-specific physics is differentiable in isolation (function-level
   adjoints in `differentiability/snow_model_diff.jl`), so the failure is isolated to the full-model
   reverse pass. Documented in `snow_model_diff.jl` for separate investigation; not required for the
   scheme to function.
8. **Reactant `:snow_column` CI run.** The CPU-vs-Reactant `:snow_column` correctness case is registered
   (`test/reactant/setup.jl` + `runtests.jl`) but was not executed in the dev session because the
   `test/reactant` environment was not instantiated. It needs a real run in CI (or a locally instantiated
   Reactant environment) to confirm correctness.

## Readiness assessment

**Ready to proceed.** The physics is fully specified (UEB-based, depth-integrated energy prognostic,
Darcy meltwater, fractional cover), the design decisions are resolved, and both prerequisite refactors
(PR-A, PR-B) have landed. Every integration point was re-verified against the current source: the
`FreeWater` maps are medium-agnostic in `thermodynamics/`, the SEB already threads an
`Optional{…} = nothing` component argument (the pattern the optional `snow` mirrors), `ground_temperature`
is a `SoilThermodynamics` accessor, and the `soil_heat_flux` BC seam exists.

Recommended sequencing (each a reviewable increment; the first three are snow-independent and can be
tested in isolation):

1. Phases 1–3 (process skeleton + `snow_properties` → enthalpy closure → tendencies) behind a standalone
   `SnowModel`, with unit + conservation + Enzyme tests. This is the bulk of the new physics and carries
   the least coupling risk.
2. Phase 4 (SEB optional-`snow` threading + snow-aware conduction accessors + `SnowAlbedo`), guarded by
   the no-snow regression.
3. Phase 5 (`LandModel` integration: `@component snow = NoSnow`, coupling order, `soil_heat_flux` /
   `infiltration` blend and snow-branch BC wiring), then phase 6 (docs).

Two design points to confirm before/while implementing (neither blocks starting phase 1):

- **`f_snow` functional form and `W_ref`.** The plan lists `W/(W+W₀)` or `tanh(W/W_ref)`; pick one and a
  default reference SWE. Both are differentiable and → 0 as `W → 0`; this only affects the blend weights.
- **Snow-base conductance for `Q_base`.** The plan specifies a series snow+soil Fourier conductance
  between `T_snow` and `T_soil_top`; confirm the discretization (half-cell thicknesses `d_snow/2` and
  `Δz_soil_top/2`) and that it reuses the soil-side `κ` from `SoilThermodynamics`' thermal properties.

## References

- Tarboton, D. G., Chowdhury, T. G., & Jackson, T. H. (1994). *A Spatially Distributed Energy Balance
  Snowmelt Model.* Utah Water Research Laboratory Working Paper WP-94-HWR-DGT/003. (Attached.)
- Male, D. H., & Gray, D. M. (1981). Snowcover ablation and runoff. In *Handbook of Snow*. (UEB melt
  outflow, eqn 9.45.)
- Dickinson, R. E., Henderson-Sellers, A., & Kennedy, P. J. (1993). BATS v1e. NCAR/TN-387+STR. (Snow
  albedo; deferred.)
