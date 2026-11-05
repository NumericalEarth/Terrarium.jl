# Prognostic phenology: prescribed LAI + GDD-based PALADYN phenology factor

> Status: **completed** (implementation + verification). Part A (`PrescribedPhenology` derives the
> phenology factor) and Part B (prognostic GDD reformulation of `PALADYNPhenology`) are both
> implemented and verified; documentation pages and Enzyme adjoint tests remain as follow-up. Part A
> required finishing the previously-unexercised `kernel(...)` lazy-auxiliary API. Covers (a) the
> `PrescribedPhenology` scheme's derivation of the phenology factor from prescribed LAI, and (b) a
> continuous-time, prognostic reformulation of the PALADYN GDD phenology that removes the need to
> store previous timesteps' inputs.

Date of initial draft: 2026-07-22

Base revision: b53747d7fb9bd12c213d3a144a3fc064cf519dfd

## Revision log

> 2026-07-22 — Initial draft.
>
> 2026-07-22 — Part A implemented and verified. `PrescribedPhenology` derives `phenology_factor`
> via the `kernel(...)` lazy-auxiliary API. Getting this to work end-to-end required finishing that
> API, which `PrescribedPhenology` is the first consumer of:
> - `src/abstract_variables.jl`: added an `auxiliary(::Variable, ::KernelFunction, ::Nothing)`
>   dispatch (the convenience constructor previously only accepted `ctor::Function` or `::Nothing`),
>   and relaxed the `AuxiliaryVariable` `ctor` field type bound from `Union{Nothing, Function}` to
>   unconstrained (a `KernelFunction` is a callable struct, not a `Function`).
> - `src/utils/kernel_utils.jl`: the `KernelFunction` call operators now (i) unwrap the Terrarium
>   grid via `get_field_grid` before constructing the Oceananigans `KernelFunctionOperation`,
>   (ii) pass location *types* (`map(typeof, location(...))`) as required by `KernelFunctionOperation`
>   indexing, and (iii) invert the `clock` kwarg so `clock = false` (the default) yields the
>   autonomous variant that does not pass the clock to the kernel function.
> - `src/processes/vegetation/phenology/prescribed_phenology.jl`: `variables(phenol::...)` binds the
>   process instance and passes it through `kernel(phenology_factor, phenol)` so the kernel function
>   receives the parameters. Verified: for prescribed `LAI = 2.0`, `max_leaf_area_index = 4.0`, the
>   computed `phenology_factor` is `0.5`. Full vegetation, state-variable, and abstract-variable test
>   suites pass (the one `parameter_handling` failure is pre-existing and unrelated).
>
> 2026-07-22 — Part B implemented. `PALADYNPhenology` rewritten as a prognostic GDD scheme:
> - New parameters `T_gdd_base`, `gdd_crit`, `T_senescence_range`, `gdd_relaxation_time`,
>   `f_deciduous` (placeholder defaults; PALADYN Table 5 values flagged TODO).
> - New prognostic variable `growing_degree_days` (K⋅day) with tendency
>   `d(gdd)/dt = max(T_air − T_gdd_base, 0)/spd − 𝟙[T_air<T_gdd_base]·gdd/(gdd_relaxation_time·spd)`,
>   where `spd = seconds_per_day` (accumulation in degree-days while the timestepper integrates in
>   seconds). Diagnostic `ϕ = f_deciduous·min(gdd/gdd_crit, senescence_ramp) + (1−f_deciduous)`,
>   `LAI = ϕ·LAI_b`.
> - Air temperature is obtained through the `air_temperature(i, j, grid, fields, atmos)` accessor, so
>   `atmosphere` is now threaded into the phenology `compute_auxiliary!`/`compute_tendencies!`. This
>   required adding `atmosphere` to `VegetationCarbonCycle`'s `compute_tendencies!` and to the
>   `VegetationModel` / `LandModel` tendency couplings (`compute_tendencies!(..., constants,
>   atmosphere)`); `PrescribedPhenology`'s no-op methods absorb the extra argument via `args...`.
> - Added the `sitchEvaluationEcosystemDynamics2003` bibliography entry (cited by the new docstrings).
> - Rewrote `test/vegetation/phenology_tests.jl`: 26 tests across green-up/mature/senescence/dormant
>   regimes, the evergreen limit, the GDD tendency (warm accumulation, cold relaxation, zero at base),
>   and in-model `PALADYNPhenology`/`PrescribedPhenology`. Full vegetation suite green. Regime values
>   verified against hand calculations (e.g. gdd=100/gdd_crit=200 ⇒ ϕ=0.5; T_air=2 °C ⇒ ϕ=0.4).
>
> 2026-07-22 — Prescribed-LAI input pipeline + global example (`PrescribedPhenology` driven by an
> ERA5-Land LAI climatology). This extended the work to the input-source layer and a runnable example.
> - **Rasters extension refactor** (`ext/TerrariumRastersExt/TerrariumRastersExt.jl`):
>   - Static (time-invariant) rasters now reduce to a plain `FieldInputSource` — `InputSource(grid,
>     raster)` dispatches on `dims(raster, Ti)`, regridding a static raster once into a `RingGrids.Field`
>     (scattering masked cells via `idxmap`, which `Field(ring_field, grid)` gathers back out) and
>     deferring to the core input source. The bespoke static branches of `RasterInputSource` are deleted.
>   - `RasterInputSource` is now time-varying-only and gained a `cycle::Bool` field. With `cycle = true`,
>     `update_from_raster_cyclic!` maps the requested time into one period (`span + mean spacing`, so no
>     phase drift) and interpolates, wrapping the last record back to the first across the period
>     boundary; the non-cyclic path is unchanged. This is the climatology-repeat mechanism.
>   - Updated `test/inputs/raster_inputs.jl`: static/multiple-raster cases expect `FieldInputSource`;
>     added a "Cyclic time-varying raster input" testset (5→record 1, 6 h→record 2, wrap-gap
>     interpolation). All 27 raster tests pass; the broader `test/inputs` suite is green.
> - **Assets** (`src/input_output/assets.jl`, `src/Terrarium.jl`): fixed a stale `include` that still
>   pointed at the renamed `get_asset.jl` (now `assets.jl`) — this had blocked precompilation of the
>   whole package — and exported the new `ERA5LandLeafAreaIndex` asset type (daily 1980–2010 LAI
>   climatology, `lai_hv`/`lai_lv`, native `FullClenshawGrid(900)` ≈ 6.48 M points).
> - **`VegetationCarbon` → `VegetationCarbonCycle` rename** propagated through the phenology tests.
> - **Example** `examples/simulations/vegetation_global.jl` (name being iterated by the author): a
>   global `VegetationModel` with `PrescribedPhenology`, driven by the regridded `lai_hv` climatology
>   as a cyclic input (`InputSource(grid, lai_raster; cycle = true)`). Config per the design decision
>   below: `vegetation_dynamics = nothing` and `carbon_vegetation` initialized to a constant.
> - **Verification.** The full data-flow (regrid → model-grid raster → cyclic input → `VegetationModel`
>   with `PrescribedPhenology` → `run!`) was validated on **synthetic** small-grid data: LAI cycles
>   seasonally (5.0 → 1.02 → 4.995 after 365 days) and `ϕ` tracks it and repeats — confirming cyclic
>   input + prescribed phenology work together end-to-end. All real-asset API calls were statically
>   checked. The real 2.5 GB LAI artifact was **not** downloaded/run locally.

## Problem description

The autotrophic respiration scheme `PALADYNAutotrophicRespiration` requires a `phenology_factor`
input `ϕ ∈ [0, 1]` (leaf-out fraction), consumed in the root-maintenance-respiration term
(`src/processes/vegetation/respiration/autotrophic_respiration.jl`, `compute_Rm`). This factor must
be produced by whichever `AbstractPhenology` implementation is coupled into the vegetation model.

Two phenology implementations exist:

- `PrescribedPhenology` — treats `leaf_area_index` as a (possibly time-varying) input. Prior to this
  work it produced *no* `phenology_factor`, so pairing it with `PALADYNAutotrophicRespiration` left
  that input unsatisfied.
- `PALADYNPhenology` — currently a stub. `compute_phenology_factor` returns a hardcoded `1.0` and
  `compute_f_deciduous` returns `0.0` (evergreen). The real PALADYN scheme is GDD-based and thus
  history-dependent; a naive implementation would require storing previous timesteps' temperature
  inputs, which conflicts with the differentiable, continuous-time design of Terrarium.

This plan specifies how each scheme derives `ϕ` in a manner consistent with the project rules:
continuous-time dynamics only, no discrete/per-day updates, no stored input history, GPU- and
Enzyme-compatible.

## Background

### PALADYN phenology (Willeit & Ganopolski, 2016; after Sitch et al., 2003)

Phenology type is set by the coldest-month temperature: if it falls below a PFT-specific threshold
`T_phen_cmon`, the PFT is cold-deciduous, otherwise evergreen. Needleleaf trees are always evergreen.
For deciduous PFTs:

- `LAI = ϕ · LAI_b`  (balanced/maximum LAI `LAI_b`).                                        (eq. 82)
- `ϕ = gdd / gddcrit`, where `gdd` accumulates degree-days above a base temperature `T_gdd_base`
  at a PFT-specific critical value `gddcrit`.                                                (eq. 83)
- After `ϕ` reaches its maximum of 1 it is held constant until air temperature drops below
  `T_gdd_base`. Senescence then proceeds, with all leaves lost at `T_gdd_base − 5 °C`.
- Raingreen (moisture-driven) phenology is not represented.

The scheme has three history-dependent elements: (1) the GDD accumulation, (2) the implicit annual
GDD reset used by LPJ/Sitch, and (3) the coldest-month-temperature classification. The senescence
ramp is an instantaneous function of current air temperature and needs no history.

### Design constraint

Terrarium prohibits discrete-time updates (e.g. "reset once per year", "sum once per day"). All
memory of the past must live in **prognostic state variables** integrated by the timestepper, and
all dynamics must be well-posed ODEs. This is what lets us discard stored input history.

## Summary of changes

### Part A — `PrescribedPhenology` derives the phenology factor

Derive `ϕ` from the prescribed LAI by inverting eq. 82 for the deciduous case, `ϕ = LAI / LAI_b`,
using a configured reference (annual-maximum) LAI:

```
ϕ = clamp(leaf_area_index / max_leaf_area_index, 0, 1)
```

Implementation:

- Add a `max_leaf_area_index` parameter (documented explicitly as the *reference / annual-maximum*
  LAI, not a generic constant — the seasonal amplitude of `ϕ` is entirely set by how well it matches
  the true annual maximum of the prescribed series).
- Emit `phenology_factor` as an `auxiliary` variable so `PrescribedPhenology` becomes a drop-in
  producer of `{leaf_area_index, phenology_factor}`, matching `PALADYNPhenology`'s outputs.
- The derivation is a pure algebraic map of a single cell's input, differentiable except for the
  `clamp` kinks (acceptable; subdifferentiable).

This part is essentially already drafted in `src/processes/vegetation/phenology/prescribed_phenology.jl`
via the `kernel(phenology_factor)` auxiliary-variable constructor. See "Testing and verification"
for a wiring check that must be completed before this is considered done.

### Part B — Prognostic GDD reformulation of `PALADYNPhenology`

Introduce a prognostic growing-degree-day state variable and derive `ϕ` diagnostically.

**Prognostic GDD (growth + continuous reset):**

```
d(gdd)/dt = max(T_air − T_gdd_base, 0) − (gdd / τ) · 𝟙[T_air < T_gdd_base]
```

with `t` in days so `gdd` carries units of degree-days (matching `gddcrit`). The first term is the
standard heat sum. The second term replaces LPJ's discrete annual reset with a continuous cold-season
leak (timescale `τ`, order weeks): `gdd` drains toward zero over winter so each spring restarts the
linear ramp from ≈0, keeping the variable bounded and event-free.

**Diagnostic phenology factor (unifies growth cap and senescence ramp):**

```
ϕ_deciduous = min( gdd / gddcrit,
                   clamp((T_air − (T_gdd_base − 5)) / 5, 0, 1) )
```

- Growth (`T_air ≥ T_gdd_base`): temperature term saturates at 1 ⇒ `ϕ = gdd/gddcrit` (eq. 83).
- Mature (`gdd/gddcrit ≥ 1`, warm): `ϕ = 1`, held.
- Senescence (`T_air < T_gdd_base`): temperature term falls below 1 and draws `ϕ` linearly to 0 at
  `T_gdd_base − 5 °C`.

**Deciduous/evergreen classification:** the coldest-month-temperature test is an annual statistic
that does not reduce to a clean ODE. Represent `f_deciduous` as a prescribed PFT/site property (or an
infrequently-updated input), consistent with the current `compute_f_deciduous` stub and with the fact
that needleleaf PFTs are always evergreen. Combine as in the existing `compute_leaf_area_index`:

```
ϕ_effective = f_deciduous · ϕ_deciduous + (1 − f_deciduous) · 1
LAI = ϕ_effective · LAI_b        (evergreen: ϕ_effective → 1, LAI = LAI_b)
```

**New parameters on `PALADYNPhenology`** (replacing the `# TODO add phenology parameters` stub),
each PFT-specific in PALADYN:

- `T_gdd_base` — base temperature for degree-day accumulation and senescence onset (°C).
- `gddcrit` — critical degree-day sum for full leaf-out (K·day).
- `τ` — cold-season GDD relaxation timescale (day).
- `f_deciduous` — prescribed deciduous fraction (or promote to an input).

**State/variable changes:** declare `gdd` as `prognostic(:growing_degree_days, XY())`, add a
`compute_tendencies!(state, grid, phenol::PALADYNPhenology, ...)` dispatch and a
`compute_gdd_tendency` kernel function (following the `PALADYNVegetationDynamics` /
`compute_ν_tendency` pattern in `src/processes/vegetation/dynamics/vegetation_dynamics.jl`), and keep
`phenology_factor` and `leaf_area_index` as auxiliaries computed in `compute_auxiliary!`.

## Testing and verification

- **Wiring check for the `kernel(...)` auxiliary-variable API (Part A).** DONE — see the revision
  log. `PrescribedPhenology` was the first user of the `kernel(func)` constructor in a
  `variables(...)` declaration, so the full path had to be repaired (auxiliary dispatch, ctor field
  bound, grid unwrapping, location types, `clock` kwarg inversion). Verified end-to-end: the lazy
  `KernelFunctionOperation`-backed `phenology_factor` field computes `LAI / max_leaf_area_index`
  (clamped) when read inside the downstream autotrophic-respiration kernel.
- **Unit / regime tests (Part B).** Verify `ϕ` reproduces the three PALADYN regimes: linear ramp
  during warm-up, held at 1 when mature, linear decline to 0 between `T_gdd_base` and
  `T_gdd_base − 5`. Verify `gdd` drains over a synthetic cold period and re-ramps the following warm
  period (no cross-year ratcheting).
- **Conservation / bounds.** Assert `0 ≤ ϕ ≤ 1` and `gdd ≥ 0` across a seasonal forcing cycle.
- **Type stability & allocations.** `@code_warntype` / allocation checks on the new kernel functions.
- **Differentiability.** Enzyme adjoint test through `compute_tendencies!` for `gdd` and through the
  diagnostic `ϕ`, mirroring `test/differentiability`. If the kinks at `T_gdd_base` degrade the
  adjoint, substitute softplus for `max(·,0)` and a smooth logistic for the cold-season indicator.
- **Evergreen limit.** With `f_deciduous = 0`, confirm `ϕ_effective = 1` and `LAI = LAI_b`
  independent of temperature.

## Documentation changes

- Add / update the phenology doc page following the standard Overview / Implementations / Methods /
  Kernel functions structure, including non-canonical docstrings for `PrescribedPhenology` and
  `PALADYNPhenology` and their `compute_auxiliary!` / `compute_tendencies!` dispatches.
- Document `max_leaf_area_index` as the reference (annual-maximum) LAI, and note the evergreen caveat
  for `PrescribedPhenology` (below).
- Cite eqs. 82–83 of Willeit & Ganopolski (2016) and Sitch et al. (2003) in the PALADYN section.
- Use `jldoctest` blocks and unicode math per project rules.
- Added a global example, `examples/simulations/vegetation_global.jl` (Literate style), driving a
  `VegetationModel` + `PrescribedPhenology` with the cyclic `lai_hv` climatology. It is intentionally
  **not** registered in `docs/make.jl`'s `running_scripts` (like the other asset-heavy examples), since
  it downloads the 2.5 GB LAI artifact and regrids it.

## Known limitations

- **`PrescribedPhenology` assumes deciduous-like behavior.** `ϕ = LAI/maximum_leaf_area_index` conflates a
  *structural* LAI ratio with a *phenological* leaf-out state. For an evergreen PFT (where PALADYN
  sets `ϕ = 1` regardless of LAI), a dip in prescribed LAI would spuriously suppress root
  respiration. Document this; if evergreen support is needed, allow `phenology_factor` to be a second
  prescribed input or add an evergreen flag.
- **Spring-frost clipping (Part B).** The diagnostic `min(...)` has no phase memory, so a hard frost
  during spring green-up would transiently clip `ϕ` where strict LPJ (a state machine) would not.
  Judged physically acceptable (frost damage) and avoids carrying a discrete phase flag, but it is a
  deviation from the reference.
- **Coldest-month classification is not dynamic.** Treating `f_deciduous` as prescribed omits online
  biogeographic reclassification. Acceptable given needleleaf-always-evergreen and climate-timescale
  variation; revisit if dynamic vegetation requires it.
- **`gdd_relaxation_time` is a modeling choice**, not from the PALADYN paper; it emulates the annual
  reset and needs tuning so a full winter drains `gdd` but short warm spells do not.
- **`growing_degree_days` is unbounded in a permanently-warm deciduous climate** (no cold season to
  trigger the relaxation). The `min(gdd/gdd_crit, …)` cap means this has no effect on `ϕ`, but the
  raw accumulator grows without bound over very long runs; acceptable for now.
- **Parameter defaults are placeholders.** `T_gdd_base`, `gdd_crit`, `T_senescence_range` and the
  deciduous classification are PFT-specific (PALADYN Table 5) and must be calibrated per PFT.
- **Bare-`VegetationModel` timestepping needs a nonzero initial `carbon_vegetation`.** A
  pre-existing assertion (`@assert isfinite(LAI)` in `medlyn_stomatal_conductance.jl`) fires when
  `carbon_vegetation` is zero-initialized, making `balanced_leaf_area_index` non-finite. Workaround
  (used by the example and confirmed working): initialize `carbon_vegetation` to a constant and set
  `vegetation_dynamics = nothing`. Still worth revisiting — it is also a throw path in a kernel,
  which the kernel rules disallow.
- **Prescribed phenology paired with prognostic `carbon_dynamics` is physically inconsistent.**
  Normally leaf area and carbon are coupled (`LAI_b = f(C_veg)`); prescribed phenology deliberately
  breaks that link by imposing observed LAI, so an evolving carbon pool alongside it can drift out of
  consistency. The coherent modes are (a) prescribed LAI with carbon dynamics off, or (b) prognostic
  carbon with `PALADYNPhenology`. The example uses mode (a)-like behavior but with the default carbon
  dynamics left on (see the decision in Future work), which is why `carbon_vegetation` is just
  initialized to a constant and annotated as not coupled to the imposed LAI.

## Future work / remaining TODOs observed nearby

**Deferred: make `carbon_dynamics` optional for prescribed phenology.** The clean design for mode
(a) above is `carbon_dynamics = nothing` with `PrescribedPhenology` producing `carbon_vegetation`
from a linear LAI relationship (`C_veg = c · LAI`). This was scoped for this session but deferred to
keep the PR minimal, because it is larger than a one-field `Optional` change: autotrophic respiration
depends on `carbon_dynamics` for more than `C_veg` — `compute_Rm` reads the `SLA` and `awl`
parameters from `PALADYNCarbonDynamics` and consumes the NPP feedback. So dropping `carbon_dynamics`
requires either relocating those PFT parameters (`SLA`, `awl`) so respiration stands alone, or making
respiration itself optional. Recommended as its own focused change with its own plan entry.

These are pre-existing TODOs in the vegetation code, surfaced here for tracking; they are **out of
scope** for this plan unless noted:

- `paladyn_phenology.jl` — replace the stubbed `compute_f_deciduous` (returns 0) and
  `compute_phenology_factor` (returns 1) with the implementation above; add the phenology parameters.
- `respiration/autotrophic_respiration.jl`:
  - `cn_sapwood`, `cn_root` — physical meaning and units unconfirmed (`# TODO check ... + add unit`).
  - `compute_f_temp` hardcodes the constants `308.56`, `56.02`, `46.02`; per the rule against
    hardcoded literals these should move into the process struct or `PhysicalConstants`.
  - `compute_resp10` returns a placeholder constant `0.066`; needs a real implementation, meaning,
    and units.
- `dynamics/vegetation_dynamics.jl` & `dynamics/carbon_dynamics.jl` — several parameters are annual
  and marked to be converted to daily; the disturbance rate `γv` is a placeholder minimum awaiting
  the soil-moisture-dependent PALADYN formulation; leaf/stem/root carbon splitting is not yet
  implemented.
