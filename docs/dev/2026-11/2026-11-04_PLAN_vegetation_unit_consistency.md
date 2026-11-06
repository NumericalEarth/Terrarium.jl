# Unit and physics corrections in the vegetation processes, and a stronger vegetation test regime

> Status: **completed**. All four phases are implemented on `bg/revise-vegetation` (PR #233) and pass the test suite and CI.

Date of initial draft: 2026-11-04

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code

Reviewer: Brian Groenke (brian.groenke@pik-potsdam.de)

Base revision: `3ba9c3c08500a7988ddf6202eff17a8809db21f8` (`main`)

## Originating prompt

> Go through all of the vegetation processes and look for unit inconsistencies. Also look through the tests and propose an improved regime. Ask clarifying questions if needed.

> Please draft an implementation plan and summarize your findings there. Here are the answers to your questions:
> 1. Let's standardize everyting in kgC, including `atomic_weight_carbon`
> 2. You can find the CLIMBER-X/PALADYN source code here: ~/lab/repos/climber-x/src/lnd
> 3. Yes draft a plan

## Revision log

- **Rev 1 (2026-11-04, Claude Opus 5.5, Claude Code):** Initial draft.
- **Rev 2 (2026-11-04, Claude Opus 5.5, Claude Code):** Requested by Brian Groenke, who pointed out that the former Phase 1 was not only a change of units.
  Split it into an output-neutral unit standardization (new Phase 1: F5, F6, and F7) and the physics corrections (new Phase 2: F1 to F4, one commit each).
  Added a snapshot regression check that Phase 1 leaves model output unchanged, renumbered the test regime and differentiability phases to 3 and 4, and retitled the plan.
- **Rev 3 (2026-11-04, Claude Opus 5.5, Claude Code):** Requested by Brian Groenke.
  Clarified Phase 2.2: vegetation carbon stays a single prognostic pool, and the leaf, root, and stem pools are diagnosed pointwise from it with the existing allometry, without new state variables.
  Added the partitioning identity to the Phase 3 tests.
- **Rev 4 (2026-11-04, Claude Opus 5.5, Claude Code):** Deviations made during implementation; approved by Brian Groenke on 2026-11-05.
  No commits were made, so the per-phase and per-correction commits of the plan are replaced by the snapshot comparisons listed under Results, which attribute each change in output to its correction.
  The scaling, identity, and plausibility tests of Phase 3 were added to the existing per-process test files, and the budget and multi-year tests to a new file, `test/vegetation/carbon_cycle_column_tests.jl`; only the Unitful references, the declared-unit guard, and the Float32 test are in `unit_consistency_tests.jl`.
  Phase 4 tests the parameter gradients (with respect to `α_C3`, `g₁`, and `γL`) at the level of the kernel functions, against central finite differences, and the `VegetationModel` adjoint of `timestep!` with respect to the state only; reading parameter gradients from the shadow of a CPU `ModelIntegrator` is not established elsewhere in the test suite.
  The tissue carbon helpers are listed under Methods rather than Kernel functions on the carbon dynamics doc page, since they do not take grid indices.
  `compute_Λ_loc` was rewritten in terms of the tissue carbon helpers (the optional refactor of Phase 2.2), and `compute_resp10` was removed.
  The respiration doc page cites Lloyd and Taylor (1994) for the temperature factor, which required a new entry in `docs/src/references.bib`.
- **Rev 5 (2026-11-06, Claude Opus 5.5, Claude Code):** Changes from the review of PR #233 by @mahabadri and Brian Groenke, requested by Brian Groenke.
  `compute_resp10` is kept as a placeholder that returns the `resp10` parameter, with its test.
  The Unitful reference checks and the declared-unit guard are removed; only the Float32 test remains, in `test/vegetation/float32_tests.jl`.
  The multi-year column run is replaced by tests that all component variables of `VegetationCarbonCycle` are present and that the leaf area index changes under time stepping; the carbon budget closure test is kept.
  The differentiability tests now cover the prescribed vegetation setup: gradients of GPP with respect to `α_C3`, `g₁`, and the leaf area index, and the adjoint of a `PrescribedVegetation` time step with respect to the prescribed leaf area index; the gradients of NPP and the carbon tendency and the `VegetationCarbonCycle` adjoint are removed.
  The temperature factor of autotrophic respiration is documented as a separate equation.

## Problem description

A unit audit of `src/processes/vegetation/` found one error that dominates the vegetation carbon budget and several smaller ones.
With the default parameters, the maintenance respiration of `PALADYNAutotrophicRespiration` exceeds midday GPP by a factor of roughly 120 to 12,000, depending on the vegetation carbon pool.
Net primary production in `VegetationCarbonCycle` is therefore strongly negative under all conditions, and the prognostic carbon pool cannot reach a meaningful steady state.

The existing vegetation tests did not detect any of these problems.
Most of them only check `isfinite` and positivity, with mock inputs that have no physical meaning (for example `GPP = 0.5` kgC/m²/s, about 40 tC/m²/day).
Neither a missing factor of 86,400 nor an inverted dependence on biomass can fail such tests.

## Background

### Findings

The findings were checked against the PALADYN implementation in CLIMBER-X (`~/lab/repos/climber-x/src/lnd`, mainly `photosynthesis.f90`, `dyn_veg.f90`, and `lnd_params.f90`).
PALADYN uses a daily time step, so all of its fluxes are in kgC/m²/day or gC/m²/day; Terrarium works in per-second rates.

A short numerical check on the base revision with default parameters gave the following, where `Rm` is evaluated at `T_air = 15 °C`, `T_soil = 10 °C`, and `ϕ = 1`, and GPP is the midday value at 20 °C, 400 W/m² of shortwave radiation, and LAI 4 (1.5 × 10⁻⁷ kgC/m²/s):

| `C_veg` (kgC/m²) | `Rm` as returned (treated as kgC/m²/s) | `Rm / GPP` |
|---|---|---|
| 0.1 | 1.8 × 10⁻³ | ≈ 12,000 |
| 1.0 | 1.8 × 10⁻⁴ | ≈ 1,200 |
| 10.0 | 1.8 × 10⁻⁵ | ≈ 120 |

#### F1 (critical): stem and root respiration are per day but used as per second

`compute_Rm` in `respiration/autotrophic_respiration.jl` multiplies by `resp10 = 0.066`, which is the PALADYN/LPJ tissue respiration rate at 10 °C in kgC/kgN/day (`photosynthesis.f90`, line 337).
Terrarium never converts it to per second, so `R_stem` and `R_root` are 86,400 times too large relative to GPP.
The docstring of `compute_Rm` says "kgC/m²/day", while those of `compute_Rg`, `compute_Ra`, and `compute_NPP` say "kgC/m²/s".

#### F2 (critical): stem and root respiration scale inversely with the carbon pool

Terrarium computes

- `R_stem = resp10 · f(T_air) · awl·(2/SLA + awl) / (C_veg · aws · cn_sapwood)`
- `R_root = resp10 · f(T_soil) · ϕ · (2/SLA) / (SLA · C_veg · cn_root)`

whereas PALADYN (`photosynthesis.f90`, lines 343–345, with the allometry of `dyn_veg.f90`, lines 206–208) computes

- `R_stem = resp10 · f(T_air) · C_stem / (aws · cn_sapwood)`, with `C_stem = awl · LAI_b`
- `R_root = resp10 · f(T_soil) · ϕ · C_root / cn_root`, with `C_root = C_leaf = LAI_b / SLA`

and `LAI_b = C_veg / (2/SLA + awl)`.
Both forms have units of kgC/m², so dimensional analysis cannot catch the error.
In the Terrarium form, respiration falls as biomass grows and diverges as `C_veg → 0`, and the root term carries a spurious factor of 2.

#### F3 (critical): leaf respiration never enters autotrophic respiration

`PALADYNAutotrophicRespiration` reads the input `daily_leaf_respiration`, which no process produces, so it keeps its zero initial value.
Photosynthesis writes the auxiliary `leaf_respiration` instead.
The input is declared in g/m²/s, while the comment at the conversion says "gC/m²/day to kgC/m²/day".
PALADYN uses its leaf respiration directly (`resp_leaf = rd/1000`).

#### F4: the leaf respiration coefficient is the quantum efficiency

`compute_Rd` in `photosynthesis/lue_photosynthesis.jl` computes `Rd = α_C3 · Vc_max · β`, with the intrinsic quantum efficiency `α_C3 = 0.08`.
PALADYN uses the separate constant `a_c3 = 0.015` ("leaf respiration as fraction of Vmax for C3 plants", `photosynthesis.f90`, line 36), following Haxeltine and Prentice (1996), Eq. 10.
Both are dimensionless, so this is a misused parameter rather than a unit error, but it puts `Rd / Ag` at about 0.32 instead of about 0.06.

#### F5: time conversions with Float64 literals

`compute_Λ_loc` (`dynamics/carbon_dynamics.jl`) and `compute_γv` (`dynamics/vegetation_dynamics.jl`) convert per-year rates by dividing by `365.25 * 24 * 3600`.
That literal is a Float64, so `compute_Λ_loc`, `compute_γv`, and `compute_C_veg_tend` return Float64 for a Float32 model (verified).
The `ground_temperature` input default of `PALADYNAutotrophicRespiration` (`10.0`) is also a Float64 literal.

#### F6: mixed carbon mass units

`net_assimilation` and `leaf_respiration` are in gC/m²/s, but `gross_primary_production` is in kgC/m²/s.
`atomic_weight_carbon = 12.0` is in g/mol.
Conversions between the two are hard-coded (`NF(1.0e-3)` in `compute_GPP`, `/ NF(1000.0)` in `compute_Rm`), which is how F1 and F3 went unnoticed.
The reviewer decided to standardize on kgC throughout (see the originating prompt).

#### F7: labels and documentation

- `medlyn_stomatal_conductance.jl`: the comment on `M_air` says "gC / mol -> kgC / mol" for dry air (the arithmetic is correct).
- `plant_traits.jl`: `specific_leaf_area` is declared in m²/kg; the allometry requires m²/kgC (as `lnd_params.f90`, line 334, states).
- `compute_Vc_max` is described as the "maximum rate of net photosynthesis"; it is the maximum carboxylation capacity.
- `PALADYNAutotrophicRespiration` has TODOs asking for the meaning and units of `cn_sapwood` and `cn_root`; they are C:N mass ratios (kgC/kgN).
- `PALADYNVegetationDynamics` has a stale TODO ("this parameter is yearly, should be changed to daily for now").
- The `γS` docstring in `PALADYNCarbonDynamics` contains a pasted path ("src/processes/vegetation/hydraulicsturnover").
- `test/vegetation/stomatal_conductance_tests.jl` describes 1 × 10⁻⁴ gC/m²/s as "≈ 25 μmol CO₂/m²/s"; it is about 8.3 μmol.
- `examples/simulations/vegetation_global.jl` plots `gross_primary_production * 24 * 3600` (kgC/m²/day) with a color range of 0–6, which only suits gC/m²/day.

#### Verified as consistent

GDD tendencies (K·d/s), PAR and APAR (mol/m²/s), Γ*, Kc, and Ko (Pa), the `c₁ · APAR` chain, the Medlyn conductance (m/s, including the √kPa convention for `g₁`), λc, the PAW integral, the root density, `Λ_loc` and `λ_NPP` (apart from F5), the `ν` tendency, and the Lloyd–Taylor temperature factor in °C.

### Current test coverage

| File | Character |
|---|---|
| `photosynthesis_tests.jl` | Sign and finiteness checks with mock inputs; one exact check of the `1e-3` in `compute_GPP`; a commented-out `And` test block |
| `stomatal_conductance_tests.jl` | The strongest file: Fick's law consistency, the shared √kPa convention, and a tight magnitude bracket |
| `autotrophic_respiration_tests.jl` | Finiteness and positivity only, with mock `Rd`, `GPP`, and `C_veg` |
| `carbon_dynamics_tests.jl`, `vegetation_dynamics_tests.jl` | Piecewise limits of `λ_NPP` and `ν*`; finiteness otherwise |
| `phenology_tests.jl` | Exact values of the phenology factor and GDD tendency; a model-level auxiliary check |
| `root_distribution_tests.jl`, `plant_available_water_tests.jl` | Normalization and limit cases |
| `integration_tests.jl` | Fused kernel equals the per-process path |
| `test/differentiability/vegetation_model_diff.jl` | A `# TODO` stub |

Nothing runs the vegetation carbon cycle forward in time, checks magnitudes of carbon fluxes, or checks precision preservation.

## Summary of changes

The findings fall into two kinds.
F5, F6, and F7 are changes of representation that leave model output unchanged (apart from rounding, and apart from Float32 models no longer being promoted to Float64).
F1 to F4 are corrections of the physics that change model output, even where the cause is a missing unit conversion (F1).
The work is split into four phases so that the two kinds are never mixed: the output-neutral phase is verified by showing that output does not change, and each correction can then be attributed to its own effect.

Phases 1 to 3 belong in one PR, with each phase (and each correction within Phase 2) in its own commit.
Phase 4 (differentiability) can go into the same PR or a follow-up, at the reviewer's discretion.

### Phase 1: standardize units without changing model output

1. **Carbon in kgC throughout (F6).**
   Set `atomic_weight_carbon = 0.012` (kgC/mol) in `MaterialConstants` and update its docstring.
   Its only users are `compute_assimilation_factors` (so `c₁` becomes kgC/mol and `JE`, `JC`, `Vc_max`, `Ag`, `Rd`, and `An` become kgC/m²/s) and `compute_stomatal_conductance` (where `An / M_C` stays in mol/m²/s).
   Declare `net_assimilation` and `leaf_respiration` in `u"kg/m^2/s"` in `LUEPhotosynthesis` and `MedlynStomatalConductance`.
   `compute_GPP` becomes `GPP = An + Rd`, without the conversion factor.
   Update all docstrings and comments that say gC.
   The `/ 1000` applied to `daily_leaf_respiration` in `compute_Rm` stays until Phase 2 replaces that input, since removing it here would change output.
2. **Time conversions (F5).**
   Add `seconds_per_year(::Type{NF})` to `unit_conversions.jl`, defined as `ustrip(u"s", NF(1)u"yr")` like `seconds_per_day` (Unitful's `yr` is 365.25 days, the same value as the current literal).
   Use it in `compute_Λ_loc` and `compute_γv` and remove the stale TODOs.
   Change the `ground_temperature` input default of `PALADYNAutotrophicRespiration` to `NF(10)` or an integer literal, matching the other declarations of that input.
3. **Labels (F7).**
   Fix the comments, the SLA description (m²/kgC), the `Vc_max` docstring, and the pasted path listed under F7.
   Document `cn_sapwood` and `cn_root` as C:N mass ratios (kgC/kgN) and remove the TODOs.

Existing tests that encode gC values are updated in this phase: the `compute_GPP` identity, and the `An` inputs of the stomatal conductance tests, which move from 10⁻⁴ gC/m²/s to 10⁻⁷ kgC/m²/s.
No other test result should change.

### Phase 2: correct the physics

Each item changes model output and goes in its own commit, together with a targeted test (taken from Phase 3) that fails before the commit and passes after it.

1. **Per-day respiration rate (F1).**
   Make `resp10` a parameter of `PALADYNAutotrophicRespiration`, `@param resp10::NF = 0.066 (units = u"d^-1", bounds = Positive)`, documented as kgC/kgN/day at 10 °C.
   Convert it to per second inside `compute_Rm` with `seconds_per_day(NF)`.
   Remove `compute_resp10` (and its doc page entry), or keep it as a thin accessor if the reviewer prefers.
2. **Tissue carbon pools in stem and root respiration (F2).**
   The vegetation carbon pool `carbon_vegetation` (`C_veg`) remains the only prognostic carbon variable.
   The leaf, root, and stem pools are not stored anywhere; they are diagnosed pointwise from `C_veg` inside the kernel functions, using the allometry that `compute_balanced_leaf_area_index` already assumes (PALADYN Eqs. 76–79 with `bwl = 1`, as in `dyn_veg.f90`, lines 206–208):

   - `LAI_b = C_veg / (2/SLA + awl)` (existing `compute_balanced_leaf_area_index`)
   - `C_leaf = LAI_b / SLA`
   - `C_root = C_leaf`
   - `C_stem = awl · LAI_b`

   By construction, `C_leaf + C_root + C_stem = C_veg`, so this is a fixed partitioning of the single pool rather than separate pools.
   PALADYN does the same: its prognostic variable is the total vegetation carbon, and it rediagnoses the tissue pools from `lai_bal` at each step.

   The partitioning is added as three `@inline` functions in `carbon_dynamics.jl`, next to `compute_balanced_leaf_area_index` and with the same signature pattern, `compute_leaf_carbon(vegcarbon_dynamics, traits, LAI_b)`, `compute_root_carbon(...)`, and `compute_stem_carbon(...)`.
   They take `LAI_b` rather than `C_veg`, so that `compute_Λ_loc` (which is the sum of the per-pool turnover rates times these pools) can be expressed with them as well; that refactor is output-neutral and optional.
   No new state variables, fields, or kernel launches are introduced.

   `compute_Rm` already receives `C_veg` (read from `fields.carbon_vegetation`) and `vegcarbon_dynamics`, so it computes `LAI_b` with `compute_balanced_leaf_area_index` and then
   `R_stem = resp10 · f(T_air) · C_stem / (aws · cn_sapwood)` and `R_root = resp10 · f(T_soil) · ϕ · C_root / cn_root`.
   Recomputing `LAI_b` from `C_veg`, rather than reading the auxiliary `balanced_leaf_area_index`, keeps respiration independent of the order in which auxiliaries are computed, so the fused kernel and the per-process path in `integration_tests.jl` stay identical.
   With these pools, `R_stem` and `R_root` are proportional to `C_veg`, as in PALADYN.
3. **Leaf respiration in autotrophic respiration (F3).**
   Replace the input `daily_leaf_respiration` with the input `leaf_respiration` (kgC/m²/s), which photosynthesis already produces before respiration in the fused kernel, and drop the `/ 1000`.
4. **Leaf respiration coefficient (F4).**
   Add `@param leaf_respiration_fraction::NF = 0.015 (bounds = UnitInterval,)` to `LUEPhotosynthesis`, citing Haxeltine and Prentice (1996), Eq. 10, and use it in `compute_Rd` in place of `α_C3`.

Existing tests that encode the old behavior are updated with the commit that changes it: the `compute_Rd` coefficient (item 4) and the `resp10` constant (item 1).

### Phase 3: test regime

All new tests use realistic inputs with units stated in comments; mock values without physical meaning are removed.
The `compute_γv` test, which spells out the conversion literal, is rewritten in terms of `seconds_per_year`.

1. **Unitful reference checks** (new file `test/vegetation/unit_consistency_tests.jl`).
   For each formula that contains a unit conversion, write a test-side reference in Unitful quantities, transcribed from the PALADYN source where one exists (with PALADYN's per-day units), and compare `ustrip` of the reference in the declared field units with the kernel function.
   Covered: `c₁ · APAR` (JE), the Medlyn conversion from assimilation to conductance, `R_leaf`, `R_stem`, `R_root`, `Λ_loc`, `γv`, and the GDD tendency.
2. **Declared-unit guard** (same file).
   Collect `Variables(VegetationCarbonCycle(Float64))` and `Variables(PrescribedVegetation(Float64))` and assert that every variable whose units have the dimension of an areal mass flux (`𝐌 𝐋⁻² 𝐓⁻¹`) is declared in exactly `u"kg/m^2/s"`, and every areal mass in `u"kg/m^2"`.
   This enforces the kgC convention for future processes.
3. **Scaling and homogeneity.**
   `R_stem` and `R_root` are proportional to `C_veg`; `Λ_loc` is proportional to `LAI_b`; `APAR` is linear in shortwave radiation; `JE == JC` under the coordination hypothesis; `Rd / Vc_max == leaf_respiration_fraction · β`.
   These catch structural errors like F2 that dimensional analysis cannot.
4. **Identities.**
   `GPP == An + Rd`, `Ra == Rm + Rg`, `NPP == GPP − Ra`, `Rg == 0.25 (GPP − Rm)`, and `C_leaf + C_root + C_stem ≈ C_veg` for the diagnosed tissue pools.
5. **Plausibility brackets** from the literature, at defined midday or daily conditions for the default needleleaf parameters.
   Midday GPP between 5 and 30 μmol CO₂/m²/s, `Rd / Ag` between 0.03 and 0.15, and `R_stem + R_root` well below midday GPP for `C_veg` between 1 and 15 kgC/m².
   The exact bounds will be set once the fixed implementation runs and documented with their sources in the test file.
6. **Precision preservation.**
   One table-driven test that calls each vegetation kernel function with Float32 parameters and inputs and asserts, with `@inferred`, that the result is Float32.
   This catches F5 and any future Float64 literal.
7. **Carbon budget closure.**
   Run a `VegetationModel` with `VegetationCarbonCycle` for a number of forward Euler steps under constant forcing and check that the change in `carbon_vegetation` equals the accumulated `((1 − λ_NPP) · NPP − Λ_loc) · Δt` to round-off.
   This is the counterpart of PALADYN's local carbon balance check (`dyn_veg.f90`, line 460).
8. **Multi-year column run.**
   Run a `VegetationModel` with `VegetationCarbonCycle` for three simulated years with daily steps and seasonal (sinusoidal) air temperature and shortwave forcing, supplied as cyclical `FieldTimeSeries` inputs as in the `:vegetation_column_lai_cycle` configuration of `test/reactant/setup.jl`.
   Check that `carbon_vegetation` stays positive and finite, that `vegetation_area_fraction` stays in [0, 1], that the phenology factor completes an annual cycle, and that annual GPP and the ratio `Ra / GPP` fall in plausible bands (indicatively 0.3–2 kgC/m²/yr and 0.3–0.7).
   If the fixed model fails these bands, that is a finding to report to the reviewer, not a reason to widen the bands.
9. **Cleanup.**
   Delete the commented-out `And` test block in `photosynthesis_tests.jl` and fix the μmol comment in `stomatal_conductance_tests.jl`.
   Register `unit_consistency_tests.jl` in `test/vegetation/vegetation_model_tests.jl`.

### Phase 4: differentiability

Replace the stub in `test/differentiability/vegetation_model_diff.jl` with Enzyme reverse-mode gradients of GPP and NPP with respect to `g₁`, `α_C3`, and `γL` for a single-column `VegetationModel`, compared with central finite differences, following the existing files in `test/differentiability/`.
Since this edits an Enzyme test, run the Enzyme suite locally (`Pkg.test(; test_args=["enzyme"])`).

## Testing and verification

- Phase 1 regression check: before starting, save a Float64 snapshot of all vegetation auxiliary and tendency fields from the `vegetated_land` setup of `test/vegetation/integration_tests.jl`, after initialization and after 10 time steps, at the base revision (a scratch script, not committed).
  After Phase 1, the same script must reproduce the snapshot to a relative tolerance of 10⁻¹², with `net_assimilation` and `leaf_respiration` scaled by 10⁻³.
  Any larger difference means Phase 1 changed behavior and must be fixed before Phase 2 starts.
- After each Phase 2 commit, rerun the same script and record which fields changed and by how much, for the PR description.
- The vegetation tests via TestEnv (`include("test/vegetation/vegetation_model_tests.jl")`), with output written to a temporary file and checked with `grep`.
- The full suite (`julia --project=. -e 'using Pkg; Pkg.test()'`), since the change of `atomic_weight_carbon` and of the respiration fluxes reaches the coupled `LandModel` tests.
- The Enzyme suite, only if Phase 4 is included.
- Not run locally: the Reactant suite. The Reactant correctness tests compare Reactant against native results for the same model, so the unit changes affect both sides equally; CI covers them.
- A draft doc build (`julia --project=docs docs/make.jl --local --draft`), to catch stale docstring references such as `compute_resp10`.
- Before and after the fix, report `Rm / GPP`, `Rd / Ag`, and annual GPP and NPP from the multi-year column run in the PR description.

## Documentation changes

- `docs/src/processes/vegetation/autotrophic_respiration.md`: state the equations for `R_leaf`, `R_stem`, and `R_root` explicitly, with the tissue pools from the allometry and the per-day reference rate.
- `docs/src/processes/vegetation/photosynthesis.md`: replace "gC/m²/s" with "kgC/m²/s" and document `leaf_respiration_fraction`.
- `docs/src/processes/vegetation/carbon_dynamics.md`: list the new tissue carbon helpers as kernel functions.
- `docs/src/processes/utils/physical_constants.md`: update the `atomic_weight_carbon` row (0.012, kgC/mol).
- `examples/simulations/vegetation_global.jl`: plot GPP in gC/m²/day (multiply by 1000 as well) so that the color range of 0–6 is meaningful, and label the units.

## Results

Snapshot of the coupled `LandModel` column of `test/vegetation/integration_tests.jl` (Float64, 400 W/m², 20 °C, `C_veg = 0.5` kgC/m²) at the initial state, after each phase (fluxes in kgC/m²/s; the base revision's gC fields are shown as stored):

| Quantity | Base | Phase 1 | 2.1 (F1) | 2.2 (F2) | 2.3 (F3) | 2.4 (F4) |
|---|---|---|---|---|---|---|
| GPP | 9.00e-9 | 9.00e-9 | 9.00e-9 | 9.00e-9 | 9.00e-9 | 9.00e-9 |
| Leaf respiration | 3.29e-6 (gC) | 3.29e-9 | 3.29e-9 | 3.29e-9 | 3.29e-9 | 6.16e-10 |
| Net assimilation | 5.71e-6 (gC) | 5.71e-9 | 5.71e-9 | 5.71e-9 | 5.71e-9 | 8.38e-9 |
| Autotrophic respiration | 3.58e-4 | 3.58e-4 | 6.39e-9 | 2.78e-9 | 5.25e-9 | 3.25e-9 |
| NPP | −3.58e-4 | −3.58e-4 | 2.61e-9 | 6.21e-9 | 3.75e-9 | 5.75e-9 |
| Canopy water conductance (m/s) | 1.51e-4 | 1.51e-4 | 1.51e-4 | 1.51e-4 | 1.51e-4 | 2.09e-4 |
| `C_veg` after 10 steps of 600 s (kgC/m²) | −2.28 | −2.28 | 0.500 | 0.500 | 0.500 | 0.500 |

Phase 1 reproduced all 156 arrays of the base revision to a relative tolerance of 10⁻¹², after scaling the two gC fields by 10⁻³.
At the base revision, maintenance respiration drained the vegetation carbon pool below zero within 10 steps of 600 s.

With all corrections, the default needleleaf parameters give a midday GPP of 6–23 μmol CO₂/m²/s and `Rd/Ag` of 0.05–0.08 across cool to warm conditions, and stem and root respiration of 0.05–0.7 kgC/m²/yr for `C_veg` between 1 and 15 kgC/m².
In a ten-year standalone `VegetationModel` column with a seasonal cold-temperate climatology (−7 to 17 °C, 30–270 W/m²), starting from `C_veg = 5` kgC/m², annual GPP is 0.95–1.0 kgC/m²/yr, `Ra/GPP` is 0.38, `C_veg` approaches a steady state slowly (5.0 to 5.5 kgC/m²), and the phenology factor completes a full cycle each year.

## Known limitations

- Vegetation litterfall (`Λ_loc`) is not passed to `SinglePoolSoilCarbon`, so the budget check in Phase 3 covers the vegetation pool only, not the coupled vegetation–soil carbon budget.
- The plausibility bands are indicative for a single default PFT and are not a calibration.
- Remaining differences from the current PALADYN code that are not unit errors are out of scope:
  `α_a` is 0.5 in Terrarium and 0.4 in PALADYN;
  PALADYN now uses `awl = 0.65` with `bwl = 5/3` rather than `awl = 2` with `bwl = 1`;
  PALADYN sets the soil temperature factor to zero below 240 K, computes it from the second soil layer, and clamps NPP at zero;
  PALADYN zeroes `Rd` when net assimilation is negative.
  The last two are discrete operations that Terrarium's continuous-time design would need to replace with smooth alternatives.
- The −3 °C photosynthesis cutoff stays hard-coded, as in PALADYN.

## Future work

- Smooth (differentiable) replacements for the short-circuiting `if` branches in the photosynthesis kernel functions, which AGENTS.md bans in kernels.
- Couple vegetation litterfall to soil carbon and extend the budget test to the coupled column.
- Revisit the PALADYN parameter differences listed above once PFT support is added.
- Consider automatic normalization of `@param` units to SI at construction, so that per-day and per-year parameters no longer need manual conversion in kernels.
- Address the loss of precision when slow prognostic pools are stepped in Float32 with short time steps.
  The per-step increment of `carbon_vegetation` is then only a few units in the last place of the pool, so rounding biases the accumulated change.
  In the seasonal column run with 600 s steps, the one-year change in `carbon_vegetation` differs by 1.1 % between Float32 and Float64 (and by 1.6 × 10⁻⁵ with daily steps); the fluxes themselves agree to about 10⁻⁴.
  The issue is independent of the carbon units, and updating in Float64 before storing in Float32 does not help, because the information is lost when the state is rounded for storage.
  Candidate remedies are compensated (Kahan) summation in the time stepper, which recovered Float64 accuracy with Float32 storage in a scalar test, or storing selected slow pools in Float64; both affect the time steppers and their Reactant and Enzyme compatibility, so they need a separate plan.
  Advancing the vegetation pools on a longer time step reduces the error but does not remove it.
