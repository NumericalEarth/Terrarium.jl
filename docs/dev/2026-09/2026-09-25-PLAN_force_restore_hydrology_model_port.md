# Porting DifferentiableEvaporation to Terrarium: a force-restore hydrology model

> Status: **planned**. Rev 16 (consistency pass over the whole document) is the current revision; nothing is implemented yet.

Date of initial draft: 2026-09-25

Base revision: 8af55f5e8c81b9ae7175e652e6f7a156b22fd27a

## Originating prompt

> My goal is to port the model currently implemented in https://github.com/olivierbonte/DifferentiableEvaporation
> to Terrarium. Besides this original repository, another important source is the model description as referred
> to in @docs/dev/2026-09/2026-09-25-hydrology_model_port_manual.md. You can also find this on my local computer at
> "C:\Users\olivi\OneDrive - UGent\Doctoraat\Doctoraat_obsidian\diff_evaporation_model\Model equations version clean.md"
> (if you access from WSL, look for this windows path).
> Take into account the design principes of Terrarium (see @AGENTS.md).
> Please propose a porting plan. To my understanding, some good porting patterns are already laid out in
> @docs/dev/2026-09/2026-09-25-hydrology_model_port_manual.md. Look at those, evaluate them critically and give me a
> new plan in a separate md file that we can iterate on.

Clarifications given while drafting:

> 1) [C₁] it should indeed be (wsat/w1)^(b/2+1) following the publication, so use that
> 2) please do Santanello-Friedl on R_ns
> 3) HBV I am not sure yet, you would have to look at Trautman 2022 to see what is most correct, intuitively I
>    would think on P_s.
>
> PR split: sequence of focused PRs.

## Revision log

1. **Rev 0 (2026-09-25)**: Initial draft.
2. **Rev 1 (2026-09-28)**: Aligned with open PR #200 (`AbstractGroundHeatFlux` as an SEB sub-process). The plan no longer introduces its own ground-heat-flux abstraction or `PrescribedGroundHeatFlux`. `SantanelloFriedlGroundHeatFlux` subtypes PR #200's type, PR 5a (now PR 9) depends on #200 being merged, and open question Q12 (energy container) is added. The net-radiation gap is unchanged, since PR #200 does not touch the radiative fluxes.
   > Given your net radiation critique: consider that https://github.com/NumericalEarth/Terrarium.jl/pull/200 is in the making
3. **Rev 2 (2026-09-29)**: Incorporates the review below.
   > - Add Insolation.jl as it is required to calculate the solar time. However as for the use of Thermodynamics.jl, we need to find a way to override the use of ClimaParams.jl and instead our own minimal set of constants needed to do the calculation of solar time
   > - The net radiation as input can be a follow up to the PR that gives ground heat flux as an input?
   > - The implementation of the LambertBeer as a simple util function + the refactoring of all its uses so far can be a small separate PR?
   > - For Jarvis, the IFS Table 8.1 is more something for PlantTraits I believe? So sticking with one value for now is okay. I don't want the vegetation_type = ... constructor, this is against the current design principles. this should be set via PlantTraits.
   > - AbstractHydrologyModel: is that a new type?
   > - Instead of flipping the observed net radiation its sign, make clear to the user that is should be inputted following the convention positive up.
   > - For the slope of the saturation vapor pressure curve, look at ∂q_vap_sat_∂T from Thermodynamics.jl
   > - is potential_latent_heat_flux really needed as a separate function? Can't we just reuse penman_monteith and set g_s = 0 and fill in g_a by what we want?
   > - I am not convinced that surface roughness properties belong in the atmosphere part of the code. Instead I'd rather have a separate PR that both relocates the current drag coefficient based scheme + adds the new methods.
   > - smooth clamping: isn't there already some functionality for this in Terrarium? same goes for max
   > - Why does the ForceRestoreSoil needs biogeochemistry?
   > - For the smoothing of the canopy interception: think of this function can't be generalised in to a reusable group of smoothing functions
   > - Rename eddy_diffusivity_extinction to attenuation_factor (is closer to terminology of Bonan)
   > - Insolation.jl explicitly mentions in its docs that it is GPU compatible. Therefore, I would think that it could also work inside a Kernel?
   > - You can put the Reactant work as future work for now

   Changes:
   - **Insolation.jl becomes a root dependency** (explicitly requested; this overrides the AGENTS.md pitfall 13 default). Solar time is computed in-kernel, and Terrarium supplies its own `OrbitalConstants` in place of ClimaParams (new PR 8).
   - **`PrescribedNetRadiation` is its own small follow-up PR** to #200 (PR 7). Users are told to supply R_n in Terrarium's positive-upward convention; Q3 is resolved.
   - **Lambert–Beer becomes a util plus a refactor of all existing uses** (PR 2).
   - **Jarvis**: the vegetation-type constructor is dropped in favour of single default values. PFT-dependent values (IFS Table 8.1) are deferred to `PlantTraits` (future work).
   - ~~**Δ** is replaced by Thermodynamics' `∂q_vap_sat_∂T`, converted to constant pressure. Penman–Monteith is written in specific-humidity form.~~ *(Superseded by Rev 3.)*
   - **`potential_latent_heat_flux` is removed**; `penman_monteith` is reused with r_s = 0, i.e. g_s = ∞.
   - **Aerodynamics** (the current drag-coefficient scheme, the neutral scheme and roughness) moves out of `atmosphere/` in a dedicated PR with its own plan (PR 4).
   - **Smoothing**: a new reusable `src/utils/smoothing.jl` (PR 1). Terrarium has no smooth min/max/clamp yet.
   - **`ForceRestoreSoil` drops biogeochemistry.**
   - `eddy_diffusivity_extinction` is renamed to **`attenuation_factor`**.
   - **Reactant** moves to future work.
   - Also added: a third defect of the original code (uncapped Lee–Pielke β), the note that the original already uses the r_ss form, the Santanello–Friedl midnight discontinuity, and a "Tracking" section.
4. **Rev 3 (2026-09-29)**: Penman–Monteith returns to the classic vapour-pressure form (this supersedes the Δ/q-form item of Rev 2). VPD comes from the existing `compute_vapor_pressure_deficit`, γ from the existing `psychrometric_constant`, and Δ = de_s/dT via Clausius–Clapeyron from `saturation_vapor_pressure`. `∂q_vap_sat_∂T` is not used: it is a constant-density derivative of specific humidity, which the vapour-pressure form does not need.
   > Okay, but I don't want penman moteith in the specific humidity form. Terrarium can also calculate the vapor pressure deficit at one specific location (based on 1 temperature), so we can use this for the classis penman monteith formulation right?
5. **Rev 4 (2026-09-29)**: The Santanello–Friedl formulation of the model description is kept after reviewing the linked implementations (the paper, the ALEXI ATBD/DisALEXI, STIC/Mallick 2022, and the JPL `santanello-soil-heat-flux` package used by PM-JPL and STIC-JPL). That means R_n,s; c_g ∈ [0.31, 0.35] and t_g ∈ [74 000, 100 000] s; linear interpolation in Θ = w₁/w_sat; and t negative before solar noon. The docstring citation now credits all three sources, and Mallick et al. 2022 is added to the references.
   > Agree to keeo model description that we had, I like the docstring, the warning is not needed. These changes can go to the plan
6. **Rev 5 (2026-09-29)**: Resolves open questions 2, 4, 6, 7, 8, 9, 13 and 14:
   - **Q2**: the Bergström ratio uses the **maximum storage w_sat**, i.e. w₂/w_sat. This was verified against the HBV-light manual (Seibert 2005): "FC = maximum soil moisture storage", "FC is a model parameter and not necessarily equal to measured values of 'field capacity'", and the ratio is SM/FC with SM ≤ FC. wflow's HBV and SINDBAD (Trautmann et al. 2022, wSoil_max) agree. Bergström (1976, 1992) itself was not accessible.
   - **Q4**: no new humidity type. Terrarium computes the VPD in-kernel from the specific-humidity input, so observed VPD only has to be converted to specific humidity once, host-side.
   - **Q6**: smooth floor of w₁ at w_wp inside C₁. Improved dry-soil formulations are future work.
   - **Q7**: the canopy smoothing kernels are dropped for now. They were introduced together with the autodiff experiments (commit `7663859`, 2025-08-01, which also replaced hard `max`/`min` clamps in f_wet by smooth ones). When the kernels were added to `ODE_clean.jl` (`54f0fd6`, 2025-08-11), the explicit Euler step was reduced tenfold in the same commit. This points to stability problems, but the commit messages do not state the reason. Dropping them removes both the mass leak and the E_i double bookkeeping. The upper kernel is also redundant for k_ext = 0.5, see the De Ridder section.
   - **Q8**: w_sat and texture come from `ForceRestoreSoil.strat`.
   - **Q9**: f_wet is regularized. Deardorff (1978) chose the 2/3 exponent so that retained water evaporates in *finite time*; exponent 1 gives an exponential decay that never vanishes, exponent 0 a thin film that disappears too fast. The infinite slope at wᵣ = 0 is therefore intended physics, but it breaks Enzyme derivatives and ODE uniqueness, and a negative wᵣ from an explicit step makes `x^(2/3)` throw a `DomainError`, which is a banned throw path in kernels.
   - **Q13**: f_veg is computed pointwise via `canopy_cover_fraction(i, j, grid, fields, vegetation)`, mirroring the existing `vegetation_area_fraction(i, j, grid, fields, vegetation)` pattern used by the albedo.
   - **Q14**: `OrbitalConstants` becomes a field `orbital` of `PhysicalConstants`, as for `ThermodynamicConstants`.
   - Also found: the smoothing parameter of the original `smooth_max(w_r, 0, w_rmax/1000)` has the wrong dimension (it enters squared under the square root). In metres it would floor wᵣ at about 65 % of wᵣ,max. The Terrarium utilities therefore take a length scale δ with the units of their arguments.
   > Responses to open questions: 2: not sure myself, I would think the max makes more sense, as the original FC is a model paramters in a conceptual rainfall runoff model no? Please verify this questions at the original source. 4: not sure of the relevance of this question? You can alwas compute the VPD in kernel no? 6: Don't yet implement improved representations of this, smooth floor is okay. 7: on the smoothing kernels for canopy, I think I initially implemented these because of issues with numerical stability (you can probably verify this in the git history). As a start, working without it is also fine. 8: proposed is good. 9: Look in my obsidian notes at the original Deardroff paper on the choice of this 2/3. Maybe this gives insight into wheter or not the smooth it. 13: what is most consistent with current code patterns? 14: do same as for Thermodynamics
7. **Rev 6 (2026-09-29)**: Q15 is now informed by De Ridder (2001). The ½ in both the throughfall (Eq. 6) and the drainage coefficient b = 1/(2c) (Eqs. 39–41) follows from its uniform-leaf-angle, vertical-rain assumption (Appendix A). The proposal is b = k/c with the same k as f_veg, a generalization derived here that is identical to the paper for k = 0.5. Also added: the canopy stiffness limit Δt ≪ t_c from the paper to the known limitations.
   > For open question 15: look at the original paper on my vault. I believe that there is an assumption about the the extinction coefficient in the derivation here (=0.5 assumed because of leaf angle distribution)?
8. **Rev 7 (2026-09-29)**: Q15 resolved. `DeRidderCanopyInterception` uses b = k/c with k = `PlantTraits.extinction_coefficient`. Its docstring states the k = ½ (uniform leaf angle distribution) assumption, and a host-side check at model construction warns (`maxlog = 1`) when k ≠ ½. Tests cover the warning.
   > For de ridder, I think it makes sense to use the extinction coefficient from plant traits BUT there should be a warning ( + a mention in the docstrings) if the extinction coefficient is not 1/2 as the theory behind this model requires
9. **Rev 8 (2026-09-29)**: Adds a general "keep documentation concise" principle to the Documentation changes section, adapted from NumericalEarth.jl's `style-rules.md` (Comments) and `restraint-rules.md` (Rules 7, 8, 9, 12) at commit `fffb946`. The De Ridder docstring and warning are shortened to one sentence each accordingly.
   > For documentation: add as a general principle that documentation should try to be concise. Look at the rules in comments in NumericalEarth.jl style-rules.md and restraint-rules.md regarding documentation
10. **Rev 9 (2026-09-29)**: Reviewed Kavetski & Kuczera (2007):
    - Listed the functions it proposes (Eqs. 8–20) and which the original and Terrarium implement.
    - Smoothing becomes a switchable `AbstractSmoothing` strategy (`NoSmoothing`, `QuadraticSmoothing` = Eqs. 11/18 with m = δ², `LogisticSmoothing` = Eq. 13), held as a component by each consuming process.
    - The canopy storage limiters (Eq. 20) become optional, and the f_wet regularization ε becomes a parameter.
    - The Bergström cap is justified as an overshoot guard only, since w_sat is invariant in the exact ODE.
    - Added PR 11 (smoothing experiments in the research repo) to decide the defaults, and Kavetski's implicit-Euler recommendation to the known limitations.
    > Regarding this smootihg (both of the canopy and bergstrom), I want you to look at the Kavetski 2007 paper on this topic. What functions does he propose? are these implemented now? I would prefere if there could be experimentation with whether or not smoothing is necesarry/beneficial
11. **Rev 10 (2026-09-29)**: **No smoothing in this plan** (supersedes the smoothing parts of Rev 2, Rev 5 and Rev 9).
    - Thresholds are hard: C₁ floor `max(w₁, w_wp)`, K₂ `max(w₂ − w_fc, 0)`, f_wet `min((max(wᵣ, 0)/wᵣ,max)^(2/3), 1)`, Jarvis and PAW `clamp`, no Bergström cap, no canopy kernels.
    - PR 1 (smoothing utilities) and PR 11 (smoothing experiments) are removed, and their content moves to Future work.
    - PR 0 now runs the original model with these hard formulations. Smoothing is introduced for a specific threshold only if those simulations fail.
    > I think this smoothing is future work. For this plan, keep it to without smoothing. only introduce smoothing if the original simulations fail otherwise (after PR0).
12. **Rev 11 (2026-09-29)**: Q1 resolved: the proposed names are accepted.
    > Open question 1: naming is okay for me
13. **Rev 12 (2026-09-29)**: Q10 now records the literature: no residual in Lee & Pielke/CLM/ISBA; a residual dry end in the linear IFS/H-TESSEL (Albergel et al. 2012) and GLEAM (Martens et al. 2017) stress functions. It also flags the negative-argument behaviour of Terrarium's existing factor below θ_res.
    > For the Lee-Pieilke: when looking at the original 1992 paper, I don't think I see a residual saturation term in there. However, I think it does makes sense conceptually that soil moisture can't drop below residual. Can you find literature supporting an adpatation for including this residual in the formulation?
14. **Rev 13 (2026-09-29)**: Q10 resolved. The force-restore dispatch of `SoilMoistureResistanceFactor` passes θ_res = 0, giving the original Lee & Pielke (1992) β; no residual adaptation of the cosine form.
    > Okay, the don't do this and just set saturation = 0 in our application of this resistance
15. **Rev 14 (2026-09-29)**: Aligned with merged PR #204, which fixes #203 (ET not applied to the soil water in `LandModel`). The background and future work are updated. The force-restore model passes `surface_hydrology` to the soil tendencies with the same signature, and the soil reads E_s and E_t separately.
    > Look at https://github.com/NumericalEarth/Terrarium.jl/pull/204, this fixes the issue with ET for landmodel
16. **Rev 15 (2026-09-29)**: Adds PR 12, grids without vertical resolution (z-less `ColumnGrid`/`ColumnRingGrid` constructors, plus one `XYZ`-on-`Flat` rejection), after #193 and #194. PR 6 gains a depth-free single-horizon `with_soil_horizon` method for such grids, and the example uses the z-less grid.
    > I think it is important for the user here that when using this model, it does not make sense for them to set a vertical resolution. So there should be a convient way to construct a ColumnGrid or ColumnRingGrid that has no vertical resolution. I suppose this is best fit as a separate PR, following up pr 193 and 194?
17. **Rev 16 (2026-09-29)**: Consistency pass; removed leftovers from superseded revisions. Specifically:
    - Naming marked as accepted (review row 5, composition, PR 2).
    - Removed the reference to PR 1 smooth limiters (row 13) and the ε/`w⁺` f_wet text (superseded by Rev 10).
    - Dropped w_res from the reused hydraulics (θ_res = 0, Rev 13).
    - The De Ridder drainage is now written with b = k/c (Rev 7) and its equilibrium statement corrected.
    - Bergström uses the generic `relative_root_zone_wetness`.
    - PR dependencies no longer reference the removed PR 1, and PR 2's file list matches its design (`vegetation_base.jl`).
    - Clarified the sign notation in the order of operations.
    - Corrected the Rev 1/Rev 2 cross-references (PR 5a is now PR 9; Q3 is resolved, not removed).
    > Check that the entire plan document is consistent (no internal contradictions)

## Problem description

The DifferentiableEvaporation model (`src/EvaporationModel` in that repository) has three prognostic states:

- surface soil moisture w₁ and root-zone soil moisture w₂, from a two-layer ISBA force-restore scheme;
- canopy water wᵣ.

These are driven by a three-source evapotranspiration scheme (Shuttleworth–Wallace, extended to include interception after Lhomme et al. 2012). The model is currently written as a SciML `ODEProblem` on scalar `ComponentArray`s, using Bigleaf.jl for thermodynamics and resistances. The goal is to re-implement it inside Terrarium as a set of reusable, modular processes plus a thin model wrapper. The result must satisfy Terrarium's design rules: kernels run on GPU, dynamics are continuous-time and differentiable with Enzyme, and code uses dispatch rather than conditionals.

The goal is **not** a one-to-one transliteration. Where Terrarium already has an abstraction for a concept (stomatal conductance, canopy interception, runoff, aerodynamics, plant-available water, ground evaporation resistance, ground heat flux), the new physics should be a new implementation of that abstraction, so it can be combined with existing components later.

## Background

### The model to port (source of truth)

The model description ("Model equations version clean") is the reference. Where the original code deviates from it, we follow the description, as decided in the clarifications above:

| Topic | Original code | Port follows |
|---|---|---|
| Force coefficient C₁ | `C1sat·(w₁/w_sat)^(b/2+1)` (inverted ratio) | `C1sat·(w_sat/w₁)^(b/2+1)` ([Noilhan & Planton 1989], [Noilhan & Mahfouf 1996] eq. 20) |
| Ground heat flux G | Allen et al. (2007)/METRIC on R_n | Santanello & Friedl (2003) eq. 4 on R_ns, with the soil-wetness-dependent c_g and t_g of Anderson et al. (2018) ALEXI |
| Runoff input | β-function applied to P (above canopy) | β-function applied to **P_s** (below canopy) |

The runoff choice is confirmed by [Trautmann et al. 2022] (HESS 26, 1089, eq. 1): `I_exc = I_in·(Σwsoil/Σwsoil_max)^p_berg`, where `I_in` is "incoming water from throughfall and snowmelt", i.e. P_s in this model. In SINDBAD, as in HBV (where FC is the *maximum* soil moisture storage, a model parameter), the ratio is taken against the maximum storage, so it cannot exceed 1. The port therefore uses w₂/w_sat rather than the original w₂/w_fc (Rev 5).

The original code already uses the **r_ss form** for soil evaporation, consistently in the Lhomme total and the soil component (`model.jl:115-116`, `evaporation.jl:101,128`): β is converted to `r_ss = r_as/β − r_as`. The port keeps this, written in conductance form (see the Penman–Monteith section).

Three more defects in the original code surfaced while cross-reading it. The port fixes all of them:

- **Canopy mass leak.** The canopy tendency receives `f_veg·P·k_up(wᵣ)`, where `k_up` is the upper-bound smoothing kernel, but `P_s = (1−f_veg)·P + D_c`. The fraction `f_veg·P·(1−k_up)` therefore vanishes. The port drops the kernels (Rev 5) and uses one interception flux `I = f_veg·P` in both the canopy tendency and `P_s = P − I + D_c`.
- **E_i double bookkeeping.** The canopy loses `E_i·k_low(wᵣ)`, but the full `E_i` is reported in λE and in the energy balance. Since `f_wet → 0` as `wᵣ → 0`, `E_i` already vanishes at an empty canopy. The lower-bound kernel is only needed against dew (E_i < 0) and numerical overshoot. The port drops it (Rev 5).
- **Uncapped Lee–Pielke β.** `0.25·(1 − cos(π·w₁/w_fc))²` is only valid for w₁ < w_fc. Uncapped, β decreases again above field capacity, reaching 0 at w₁ = 2·w_fc. Terrarium's existing `SoilMoistureResistanceFactor` sets β = 1 for θ ≥ θ_fc, and the port reuses it.

### Relevant state of Terrarium (as of the base revision)

- `SurfaceHydrology` (`src/processes/surface/surface_hydrology.jl`) has three slots: `canopy_interception`, `evapotranspiration` and `surface_runoff`. These map one-to-one onto the ported processes.
- Evapotranspiration is formulated with **humidity gradients at a skin temperature** that the SEB Newton solve provides (`surface_energy_balance.jl:135-156`).
  - There is no Penman–Monteith / combination equation anywhere.
  - `psychrometric_constant` exists but is unused (`thermodynamics.jl:34`).
- **ET → soil coupling in `LandModel`** was missing at the base revision (issue #203), and is fixed by the merged PR [#204](https://github.com/NumericalEarth/Terrarium.jl/pull/204).
  - `LandModel.compute_tendencies!` now calls `compute_tendencies!(state, grid, soil, constants, surface_hydrology)`, and `SoilEnergyWaterCarbon` passes `get_evapotranspiration(surface_hydrology)` to the Richards hydrology. `surface_hydrology = nothing` means no ET sink, for standalone soil.
  - The ET forcing still removes all of `ground_evapotranspiration_flux`, i.e. ground evaporation plus transpiration, from the **top layer only**; there is no root-weighted uptake.
- **Aerodynamics.** `aerodynamic_resistance(i, j, grid, fields, atmos) = 1/(drag_coefficient(i, j, grid, fields, atmos.aerodynamics)·Vₐ)`. The only implementation is `ConstantAerodynamics` (Cₕ = 1.2e-3), owned by `PrescribedAtmosphere` (`src/processes/atmosphere/aerodynamics.jl`).
  - `PrescribedAtmosphere.altitude` ("surface-relative altitude at which the forcings are applied", default 10 m) is exactly z_a, but it is currently unused.
- **Radiation and G.**
  - `PrescribedRadiativeFluxes` derives `surface_net_radiation` from prescribed *upward* fluxes; net radiation cannot be prescribed directly.
  - **Open PR [#200](https://github.com/NumericalEarth/Terrarium.jl/pull/200)** (`bg/prescribed-seb`) moves G into its own SEB sub-process in `src/processes/surface/ground_heat_flux.jl`:
    - It adds `AbstractGroundHeatFlux{NF} <: AbstractProcess{NF}` with `DiagnosedGroundHeatFlux` (auxiliary) and `PrescribedGroundHeatFlux` (input).
    - `SurfaceEnergyBalance` gets a fifth `ground_heat_flux` field.
    - It does **not** touch the radiative fluxes.
    - It declares variables with the new location types (`Ground(Top())`) from the grid-integration work, not `XY()`. This plan writes `XY()` throughout; follow whatever has landed by implementation time.
  - **All Terrarium surface fluxes are positive upward.**
- **Lambert–Beer `1 − exp(−k·L)` is computed inline** in three places:
  - `PALADYNCanopyInterception` (with its own duplicated `k_ext`, marked TODO, on LAI + SAI);
  - Medlyn's `g₀` (on LAI);
  - PALADYN's `rₐ_can` (on LAI + SAI).
- `vegetation_area_fraction` means the PFT area fraction ν. It is zero for `PrescribedVegetation`, and is **not** the Lambert–Beer canopy cover.
- **`PrescribedVegetation`.** Its type parameters are unconstrained, so `photosynthesis = nothing` and `root_distribution = nothing` dispatch to no-ops.
  - Stomatal conductance is exposed as the XY auxiliary `canopy_water_conductance`.
  - The soil moisture stress `soil_moisture_limiting_factor` comes from the PAW process. `FieldCapacityLimitedPAW` computes `clamp((θ−θ_wp)/(θ_fc−θ_wp), 0, 1)`, which is **exactly** the Jarvis f₂ of the original model, but its fields are XYZ (root-fraction-weighted `Integral`).
- **Pedotransfer functions and porosity.**
  - `SoilHydraulicsSURFEX` already implements the Noilhan & Mahfouf (1996) pedotransfer functions for w_fc and w_wp from clay %, and `SoilTexture` holds clay/sand fractions. The remaining ISBA coefficients (C1sat, C2ref, C3, a, p, b) are also clay-based pedotransfer functions from the same paper.
  - `porosity(i, j, k, grid, fields, strat, bgc)` requires a biogeochemistry argument, because it blends mineral and organic porosity using the organic fraction.
- **No smoothing utilities.** Terrarium has no smooth min/max/clamp or smooth limiter functions (`src/utils/` has `math.jl`, `kernel_utils.jl` and others, none of which provide them). Existing code uses hard `max`/`min`/`clamp`, e.g. snow meltwater outflow and `FieldCapacityLimitedPAW`.
- **Grids and models.**
  - XY-only models are supported (`examples/extending/linear_ode_exp_growth.jl`, `simple_snow_ddm.jl`). A `ColumnGrid` still needs a vertical coordinate (Nz ≥ 1); PR 12 adds grids without vertical resolution.
  - **`AbstractHydrologyModel` is an existing type** (`src/models/abstract_types.jl:38`, "Base type for surface hydrology models", listed in `docs/src/extending/core_interfaces.md`). It currently has **no concrete subtype**.
  - The dead `src/models/surface/surface_hydrology_model.jl` is never included and subtypes a non-existent `AbstractSurfaceHydrologyModel`.
- **Timestepping and clock.**
  - Only fixed-step `ForwardEuler` and `Heun` exist, plus adaptive stepping via `Simulation` + `TimeStepWizard` driven by `cell_diffusion_timescale`. There is no SciML interface.
  - The model clock is `Clock(time = zero(NF))`: `NF` seconds from zero, with **no reference date** stored in the model.
- **Dependencies.**
  - Bigleaf.jl is not a dependency; it will be test-only.
  - Thermodynamics.jl is used **without ClimaParams**: `ThermodynamicConstants{NF} <: Thermodynamics.Parameters.AbstractThermodynamicsParameters{NF}` plus accessor overloads (`src/processes/constants.jl:18`, `thermodynamics.jl:3-18`).
  - Existing helpers for the vapour-pressure form of Penman–Monteith: `saturation_vapor_pressure`, `vapor_pressure_deficit` / `compute_vapor_pressure_deficit` and `psychrometric_constant` (`thermodynamics.jl`, `prescribed_atmosphere.jl`). Only Δ = de_s/dT is missing.
- **Insolation.jl** (v1.2.1; not yet a dependency):
  - `[deps]`: Adapt, Artifacts, Dates, DelimitedFiles, Interpolations (compat 0.14–0.16; Terrarium pins 0.16).
  - **ClimaParams is only a weak dependency.** Parameters use `Insolation.Parameters.AbstractInsolationParams` with getter functions, which is the same override route as Thermodynamics.
  - It ships two **non-lazy artifacts**: `laskar2004` from data.caltech.edu, and the CMIP monthly TSI from caltech.box.com. Both are downloaded at install time.
  - The exported API (`insolation`, `solar_geometry`) takes a `DateTime`. The functions we need are **not exported**: `hour_angle`, `equation_of_time`, `mean_anomaly` and `years_since_epoch`.

## Critical review of the porting manual

| # | Manual proposal | Verdict | Rationale / change |
|---|---|---|---|
| 1 | HBV runoff in `src/processes/surface/runoff`, "only compatible with force-restore" | ✅ location, ⚠️ scope | The Bergström β-function only needs a *relative root-zone wetness*. Expose it through a soil-dispatched pointwise function so the scheme is not tied to force-restore (Richards later). The ratio is taken against the maximum storage w_sat, as HBV's FC is a maximum storage, so it stays ≤ 1 and Q_s ≤ P_s. Name it after Bergström ([Bergström & Lindström 2015]) rather than HBV, since that is what the literature calls the β-function. |
| 2 | Net radiation via `PrescribedRadiativeFluxes`, or just an `InputSource` | ❌ | Neither works as-is: the existing type derives R_n from upward fluxes. Add a `PrescribedNetRadiation <: AbstractRadiativeFluxes` with input `surface_net_radiation` as a small follow-up PR to #200 (PR 7). G is a *parameterization*, not an input: add `SantanelloFriedlGroundHeatFlux <: AbstractGroundHeatFlux` (PR #200's type). |
| 3 | `k_ext` already in `PlantTraits`; Lambert–Beer not yet implemented | ⚠️ | Lambert–Beer *is* implemented, but inline in three places, one with a duplicated `k_ext`. A small separate PR (PR 2) adds a util and refactors all three uses. |
| 4 | G: unsure whether in Terrarium; Insolation.jl for solar time | ✅ | G belongs in Terrarium: it partitions the available energy and depends on w₁ and f_veg (model state). Insolation.jl becomes a root dependency with Terrarium-owned orbital constants in place of ClimaParams. Solar time is computed **in-kernel** (PR 8, with caveats listed there). |
| 5 | `ThreeSourceEvapotranspiration <: AbstractEvapotranspiration`; PM + Lhomme in `evapotranspiration/`; Bigleaf as test reference | ✅ | Agree on the location and on Bigleaf as a **test-only** reference (`test/Project.toml`). Named `ShuttleworthWallaceEvapotranspiration` (accepted in Rev 11), since "three-source" is not a standard term and Lhomme calls it "multi-source". |
| 6 | VPD_m computed inside the ET scheme, not in atmosphere | ✅ | VPD_m is a diagnostic of the series-resistance network. |
| 7 | r_aa via `NeutralStabilityAerodynamics <: AbstractAerodynamics` | ✅ | The neutral log-profile bulk transfer coefficient `k²/ln²((z_a−d)/z₀ₘ)` *is* a drag coefficient, so it fits the existing `drag_coefficient` interface. Existing schemes (bare-ground ET, SEB turbulent fluxes) get it for free. |
| 8 | Roughness as a new field of `SurfaceHydrology`; maybe move aerodynamics to surface | ⚠️ → dedicated PR | Not a `SurfaceHydrology` field: that struct is shared by every `LandModel`, and rₐ is also used by the SEB. Instead, **PR 4 relocates the whole aerodynamics family** (the current drag-coefficient scheme, the neutral scheme and roughness) out of `atmosphere/`, with its own plan document for the ownership question. |
| 9 | New methods `leaf_canopy_air_space_aerodynamic_resistance`, `ground_canopy_air_space_aerodynamic_resistance`; move all resistances to a new file | ✅ methods, ⚠️ file | r_ac (canopy boundary layer, kB⁻¹) and r_as (ground to canopy source height, Choudhury & Monteith 1988) are specific to the series network. They live with the new ET scheme under shorter names (`canopy_boundary_layer_resistance`, `ground_aerodynamic_resistance`). The shared rₐ moves in PR 4. |
| 10 | Soil β reuses `ground_evaporation_resistance_factor` via a new soil dispatch; "first try β × potential evaporation" | ✅ reuse, ❌ β × E_p | Multiplying β onto a Penman flux breaks the Lhomme closed form: λE_total = λE_t + λE_i + λE_s only holds if the *same* r_ss enters the total and the components. The original code already does this correctly. Keep `r_ss = r_as(1/β − 1)`, written in **conductance form** so that β → 0 (and LAI → 0 for r_sc) stays finite and differentiable. |
| 11 | `JarvisStomatalConductance <: AbstractStomatalConductance`; soil constraint usable with Richards and force-restore | ✅ | Output `canopy_water_conductance` (a conductance, as Medlyn does) so it is a drop-in for the existing interface; this also avoids the `r_smin/LAI` singularity. f₂ reads `soil_moisture_limiting_factor`, so it automatically works with `FieldCapacityLimitedPAW` (Richards) and with a new XY PAW for force-restore. Single default parameter values for now; PFT-dependent values belong in `PlantTraits` (future work). |
| 12 | `PrescribedVegetation` with `PrescribedPhenology`, no photosynthesis, `RootDistribution` unclear, `PlantTraits` reused | ✅ | Verified: `photosynthesis = nothing` and `root_distribution = nothing` both work. Roots are implicit in d₂, so use no root distribution plus a new bulk root-zone PAW. |
| 13 | `DeRidderInterception <: AbstractCanopyInterception`; new methods for saturation fraction and removal; reuse PALADYN partitioning with "α to zero", SAI = 0 | ✅ type, ⚠️ PALADYN reuse | To get `I = f_veg·P` from PALADYN, α_int must be **1**, not 0 (α_int = 0 switches interception off). Better: give DeRidder its own interception function. Reuse `compute_precip_ground` (P − I + R) and the generic function names. Units: wᵣ is kg m⁻², Terrarium's `canopy_water` is m. c = 0.2 kg m⁻² per LAI = 2×10⁻⁴ m equals PALADYN's `W_can_max`. |
| 14 | `SurfaceHydrology` with the three new options | ✅ | The existing `compute_auxiliary!` order (interception → ET → runoff) fits. `vegetation` must be forwarded to interception, which PR 2 does anyway. |
| 15 | New soil, not `SoilEnergyWaterCarbon`: e.g. `ForceRestoreWater`, state as one 2-layer XYZ field or two XY fields | ✅ new soil, **two XY prognostics** | In ISBA-2L the layers **overlap**: w₂ is the bulk water content of 0…d₂, *including* the surface layer (hence I_s and E_s appear in both equations). A 2-cell grid with interfaces [−d₂, −d₁, 0] implies non-overlapping cells of thickness d₁ and d₂ − d₁. Every θ·Δz-based piece of machinery would then be wrong (mass, the forcing ÷ Δz, the root-fraction integral). Force-restore is also not a spatial discretization, so no vertical operator applies. |
| 16 | Subtype `AbstractSoilHydraulics` with C₁, C₂, C₃ and a "none" unsaturated K | ❌ | `AbstractSoilHydraulics` is the Richards-flavoured hydraulic-conductivity interface; a dummy `UnsatK` is a smell. Keep the ISBA coefficients in a separate `ForceRestoreCoefficients` struct with a pedotransfer constructor. **Reuse** the existing hydraulics only for what they already provide: w_fc and w_wp via `field_capacity`/`wilting_point`. |
| 17 | `ForceRestoreVerticalFlow <: AbstractVerticalFlow` inside `SoilHydrology`, tendencies in `soil_hydrology.jl` | ⚠️ | `SoilHydrology`'s other slots (saturation closure, XYZ variables, `vwc_forcing`) are Richards-specific. Use a standalone `ForceRestoreSoilHydrology <: AbstractSoilHydrology` in its own file, mirroring `soil_hydrology_rre.jl`. |
| 18 | Wrap everything in a new model (`ForceRestoreHydrologyModel` / `SimpleHydrologyModel`) | ✅ | This is required anyway: the combination-equation ET eliminates the skin temperature, so it cannot be combined with `LandModel`'s SEB solve. Subtype the **existing** `AbstractHydrologyModel`, which then gets its first concrete subtype. The dead `SurfaceHydrologyModel` is left for a separate cleanup PR. |

Items missing from the manual and addressed below: units and signs, VPD versus specific-humidity forcing, thermodynamic consistency with Bigleaf, **stiffness of the corrected C₁**, the canopy mass leak, and the uncapped β.

## Proposed architecture

### Model composition

```julia
@kwdef struct ForceRestoreHydrologyModel{NF, Grid, Atmosphere, Aerodynamics, Vegetation, Soil, SurfaceEnergy,
                                         SurfaceHydrology, Initializer, TimeStepper} <: AbstractHydrologyModel{NF, Grid}
    grid::Grid                                   # ColumnGrid / ColumnRingGrid without vertical resolution (PR 12); any Nz also works, since only XY fields are used
    atmosphere = PrescribedAtmosphere(NF)
    aerodynamics = NeutralAerodynamics(NF; roughness = LAIDependentCanopyRoughness(NF))  # ownership decided in PR 4
    vegetation = PrescribedVegetation(NF;
        photosynthesis = nothing,
        stomatal_conductance = JarvisStomatalConductance(NF),
        root_distribution = nothing,
        plant_available_water = BulkRootZonePAW(NF))
    soil = ForceRestoreSoil(NF)
    surface_energy_balance = NetRadiationEnergyBalance(NF;
        radiative_fluxes = PrescribedNetRadiation(NF),
        ground_heat_flux = SantanelloFriedlGroundHeatFlux(NF; reftime = DateTime(2000)))
    surface_hydrology = SurfaceHydrology(NF;
        canopy_interception = DeRidderCanopyInterception(NF),
        evapotranspiration = ShuttleworthWallaceEvapotranspiration(NF),
        surface_runoff = BergstromSurfaceRunoff(NF))
    constants = PhysicalConstants(NF)
    initializer = DefaultInitializer(NF)         # w₁, w₂, wᵣ set via initialize(model; initializers = ...)
    timestepper = Heun(NF)
end
```

Names follow Q1 (accepted in Rev 11). Where `aerodynamics` lives (a model field, as sketched, or elsewhere) is decided in PR 4. There is no snow component: precipitation is treated as liquid `rainfall`, which is a documented limitation.

### Order of operations

`compute_auxiliary!`:
1. `atmosphere`: no-op.
2. `soil`: C₁, C₂, w₁,eq, D₁, K₂ (and the drainage flux).
3. `vegetation` (`PrescribedVegetation`'s own order): PAW(soil) → phenology (LAI) → Jarvis `canopy_water_conductance`.
4. `surface_energy_balance`: R_n (input) → t_sol (in-kernel, Insolation) → G(w₁, f_veg, t_sol) → A, A_c = R_n,c, A_s = R_n,s − G (textbook downward-positive notation; stored positive upward, see Units).
5. `surface_hydrology`: interception (f_wet, I, D_c, P_s) → ET (resistances, λE_total, VPD_m, λE_t, λE_i, λE_s, H) → runoff (Q_s, infiltration).

`compute_tendencies!`: `surface_hydrology` (canopy water) → `soil` (w₁, w₂).

Coupling is entirely through the shared-field mechanism: one process's `input` is satisfied by another process's `auxiliary`, and no boundary conditions are needed. The soil declares `infiltration`, `evaporation_ground` and `transpiration` as inputs (default 0), so the force-restore soil can also run standalone in a `SoilModel` with prescribed fluxes.

For consistency with PR #204, `ForceRestoreHydrologyModel.compute_tendencies!` calls the soil as `compute_tendencies!(state, grid, soil, constants, surface_hydrology)`, with `surface_hydrology = nothing` as the standalone default. The force-restore soil needs E_s and E_t **separately** (E_s enters both w₁ and w₂, E_t only w₂). It therefore reads the `evaporation_ground` and `transpiration` fields, rather than the summed `ground_evapotranspiration_flux` used by the Richards forcing.

The canopy cover fraction f_veg = 1 − exp(−k_ext·LAI) is evaluated pointwise wherever it is needed, via `canopy_cover_fraction(i, j, grid, fields, vegetation)` (PR 2). This mirrors the existing `vegetation_area_fraction(i, j, grid, fields, vegetation)` used by the albedo, including a zero fallback for `vegetation = nothing`. It requires `vegetation` to be passed to the interception, energy and runoff calls.

### Units and sign conventions

- **Water fluxes**: m s⁻¹ of liquid water, as in existing Terrarium ET/runoff (the original uses kg m⁻² s⁻¹; conversion ÷ρ_w). With fluxes in m s⁻¹, the ρ_w in the ISBA equations cancels: `dw₁/dt = C₁/d₁·(I_s − E_s) − D₁`.
- **Energy fluxes**: the Terrarium convention, **positive upward**, everywhere, including the inputs.
  - `surface_net_radiation` must be supplied positive upward, so a typical daytime value is **negative**. Observations reported positive-downward must be converted by the user. This is stated prominently in the `PrescribedNetRadiation` docstring, the model doc page and the example; the code never flips signs silently.
  - Internally, available energy in this convention is `A = G − R_net` (`ground_heat_flux` and `surface_net_radiation` both positive upward).
  - `latent_heat_flux` and `sensible_heat_flux` are positive upward.
- **Canopy water**: m (original kg m⁻²).
- **Forcing**: air temperature in °C and specific humidity. VPD is computed in-kernel from these by the existing `compute_vapor_pressure_deficit`. Observed VPD_a (as in the original forcing) is converted to specific humidity once, host-side, with the existing `vapor_pressure_to_specific_humidity`. No new humidity type is needed.
- **Thermodynamic coefficients** come from Terrarium's `PhysicalConstants` and Thermodynamics.jl at T_a, following the "Δ(T_a) approximation" of the model description. They differ slightly from Bigleaf (λ, cₚ 1004.5 versus 1004.834, e_s formula), so reference comparisons use tolerances, not exact equality.

### Process designs

#### 1. (Removed in Rev 10)

This plan uses **no smoothing**: thresholds are hard (`max`, `min`, `clamp`, `ifelse`). Smoothing is introduced for a specific threshold **only if the PR 0 simulations fail without it**, and then as a plan revision. The design options (Kavetski & Kuczera 2007) are kept under Future work.

#### 2. Lambert–Beer util and refactor

- `lambert_beer_cover_fraction(k, area_index) = 1 − exp(−k·area_index)` goes in `src/utils/math.jl`, with a pointwise wrapper `canopy_cover_fraction(i, j, grid, fields, vegetation)` in `vegetation_base.jl`. It uses `vegetation.traits.extinction_coefficient` and `fields.leaf_area_index`, and returns zero for `vegetation = nothing`, following `vegetation_area_fraction`.
- Refactor the three existing uses (PALADYN interception, Medlyn g₀, PALADYN rₐ_can) to call it:
  - **Remove the duplicated `PALADYNCanopyInterception.k_ext`** in favour of `PlantTraits.extinction_coefficient`. This is a breaking parameter removal.
  - Forward `vegetation` to the interception in `SurfaceHydrology.compute_auxiliary!`.
- Acceptance: bit-for-bit identical `LandModel` output, since both defaults are 0.5.

#### 3. Combination-equation building blocks (`src/processes/surface/evapotranspiration/penman_monteith.jl`, `thermodynamics.jl`)

Penman–Monteith is written in the **classic vapour-pressure form**, as in the model description and Bigleaf. All the inputs already exist in Terrarium except Δ:

- **VPD at the reference level**: the existing `compute_vapor_pressure_deficit(i, j, grid, fields, atmos, c)`, i.e. e_s(T_a) − e_a from Thermodynamics.jl.
- **γ**: the existing (currently unused) `psychrometric_constant(c, p) = cₚ·p/(ε·L_v)`.
- **ρₐ**: the existing `air_density(i, j, grid, fields, atmos, constants)`.
- **Δ = de_s/dT (Pa K⁻¹)**: a new `saturation_vapor_pressure_slope(c, T)` in `thermodynamics.jl`. It is the Clausius–Clapeyron derivative `L·e_s/(R_v T²)` of the existing `saturation_vapor_pressure`.
  - It uses the same liquid/ice branch as that function (L_v above 0 °C, L_s below), so Δ is consistent with the e_s that enters the VPD.
  - e_s depends on temperature only, so Δ is an ordinary derivative. The constant-p versus constant-ρ question of Rev 2 does not arise in this form.

Pure scalar functions (not kernel functions; all coefficients are arguments), in `penman_monteith.jl`:
- `penman_monteith(Δ, γ, ρₐcₚ, A, VPD, g_a, g_s)` gives `λE = (Δ·A + ρₐcₚ·g_a·VPD)/(Δ + γ·(1 + g_a/g_s))`, in **conductance form**.
  - Potential (Penman) evaporation is `penman_monteith(…, g_a, g_s = Inf)`, i.e. r_s = 0. In conductance form that is **g_s = ∞**, not g_s = 0; g_s = 0 means a closed surface and gives λE = 0. So there is no separate `potential_latent_heat_flux` function.
- `multi_source_latent_heat_flux(Δ, γ, ρₐcₚ, A, A_c, A_s, VPD_a, r_aa, r_ac, r_as, g_sc, g_ss, f_wet)`: the Lhomme et al. (2012) closed form (model description, "Latent heat flux").
  - It is reformulated with `1/R_c` and `1/R_s` so that `g_sc → 0` and `g_ss → 0` (β → 0) have finite limits.
- `canopy_air_vapor_pressure_deficit(VPD_a, Δ, γ, ρₐcₚ, A, λE, r_aa)`: VPD_m, from Shuttleworth & Wallace (1985) eq. 8. It is algebraic in VPD_a, so it needs no humidity at the canopy source height.
- These functions map one-to-one onto Bigleaf's `potential_ET(PenmanMonteith())` for testing (Bigleaf takes kPa and °C; convert units in the test only).

#### 4. Surface aerodynamics relocation + neutral scheme + roughness (own plan document)

This PR gets its own plan document (`docs/dev/YYYY-MM/…_PLAN_surface_aerodynamics.md`), because it changes call signatures used by the SEB and by all ET schemes.

- **Relocate** `AbstractAerodynamics`, `ConstantAerodynamics`, `drag_coefficient` and `aerodynamic_resistance` from `src/processes/atmosphere/` to `src/processes/surface/aerodynamics/`.
  - **Key decision for that plan: who owns the aerodynamics object.** Options:
    - (a) a new model-level component passed to the SEB and surface hydrology the way `atmosphere` is today;
    - (b) a sub-process of `SurfaceEnergyBalance`, with surface hydrology receiving it from there.
  - Either way, `PrescribedAtmosphere(; aerodynamics)` goes away (breaking), and z_a remains `atmos.altitude`.
- **Add** `NeutralAerodynamics{NF, Roughness} <: AbstractAerodynamics{NF}`:
  - `@component roughness`;
  - `drag_coefficient = k²/ln²((z_a − d)/z₀ₘ)`;
  - `friction_velocity = k·Vₐ/ln((z_a − d)/z₀ₘ)`.
  - Result: r_aa = `compute_Ram(ResistanceWindZr)` of Bigleaf.
- **Add** `AbstractCanopyRoughness` with:
  - `StaticCanopyRoughness{NF}` (`canopy_height`, `displacement_ratio = 2/3`, `roughness_ratio = 0.1`);
  - `LAIDependentCanopyRoughness{NF}` (`canopy_height`, `c_d = 0.2`, `ground_roughness_length = 0.01` m), after Choudhury & Monteith (1988) / Shaw & Pereira (1982), equivalent to Bigleaf `RoughnessCanopyHeightLAI`. The X ≤ 0.2 branch uses `ifelse`.
- Acceptance: bit-for-bit identical `LandModel` output with `ConstantAerodynamics`.

#### 5. Jarvis stomatal conductance (`src/processes/vegetation/stomatal_conductance/jarvis_stomatal_conductance.jl`)

- `JarvisStomatalConductance{NF} <: AbstractStomatalConductance{NF}` with `@param`s:
  - `minimum_stomatal_resistance` (default 395 s m⁻¹, as in the original test setup);
  - `vpd_sensitivity` g_D (3×10⁻⁴ Pa⁻¹);
  - `optimal_temperature` (25 °C);
  - the f₁ radiation constants (0.004, 0.05, 0.81);
  - `minimum_canopy_conductance` (replaces r_smax).
  - There is **no vegetation-type constructor**. PFT-dependent values (IFS Table 8.1) are future work, via `PlantTraits`.
- `g_sc = LAI/r_smin · f₁(SW↓)·f₂·f₃(VPD_a)·f₄(T_a)`. The factors are:
  - f₁ from IFS eq. 8.9;
  - f₂ = `soil_moisture_limiting_factor`;
  - f₃ = exp(−g_D·VPD) (IFS);
  - f₄ = 1 − 0.0016(T_opt − T_a)² (Noilhan & Planton 1989).
  - All use a hard `clamp` to [0, 1].
- Declares `auxiliary(:canopy_water_conductance, XY())` and the inputs `leaf_area_index` and `soil_moisture_limiting_factor`.

#### 6. Force-restore soil (`src/processes/soil/soil_force_restore.jl`, `src/processes/soil/hydrology/soil_hydrology_force_restore.jl`)

- **`ForceRestoreSoil{NF, Stratigraphy, Hydrology} <: AbstractSoil{NF}`**, with fields `strat` and `hydrology`, and **no biogeochemistry**.
  - The only reason Rev 0 had a biogeochemistry field was that `porosity(…, strat, bgc)` needs the organic fraction. Instead, add a method `porosity(i, j, k, grid, fields, strat, ::Nothing)` = mineral porosity of the horizon. This keeps `SoilPorositySURFEX` (the ISBA w_sat from sand) available.
  - For grids without a vertical axis (PR 12), add a depth-free `with_soil_horizon` method for single-horizon stratigraphies (`SoilStratigraphy{NF, 1}`). The current method picks the horizon via `znode`, which returns `nothing` on a `Flat` z, so `nothing <= nothing` would throw. This method is only needed once PR 12 lands.
  - `get_biogeochemistry` and `get_energy_balance` return `nothing`.
- **`ForceRestoreSoilHydrology{NF, Hydraulics, Coefficients} <: AbstractSoilHydrology{NF}`**:
  - `@param` `surface_layer_depth` d₁ (0.01 m), `root_zone_depth` d₂, `restore_timescale` τ (86400 s).
  - `@component hydraulic_properties` (`SoilHydraulicsSURFEX` by default, which is exactly N&M 1996 for w_fc and w_wp; `ConstantSoilHydraulics` to set w_fc/w_wp directly).
  - `@component coefficients::ForceRestoreCoefficients`.
- **`ForceRestoreCoefficients{NF}`**:
  - Parameters `C1sat`, `C2ref`, `C3`, `a`, `p`, `b` and `w_l = 0.01`.
  - Host-side constructor `ForceRestoreCoefficients(NF, texture::SoilTexture)` from the N&M 1996 eqs. 30–36 pedotransfer functions.
  - Optionally `b` from van Genuchten n (Morel-Seytoux 1996).
- **Variables** (all XY):
  - prognostics `surface_volumetric_water_content` (w₁) and `root_zone_volumetric_water_content` (w₂), both m³ m⁻³;
  - auxiliaries `force_coefficient` (C₁), `restore_coefficient` (C₂), `equilibrium_surface_water_content` (w₁,eq), `surface_restore_rate` (D₁, s⁻¹) and `drainage` (d₂·K₂, m s⁻¹);
  - inputs `infiltration`, `evaporation_ground` and `transpiration`.
- **Tendencies** (Mahfouf 1996, Boone 1999 notation):
  - `dw₁/dt = C₁/d₁·(I − E_s) − C₂/τ·(w₁ − w₁,eq)`
  - `dw₂/dt = (I − E_s − E_t)/d₂ − C₃/(d₂τ)·max(w₂ − w_fc, 0)`, as in the original
- **Regularization of C₁**: `(w_sat/w₁)^(b/2+1)` diverges as w₁ → 0. Evaluate it at `max(w₁, w_wp)`, the validity range of the N&M formula. Improved dry-soil formulations are future work (Rev 5).
- **Soil-side coupling functions** (dispatch on `ForceRestoreSoil`), used by other processes:
  - `surface_water_content`, `root_zone_water_content`;
  - `relative_root_zone_wetness` = w₂/w_sat (for runoff);
  - `surface_relative_saturation` Θ = w₁/w_sat (for G).
- **New PAW**: `BulkRootZonePAW{NF} <: AbstractPlantAvailableWater` declares `auxiliary(:soil_moisture_limiting_factor, XY())` = `clamp((w₂ − w_wp)/(w_fc − w_wp), 0, 1)`, as in `FieldCapacityLimitedPAW`.
- **New dispatch** `ground_evaporation_resistance_factor(i, j, grid, fields, ::SoilMoistureResistanceFactor, soil::ForceRestoreSoil)` gives the Lee–Pielke β on w₁, capped at w_fc, with **θ_res = 0**: it calls the existing scalar `ground_evaporation_resistance_factor(res, w₁, w_fc, zero(NF))`, so the original Lee & Pielke (1992) form is recovered (Rev 13).

#### 7. Prescribed net radiation (follow-up to PR #200)

- `PrescribedNetRadiation{NF} <: AbstractRadiativeFluxes`, with `input(:surface_net_radiation, XY())` **positive upward**, in `radiative_fluxes.jl`.
- It works in `SurfaceEnergyBalance` too, for any configuration that only needs R_net (e.g. PR #200's `PrescribedSurfaceEnergyBalance` with a prescribed R_net).
- The docstring states the sign convention explicitly, with a `jldoctest` showing a daytime (negative) value.

#### 8. Insolation.jl dependency and in-kernel solar time

**Dependency.**
- Add Insolation.jl to the root `[deps]` and `[compat]` (`Insolation = "1.2"`).
- Costs to accept explicitly:
  - two **non-lazy artifacts** are downloaded at install (affects CI and offline installs);
  - DelimitedFiles and Artifacts enter the dependency tree;
  - the functions we need are **internal** (unexported), so compat must be pinned tightly. Also open an upstream issue asking to export `hour_angle` / `equation_of_time` or a `solar_time` helper.

**Constants without ClimaParams**, mirroring `ThermodynamicConstants`:
- `OrbitalConstants{NF} <: Insolation.Parameters.AbstractInsolationParams` in `src/processes/constants.jl`.
- Only the fields needed for solar time: `year_anom`, `day`, `eccentricity_epoch`, `obliq_epoch`, `lon_perihelion_epoch`, `epoch::DateTime`, `mean_anom_epoch`, with J2000 defaults.
- Getter overloads (`Insolation.Parameters.year_anom(c::OrbitalConstants) = c.year_anom`, …).
- It becomes a new field `orbital::OrbitalConstants{NF}` of `PhysicalConstants`, with a default in the keyword constructor, exactly as `thermodynamics::ThermodynamicConstants{NF}` (Rev 5).

**Why not simply call `insolation(date, …)` in a kernel.** Insolation.jl is GPU-compatible, but three things block using its exported `DateTime` API inside Terrarium kernels as-is:
1. The model clock has no date. Building a `DateTime` in-kernel from `reftime + t` needs `round`/`convert` to `Int`, a reachable `InexactError` that AGENTS.md bans.
2. `get_orbital_parameters` (called by `insolation`) contains an `error(...)` branch that is reachable when `milankovitch` is a runtime `Bool`.
3. The quantity we need (hour angle → seconds since solar noon) is not returned by the exported `solar_geometry` (d, θ, ζ).

**Proposed in-kernel path.**
- A `SolarTime{NF}` component stores isbits floats precomputed **host-side** in its keyword constructor from `reftime::DateTime`:
  - `years_since_epoch_at_reftime`;
  - `time_of_day_at_reftime` (fraction of a day).
- In-kernel, from `clock.time = t`:
  - `Δt_years = years_since_epoch_at_reftime + t/year_anom`;
  - `MA = Insolation.mean_anomaly(Δt_years, orbital)`;
  - `Δη = Insolation.equation_of_time(MA, Insolation.orbital_params(orbital))`;
  - `η = mod(2π·(time_of_day_at_reftime + t/day) + Δη + λ, 2π)`;
  - `t_sol = (mod(η + π, 2π) − π)/(2π)·day`.
- Longitude λ comes from an XY `input(:longitude)` (degrees). `ColumnGrid` has no longitudes; deriving it from `ColumnRingGrid` is future work.
- There is no `DateTime` inside kernels and no throw path. This is verified by a Float32 CPU run and by `@code_llvm` containing no `throw`/`error` call.

#### 9. Surface processes

**`SantanelloFriedlGroundHeatFlux{NF, SolarTime} <: AbstractGroundHeatFlux{NF}`** (PR #200's `ground_heat_flux.jl`):
- Parameters: `c_gmin = 0.31`, `c_gmax = 0.35`, `t_gmin = 74000 s`, `t_gmax = 100000 s`, `phase_shift = 10800 s`.
- `@component solar_time::SolarTime`.
- Declares `auxiliary(:ground_heat_flux)` and `input(:longitude)`.
- Computes `G = c_g(Θ)·cos(2π(t_sol + phase_shift)/t_g(Θ))·R_n,s` in the textbook convention, then stores it positive upward.
  - t_sol is the time relative to solar noon, **negative before noon**, so G/R_n,s peaks 3 h before solar noon (Santanello & Friedl 2003).
  - `c_g = (1 − Θ)·c_gmax + Θ·c_gmin` and `t_g = (1 − Θ)·t_gmax + Θ·t_gmin`, linear in Θ = w₁/w_sat, as in the model description.
  - Θ comes from the prognostic w₁, so G is explicit. ALEXI and STIC instead iterate G with a flux-dependent wetness measure.
- Docstring citation (the form is a combination of three sources): Santanello & Friedl (2003) Eq. 4, applied to R_n,s as in ALEXI (Anderson et al. 2018, Eq. 15), with the linear wetness interpolation of Mallick et al. (2022, Eqs. S1.19–S1.20) using Θ = w₁/w_sat instead of STIC's moisture index I_SM.
- Unlike `DiagnosedGroundHeatFlux`, it is not a residual, so it must be evaluated *before* the ET.

**`NetRadiationEnergyBalance{NF, RadiativeFluxes, GroundHeatFlux} <: AbstractSurfaceEnergyBalance{NF}`** (`src/processes/surface/net_radiation_energy_balance.jl`):
- Auxiliaries: `available_energy` (A), `available_energy_canopy` (A_c) and `available_energy_ground` (A_s).
- **Why a separate container rather than `SurfaceEnergyBalance`** (Q12):
  - `SurfaceEnergyBalance` always carries a skin temperature and runs its fluxes in `solve_surface_energy_balance!` during `compute_boundary_conditions!`, i.e. *after* the surface-hydrology auxiliaries.
  - The container reuses PR #200's sub-process types, so ground-heat-flux implementations stay interchangeable.

**`DeRidderCanopyInterception{NF} <: AbstractCanopyInterception{NF}`** (`canopy_interception/deridder_canopy_interception.jl`):
- Parameters:
  - `canopy_capacity_per_lai` c = 2×10⁻⁴ m;
  - `wetness_exponent = 2/3` (Deardorff 1978).
- Same variable names as PALADYN: prognostic `canopy_water`, auxiliaries `canopy_water_interception`, `canopy_water_removal` (D_c), `saturation_canopy_water` (f_wet) and `rainfall_ground` (P_s).
- `I = f_veg·P` and `D_c = (1 − f_veg)·P·(exp(b·wᵣ) − 1)` with b = k_ext/c (De Ridder 2001; b = 1/(2c) for k_ext = ½, see below). There are **no limiter kernels** (Rev 5).
  - The drainage is self-limiting at capacity. Without evaporation, I = D_c gives exp(b·wᵣ) = 1/(1 − f_veg) = exp(k_ext·LAI), so wᵣ,eq = k_ext·LAI/b = c·LAI = wᵣ,max. This is why the original upper-bound kernel is redundant. With the published b = 1/(2c) this holds only for k_ext = ½.
  - **The ½ is not a free choice** (De Ridder 2001, Appendix A). Rain is treated as vertical beams hitting randomly located leaves with a *uniform leaf angle distribution*. An infinitesimal layer of cumulative LAI Δλ then intercepts a fraction Δλ/2, giving the throughfall 1 − f = e^(−L/2) (Eq. 6). The same ½ reappears in the storage equation ∂φ/∂t = −(1/(2c))·R·φ (Eq. 11) and hence in the Rutter drainage coefficient b = 1/(2c) (Eqs. 39–41).
  - Redoing the derivation with a general interception fraction k per unit LAI (my generalization; the paper only treats k = ½) gives 1 − f = e^(−kL) and b = k/c, so the no-evaporation equilibrium is wᵣ = cL for **any** k. The published b = 1/(2c) is this with k = ½.
  - **Decision (Rev 7): b = k/c with k = `PlantTraits.extinction_coefficient`**, the same k as in f_veg. This reduces exactly to De Ridder for the default k = 0.5 and keeps throughfall and drainage mutually consistent.
  - **Docstring** (one sentence): "Assumes a uniform leaf angle distribution and vertically incident rain, i.e. an extinction coefficient of ½ (De Ridder 2001, Appendix A); b = k/c extends the drainage to other k."
  - **Warning**: a host-side check `check_canopy_interception_traits(interception::DeRidderCanopyInterception, vegetation)` emits `@warn "DeRidderCanopyInterception assumes an extinction coefficient of 1/2, got $k" maxlog = 1` when `!(k ≈ 1/2)`.
    - It is called from the `ForceRestoreHydrologyModel` keyword constructor, and from any other model constructor that combines this interception scheme with vegetation. A no-op fallback covers other interception types.
    - It never runs in kernels (AGENTS.md: validation belongs in host-side constructors).
    - `maxlog = 1` avoids flooding the log when models are rebuilt repeatedly, e.g. in calibration loops.
    - It is a warning, not an error, so k stays calibratable.
- `P_s = P − I + D_c` reuses `compute_precip_ground`, which fixes the mass leak.
- New methods of `compute_canopy_saturation_fraction` and `compute_canopy_water_removal`.
- **f_wet** (Rev 10): `f_wet = min((max(wᵣ, 0)/wᵣ,max)^(2/3), 1)`, i.e. Deardorff exactly, with hard bounds.
  - `max(wᵣ, 0)` keeps an explicit-step undershoot away from `x^(2/3)`, which would otherwise throw a `DomainError` in the kernel.
  - `min(·, 1)` is needed without the upper-bound kernel, so that (1 − f_wet) stays non-negative in λE_t.
  - Known risk: d f_wet/d wᵣ is infinite at wᵣ = 0, e.g. for a dry initial canopy, which can give non-finite Enzyme gradients there. PR 0 and PR 10 check whether this matters in practice.

**`ShuttleworthWallaceEvapotranspiration{NF, GroundResistance} <: AbstractEvapotranspiration{NF}`** (`evapotranspiration/shuttleworth_wallace_evapotranspiration.jl`):
- Parameters: `kB⁻¹ = ln(10)` and **`attenuation_factor` η = 3** (exponential decay of the in-canopy eddy diffusivity; Bonan's terminology).
- `@component ground_resistance = SoilMoistureResistanceFactor(NF)`.
- Resistances:
  - r_aa = `aerodynamic_resistance` (PR 4);
  - r_ac = `canopy_boundary_layer_resistance` = kB⁻¹/(k·u*);
  - r_as = `ground_aerodynamic_resistance` (Choudhury & Monteith 1988 eq. 25);
  - g_sc = `canopy_water_conductance`;
  - g_ss from β.
- Algorithm: λE_total (Lhomme) → VPD_m → λE_t = (1 − f_wet)·PM(A_c, VPD_m, 1/r_ac, g_sc), λE_i = f_wet·PM(A_c, VPD_m, 1/r_ac, ∞) and λE_s = PM(A_s, VPD_m, 1/r_as, g_ss) → E = λE/(L_v·ρ_w) → H = A − λE.
- Auxiliaries:
  - `evaporation_ground`, `transpiration` and `evaporation_canopy` (same names as PALADYN);
  - `latent_heat_flux`, `sensible_heat_flux`, `potential_latent_heat_flux` (diagnostic field only; computed with `penman_monteith`) and `canopy_air_vapor_pressure_deficit`;
  - the five resistances, as diagnostics.
- `ground_evapotranspiration_flux` returns E_s + E_t.

**`BergstromSurfaceRunoff{NF, Exponent} <: AbstractSurfaceRunoff{NF}`** (`runoff/bergstrom_surface_runoff.jl`):
- `Q_s = relative_root_zone_wetness^p·P_s` (= (w₂/w_sat)^p·P_s for the force-restore soil), with `infiltration = P_s − Q_s`. The ratio is taken against the maximum storage, following HBV, where FC is the "maximum soil moisture storage" and a model parameter (Seibert 2005), and SINDBAD's wSoil_max. In the exact ODE, w_sat is an invariant upper bound: at w₂ = w_sat the infiltration is zero, so dw₂/dt ≤ 0. Only an explicit step can overshoot. No cap is applied (Rev 10). PR 0 checks whether overshoot occurs at the chosen Δt.
- Exponent options (Trautmann et al. 2022):
  - `ConstantInfiltrationExponent(p = 2)`;
  - `VegetationInfiltrationExponent(s = 3)`, with p = s·f_veg.
- Same auxiliary names as `DirectSurfaceRunoff` (`surface_runoff`, `infiltration`).

## Summary of changes

A sequence of focused PRs, tracked in one GitHub issue (see "Tracking"). Each PR is independently testable, documented, and signed off separately before implementation.

| PR | Content | Main files | Depends on | Standalone test vehicle |
|---|---|---|---|---|
| **0** (outside Terrarium) | Patch the original EvaporationModel with the agreed fixes (C₁, Santanello–Friedl on R_ns, runoff on P_s with w₂/w_sat, canopy limiter kernels removed, f_wet with hard bounds, β cap). Generate reference trajectories with fixed-step Heun on the synthetic forcing and 10 days of BE-Bra. Find the stable Δt for the corrected C₁. **Run with the hard-threshold formulations of this plan** (no `smooth_*`, no canopy kernels). A failure here is the only trigger for introducing smoothing. | DifferentiableEvaporation repo | — | the original ODE |
| ~~1~~ | *(Removed in Rev 10: smoothing is future work)* | — | — | — |
| **2** | Lambert–Beer util + refactor of the 3 existing uses; remove duplicated `k_ext`; forward `vegetation` to interception | `utils/math.jl`, `vegetation/vegetation_base.jl`, `canopy_interception/canopy_interception.jl`, `stomatal_conductance/medlyn_stomatal_conductance.jl`, `evapotranspiration/canopy_evapotranspiration.jl`, `surface/surface_hydrology.jl` | — | bit-for-bit `LandModel` regression |
| **3** | Combination-equation building blocks (Δ = de_s/dT, PM in vapour-pressure form, Lhomme, VPD_m) | `thermodynamics/thermodynamics.jl`, `evapotranspiration/penman_monteith.jl` | — | unit tests vs Bigleaf |
| **4** | Aerodynamics relocation + `NeutralAerodynamics` + canopy roughness (**own plan doc**) | `surface/aerodynamics/*`, `atmosphere/*`, all rₐ call sites | — | Bigleaf tests; bit-for-bit `LandModel` regression |
| **5** | `JarvisStomatalConductance` | `vegetation/stomatal_conductance/jarvis_stomatal_conductance.jl` | — | `VegetationModel` with `photosynthesis = nothing` and existing Richards soil PAW |
| **6** | Force-restore soil, `BulkRootZonePAW`, β dispatch, soil coupling functions, `porosity(…, ::Nothing)` | `soil/soil_force_restore.jl`, `soil/hydrology/soil_hydrology_force_restore.jl`, `soil/stratigraphy/soil_stratigraphy.jl`, `vegetation/hydraulics/plant_available_water.jl`, `evapotranspiration/ground_resistance_factor.jl` | — (the single-horizon `with_soil_horizon` method is only needed with 12) | `SoilModel` with prescribed infiltration/ET inputs |
| **#200** (external) | Ground heat flux as an SEB sub-process | — | — | — |
| **7** | `PrescribedNetRadiation` | `surface/radiative_fluxes.jl` | #200 | SEB unit tests |
| **8** | Insolation.jl dependency, `OrbitalConstants`, in-kernel `SolarTime` | `Project.toml`, `processes/constants.jl`, new solar-time file | — | comparison vs Insolation's exported `DateTime` API and NOAA values; Float32 run |
| **9** | `SantanelloFriedlGroundHeatFlux`, `NetRadiationEnergyBalance`, `DeRidderCanopyInterception`, `BergstromSurfaceRunoff`, `ShuttleworthWallaceEvapotranspiration` (may split in two) | `surface/ground_heat_flux.jl`, `surface/net_radiation_energy_balance.jl`, `canopy_interception/deridder_canopy_interception.jl`, `runoff/bergstrom_surface_runoff.jl`, `evapotranspiration/shuttleworth_wallace_evapotranspiration.jl` | 2–8 (and #200 via 7) | kernel-function unit tests |
| **10** | `ForceRestoreHydrologyModel`, example, reference comparison, Enzyme test | `models/hydrology/force_restore_hydrology_model.jl`, `models/models.jl`, `src/Terrarium.jl` (exports), `examples/simulations/force_restore_column.jl` | 9 (12 for the example's grid) | full model |
| **12** (own plan doc; after #194) | Grids **without vertical resolution**: z-less `ColumnGrid` and `ColumnRingGrid` constructors | `grids/column_grid.jl`, `grids/column_ring_grid.jl`, `grids/grid_utils.jl`, `grids/land_grid.jl` | #193 (merged), #194 | XY-only model on both grids |

PRs 2–6 and 8 are independent and can proceed in parallel.

### PR 12: grids without vertical resolution

For this model a vertical resolution is meaningless, so users should not have to choose one. PR 12 follows #193 (grid types, merged) and #194 (`VarLocation`/`VarDomain`, open), which touch the same files, and gets its own plan document.

- **Tested at `cd2f122` (2026-09-29)**: the XY-only example model runs unchanged on a plain `RectilinearGrid` with topology (Periodic, Flat, Flat). Oceananigans treats the `Flat` z as a single level: Nz = 1, `znode` returns `nothing`, and `Δzᵃᵃᶜ` returns 1.0.
- **Constructors**:
  - `ColumnGrid(arch, NF, num_columns::Int = 1)` without a vertical coordinate builds (Periodic, Flat, Flat). This does not clash with the existing `ColumnGrid(arch, NF, vertical_coordinate, num_columns)`, because the vertical coordinate is an `AbstractVector` or discretization.
  - `ColumnRingGrid` gets the same option. Its supertype `AbstractGrid{NF, Periodic, Flat, Bounded, …}` needs a z-topology parameter.
  - The `ColumnGrid` alias widens to `Bounded` or `Flat` z. Nothing in `src/` dispatches on it.
- **One loud check**: allocating an `XYZ` variable on a ground domain with `Flat` z raises an `ArgumentError`, in the same place #194 already rejects `XYZ` variables on undiscretized domains (`default_domain_grid`).
  - This replaces the silent Δz = 1 m that vertically resolved processes (Richards soil, heat conduction, `ImplicitSkinTemperature`, snow) would otherwise see.
  - One mechanism, one place (restraint Rule 2).
- **To decide in its plan**: how #194's `Top`/`Bottom` coordinates behave on a `Flat` z. Either they map to the single level, or they are rejected.
- **To verify**: `FieldTimeSeries`/`InputSource` inputs, output writers and `Simulation` on a `Flat` z.
- **Tests**: XY-only models on both z-less grids, the `XYZ` rejection, and an input time series.


Every PR adds exports for new public types, explicit imports, and `@kwdef`/`@parameterized` keyword constructors with `(::Type{NF}; kwargs...)`. The only root `Project.toml` change is Insolation.jl (PR 8); Bigleaf.jl is added to `test/Project.toml` only.

## Tracking

- **One GitHub tracking issue** ("Port DifferentiableEvaporation force-restore hydrology model"). It holds a task list of PRs 2–10 and 12 (plus #200, #194 and PR 0), links to this plan, and records the dependency order.
- **This umbrella plan stays in `docs/dev/2026-09/`**, per AGENTS.md: dated by its initial draft.
  - It should reach `main` early through a docs-only PR (e.g. the current `ob/hydrology-model` branch with the manual and this plan), so the issue can link a stable path.
  - Each implementation PR updates the Status line and appends to the Revision log.
- **PR-specific plans** (definitely PR 4 and PR 12, probably PR 8) are separate files in `docs/dev/YYYY-MM/` for their drafting date. They link back here, and this plan links to them.

## Testing and verification

- **Unit tests (per PR)** for every pointwise and kernel function:
  - **Bigleaf.jl as reference** (test-only dependency): `potential_ET(PenmanMonteith())`, `roughness_parameters`, `compute_Ram`, `Gb_constant_kB1`, `Esat_from_Tair_deriv`.
  - **Hard-coded reference values** from the patched original (PR 0) for the rest.
- **Consistency invariants**:
  - λE_total = λE_t + λE_i + λE_s;
  - H + λE = A;
  - PM(g_s = ∞) = Penman;
  - the Lhomme form reduces to single-source PM for f_veg = 1 and f_wet = 0;
  - `saturation_vapor_pressure_slope` versus a finite difference of `saturation_vapor_pressure`, and versus Bigleaf `Esat_from_Tair_deriv`.
- **De Ridder k check**: `@test_logs (:warn, r"extinction")` when constructing the model with `PlantTraits(extinction_coefficient = 0.7)`, and `@test_logs` with no warning at the default 0.5.
- **Limit / robustness tests**: β → 0, LAI → 0, wᵣ → 0 and w₁ → w_wp all give finite values. Enzyme derivatives are finite away from the hard thresholds; the f_wet slope at wᵣ = 0 is the known exception.
- **Solar time (PR 8)**:
  - agrees with `Insolation.solar_geometry`/`hour_angle` on host `DateTime`s;
  - matches the original `local_to_solar_time` doctest (within a minute);
  - no throw paths in the kernel.
- **Analytic soil tests (PR 6)**:
  - with no fluxes, constant w₂ and w₂ ≤ w_fc, w₁ relaxes to w₁,eq with rate C₂/τ;
  - drainage-only decay.
- **Mass conservation (PR 10)**: `d/dt(d₂·w₂ + wᵣ) = P − Q_s − E_s − E_t − E_i − d₂·K₂`, checked to round-off.
- **Reference comparison (PR 10)**: 1-day synthetic and 10-day BE-Bra runs versus the PR 0 trajectories at equal Δt (tolerance Q11).
- **Type stability and allocations**: `@inferred` on kernel functions; a Float32 model run.
- **Enzyme (PR 10)**: gradient of cumulative λE and of final w₂ with respect to r_smin, p, d₂ and C2ref, against finite differences.
- Full `Pkg.test()`, then the draft doc build `julia --project=docs docs/make.jl --local --draft`.

## Documentation changes

**General principle: keep documentation concise.** Docstrings, doc pages, comments and log messages say what is there, briefly. Rationale and history live in this plan, commit messages and PR descriptions. The concrete rules are adapted from NumericalEarth.jl's [`style-rules.md`](https://github.com/NumericalEarth/NumericalEarth.jl/blob/fffb946b7a2c9cbb5ae2dd05b4a69ce6884cd988/.claude/rules/style-rules.md) ("Comments") and [`restraint-rules.md`](https://github.com/NumericalEarth/NumericalEarth.jl/blob/fffb946b7a2c9cbb5ae2dd05b4a69ce6884cd988/.claude/rules/restraint-rules.md) (Rules 7, 8, 9, 12):

- **Docstrings describe, they do not argue.** State what the function or type does, its arguments with units, the governing equation, and the citation. No defence of the function's existence, no bulleted rationale, no tuning essays, and no clusters of `@ref`s to siblings needed to understand it.
- **No history or contrast wording** in docstrings or comments ("no longer", "previously", "instead of", "now does"). Original-code deviations and design choices are recorded in this plan's revision log and in PR descriptions, not in the source.
- **Comments: default to none.** Add a one-line comment only at a genuinely non-obvious step, such as a sign convention, a numerical-stability detail or an index trick. Never restate the next line, and never describe other functions or callers.
- **Warning and error messages are one sentence**: name the object, state the requirement, stop.
- **Doctests** show the object (`show`/`summary`) or a computed value. They do not assert that a constructor or `set!` worked.
- **Doc pages** keep the AGENTS.md section structure (Overview, Implementations, Methods, Kernel functions). Each section is equations plus a short explanation and citations, not narrative.

- **New model page**: `docs/src/models/force_restore_hydrology_model.md`, including a prominent note on the positive-upward R_n convention.
- **Process pages**, each extended with Implementations, Methods and Kernel-functions sections using `canonical = false`:
  - soil hydrology (force-restore);
  - canopy interception (De Ridder);
  - evapotranspiration (Shuttleworth–Wallace + Penman–Monteith);
  - surface runoff (Bergström);
  - surface energy balance (`NetRadiationEnergyBalance`, `PrescribedNetRadiation`, Santanello–Friedl);
  - surface aerodynamics (a new page, from PR 4);
  - vegetation (Jarvis, bulk root-zone PAW).
  - Utilities page for solar time.
- **`docs/src/references.bib`** additions: Noilhan & Planton 1989, Mahfouf & Noilhan 1996, Boone et al. 1999, Deardorff 1978, Shuttleworth & Wallace 1985, Lhomme et al. 2012, Choudhury & Monteith 1988, Shaw & Pereira 1982, De Ridder 2001, Bergström & Lindström 2015, Seibert 2005 (HBV-light manual), Trautmann et al. 2022, Lee & Pielke 1992, Santanello & Friedl 2003, Anderson et al. 2018, ECMWF IFS Cy47r3/Cy49r1, Knauer et al. 2018, Mallick et al. 2022 (GRL, doi:10.1029/2021GL097568), Morel-Seytoux et al. 1996, Bonan 2019. `noilhanISBA1996` already exists.
- **Literate example** `examples/simulations/force_restore_column.jl` with synthetic diurnal forcing, on a `ColumnGrid` without vertical resolution (PR 12), registered in `docs/make.jl`.

## Known limitations

- **Stiffness**: with the corrected C₁ and d₁ = 1 cm, the w₁ equation is stiff for dry soils. A rough estimate with the original parameters gives C₁ ≈ 10³ at w₁ = 0.1, i.e. an explicit Δt of seconds. This is the **main numerical risk**, and PR 0 quantifies it. Mitigations, in order of preference:
  1. regularize C₁ via the hard floor at w_wp (chosen), or a physical dry-soil formulation later;
  2. adaptive stepping via a `cell_diffusion_timescale`-style method;
  3. an implicit/IMEX treatment of w₁ (no implicit stepper exists yet).
- **Santanello–Friedl is discontinuous at solar midnight.** t_sol wraps from +12 h to −12 h, and because t_g ≠ 86400 s the cosine does not match across the wrap, so G jumps by c_g·R_n,s·Δcos. The formulation is a daytime one. Document it; if it hurts gradients, smoothing or a daylight weighting is future work.
- **Canopy stiffness under heavy rain.** De Ridder (2001) notes that forward Euler requires Δt ≪ t_c = cL/(f·R₀), the time to fill the canopy. His discrete alternative (Eq. 36) integrates over the step analytically, which is a discrete-time update and not allowed in Terrarium. For c = 0.2 kg m⁻², L = 3 and 10 mm h⁻¹ of rain, t_c ≈ 5 min, so this can bind the time step during intense rain. It may also explain the instabilities that originally motivated the canopy kernels (Q7).
- **Hard thresholds and explicit time-stepping.** Hard `max`/`min`/`clamp` make the model non-differentiable at the thresholds (C₁ floor, K₂ onset, f_wet bounds, Jarvis clamps, Lee–Pielke β), which can roughen calibration objective functions (Kavetski & Kuczera 2007). Explicit steps can also overshoot bounds. Both are accepted for this plan; smoothing is future work.
- Neutral stability only (no Monin–Obukhov).
- No snow, no frozen soil, no soil temperature.
- Thermodynamic coefficients are evaluated at T_a.
- The combination-equation ET is **incompatible with `LandModel`'s SEB skin-temperature solve** (by construction), enforced by dispatch.
- Insolation.jl's internal functions are used; this relies on a tight compat bound until the functions are exported upstream.
- Single patch, single PFT.

## Open questions

1. *(Resolved in Rev 11: the proposed names are accepted: `ForceRestoreHydrologyModel`, `ShuttleworthWallaceEvapotranspiration`, `BergstromSurfaceRunoff`, `lambert_beer_cover_fraction`, `surface_volumetric_water_content` / `root_zone_volumetric_water_content`.)*
2. *(Resolved in Rev 5: w₂/w_sat, since HBV's FC is a maximum storage.)*
3. *(Resolved in Rev 2: R_n is supplied positive upward, with no sign flip in code.)*
4. *(Resolved in Rev 5: VPD is computed in-kernel from specific humidity; observed VPD is converted host-side.)*
5. *(Resolved in Rev 2: in-kernel solar time via Insolation.jl.)*
6. *(Resolved in Rev 5, revised in Rev 10: hard floor at w_wp.)*
7. *(Resolved in Rev 5: no canopy limiter kernels for now.)*
8. *(Resolved in Rev 5: from `ForceRestoreSoil.strat`.)*
9. *(Resolved in Rev 10: Deardorff's exponent kept exactly, with hard bounds at 0 and 1; see the De Ridder section.)*
10. *(Resolved in Rev 13: the force-restore application of `SoilMoistureResistanceFactor` uses θ_res = 0, i.e. the original Lee & Pielke (1992) form. Terrarium's existing residual adaptation is left unchanged.)*
    - Literature (Rev 12):
      - The Lee–Pielke form as reproduced in Merlin et al. (2016, Eq. 13; CLM4.5) has no residual term, and neither does ISBA's α (Eq. 9).
      - A residual as the *dry end* of the bare-soil stress does have precedent, in linear form. H-TESSEL/IFS (Albergel et al. 2012, Eqs. 5–6; Balsamo et al. 2011) uses f₂′ = (w − w_min)/(w_fc − w_min) with w_min = veg·w_wilt + (1 − veg)·w_res, arguing that the wilting point applies only to vegetation and that bare soil keeps evaporating below it. The GLEAM linear β (Martens et al. 2017; the original's `Martens17` option) also uses (w − w_res)/(w_c − w_res).
      - No source found applies the residual inside the cosine form. Terrarium's version (commit `2b0f8749e`) is an uncited adaptation, and its docstring still shows the formula without the residual.
    - Existing-code caveat, to report separately: for θ < θ_res the rescaled argument turns negative and the cosine makes β rise again. A `max(·, 0)` on the argument is needed if θ < θ_res is reachable.
11. **Reference-comparison tolerance** for PR 10.
12. **Energy container**: keep the lightweight `NetRadiationEnergyBalance` (proposed), or extend `SurfaceEnergyBalance` after PR #200 so it can run without a skin temperature and evaluate a non-residual G before the surface hydrology?
13. *(Resolved in Rev 5: pointwise `canopy_cover_fraction(i, j, grid, fields, vegetation)`, like `vegetation_area_fraction`.)*
14. *(Resolved in Rev 5: a field of `PhysicalConstants`.)*
15. *(Resolved in Rev 7: b = k/c with k from `PlantTraits`, plus a docstring note and a host-side warning when k ≠ ½.)*

## Future work

- **Reactant**: registry entry `:force_restore_column` in `test/reactant/setup.jl`, and a benchmark configuration.
- **PFT-dependent Jarvis parameters** (IFS Table 8.1) via `PlantTraits`.
- Longitude from `ColumnRingGrid` for solar time; upstream export of Insolation's hour-angle helpers.
- Generalize `BergstromSurfaceRunoff`, `SoilMoistureResistanceFactor` and the Jarvis coupling to the Richards soil (`relative_root_zone_wetness` for `SoilEnergyWaterCarbon`).
- Root-weighted transpiration uptake for the Richards soil. Since PR #204 all ET reaches the soil, but only from the top layer.
- **Smoothing** (moved here in Rev 10; introduce earlier only if PR 0 fails without it):
  - Kavetski & Kuczera (2007) propose: a logistic step (Eq. 8) or arctan step (Eq. 9); for max(x, 0) the CHKS form ½(x + √(x² + m)) (Eq. 11) or softplus (Eq. 13); a logistic piecewise-linear blend (Eq. 15); a smooth min for storage-capped fluxes (Eq. 18); and the exponential storage kernel 1 − e^(−(S − S*)/m) (Eqs. 19–20). They solve the result with implicit Euler (Eq. 23).
  - m in Eqs. 11/18 has the units of x². The original EvaporationModel implements Eqs. 11/18 and Eq. 20 (plus a mirrored upper bound), but uses a non-squared m in f_wet and the r_s cap.
  - Candidate design: a switchable `AbstractSmoothing` strategy (`NoSmoothing`, `QuadraticSmoothing(δ)` with m = δ², `LogisticSmoothing(δ)`) held by each thresholding process, optional canopy storage limiters, and an ε-regularized f_wet.
  - Candidate experiment (Kavetski §8): no smoothing versus smoothing variants on synthetic, BE-Bra and stress-case forcing. Measure robustness (stable Δt, bound violations), prediction impact, loss and gradient smoothness (Enzyme versus finite differences), and twin-experiment calibration.
  - An implicit/IMEX stepper, which Kavetski recommends for smoothed ODEs.
- Remove or revive the dead `SurfaceHydrologyModel`.
- Spatially varying parameters (clay from `PrescribedSoilHorizon`, r_smin/d₂ from remote sensing), and hybrid/NN parameterization of p (Feng et al. 2022).
- Stability corrections; Murray & Verhoef (2007) ground heat flux.
