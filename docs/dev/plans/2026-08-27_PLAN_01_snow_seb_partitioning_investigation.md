# Investigation: insufficient high-latitude winter snow accumulation (SEB/snow-fraction coupling)

> Status: **completed**. Four independent bugs from the original investigation identified and fixed
> (Fix B `G`/`S` blending, the accumulation-onset energy gate, the liquid-fraction floor dilution — see
> Rev 4/5 — and the Rev 6 explicit-Euler snow-top-conduction instability, see Rev 6/7/8 below). All fixes
> verified in a real coupled SpeedyWeather+Terrarium run: a 1-year run (`outputs/run_0008`) shows a
> physically correct seasonal snow cycle at representative mid/high-latitude NH sites — winter
> accumulation, spring peak, full summer melt-out tracking skin temperature crossing 0°C, autumn
> re-accumulation — with zero single-step drainage-wipeout events anywhere on the globe over the full
> year (Rev 8). The differentiability CI failure on the new soil-dependent `compute_snow_basal_heat_flux`
> is fixed and confirmed passing (Rev 9): full `test/differentiability/snow_model_diff.jl` run, all
> testsets pass including the rewritten "Snow basal heat flux: differentiability" (4/4).

Date of initial draft: 2026-08-27

Base revision: 836bbb44b2e04c8e2ff0882be7d61ae1d2e6094f

## Originating prompt

> There currently appears to be a bug where not enough snow is accumulating in the northern latitudes
> during winter (<0.001 m SWE), likely due to it being melted too fast. Please review the current
> implementation as well as the text from the GitHub issue [pasted below] and prepare a plan for an
> investigation + potential fixes.

GitHub issue (verbatim, condensed): the surface energy balance (SEB) uses `ground_heat_flux` to force
the top of the snowpack. With fractional snow cover and no tiling, the SEB blends ground/snow inputs
(temperature, conductivity) into a single bulk `ground_heat_flux`, which is then applied unmediated as
the snowpack's top boundary flux. This likely produces unrealistic snow-top fluxes. The ideal fix is
tiling (solve the SEB separately for snow-covered and snow-free fractions); an interim alternative is to
force the *actual ground* with `ground_heat_flux` and partition it in the SEB into `G_g` (snow-free
conductive flux) and `G_s` (snow-covered conductive flux):
`R_net(Ts) + H(Ts) + LE(Ts) - G_g(Ts) - G_s(Ts) = 0`. A remaining concern even under partitioning: the
skin temperature is still solved with a single aggregated albedo in `R_net`, so the snow-covered fraction
still sees an unrealistic radiation budget.

## Revision log

- Rev 1 (2026-08-27): initial draft from code review of `skin_temperature.jl`, `surface_energy_balance.jl`,
  `snow_energy.jl`, `snow_interfaces.jl`, `albedo.jl`, and `land_model.jl`.
- Rev 2 (2026-08-27): approved by author. Fix B selected as the target implementation. Investigation
  step 3 (`snow_cover_fraction` functional-form check) dropped from scope; steps 1, 2, and 4 remain.
  Author clarified Fix B's `S` (formerly `G_s`) is the skin→snow-top conductive flux, distinct from
  the existing snow-base→soil flux `Q_base`; Fix B section rewritten accordingly.
- Rev 3 (2026-08-27): ran the step-1 diagnostic (single-column `LandModel`, permanently sub-zero forcing).
  Confirmed the unweighted `surface_heat_flux = ground_heat_flux` alias destroys the snowpack every
  timestep once it forms (0% of cumulative snowfall retained over 45 days). Steps 1 and 4 marked
  complete; step 2 (albedo) folded in as a retained known limitation, not separately quantifiable in this
  configuration.
- Rev 4 (2026-08-27): implemented Fix B. Design refined during implementation per author direction: kept
  the existing closed-form Ts-space Newton inversion (proven equivalent in iteration count to a
  flux-space residual — verified algebraically, `f_a(Ts) = f_b(Ts)/A` exactly, since the total conductance
  `A` is `Ts`-independent), but made `ImplicitSkinTemperature`'s `ground_heat_flux` an *explicit*
  conductive-flux dispatch (function of the current `Ts`) rather than deriving it from the atmosphere-side
  demand — this is what actually resolves the storage ambiguity, decoupled from the solver mechanism
  itself. Full test suite (skin-temperature stress test incl. snow cases, snow/coupled-model regressions)
  passes. **Re-ran the step-1 diagnostic after Fix B: still 0% retention** — Fix B alone did not resolve
  the reported symptom in this scenario.
- Rev 5 (2026-08-27): traced the persistent 0%-retention result (with Fix B in place) to two further,
  independent bugs, found by isolating the standalone `SnowModel` energy tendency with all SEB forcing set
  to zero:
  1. `compute_snow_energy_tendency`'s `(W > 0)` gate (`snow_energy.jl`) zeroed the *entire* tendency,
     including the cold-snow precipitation-advection term `Q_prcp`, on the first step any snow accumulates
     onto bare ground (`W` read before that step's mass increment). Fresh snow entered with `Ū_snow = 0`,
     which the `FreeWater` closure reads as instantaneously fully melted. Fixed by excluding `Q_prcp` from
     the gate (it is a pure source term, always well-defined, unlike the conductive/sublimation terms).
  2. `energy_to_temperature!`'s (`snow_energy_closures.jl`) liquid-fraction calculation shared the same
     `min_conduction_thickness`-floored volumetric energy as the temperature calculation. The floor is
     needed for temperature (protects the conductive fluxes from an unbounded result as depth → 0) but
     actively harmful for liquid fraction (`liquid_water_fraction` self-clamps to `[0,1]` regardless of
     magnitude, so no floor is needed there, and the floor was diluting a genuinely cold, thin layer's
     energy density toward the melting threshold). Fixed by computing `liq` from the *unfloored*
     depth-integrated comparison `liquid_water_fraction(FreeWater(), Ū_snow, d_snow * ρLθ)` (still safely
     gated by `W_snow > 0` to avoid a `1`-when-empty artifact), while `T` keeps the floored volumetric path.
  Both fixed; author iterated on the exact formulation live in the editor (several intermediate attempts —
  `W_snow * ρLθ` instead of `d_snow * ρLθ`, dropping the `W_snow > 0` gate, an unfloored `T` via
  `d_snow * C_snow` — were each checked and corrected before landing on the final form; see the file's
  current state for the settled implementation). Full regression suite re-run and passes. Diagnostic
  re-run after all three fixes: pathological all-or-nothing oscillation gone (`f_snow` and `T_snow` now
- Rev 6 (2026-08-27): a coupled SpeedyWeather+Terrarium year-long run (`outputs/run_0005`) still showed
  near-zero NH winter SWE and spurious summer spikes. A targeted single-column diagnostic
  (`scratch/debugging/snow_mass_leak_investigation.jl`, cold Siberian-winter forcing) traced this to a
  fourth, independent issue: `compute_snow_surface_heat_flux`'s conductive term
  `2κ_snow(T_snow − T_s)/max(d_snow, min_conduction_thickness)` is *explicit-Euler numerically unstable*
  at the default `min_conduction_thickness = 5mm` and the land model's `Δt = 300s` — the implied
  relaxation timescale `τ = C_snow·d_min²/(2κ_snow) ≈ 30s` is ~9× shorter than `Δt`, so `T_snow` overshoots
  and oscillates every step between ≈0°C and ≈−70°C. When it lands on the warm side, the depth-integrated
  energy swings positive, `snow_liquid_fraction` snaps to 1.0, and the Darcy meltwater term fires at its
  fully-saturated rate, wiping out accumulated SWE in a single step. Confirmed both the mechanism (direct
  per-step trace of `Qtop`, `U`, `liq`) and a candidate fix empirically: setting
  `min_conduction_thickness = 0.02` (pushing `τ` above `Δt`) eliminates the oscillation, `liq` stays 0,
  and SWE accumulates ~100% of forced snowfall over a 30-day cold run (vs. ~0% before). Author approved
  three follow-up changes, scoped below under "Rev 6 fix — approved, in progress".
  vary smoothly/physically instead of pinning at exactly `0`/`0°C` every other step); the diagnostic's own
  weak synthetic forcing (fixed longwave, minimal solar, no realistic winter energy balance) still yields
  low net retention in that toy scenario, which is a forcing-realism limitation of the reproducer, not the
  bug pattern under investigation.
- Rev 7 (2026-08-27): implemented the three Rev 6 fixes. During implementation, two corrections from the
  author refined the design: (1) the new `min_snow_conduction_thickness(i, j, grid, fields, ::SingleLayerSnow)`
  helper belongs in `snow_single_layer.jl` (snow code), not `skin_temperature.jl`, since it is also needed
  by the volumetric snow energy closure; (2) its signature was made consistent with the general snow-accessor
  convention (`i, j, grid, fields, ::SingleLayerSnow`) rather than a bare `(i, j, grid)`. A third correction
  fixed the new `cell_diffusion_timescale` for snow to use `max(snow_depth(...), min_snow_conduction_thickness(...))`
  for `d_snow`, matching the floor pattern used everywhere else, rather than the floor alone. Verified: the
  rewritten `test/snow/snow_energy_tests.jl` basal-flux testset (24/24, checks both thick and vanishing-SWE
  cases against an independently-reimplemented series-resistance formula); a broader regression run across
  `test/snow/*`, `test/coupled_models/land_model_tests.jl`, `test/surface/*` (7902/7902 pass); a targeted
  ad-hoc check of `cell_diffusion_timescale` across SWE = 0, 1e-6, 0.05, 0.5 m confirming it stays finite,
  floors correctly at the minimum thickness for a vanishing pack, grows quadratically with depth, and that
  `LandModel`'s combined `min(soil_τ, snow_τ)` dispatch behaves correctly; and a clean local doc build
  (fixed one incidental broken `@ref` — `[`soil_diffusion_timescales.jl`](@ref)` doesn't resolve to a
  documented symbol — introduced by the new docstring, changed to reference `[`cell_diffusion_timescale`](@ref)`
  instead). Not yet done: re-running the coupled SpeedyWeather+Terrarium scenario (`outputs/run_0005`-style)
  to confirm the original reported NH-winter-SWE symptom is actually resolved end-to-end.
- Rev 8 (2026-08-27): re-ran the coupled SpeedyWeather+Terrarium scenario with the Rev 6/7 fixes in place,
  first a 3-month run (`outputs/run_0006`, Jan–Mar) then a full 1-year run (`outputs/run_0008`), analyzed
  via `xarray`/Python (`geo` conda env, per author preference). `run_0006`: NH (50–90°N) snow depth grows
  smoothly and monotonically over the winter (mean 0→0.014 m, frac-of-cell->1mm-SWE 0%→62%), a Siberian
  point (65°N, 90°E) increases every single output step with the largest single-step change over the whole
  run being 1.5e-5 m (vs. the old pathology's full-pack wipeout every step); zero single-step drops >5mm
  anywhere in that point series. `run_0008` (full year): a naive zonal-mean check of the 50–90°N band
  showed net monotonic growth across the *entire* year including NH summer — initially flagged as a
  possible regression (no seasonal decline) — but resolved by checking individual mid/high-latitude
  seasonal-snow sites directly (S. Siberia/Mongolia 50°N, US Great Plains 45°N, Scandinavia 60°N): each
  shows a textbook seasonal cycle — winter accumulation, spring peak, full summer melt-out tightly tracking
  skin temperature crossing 0°C (e.g. Great Plains: SWE 0.023 m in March → ~5e-7 m by July as skin
  temperature rises from −6.7°C to +13.5°C), then autumn re-accumulation. The zonal-mean band average is
  dominated by the permanent high Arctic (75–90°N, real-world firn/glacier accumulation zone that would not
  be expected to melt out within a single year even in reality), so its monotonic growth is not itself
  evidence of a bug — same explanation for the analogous SH 50–90°S (Antarctica) pattern. Globally over the
  full year: zero (time, lat, lon) grid cells with a >20 mm single-step drop — the old drainage-wipeout
  signature is completely absent. Original reported symptom (near-zero NH winter SWE, sporadic summer
  spikes) confirmed resolved.
- Rev 9 (2026-08-27): GitHub CI's Enzyme differentiability job failed —
  `test/differentiability/snow_model_diff.jl`'s "Snow basal heat flux: differentiability" testset called
  the removed pure-scalar `compute_snow_basal_heat_flux(κ, T_soil, T_snow, d_snow, d_min)` signature (an
  artifact of Rev 7 moving the function to a field-based `(i, j, grid, fields, snow, soil, constants)`
  interface without updating this test). Rewritten to differentiate the real function directly via
  `Duplicated(state, dstate)` (mirroring the `closure!`-differentiation pattern already used in
  `soil_hydrology_diff.jl`), checking `∂Q_base/∂T_soil` and `∂Q_base/∂T_snow` are finite and satisfy the
  closed-form relation (equal, opposite sign, since `Q_base` is linear in both temperatures). First attempt
  differentiated a full `LandModel` `timestep!` (mirroring the file's own "Snow model: timestep!" pattern);
  author redirected to keep the test focused specifically on `compute_snow_basal_heat_flux` rather than the
  whole coupled step. Confirmed passing: full standalone run of `test/differentiability/snow_model_diff.jl`,
  all testsets pass, including the rewritten "Snow basal heat flux: differentiability" (4/4, 1m22s) and the
  pre-existing "Snow model: timestep!" (4/4, 24m17s — long Enzyme compilation of the full snow-model
  reverse-mode call graph, expected and unaffected by this change).

## Problem description

Northern-latitude winter SWE stays below 0.001 m — snow that does accumulate appears to melt (or
sublimate/never accumulate) almost immediately, rather than persisting through the cold season.

## Background: how the bulk-blend forcing actually reaches the snowpack today

Confirmed by reading the current implementation (not yet run/instrumented):

1. **Skin temperature solve blends the conduction target, not just diagnostics.**
   [`ground_thermal_interface`](../../src/processes/surface/skin_temperature.jl#L121-L132) (snow-aware
   overload) computes `Tg_eff = (1-f)·T_ground + f·T_snow`, `κ_eff = (1-f)·κₛ + f·κ_snow`,
   `Δz_eff = (1-f)·Δz_ground + f·d_snow`, and the implicit solve uses these to get
   `Ts = Tg_eff - G·Δz_eff/(2·κ_eff)` ([`skin_temperature.jl:94-97`](../../src/processes/surface/skin_temperature.jl#L94-L97)).
   `G` itself closes the SEB as `G = R_net(Ts) + H_s(Ts) + H_l(Ts)`
   ([`skin_temperature.jl:141-144`](../../src/processes/surface/skin_temperature.jl#L141-L144)). So there is
   a *single* skin temperature and a *single* bulk `G` for the whole grid cell, exactly as the issue
   describes — this is not tiled.
2. **That same bulk `G` is fed directly into the snowpack's top-of-pack energy tendency, unmediated by
   `f_snow`.** In `LandModel`, `snow_flux_aliases = (; surface_heat_flux = ground_heat_flux, ...)`
   ([`land_model.jl:81`](../../src/models/coupled/land_model.jl#L81)) — no `f_snow` weighting at all — and
   [`compute_snow_energy_tendency`](../../src/processes/snow/energy/snow_energy.jl#L79-L100) reads
   `Q_top = fields.surface_heat_flux[i,j]` directly as `dŪ_snow/dt = Q_base - Q_top + ...`. So a grid cell
   with, say, `f_snow = 0.1` still has the *entire* bulk `G` (computed from a 90%-bare-ground-weighted
   conduction target and a blended albedo) driving 100% of the snowpack's top-of-pack tendency. This is
   the mechanism most directly implicated by the issue and the most likely single cause of excess melt:
   thin/patchy snow gets hit with a flux sized for a much larger effective bare-ground fraction.
3. **The soil-top flux is already properly `f_snow`-blended** —
   [`compute_snow_soil_heat_flux`](../../src/processes/snow/snow_interfaces.jl#L37-L52) computes
   `f·Q_base + (1-f)·G` for `soil_heat_flux` — but this only fixes the *soil's* boundary condition, not the
   snowpack's own top-of-pack forcing from step 2.
4. **Albedo is also bulk-blended before the skin-temperature solve.**
   [`compute_albedo` (`DiagnosticAlbedo`)](../../src/processes/surface/albedo.jl#L94-L118) computes a single
   `α_eff = (1-f_snow)(1-f_veg)·α₀ + (1-f_snow)·f_veg·α_veg + f_snow·α_snow` used for the one skin-temperature
   solve. For small `f_snow` this washes out snow's high albedo, inflating absorbed shortwave (and hence
   `G`, `Ts`) relative to what the snow-covered patch would actually see — compounding point 2.
5. A prior design doc ([`2026-07-10_PLAN_01B_surface_ground_heat_flux_split.md`](../2026-07/2026-07-10_PLAN_01B_surface_ground_heat_flux_split.md))
   already flagged the semantic ambiguity between the SEB closure flux and the soil BC and split them by
   name (`ground_heat_flux` vs `soil_heat_flux`), fixing the soil side. It explicitly deferred the
   *snow-top* forcing question — this is that deferred problem.

This is consistent with, and sharpens, the issue text: the reporter suspected the bulk blend was the
culprit but had not yet traced it all the way to `surface_heat_flux = ground_heat_flux` with **zero**
`f_snow` weighting at the snow-tendency boundary. That zero-weighting (not just "some blending exists")
is likely the dominant term, especially for the thin/patchy snow (`f_snow` small) typical of snow onset
and ablation in shoulder seasons — exactly when spurious melt would show up as chronically-suppressed
peak SWE.

## Investigation plan (before committing to a fix)

1. **Instrument and confirm the mechanism.** Add a diagnostic run (single grid cell or small regional
   subset at a northern-latitude site with known snow cover, e.g. from ERA5/CRU forcing already used in
   existing examples/tests) that logs `f_snow`, `ground_heat_flux` (bulk `G`), `compute_snow_soil_heat_flux`
   (`Q_base` blend), `snow_temperature`, `skin_temperature`, `albedo`, and the snow energy tendency terms
   (`Q_top`, `Q_base`, `Q_prcp`, `Q_subl`) through a full autumn-onset → winter → melt cycle. Confirm that
   (a) `Q_top` tracks the *bulk* `G` rather than a snow-covered-fraction-appropriate flux, and (b) melt
   events correlate with periods of small `f_snow` (patchy cover) rather than genuine warm spells.
2. **Quantify the albedo effect in isolation.** With `f_snow` diagnostics from (1), estimate how much
   absorbed shortwave increases from blended vs. snow-only albedo at representative small-`f_snow` values,
   to judge whether fixing point 2 alone (without touching albedo) would be enough, or whether point 4
   also needs addressing per the issue's closing concern.
3. **(Skipped per author direction, rev 2.)** ~~Check `snow_cover_fraction`'s functional form~~ — out of
   scope for this investigation.
4. **Rule out independent causes** before attributing the whole deficit to SEB coupling: verify
   snowfall-phase partitioning (rain/snow split by `T_air`), sublimation magnitude
   ([`compute_snow_sublimation_flux`](../../src/processes/snow/mass/snow_mass.jl#L13-L29)), and the
   `min_conduction_thickness` floor behavior for very thin packs (already flagged as thermally transient by
   design — worth confirming it isn't itself injecting spurious heat during pack initiation).

### Step 1 result: mechanism confirmed empirically (rev 2)

Ran a minimal single-column `LandModel` diagnostic
([`scratch/debugging/snow_seb_partition_investigation.jl`](../../scratch/debugging/snow_seb_partition_investigation.jl),
log in `/tmp/snow_seb_investigation2.log`): `ColumnGrid` with `SoilEnergyWaterCarbon` + `SingleLayerSnow`
+ `DiagnosticAlbedo` (the `LandModel` default is actually `ConstantAlbedo`, not snow-aware — switched to
`DiagnosticAlbedo` so the run also exercises point 4), forced with constant light snowfall
(`1e-7 m/s SWE`), zero rain, air temperature fixed at −8 °C (no diurnal/seasonal cycle — genuine
air-temperature-driven melt is structurally impossible in this setup), 230 W/m² longwave-down, and zero
shortwave (polar-night limit). Soil/snow started frozen at −8 °C, `W_swe = 0`.

Per-step trace (`Δt = 300 s`, single column, `f_snow = W/(W + 0.01)`):

```
step 1: W=2.9999999999999997e-5 f_snow=0.00299 d_snow=0.00012  G(bulk)=42.20  snow_energy=0.0   T_snow=0.0  θ_liq=1.0
step 2: W=0.0                    f_snow=0.0     d_snow=0.0     G(bulk)=41.06  snow_energy=-0.0  T_snow=0.0  θ_liq=0.0
step 3: W=2.9999999999999997e-5 f_snow=0.00299  d_snow=0.00012  G(bulk)=39.93  snow_energy=0.0   T_snow=0.0  θ_liq=1.0
step 4: W=0.0                    f_snow=0.0     d_snow=0.0     G(bulk)=39.37  snow_energy=-0.0  T_snow=0.0  θ_liq=0.0
```

Within the **same timestep** that a sliver of snow first accumulates (`f_snow ≈ 0.3%`), the pack is
diagnosed as fully melted (`T_snow = 0 °C`, `θ_liq = 1.0`) and drains completely via the Darcy meltwater
flux — then the cycle repeats every step for the full 45-day run. Over the whole run: **0.00% of the
cumulative snowfall input (0.389 m SWE) is retained** (see the script's summary output) — this
reproduces the `< 0.001 m SWE` symptom directly, with air temperature held permanently sub-zero the
entire time, ruling out a genuine warm-air melt explanation. This directly implicates the unweighted
`surface_heat_flux = ground_heat_flux` alias (`land_model.jl:81`): the bulk `G` (~40 W/m², sized for the
whole cell) is applied unmediated to a snowpack thermal mass bounded by `min_conduction_thickness`
(5 mm), which cannot absorb a cell-scale flux without being driven straight to (and past) its melting
point in one step.

Step 1 investigation is considered complete; step 2 (isolating the albedo contribution specifically) is
superseded in practice by this result — the melt is total and immediate even before the albedo blend has
a chance to matter over multiple steps, so albedo's marginal contribution is not separable from noise in
this configuration. It remains a real, distinct problem (per Background point 4) and is retained as a
known limitation of Fix B. Step 4 (independent causes) is satisfied by construction: air temperature is
fixed and always sub-zero, so rain-phase partitioning is irrelevant, and sublimation cannot be the melt
mechanism (`θ_liq` rises with the pack still frozen).

## Candidate fixes (in increasing order of scope)

**Fix A — weight the snowpack's top-of-pack flux by `f_snow` (minimal, addresses point 2 directly).**
Instead of aliasing `surface_heat_flux = ground_heat_flux` unweighted, apply the same `f_snow` blend
already used for the soil side: derive a snow-only top flux and multiply, or (more physically) recognize
that with `f_snow` already baked into the *conduction target* (`Tg_eff`, `κ_eff`, `Δz_eff` blend) `G` is
not simply separable into "the snow's share" — this needs care, not just multiplying `G` by `f_snow`
(that alone would not correct the albedo/conduction-target distortion in point 4, only rescale the
already-distorted flux). This is the smallest change but only partially matches the issue's own proposed
interim fix.

**Fix B — issue's proposed interim fix: partition the SEB residual into a ground-conduction term and a
snow-top-conduction term.** Redefine the skin-temperature residual as
`R_net(Ts) + H(Ts) + LE(Ts) - G(Ts) - S(Ts) = 0`, where both `G` and `S` are conductive fluxes *from the
single shared skin temperature `Ts` downward*, not flux-into-soil terms:
- `G` = `(1-f_snow)·2κₛ(Ts - T_ground)/Δz_ground` — conduction from the skin into the *snow-free ground*
  (unchanged meaning from today's no-snow case; this is what already drives the soil BC directly when
  there is no snow).
- `S` = `f_snow·2κ_snow(Ts - T_snow)/d_snow` — conduction from the skin into the *top of the snowpack*.
  This is a new quantity: today there is no separate flux terminating at the snow top at all, only the
  single blended `G` from the blended conduction target in `ground_thermal_interface`.

Critically, `S` (not `G`, and not blended with it) becomes the field that drives the snowpack's own
top-of-pack energy tendency (`surface_heat_flux`, read as `Q_top` in
[`compute_snow_energy_tendency`](../../src/processes/snow/energy/snow_energy.jl#L79-L100)), replacing
today's unweighted `surface_heat_flux = ground_heat_flux` alias
([`land_model.jl:81`](../../src/models/coupled/land_model.jl#L81)). `G` continues to drive the soil
directly over the snow-free fraction, and is *unrelated* to the existing snow→soil basal flux `Q_base`
(the already-correct [`compute_snow_soil_heat_flux`](../../src/processes/snow/snow_interfaces.jl#L37-L52)
blend `f_snow·Q_base + (1-f_snow)·G`, which conducts from the snow *base* down into the soil under the
snow, and is untouched by this fix). So under snow, three distinct conductive fluxes coexist: `G`
(skin → bare ground, snow-free fraction), `S` (skin → snow top, snow-covered fraction), and `Q_base`
(snow base → soil top, snow-covered fraction) — Fix B only adds `S` and rewires the snow-tendency
consumer to use it; `G` and `Q_base` keep their present roles and definitions.

This directly fixes point 2 and is a bounded, well-specified change to `skin_temperature.jl` (the residual
gains an `S` term computed from an unblended pure-snow conduction target, alongside the existing
unblended pure-ground `G`), `land_model.jl` (wire `surface_heat_flux` to the new `S` field instead of raw
`ground_heat_flux`), and no change to `snow_interfaces.jl`'s `Q_base` blend. Leaves the
single-albedo/single-`Ts` limitation (point 4) unresolved, as the issue itself notes.

**Fix C — full tiling.** Solve the SEB independently for the snow-covered and snow-free sub-fractions
(two skin temperatures, two albedos, two turbulent-flux solves), then area-weight only the *aggregate*
diagnostics (mean skin temperature for output, total sensible/latent flux for atmosphere coupling). This
is the "ideal" fix per the issue and resolves both point 2 and point 4, but is a substantially larger
change: doubles the per-cell nonlinear solve cost, touches `SurfaceEnergyBalance`, `skin_temperature.jl`,
`turbulent_fluxes.jl`, `radiative_fluxes.jl`, `albedo.jl`, and the snow/soil coupling in `land_model.jl`
and `snow_interfaces.jl`. Also raises open questions about how vegetation fraction composes with snow
tiling (currently vegetation and snow fractions are both blended into the same single albedo formula) and
about GPU/Reactant kernel cost of two solves per cell.

## Recommendation

**Decision (rev 2): Fix B**, approved by the author. Run investigation steps 1, 2, and 4 first (mechanism
confirmation is cheap and de-risks the implementation); step 3 is out of scope. The residual single-albedo
limitation (point 4 of the Background section) is accepted as a known limitation for this PR, with
**Fix B+** (post-hoc dual-albedo `R_net`) or full tiling (**Fix C**) as candidate future work if step 2 of
the investigation shows it's still a significant residual error after B.

## Rev 6 fix — implemented and verified (Rev 7)

Root cause is a fixed, too-small `min_conduction_thickness` making the explicit snow-top conduction
update numerically unstable at the model's `Δt`. Approved fix (three parts):

1. **Remove `min_conduction_thickness` as a `SingleLayerSnow` struct field.** Replace every read of
   `snow.min_conduction_thickness` (`skin_temperature.jl:105`, `snow_energy_closures.jl:117`, and the
   basal-flux call site moved per item 3) with a dynamically-computed value: half the thickness of the
   uppermost soil grid cell, via `Δzᵃᵃᶜ(i, j, field_grid.Nz, field_grid) / 2` (same
   `get_field_grid(grid)` pattern already used in `ground_thermal_interface`). This scales the floor with
   the actual grid resolution instead of a fixed constant, and ties it to the same length scale already
   used for the ground conduction target. `NF` remains pinned in the struct via `<: AbstractSnow{NF}`
   (verified — Julia permits the "phantom" type parameter; no other struct/constructor change needed
   beyond dropping the field and kwarg).
2. **Add `cell_diffusion_timescale` for snow**, mirroring
   `src/processes/soil/soil_diffusion_timescales.jl`'s pattern (`τ = Δz²C/κ` via
   `KernelFunctionOperation` + `minimum(τ)`, `Inf` fallback where not applicable). Wire it into
   `LandModel`'s `cell_diffusion_timescale(state, model::LandModel)` (currently soil-only,
   `timesteppers/cell_diffusion_timescale.jl`) as `min(soil_τ, snow_τ)`, so `TimeStepWizard` respects the
   snow conduction stability limit directly instead of relying on a hand-tuned floor.
3. **Fix `compute_snow_basal_heat_flux` to include the soil-side resistance.** Currently
   `Q_base = 2κ_snow(T_soil − T_snow)/max(d_snow, d_min)` (`snow_energy.jl:142-143`) omits the soil
   half-cell's own resistance (`Δz_soil/(2κ_soil)`), implicitly assuming it's negligible next to the
   snow's — invalid once `d_snow`/the floor is comparable to `Δz_soil/2`. Move the function from
   `snow_energy.jl` to `snow_interfaces.jl` (its only caller, `compute_snow_soil_heat_flux`, already lives
   there and already takes `soil`; `snow_energy.jl` stays soil-independent) and change it to a proper
   series-resistance formula, `Q_base = (T_soil − T_snow) / (Δz_soil/(2κ_soil) + d_snow/(2κ_snow))`.

## Summary of changes

**`src/processes/surface/skin_temperature.jl`** (Fix B):
- `ground_thermal_interface(i, j, grid, fields, skinT::ImplicitSkinTemperature)` no longer takes `snow`:
  it is now *always* the unblended, snow-free ground conduction target `(Tg, κₛ, Δz)`.
- New `snow_conduction_interface(i, j, grid, fields, snow, constants)`: the unblended snow-top target
  `(Tsnow, κsnow, dsnow)`, floored at `min_conduction_thickness`; `snow === nothing` returns a finite,
  zero-conductivity placeholder (safe against `0 * Inf`).
- `compute_skin_temperature(::ImplicitSkinTemperature, G, Tg, κg, Δzg, f_snow, Tsnow, κsnow, dsnow)`: the
  closed-form Newton-residual inversion, generalized from one to two parallel (area-weighted) conductances;
  exact algebraic solve, not an approximation (conduction is linear in `Ts`; only `R_net`/`H`/`LE` are
  nonlinear).
- New `compute_ground_heat_flux(i, j, grid, fields, skinT::ImplicitSkinTemperature, seb)` override: stores
  the *explicit* conductive flux `2κg(Tg − Ts)/Δzg` at the current `Ts`, replacing the generic
  `R_net + H_s + H_l` dispatch inherited from `AbstractSkinTemperature` (still used by
  `PrescribedSkinTemperature`, which has no conduction target to be explicit about).
- `compute_skin_temperature_residual!` recomputes the atmosphere-side demand separately (no longer reads
  it back from `ground_heat_flux`, whose meaning changed above) and inverts it through both conduction
  targets.

**`src/processes/snow/snow_interfaces.jl`**:
- `compute_snow_soil_heat_flux` unchanged in form (`f·Q_base + (1−f)·G`) but now correct in practice
  since `G` is the genuinely-explicit bare-ground flux.
- New `compute_snow_surface_heat_flux`: the snow-top flux `S = 2κsnow(Tsnow − Ts)/dsnow`, computed
  post-solve at the converged `Ts` and stored into the new `surface_heat_flux`/`snow_surface_heat_flux`
  field, replacing the old unweighted alias to `ground_heat_flux`.

**`src/models/coupled/land_model.jl`**: `surface_heat_flux` is now its own allocated field
(`snow_surface_heat_flux`) rather than an alias of `ground_heat_flux`, mirroring the existing
`soil_heat_flux` pattern (aliased to `ground_heat_flux` only in the no-snow case).

**`src/processes/snow/energy/snow_energy.jl`** (gate fix): `compute_snow_energy_tendency`'s
`(Q_base − Q_top + Q_prcp + Q_subl) * (W > 0)` gate changed to `(Q_base − Q_top + Q_subl) * (W > 0) + Q_prcp`
— `Q_prcp` (cold-snow advection) is no longer suppressed on the accumulation-onset step.

**`src/processes/snow/energy/snow_energy_closures.jl`** (closure fix): `energy_to_temperature!` now
computes `snow_liquid_fraction` from the unfloored, depth-integrated
`liquid_water_fraction(FreeWater(), Ū_snow, d_snow * ρLθ)` (still gated by `W_snow > 0`), while
`snow_temperature` continues to use the floored volumetric `U_snow`/`ρLθ`/`C_snow` path — decoupling two
consumers of the same underlying quantity that had different numerical needs.

**`test/surface/skin_temperature.jl`**: convergence-check helper rewritten to reconstruct the residual
explicitly (atmosphere-side demand vs. area-weighted `G`/`S` conduction) rather than calling the removed
blended `ground_thermal_interface` overload.

**`test/snow/snow_model_tests.jl`**: new regression test, "snowfall onto bare ground (W=0) advects cold,
not zero", pinning the gate-fix behavior.

## Testing and verification

Performed:
- Full `test/surface/skin_temperature.jl` (includes the 7776-case `LandModel` stress sweep over
  snow/no-snow × radiation/temperature/humidity/wind combinations): **pass**.
- `test/snow/snow_model_tests.jl`, `snow_energy_tests.jl`, `snow_properties_tests.jl`,
  `test/coupled_models/land_model_tests.jl`, `test/surface/diagnostic_albedo_tests.jl`: **pass**
  (119 total, including the new regression test), re-run after each of the three fixes landed.
- Empirical mechanism confirmation and fix verification via
  `scratch/debugging/snow_seb_partition_investigation.jl` (see Rev 3–5 above): pathological
  all-or-nothing SWE oscillation eliminated; `f_snow`/`T_snow` now vary smoothly and physically.

Not yet performed (recommended before merge):
- Differentiability: re-run `test/differentiability` and `test/reactant/autodiff.jl` relevant snow/SEB
  cases, since `compute_ground_heat_flux`, `ground_thermal_interface`, and `snow_conduction_interface`
  sit inside the implicit skin-temperature solve's differentiated call graph, and
  `energy_to_temperature!`'s branch structure changed.
- Full doc build (`julia --project=docs docs/make.jl --local --draft`) — the `ImplicitSkinTemperature`
  docstring's residual equation was updated; needs a build check for stale `@docs`/doctest references.
- A realistic (ERA5-forced) end-to-end regression at a northern-latitude site, since the diagnostic
  script's synthetic forcing was not tuned to validate actual peak-SWE/season-length magnitudes — only
  to reproduce and check the specific oscillation pathology.

Rev 6/7 fix (grid-derived `min_conduction_thickness`, snow `cell_diffusion_timescale`, soil-inclusive
basal-flux series resistance) — performed:
- `test/snow/snow_energy_tests.jl` rewritten basal-flux testset: **24/24 pass**, covering both thick
  (`W=0.5`) and vanishing/zero SWE (`W ∈ {0, 1e-9, 1e-6}`) cases against an independently-reimplemented
  series-resistance formula.
- Broader regression: `test/snow/*`, `test/coupled_models/land_model_tests.jl`, `test/surface/*` —
  **7902/7902 pass**.
- Ad-hoc `cell_diffusion_timescale` sanity check (single-column `LandModel`, SWE = 0, 1e-6, 0.05, 0.5 m):
  stays finite and positive throughout; floors correctly (identical τ at W=0 and W=1e-6, both far below
  the grid-derived minimum thickness); grows quadratically with snow depth once past the floor
  (τ ≈ 1.98e3 s → 1.27e5 s → 1.27e7 s across W = 0/1e-6 → 0.05 → 0.5); `LandModel`'s
  `min(soil_τ, snow_τ)` combination dispatch behaves correctly (soil, at ≈1090 s, dominates whenever
  snow is thin/absent).
- Clean local doc build (`julia --project=docs docs/make.jl --local --draft`, exit 0): fixed one
  incidental broken `@ref` introduced by the new `cell_diffusion_timescale` docstring
  (`[`soil_diffusion_timescales.jl`](@ref)` doesn't resolve — filenames aren't `@ref` targets — changed to
  `[`cell_diffusion_timescale`](@ref)`); no other new warnings versus the pre-existing baseline.

Rev 8 (coupled-run confirmation) — performed:
- 3-month coupled run (`outputs/run_0006`, Jan–Mar): smooth monotonic NH winter accumulation, no
  single-step oscillation/drainage; see Rev 8 revision-log entry for full figures.
- 1-year coupled run (`outputs/run_0008`): individual seasonal-snow sites (S. Siberia/Mongolia 50°N,
  US Great Plains 45°N, Scandinavia 60°N) each show a correct full accumulation → peak → melt-out → 
  re-accumulation annual cycle tracking skin temperature; zero single-step drops >20mm anywhere on the
  globe over the full year. Original reported symptom (near-zero NH winter SWE, sporadic summer spikes)
  confirmed resolved end-to-end, not just in the single-column diagnostic.

Rev 9 (differentiability) — performed:
- GitHub CI's Enzyme job failed on the stale `compute_snow_basal_heat_flux` scalar-signature test (see
  Rev 9 revision-log entry). Rewritten to differentiate the real field-based function via
  `Duplicated(state, dstate)`; standalone run of `test/differentiability/snow_model_diff.jl` **passes in
  full** (all testsets, including the rewritten one, 4/4).

Still not yet performed (recommended before merge):
- Differentiability re-check for `cell_diffusion_timescale`/`min_snow_conduction_thickness` specifically
  (distinct from the `compute_snow_basal_heat_flux` test above) and `test/reactant/autodiff.jl` coverage
  for any of the Rev 6/7 changes.

## Documentation changes

- Update the SEB residual equation and prose in the `ImplicitSkinTemperature` docstring
  ([`skin_temperature.jl:42-50`](../../src/processes/surface/skin_temperature.jl#L42-L50)) to reflect the
  `G_g`/`G_s` partition.
- Update `docs/src` process pages for surface energy balance / snow coupling (per `AGENTS.md` doc-page
  conventions: Overview/Implementations/Methods/Kernel-functions sections) once the concrete
  implementation is decided.

## Known limitations

- Fix B does not address the aggregated-albedo distortion of `R_net` (issue's closing concern); this
  remains until Fix B+ or full tiling (Fix C) is implemented.
- Fix B still solves a single skin temperature, so turbulent fluxes (`H_s`, `H_l`) are not tiled either —
  only the conductive term is partitioned.
- `ImplicitSkinTemperature`'s ground conduction target still uses the assumed constant `κₛ` parameter
  rather than the soil model's own (moisture/ice-state-dependent) thermal conductivity — a
  longstanding simplification, not something this investigation's fixes touch. Assessed as a scoped
  follow-up (see Future work).
- The diagnostic script's synthetic forcing has not been validated against realistic peak-SWE magnitudes
  (see Testing and verification).

## Future work

- Full tiling (Fix C), if the investigation or Fix B's post-hoc evaluation shows residual error from the
  shared skin temperature/albedo is still significant.
- Revisit `snow_cover_fraction`'s functional form if step 3 of the investigation shows pathological
  behavior at low SWE.
- **Use the soil model's actual (state-dependent) thermal conductivity in `ground_thermal_interface`**
  instead of the assumed constant `κₛ`. Scoped assessment (2026-08-27): `compute_thermal_conductivity`
  already exists (`soil_thermal_properties.jl`/`soil_energy.jl`), computed from `soil_composition`
  (porosity, saturation, liquid fraction, solid matrix — needs the soil's stratigraphy/hydrology/
  biogeochemistry components) at a given `(i,j,k)`; the top-layer view pattern already used for
  `ground_temperature` is a direct template. Would require threading an `Optional{AbstractSoil}` through
  `solve_surface_energy_balance!` → the fused kernel → `solve_skin_temperature!` →
  `compute_skin_temperature_residual!` → `ground_thermal_interface` (mirroring the existing
  `Optional{AbstractSnow}`/`Optional{AbstractSurfaceHydrology}` pattern), merging soil's hydrology/strat/
  bgc fields into the SEB kernel's field set, and evaluating `soil_composition`/`compute_thermal_conductivity`
  at a fixed `k = Nz` inside the otherwise-2D `XY` SEB kernel. `κₛ` would stay as the fallback for
  `soil === nothing` (standalone `SurfaceEnergyModel`, used by existing smoke tests). Rough size: similar
  order of magnitude to Fix B (a handful of files, ~100–150 lines, a few hours including tests) — real but
  bounded; warrants its own short plan doc before implementation, since it's a new SEB↔soil coupling
  dependency, not a bug fix.
