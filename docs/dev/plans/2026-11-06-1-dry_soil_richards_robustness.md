# Robust Richards flow in dry soil: consistent unsaturated conductivity and water conservation

> Status: **planned**. Drafted as a follow-up to the implicit time stepping PR (#196); awaiting review.

Date of initial draft: 2026-11-06

Author: Claude Opus 5.5, Claude Code (VS Code extension)

Reviewer:

Base revision: 5b858a6909727b1b89cab09440c93f674d81f28d

## Originating prompt

> Can you create a new test that exercises the failure mode you hypothetically identified with the Somali coast? If you can use some of the actual numbers from the SpeedyWeather output at the failing point, that would be great, but not strictly necessary.

> Do you have a proposed fix?

> Yes and draft this as a separate plan, follow-up to the implicit timestepping PR.

## Revision log

> **Revision 1, 2026-11-06 (Claude Opus 5.5).** Initial draft.

## Problem description

A `LandModel` column with `RichardsEq` hydrology (the default since revision 33 of the implicit time stepping plan) breaks down within two time steps when it starts from dry soil under a realistic evaporation demand.
The regression test `LandModel: dry soil under strong evaporation demand stays physical` in `test/coupled_models/land_model_tests.jl` reproduces it: a column with the SoilGrids texture of a Somali-coast grid point (38.6% sand, 31.5% silt, 29.9% clay), dry-season forcing taken from a coupled SpeedyWeather run, an initial saturation of 0.05, no rain, and `Δt = 450 s` with the default `IMEX` time stepper.

A per-step trace of the same column (default hydraulics, `SoilPorositySURFEX`) shows, for both `ConstantEvaporationResistanceFactor` and `SoilMoistureResistanceFactor`:

| step | saturation, top two cells | pressure head, top cell | Σ sat·Δz (m) | top soil temperature |
| --- | --- | --- | --- | --- |
| 0 | 0.05, 0.05 | −3.2×10⁴ m | 21.7 | 27 °C |
| 1 | 0.051, 0.041 | −2.9×10⁴ m | 21.7 | 26 °C |
| 2 | 1.0, 1.0 | −0.035 m | 10.6 | 97 °C (β = 1) / 1368 °C (Lee–Pielke β) |

In the second step the top cells jump from almost dry to saturated, half of the column water disappears, and the soil temperature leaps because the internal energy no longer matches the water content.
From there the skin temperature and the surface fluxes run away (skin temperatures of −1000 °C and fluxes of order 10⁵ W/m² within a few hours), without ever producing a `NaN`.

This is the failure signature that revision 24 of the implicit time stepping plan recorded for prescribed infiltration into freezing soil, and which was deferred to future work on the grounds that it required a physically inconsistent forcing.
The dry column shows that it also occurs under consistent forcing and at short time steps.

## Background

### Inconsistent default hydraulic functions

The default hydraulic properties (`SoilHydraulicsSURFEX` and `ConstantSoilHydraulics`) combine the Brooks–Corey soil-water retention curve (`BrooksCorey()` of FreezeCurves.jl, `ψₛ = 0.01 m`, `λ = 0.2`) with the linear unsaturated conductivity `UnsatKLinear`, `K = K_sat θ_w / θ_sat`.
The two functions do not belong together in dry soil:

- the matric head grows as a power law of the inverse saturation, `ψ = −ψₛ S^(−1/λ) = −ψₛ S^(−5)`, so `ψ ≈ −3.2×10⁴ m` at `S = 0.05`;
- the conductivity falls only linearly, to `K ≈ 5×10⁻⁷ m/s` at `S = 0.05` (with `K_sat = 10⁻⁵ m/s`).

After the first step of evaporation, the top two cells sit at `S = 0.051` and `S = 0.041`, with heads of −2.9×10⁴ m and −8.6×10⁴ m.
The Darcy flux between them is `K Δψ / Δz ≈ 5×10⁻⁷ × 5.7×10⁴ / 0.03 ≈ 1 m/s`, enough to move hundreds of meters of water through cells that hold about a centimeter within one 450 s step.
No time stepping scheme can resolve this: the Picard iterates of the implicit Richards step overshoot, the closure clamps the saturation to `[s_res, 1]`, and the excess is lost.

The Brooks–Corey retention curve has a matching conductivity function [Brooks and Corey (1964); Campbell (1974); Clapp and Hornberger (1978)],

```
K(S) = K_sat S^(3 + 2/λ)
```

which, for `λ = 0.2`, is `K_sat S¹³`: about 10⁻²² m/s at `S = 0.05`, so that dry-soil fluxes become negligible, as they are physically.
SURFEX/ISBA uses this Clapp–Hornberger conductivity together with the Brooks–Corey/Campbell retention curve [Noilhan and Mahfouf (1996); Boone et al. (2000)]; Terrarium's `SoilHydraulicsSURFEX` uses the SURFEX retention parameters with `UnsatKLinear`, which the TODO in `src/processes/soil/hydrology/soil_hydraulic_properties.jl` already hints at.
The van Genuchten pair (`VanGenuchten` with `UnsatKVanGenuchten`, after van Genuchten (1980) and Westermann et al. (2023)) is consistent and already implemented.

### Water accounting in the saturation closure

`redistribute_saturation_profile!` (`src/processes/soil/hydrology/soil_hydrology.jl`) conserves water within the column, with two exceptions:

1. excess water in the top cell is removed as `surface_excess_water`, which a `LandModel` routes to surface runoff and a standalone `SoilModel` discards;
2. a saturation deficit in the bottom cell is filled up to the residual saturation, which creates water ("should not happen under realistic conditions").

Exception 1 is legitimate water accounting, which the regression test includes in its balance.
Exception 2 is a silent mass balance violation that strong suction fluxes, such as those above, can trigger.

### Relation to the coupled SpeedyWeather failure

This failure mode was found while investigating a `KernelException` in `examples/simulations/speedy_wet_land.jl`.
The coupled failure itself turned out to originate in the atmosphere (a one-step blow-up of the near-surface humidity and temperature next to the Aral and Caspian Seas, after which the land state becomes `NaN` and `@assert_kernel isfinite(ψm)` throws), and is outside the scope of this plan.
The dry-soil failure is a separate, real defect of the land model that the coupled runs had not reached.

## Summary of changes

### Phase 1: Consistent Brooks–Corey conductivity (the fix)

- Add `UnsatKBrooksCorey <: AbstractUnsatK`, with `hydraulic_conductivity` dispatching on `AbstractSoilHydraulics{NF, <:BrooksCorey, UnsatKBrooksCorey{NF}}`:
  `K = I_ice K_sat S^(3 + 2/λ)`, with `S = clamp(θ_w / θ_sat, 0, 1)` as in `UnsatKVanGenuchten` and the same ice impedance factor `I_ice = 10^(−Ω(1 − f))` (parameter `impedance`, default 7).
  `λ` is read from the retention curve (`hydraulics.swrc.λ`), so the two functions cannot be configured inconsistently.
- Make `UnsatKBrooksCorey` the default `unsat_hydraulic_cond` of `SoilHydraulicsSURFEX` and `ConstantSoilHydraulics` whenever the retention curve is `BrooksCorey` (the default), through a `default_unsat_hydraulic_cond(NF, swrc)` helper: `BrooksCorey → UnsatKBrooksCorey`, `VanGenuchten → UnsatKVanGenuchten`.
  `UnsatKLinear` remains available as an explicit option.
- No change to the face conductivity is planned: `compute_hydraulics!` (`src/processes/soil/hydrology/soil_hydrology_rre.jl`) already takes the minimum of the two adjacent cells, the most restrictive choice, which with the steep power law shuts off the flux out of a dry cell.
  The lagged coefficients of the Picard iteration are checked in the regression test instead: a cell that dries within an iteration has its conductivity lagged at the wetter state, which with the Brooks–Corey conductivity is a much smaller overestimate in absolute terms than with the linear one.
- Add the references (Brooks and Corey 1964, Campbell 1974, Clapp and Hornberger 1978, Boone et al. 2000) to `docs/src/references.bib`.

### Phase 2: Water conservation of the saturation closure

- Replace the bottom-cell deficit fill of `redistribute_saturation_profile!` with a conservative alternative, or, if a conservative fix is not possible without a lower boundary reservoir, account for the created water in a diagnostic and document it.
- Add a check that the saturation closure conserves column water (soil water plus `surface_excess_water`) for oversaturated and undersaturated profiles.

### Phase 3 (conditional): Evaporation that vanishes at residual saturation

If Phases 1 and 2 do not make the regression test pass with `ConstantEvaporationResistanceFactor`, add a continuous-time limitation of ground evaporation that vanishes as the top cell approaches residual saturation, independently of the chosen resistance factor (for example a smooth factor of `(θ₁ − θ_res)/(θ_fc − θ_res)` applied to the evaporation sink in the soil water tendency).
A per-step supply limit `E ≤ φ(s₁ − s_res)Δz₁/Δt` is not an option, since process equations must not depend on the time step.

### Phase 4 (conditional): Picard safeguard from revision 24

If the regression test still fails, implement one of the safeguards that revision 24 of the implicit time stepping plan left open (a limiter on the net balance of each cell, or a saturation-based linearization where the pressure-head form leaves the bounds), with both the dry column and the revision 24 configuration as regression cases.
This phase is expected to be unnecessary for the dry column once the conductivity is consistent.

## Testing and verification

- The new regression test `LandModel: dry soil under strong evaporation demand stays physical` (both resistance factors) must pass: finite state, saturation in `[s_res, 1]`, column water balance (soil water plus surface pool, against evaporation and surface runoff) to a relative tolerance of 10⁻³, and soil and skin temperatures within 20 K of the air temperature over 16 steps of 450 s.
- Unit tests for `UnsatKBrooksCorey`: `K(1) = K_sat`, monotonicity in `S`, the power law at small `S`, the ice impedance, `Float32` type stability, and the default selection by retention curve.
- Existing Richards tests must keep passing, in particular the dry-column infiltration benchmark of Celia et al. (1990) and the mass conservation tests in `test/soil/soil_hydrology_implicit_tests.jl`; any change of their reference values caused by the new default conductivity is explained in the revision log.
- Enzyme and Reactant: `UnsatKBrooksCorey` is a pure scalar kernel function without branches or throw paths; the hydrology differentiability test (`test/differentiability/soil_hydrology_diff.jl`) runs with the new default in CI.
- A one-month run of `examples/simulations/speedy_wet_land.jl` is compared before and after the change for the global soil moisture distribution, as a plausibility check of the new default (not a pass/fail criterion).

## Documentation changes

- `docs/src/processes/soil/soil_hydrology.md`: a section for `UnsatKBrooksCorey` with the equation and references next to `UnsatKLinear` and `UnsatKVanGenuchten`; a note that the retention curve and the conductivity function should come from the same family, and that `UnsatKLinear` is not suitable for dry soil with a Brooks–Corey retention curve.
- Docstrings for `UnsatKBrooksCorey`, its `hydraulic_conductivity` method, and `default_unsat_hydraulic_cond`.
- The time stepping docs drop or narrow the limitation recorded by revision 24 if Phase 4 is implemented.

## Known limitations

- The default soil water dynamics change for every `RichardsEq` simulation: drainage of moderately wet soil is slower with the Brooks–Corey conductivity than with the linear one, so existing example outputs and calibrated parameters may shift.
- `ConstantEvaporationResistanceFactor` still allows potential-rate ground evaporation from soil near residual saturation unless Phase 3 is implemented; with a consistent conductivity, the top cell can then dry out quickly, but the solver should stay bounded.

## Future work

- The atmospheric blow-up in the coupled SpeedyWeather simulation (near the Aral and Caspian Seas), to be investigated on the SpeedyWeather side.
- Performance of the `soil_moisture_limiting_factor` reduction on CPU (an Oceananigans `Integral` over an `AbstractOperation`, about 60 ms per step for a single column), which may be resolved by Oceananigans 0.117 and otherwise by a dedicated column kernel.
- A finite-at-dry-end retention curve (Fayer and Simmons 1995; Webb 2000), which bounds the matric head at oven-dry conditions.
