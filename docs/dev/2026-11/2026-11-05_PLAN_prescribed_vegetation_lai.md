# Prescribed leaf area index as an input of `PrescribedVegetation`

> Status: **in progress**. Implemented as planned; the vegetation tests and the full test suite pass locally, awaiting CI.

Date of initial draft: 2026-11-05

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code

Reviewer: Brian Groenke (brian.groenke@pik-potsdam.de)

Base revision: `2d81a1766d037bf5aafa5a129a80367bcb0a1df8` (`bg/revise-vegetation`, PR #233); this branch, `bg/prescribed-vegetation-lai`, is stacked on it.

## Originating prompt

> Create a new branch stacked on bg/revise-vegetation for refactoring the prescribed vegetation and phenology. The goal is to remove PrescribedPhenology and instead build the prescribed LAI as input directly into PrescribedVegetation. First make sure that the phenology factor is not needed by any downstream processes. This will resolve an inconsistency where PrescribedPhenology could currently be used on VegetationCarbonCycle and vice versa for PALADYNPhenology on PrescribedVegetation.

## Revision log

- **Rev 1 (2026-11-05, Claude Opus 5.5, Claude Code):** Initial draft.

## Problem description

`PrescribedPhenology` exists only to declare the leaf area index as an input and to derive a phenology factor `ϕ = clamp(LAI / LAI_max, 0, 1)` from it.
Both phenology schemes are subtypes of `AbstractPhenology`, so either can be passed to either vegetation type, but only the intended pairing works:

- `VegetationCarbonCycle(NF; phenology = PrescribedPhenology(NF))` constructs, builds a model, and initializes, then fails at the first time step with `MethodError: no method matching compute_phenology!(...)`.
- `PrescribedVegetation(NF; phenology = PALADYNPhenology(NF))` constructs, then fails in `initialize` with `MethodError: no method matching variables(::PALADYNPhenology{Float64}, ::PlantTraits{Float64})`.

Both were verified on the base revision.
The phenology slot of `PrescribedVegetation` is therefore not a real extension point, and the type system should not suggest that it is.

## Background

### Who reads the phenology factor

A search of `src/`, `ext/`, `test/`, `examples/`, `benchmark/`, and `docs/src/` on the base revision finds one consumer of `phenology_factor`: `compute_autotrophic_respiration` (`PALADYNAutotrophicRespiration`), which scales root respiration with it.
Autotrophic respiration is part of `VegetationCarbonCycle` only, where `PALADYNPhenology` produces the factor.
`PrescribedVegetation` has no autotrophic respiration, so the `phenology_factor` auxiliary that `PrescribedPhenology` adds to it is read by nothing.

The vegetated area fraction of `PrescribedVegetation` (used by the surface energy balance, albedo, and evapotranspiration) is computed by `compute_relative_leaf_area` directly from `leaf_area_index` and the traits, not from the `phenology_factor` field, so it is unaffected.

Outside Terrarium, `NumericalEarth.jl`, `NumericalEarthBase.jl`, `SpeedyWeather.jl`, and `ReadyESM` do not use `PrescribedPhenology`, `phenology_factor`, or a `phenology` keyword of `PrescribedVegetation`.
`ReadyESM` vendors a snapshot of Terrarium that contains `PrescribedPhenology`, but its own code does not reference it.

The remaining in-repo uses are:

- `test/vegetation/phenology_tests.jl` (a `PrescribedPhenology` test set, the only caller passing `phenology =` to `PrescribedVegetation`);
- `docs/src/processes/vegetation/phenology.md` and `vegetation.md`, the `PrescribedVegetation` docstring, and the introductory text of `examples/simulations/vegetation_global.jl`;
- a comment in `test/reactant/correctness.jl` that names `phenology_factor` as an example of a `kernel(...)`-backed auxiliary.

All examples and Reactant configurations use `PrescribedVegetation(NF)` with its defaults and supply the input `leaf_area_index` by name, which this change keeps.

## Summary of changes

1. **Remove `PrescribedPhenology`.**
   Delete `src/processes/vegetation/phenology/prescribed_phenology.jl`, its `include`, and its export from `src/processes/processes.jl`.
   `PALADYNPhenology` remains the only `AbstractPhenology`, used by `VegetationCarbonCycle`.
2. **Declare the prescribed leaf area index in `PrescribedVegetation`.**
   Remove the `phenology` component, its type parameter, and its constructor keyword.
   `variables(::PrescribedVegetation)` declares `input(:leaf_area_index, Canopy(XY()), units = u"m^2/m^2", desc = "Prescribed leaf area index")` itself, alongside the variables of photosynthesis, stomatal conductance, root distribution, and plant-available water (the traits declare no variables of their own).
   `compute_auxiliary!` no longer calls the phenology no-op.
   The vegetated area fraction is unchanged (`compute_relative_leaf_area`).
3. **Drop `phenology_factor` from `PrescribedVegetation`.**
   No process of `PrescribedVegetation` reads it.
   Users who plotted or saved it can compute `clamp(LAI / LAI_max, 0, 1)` from the state and `traits.maximum_leaf_area_index`.
4. **Update docstrings and comments.**
   The `PrescribedVegetation` docstring and the comment in `test/reactant/correctness.jl` (which can name `soil_moisture_limiting_factor` or `root_fraction` as the example of a derived auxiliary).

This is a breaking change of the public API (an exported type is removed, and a keyword of an exported constructor).
It should be noted in the PR description and in the changelog or release notes, if the project keeps one.

## Testing and verification

- Replace the `PrescribedPhenology` test set in `phenology_tests.jl` with a `PrescribedVegetation` test set (in a new `test/vegetation/prescribed_vegetation_tests.jl`, registered in `vegetation_model_tests.jl`) that checks:
  - `Variables(PrescribedVegetation(Float64))` has `leaf_area_index` as an input and no `phenology_factor`;
  - `PrescribedVegetation(Float64; phenology = ...)` is rejected (a `MethodError` from the keyword constructor);
  - a prescribed LAI drives photosynthesis in a `VegetationModel` (GPP is zero for LAI = 0, positive for LAI > 0, and increases with LAI);
  - the vegetated area fraction equals `clamp(LAI / LAI_max, 0, 1)`, as `diagnostic_albedo_tests.jl` already checks through `vegetation_area_fraction`.
- The vegetation tests, then the full test suite.
- The Reactant suite is not run locally (no Reactant code changes); its `:vegetation_column` and `:vegetation_column_lai_cycle` configurations exercise the new input path in CI.
- A draft doc build, to catch stale references to `PrescribedPhenology` (CI, if the local build environment issues from PR #233 persist).

## Documentation changes

- `docs/src/processes/vegetation/phenology.md`: remove the "Prescribed phenology" section.
- `docs/src/processes/vegetation/vegetation.md`: describe the prescribed leaf area index as an input of `PrescribedVegetation` instead of a phenology scheme.
- `examples/simulations/vegetation_global.jl`: update the two sentences that mention `PrescribedPhenology`.

## Known limitations

- `VegetationCarbonCycle` still accepts any `AbstractPhenology`; with `PrescribedPhenology` removed, `PALADYNPhenology` is the only one, so the mismatch can no longer arise, but a future phenology scheme would again need to implement the `VegetationCarbonCycle` interface (`compute_phenology!` with the carbon dynamics, and `phenology_factor`).
- The "Phenology is not fully implemented yet" warning on the phenology doc page is out of date since `PALADYNPhenology` was implemented; correcting it is unrelated to this change and left out of scope.

## Future work

- If prescribed vegetation later needs a phenology factor (for example, for a prescribed-LAI variant with autotrophic respiration), derive it where it is consumed rather than reintroducing a phenology scheme.
