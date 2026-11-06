# Rename `VegetationCarbonCycle` to `PrognosticVegetation`

> Status: **in progress**. Approved by the reviewer in the originating prompt; implementation started on `bg/prognostic-vegetation`.

Date of initial draft: 2026-11-06

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code

Reviewer: Brian Groenke (brian.groenke@pik-potsdam.de)

Base revision: `772d0c1d4` (`bg/prescribed-vegetation-lai`, PR #235); this branch, `bg/prognostic-vegetation`, is stacked on it.

## Originating prompt

> How about renaming VegetationCarbonCycle to DynamicVegetation?

> Ok, do the rename to `PrognosticVegetation` and add an alias with deprecation warning.

## Revision log

- **Rev 1 (2026-11-06, Claude Opus 5.5, Claude Code):** Initial draft.

## Problem description

In the review of PR #235, @mahabadri pointed out that the name `VegetationCarbonCycle` is misleading, since `PrescribedVegetation` also contains parts of the vegetation carbon cycle (photosynthesis).
What distinguishes the two is that the carbon pool, leaf area, and (optionally) vegetation cover are prognostic in one and prescribed in the other.
`PrognosticVegetation` names that distinction and pairs with `PrescribedVegetation`.
`DynamicVegetation` was considered and rejected, because "dynamic vegetation" usually means changing vegetation cover, which is the optional `vegetation_dynamics` component, and the name would sit awkwardly next to `AbstractVegetationDynamics`.

## Summary of changes

1. Rename the type `VegetationCarbonCycle` to `PrognosticVegetation`, and its file `vegetation_carbon_cycle.jl` to `prognostic_vegetation.jl`.
2. Keep the old name as a deprecated alias in a new `src/deprecated.jl`, with `Base.@deprecate_binding VegetationCarbonCycle PrognosticVegetation`.
   The alias is the same type, so existing code that constructs it or dispatches on it keeps working, and it stays exported.
   Julia prints the deprecation warning only when deprecation warnings are enabled (`--depwarn=yes`, as in `Pkg.test`).
3. Replace the name in `src/`, `test/`, `examples/`, `benchmark/`, and `docs/src/` (43 occurrences in 25 files), including prose that describes the type.
   Historical plan documents in `docs/dev/` are left unchanged.
   Remove the TODO in `vegetation_base.jl` that suggests a rename.

No other repository uses the name (`NumericalEarth.jl`, `SpeedyWeather.jl`, and `ReadyESM` were checked).

## Testing and verification

- A test that the deprecated alias is the same type as `PrognosticVegetation` and that using it emits a deprecation warning.
- The vegetation tests and the full test suite.
- Doc references checked as for PR #233 and PR #235; the documentation build itself runs in CI.

## Documentation changes

- `docs/src/processes/vegetation/vegetation.md`, `docs/src/running/configuring.md`, and `docs/src/extending/coupling_processes.md` use the new name.

## Known limitations

- Deprecation warnings are not shown in a default Julia session, so users who never run with `--depwarn=yes` will not see the hint before the alias is removed.

## Future work

- Remove the deprecated alias in a future breaking release.
