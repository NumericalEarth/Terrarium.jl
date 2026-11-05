# Area-consistent snow boundary heat fluxes in the coupled snow–soil model

> Status: **completed**. Implemented with Revs 3 and 4 and verified by the test suite and a four-month coupled SpeedyWeather run.

Date of initial draft: 2026-11-01

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code

Reviewer: Brian Groenke (brian.groenke@pik-potsdam.de)

Base revision: `887c8849f8c30acbce4b59284802419821094bae` (`joss-paper`); the fix should be branched from `main` (`077c8a9ed`)

## Originating prompt

> There seems to be very little to no snow in the northern US and southern Canada. That seems wrong.
> Draft an implementation plan and I will have another agent work on applying it.

## Revision log

- **Rev 1 (2026-11-01):** Initial draft (Claude Opus 5.5).
- **Rev 2 (2026-11-01, Claude Fable 5.1, Claude Code):** Review against the code on `main` (`077c8a9ed`).
  Corrected the field-allocation note in change 1 (the input field is allocated automatically once the alias is dropped).
  Clarified the `(W > 0)` gate in change 3, the standalone `SnowModel` docstring in change 6, and how to obtain the atmosphere-side flux `G₀` in test 2.
  Noted the stale `W_ref = 0.1 m` comments in existing tests as out of scope.
  Approved by Brian Groenke on 2026-11-01.
- **Rev 3 (2026-11-01, Claude Sonnet, Claude Code; logged by Claude Opus 5.5):** Deviation made during implementation.
  Removed the kernel function `compute_snow_soil_heat_flux` and its doc page entry instead of keeping it unchanged as change 2 specified.
  The blend `f·Q_base + (1 − f)·G` is now computed inline in `compute_snow_interface_fluxes!`, so `Q_base` is evaluated once per cell.
  The formula itself is unchanged.
  Approved by Brian Groenke on 2026-11-01.
- **Rev 4 (2026-11-01, Claude Opus 5.5, Claude Code):** Requested and approved by Brian Groenke on 2026-11-01.
  Renamed the `SingleLayerSnow` inputs `surface_heat_flux` and `basal_heat_flux` to `snow_surface_heat_flux` and `snow_basal_heat_flux`.
  The new input name coincides with the former `LandModel` auxiliary `snow_surface_heat_flux`, so that auxiliary and the input alias are removed; both snow fluxes are now snow inputs that `StateVariables` allocates automatically and `compute_snow_interface_fluxes!` fills.
  Without snow, the state no longer contains a `snow_surface_heat_flux` field (it was previously an alias of `ground_heat_flux`).

## Problem description

In the coupled SpeedyWeather–Terrarium run for the JOSS paper (T64, `SingleLayerSnow`, `NoFlow` soil hydrology), snow does not accumulate in the northern US, southern Canada, and eastern Europe, although snow falls there at subfreezing temperatures.
For 42–55°N, 250–290°E, December–February: snowfall 0.9–1.0 mm/day at a mean air temperature of about −5 °C, but a mean snow water equivalent (SWE) of 8 mm in the first winter and about 0 in the second; roughly 80–90 mm would be expected.
At a single grid point (47.6°N, 264.4°E), snow falling at −6 to −7 °C on 2025-12-02 raised the SWE to 0.02 mm, and four hours later it was gone at a skin temperature of −7.2 °C, with negligible sublimation.
Throughout the week, the SWE simply tracks the instantaneous snowfall rate.
Snow only accumulates where the soil is already cold (Siberia, northern Canada) or snowfall is heavy enough to raise the snow cover fraction before the pack melts.

## Background

`SingleLayerSnow` has prognostic `snow_water_equivalent` (`W`, m) and `snow_energy` (`Ū`, J/m²).
Both are **grid-cell means**: `compute_snow_sublimation_flux` documents this, and the sublimation passed to the snow (`compute_snow_interface_fluxes!`) is scaled by the snow cover fraction `f = W/(W + W_ref)` (`FractionalSnowCover`, `W_ref = 0.01 m`).
Snowfall and the rain intercepted by snow (`f·R`) are also cell means.

The snow energy tendency (`src/processes/snow/energy/snow_energy.jl`, `compute_snow_energy_tendency`) is

```
dŪ/dt = (Q_base − Q_top + Q_subl)·(W > 0) + Q_prcp
```

with fluxes positive upward: `Q_base` is the flux from the soil into the snow base, and `Q_top` is the flux leaving the snow top into the skin.
It reads them from the input fields `basal_heat_flux` and `surface_heat_flux`, which `LandModel` aliases in `src/models/coupled/land_model.jl` (around line 102):

```julia
snow_flux_aliases = (; surface_heat_flux = snow_surface_heat_flux, basal_heat_flux = soil_heat_flux)
```

These fields are filled by `compute_snow_interface_fluxes!` (`src/processes/snow/snow_interfaces.jl`):

- `basal_heat_flux` (= `soil_heat_flux`, the soil's top boundary condition) is the **blended** flux `f·Q_base + (1 − f)·G` (`compute_snow_soil_heat_flux`), where `Q_base` is the snow–soil series conduction (`compute_snow_basal_heat_flux`, per unit snow-covered area) and `G` the bare-ground conductive flux (`ground_heat_flux`, per unit bare area).
- `surface_heat_flux` (= `snow_surface_heat_flux`) is `S = 2κ_snow(T_snow − T_skin)/max(d_snow, d_min)`, **per unit snow-covered area** and not scaled by `f` (`compute_snow_surface_heat_flux`).

The implicit skin temperature solve (`src/processes/surface/skin_temperature.jl`) closes the atmosphere-side demand `G₀` against the area-weighted conduction `(1 − f)·G + f·S`.

So the snowpack receives two inconsistent boundary fluxes:

1. **Basal:** the blended soil flux, which for a thin pack (`f → 0`) is the bare-ground flux `G`, applied in full to the cell-mean snow energy. In early winter `G` is typically positive upward (the soil releases its summer heat) at 10–30 W/m², which melts a fresh 0.02 mm pack (latent energy about −7 kJ/m²) within minutes. This is the bug behind the missing snow.
2. **Top:** `S` per unit snow area applied to the cell-mean snow energy, overstating the top exchange by `1/f`.

The blend `f·Q_base + (1 − f)·G` itself is correct as the **soil**'s top boundary condition and should stay.

Energy accounting with all quantities as cell means (skin massless):

- soil top: loses `f·Q_base + (1 − f)·G`;
- snow: gains `f·Q_base` at the base and loses `f·S` at the top;
- skin: receives `(1 − f)·G + f·S = G₀` and passes it to the atmosphere.

The column (soil + snow) then loses exactly `G₀`, so energy is conserved.
The current code instead gives the snow `f·Q_base + (1 − f)·G − S`, which does not close.

## Summary of changes

1. **Separate the snow's basal flux from the soil boundary flux.**
   In `LandModel`, stop aliasing `basal_heat_flux` to `soil_heat_flux`.
   `basal_heat_flux` is already declared as `input(:basal_heat_flux, Snow(Bottom()))` on `SingleLayerSnow` without a default, so `StateVariables` allocates it automatically as soon as it is no longer supplied through the `fields` merge; no explicit `initialize(vars.basal_heat_flux, …)` call is needed.
   Keep `soil_heat_flux` for the soil boundary condition.
   The `surface_heat_flux = snow_surface_heat_flux` alias can stay provided `snow_surface_heat_flux` stores the cell-mean `f·S` (see change 2).
   The comment in `land_model.jl` that justifies the aliases ("so no additional state fields are allocated") must be updated.
2. **Fill both fields in `compute_snow_interface_fluxes!`.**
   - `soil_heat_flux` ← `f·Q_base + (1 − f)·G` (unchanged formula, see `compute_snow_soil_heat_flux`);
   - `basal_heat_flux` ← `f·Q_base` (cell-mean flux into the snow base);
   - `surface_heat_flux` ← `f·S` (cell-mean flux out of the snow top).
   The `out` named tuple of the kernel needs the additional `soil_heat_flux` field; check `get_fields`/`interface_variables` so that the field is available in the kernel without being an aliased input.
   Keep the per-area formulas (`compute_snow_basal_heat_flux`, `compute_snow_surface_heat_flux`) unchanged and apply `f` at the call site, mirroring the sublimation flux.
   Decide whether `snow_surface_heat_flux` (an auxiliary that is also documented as the per-area target of the skin solve) should store `S` or `f·S`; the skin solve recomputes `S` from `snow_thermal_interface`, so storing the cell-mean `f·S` is safe, but update its `desc` and the docstring of `compute_snow_surface_heat_flux` accordingly.
3. **Snow energy tendency.**
   `compute_snow_energy_tendency` stays as it is (`Q_base − Q_top` with cell-mean fluxes).
   Keep the `(W > 0)` gate: although both conductive fluxes now vanish with `f` as `W → 0`, the gate also zeroes the `Q_subl` term and guards the first step in which snow falls onto bare ground (see the existing docstring and the standalone test "snowfall onto bare ground (W=0) advects cold, not zero").
   Update its docstring to state that the boundary fluxes are cell means scaled by `f`.
4. **Without snow** (`model.snow === nothing`), nothing changes: `soil_heat_flux` is `ground_heat_flux` and no snow fields exist.
5. **Kernel rules.** All of this is inside existing kernels; no throw paths, no `Float64` literals, `ifelse` only (see AGENTS.md).
6. **Standalone `SnowModel`** (`src/models/snow/snow_model.jl`) prescribes `surface_heat_flux` and `basal_heat_flux` directly, so its behavior is unchanged; update its docstring to state that these inputs are grid-cell means.

Do not change the skin temperature solve, the snow cover parameterization, or the soil boundary condition formula.

## Testing and verification

1. **Regression test (new),** in `test/coupled_models/land_model_tests.jl`, similar to "thin melting snowpack conserves water independently of Δt": a `LandModel` with `SingleLayerSnow`, no vegetation, `NoFlow` soil hydrology, prescribed atmosphere with air temperature about −10 °C and no precipitation.
   Initialize a thin pack (e.g. `W = 1 mm`, snow temperature about −5 °C) over warm soil (e.g. +5 °C in the top layers, so the bare-ground flux is strongly upward).
   Run a few hours.
   Before the fix the pack disappears; after the fix the SWE must stay close to its initial value (no meltwater outflow, `snow_liquid_fraction ≈ 0`).
2. **Energy conservation test (new):** for the same setup (and ideally a second case with a deeper pack, e.g. `W = 0.1 m`), check that the change in total column energy (soil `internal_energy` integrated over depth plus `snow_energy`) over a run equals `−∫G₀ dt` (the atmosphere-side flux, i.e. `ground_heat_flux_demand`/`R_net + H + LE`) plus the precipitation enthalpy, to a tight relative tolerance.
   The atmosphere-side flux is not stored directly: `ground_heat_flux` holds the bare-ground per-area flux `G`, so compute `G₀ = (1 − f)·G + f·S` from `ground_heat_flux`, `snow_cover_fraction`, and `snow_surface_heat_flux` (which holds `f·S` after the fix), or equivalently from `R_net + H + LE`; document the choice in the test.
   This test should fail on the current code and pass after the fix.
3. **Existing tests** that must keep passing (note that the comments in "LandModel: Snow-coupled soil" and "latent flux partitioned …" quote `W_ref = 0.1 m`, while the code default is `0.01 m`; the assertions still hold and fixing the comments is out of scope), especially "LandModel: thin snow over frozen soil stays stable", "snow meltwater reaches the soil surface", "thin melting snowpack conserves water independently of Δt", and the snow energy conservation tests in `test/snow/snow_model_tests.jl`.
4. Run the full test suite and the draft doc build (`julia --project=docs docs/make.jl --local --draft`).
   Run the Enzyme suite (`Pkg.test(; test_args = ["enzyme"])`) and `test/reactant` only if the differentiability tests or extensions are touched; otherwise leave them to CI.
5. **Optional end-to-end check:** a short coupled run (`docs/joss/run_terrarium_speedy_simulation.jl` with a shorter period, e.g. October–February) should show snow accumulating in the northern US, southern Canada, and eastern Europe in winter.

## Documentation changes

- `docs/src/processes/snow/snow_energy.md`: state that the basal and top fluxes are cell means (`f·Q_base`, `f·S`) and give the column energy balance above.
- `docs/src/processes/surface_energy/skin_temperature.md`: only if it describes `snow_surface_heat_flux` as the per-area flux.
- Docstrings of `compute_snow_soil_heat_flux`, `compute_snow_surface_heat_flux`, `compute_snow_interface_fluxes!`, `compute_snow_energy_tendency`, and the `desc` strings of `snow_surface_heat_flux`/`soil_heat_flux` in `land_model.jl`.

## Known limitations

- The snow and bare fractions share one skin temperature, so the snow-top flux `S` depends on a cell-mean skin temperature; this plan keeps that simplification.
- With `FractionalSnowCover` (`W_ref = 0.01 m`), thin packs have small `f` and therefore weak thermal coupling to both soil and skin; that is consistent with the cell-mean formulation but may make thin packs slow to warm or cool.
- The per-area/cell-mean convention is now consistent for the energy fluxes, but rain on snow (`f·R`) and snowfall (not scaled by `f`) should be double-checked for the same convention while implementing; snowfall onto the bare fraction creating new snow is intended.

## Future work

- A multi-layer snow scheme or separate skin temperatures for the snow and bare fractions.
- A process-level energy-conservation test harness for all `LandModel` configurations.
