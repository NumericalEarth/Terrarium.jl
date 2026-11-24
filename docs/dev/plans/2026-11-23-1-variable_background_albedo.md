# Soil-moisture-dependent ground albedo in `DiagnosticAlbedo`

> Status: **in progress**. The plan is approved and implemented; tests are running.

Date of initial draft: 2026-11-23

Author: Claude Opus 5.5, Claude Code (VS Code extension)

Reviewer: Brian Groenke (brian.groenke@pik-potsdam.de)

Base revision: f171fa2923fa3cecdf0673c0764ecb63d210ab59

## Originating prompt

> We have a problem in the JOSS simulation with the skin temperautre being too hot in the tropics. We should add spatially varying background albedo to address this. A simple stopgap solution would be to make the background albedo an input variable with default value set using the current parameter in the DiagnosticAlbedo struct.

> You should first draft an implementation plan and make the changes on a new branch and worktree. It looks like you already made changes, fine to keep them but move them to the worktree.

## Problem description

In the coupled Terrarium–SpeedyWeather simulation of the JOSS paper (`docs/joss/run_terrarium_speedy_simulation.jl`), the skin temperature in the tropics is too high.
The simulation uses `DiagnosticAlbedo`, whose bare-ground (background) albedo `α_bg` is a single scalar parameter (`background_albedo = 0.25`, replaced by `dry_ground_albedo` and `wet_ground_albedo` in this plan) applied everywhere.
Real snow-free land albedo varies strongly in space, from below 0.15 over dark soils to almost 0.4 over the Sahara, so a global constant cannot represent both.

## Background

`DiagnosticAlbedo` diagnoses the surface albedo as an area-weighted blend of bare ground, vegetation, and snow:
`α = (1 − f_snow)·(1 − f_veg)·α_bg + (1 − f_snow)·f_veg·α_veg + f_snow·α_snow`.
`α_veg` comes from `PlantTraits.albedo` (0.15 by default) and `f_veg` from the vegetation component (from the LAI for `PrescribedVegetation`).

Bare soil is darker when wet than when dry.
Land surface models such as CLM therefore interpolate the soil albedo between a dry and a saturated value using the water content of the uppermost soil layer, with the dry and saturated values taken from a prescribed soil color map.
Terrarium already computes a relative water content of the uppermost soil layer (between residual water content and field capacity) for `SoilMoistureResistanceFactor`.

## Summary of changes

### 1. Moisture-dependent background albedo (`src/processes/surface/albedo.jl`)

The background albedo varies linearly between a dry and a wet value with the moisture of the uppermost soil layer:

    α_bg = α_dry + (α_wet − α_dry)·x,  x = clamp((θ₁ − θ_res) / (θ_fc − θ_res), 0, 1)

where `θ₁` is the liquid water content of the uppermost soil layer, `θ_fc` the field capacity, and `θ_res` the residual water content.
`x` is the same relative water content as in `SoilMoistureResistanceFactor`, so the soil is dry at and below the residual water content and wet at and above field capacity.
This follows the dry/saturated interpolation of the CLM soil albedo (darker when wet), simplified to a single broadband value.

- Replace the parameter `default_albedo` with two parameters, `dry_ground_albedo` (default 0.30) and `wet_ground_albedo` (default 0.15), naming explicitly that they are albedos of the (bare) ground.
- Both are spatially constant parameters read directly by the kernel function; the `background_albedo` input variable of the initial draft is removed.
  Spatially varying dry and wet albedos are left to future work.
- Compute `x` in a kernel function that dispatches on the soil component: with an `AbstractSoil`, from the soil composition of the top layer (as in `ground_evaporation_resistance_factor`); with `nothing`, `x = 0`, i.e. the dry value.
- Pass the soil component through: `compute_auxiliary!(state, grid, ::SurfaceEnergyBalance, vegetation, snow, soil)` forwards it to `compute_auxiliary!(state, grid, ::DiagnosticAlbedo, vegetation, snow, soil)`, and `LandModel` passes `model.soil`.
  The soil auxiliary variables are already computed before the SEB in `compute_auxiliary!(state, ::LandModel)`, so no reordering is needed.
- The background emissivity stays a constant scalar parameter, as before.

The defaults change model behavior: the previous constant 0.25 is replaced by a value between 0.15 and 0.30 that depends on soil moisture.

### 2. JOSS simulation (`docs/joss/run_terrarium_speedy_simulation.jl`)

- No changes to the script are needed beyond picking up the new model version; `DiagnosticAlbedo` uses the default dry and wet albedos.
- Rerun to see how much the moisture dependence changes the tropical skin temperature.

The JOSS script only exists on `joss-paper`, whereas `bg/variable-background-albedo` is based on `main`.
The rerun is therefore made on `joss-paper` after the model change is merged into it (from `main` or directly from this branch).

## Testing and verification

- Update `test/surface/diagnostic_albedo_tests.jl`: without soil, the background albedo equals the dry value; with a soil component, it equals the wet value for saturated soil and the dry value for dry soil, and decreases monotonically in between.
- Run `test/surface/diagnostic_albedo_tests.jl`, `test/surface/skin_temperature.jl`, `test/surface/ground_heat_flux.jl`, and the `LandModel` tests via TestEnv, then the full suite and a draft doc build.
- The Enzyme and Reactant suites are not run locally (no extension or AD test changes); left to CI.
- Rerun the JOSS simulation (Slurm job submitted by the user) and compare tropical skin temperature and albedo against the previous run.

## Documentation changes

- Docstring of `DiagnosticAlbedo` and one sentence on the albedo doc page (`docs/src/processes/surface_energy/albedo.md`) describing the moisture dependence and the dry and wet parameters.

## Known limitations

- The dry and wet albedos are rough broadband values, not calibrated, and spatially uniform, so the background albedo varies in space only through soil moisture.
- Where vegetation cover is dense (`f_veg → 1`), the diagnosed albedo is dominated by `α_veg` and the background albedo has little effect.
  If the hot tropical cells are mostly densely vegetated (e.g. the Amazon or Congo), this change alone may not fix the warm bias, and `PlantTraits.albedo` or the vegetation fraction would need attention instead.
  The rerun should check which regions are affected before drawing conclusions.
- The background emissivity remains spatially constant.

## Future work

- Spatially varying dry and wet albedos (as input variables) from a soil color map with vegetation removed (e.g. CLM's, derived from MODIS by Lawrence and Chase, 2007), and separate visible and near-infrared bands.
  SpeedyWeather's `alb` climatology is not suitable for this, since it is the total snow-free surface albedo including present-day vegetation, which would double count vegetation where `f_veg < 1`.
- Compare with the bare-soil albedo of PALADYN (CLIMBER-X).

## Revision log
