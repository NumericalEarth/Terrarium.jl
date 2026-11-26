# Snow and snow-free surface tiles with a melting-point-limited snow skin

> Status: **planned**. Draft awaiting review; no code has been changed.

Date of initial draft: 2026-11-24

Author: Claude Opus 5.5 (claude-opus-5-5), Claude Code

Reviewer:

Base revision: 7368cf643bcaf962668082124f1f1a367b4f2f83

## Originating prompt

> Create a new worktree for the snow issue and draft a plan

Approach chosen by the user: separate snow and snow-free tiles with a 0 °C cap on the snow skin temperature (not the thin conduction layer stopgap), based on `main`.

## Problem description

Snow in a `LandModel` melts far too slowly, and in multi-year coupled runs it accumulates from year to year at high latitudes.
The issue draft `scratch/issues/13-snowmelt-limited-by-snow-conduction.md` records the evidence:

- A standalone column with `SingleLayerSnow` under warm forcing (15 °C air, 250 W/m² daily-mean shortwave) melts at about 1 mm/day, where 10–30 mm/day would be expected.
  The skin temperature settles at about +8 °C over a 97 % snow-covered surface.
  Lowering the snow albedo from 0.8 to 0.5 raises melt only to about 1.4 mm/day.
- In a 5-year coupled SpeedyWeather–Terrarium run (T64), SWE grows by about 0.16–0.21 m per year across Siberia, northern Canada, and Alaska, and about 1950 Northern Hemisphere land points keep snow through September.

The cause is structural.
`ImplicitSkinTemperature` solves one skin temperature `Ts` per grid cell from

    R_net(Ts) + H_s(Ts) + H_l(Ts) − (1 − f_snow)·G(Ts, T_g) − f_snow·S(Ts, T_snow) = 0,

where the snow-top flux `S = 2κ_snow(T_snow − Ts)/d_snow` conducts through half the bulk snow depth.
Nothing limits `Ts` to the melting point over snow.
When the atmosphere supplies more energy than the conduction path can take, `Ts` rises above 0 °C, the surplus returns to the atmosphere through longwave emission and weaker sensible heating, and only the small conductive flux reaches the pack.
For a 1 m deep pack and an 8 K difference, `S` is about 5 W/m², which matches the simulated melt rates.

## Background

Relevant code at the base revision:

- `src/processes/surface/skin_temperature.jl`: `ImplicitSkinTemperature`, the blended conduction inverse `compute_skin_temperature(..., snow::AbstractSnow)`, and the Newton solve `solve_skin_temperature!` on the residual `Ts_prev − Ts_implicit`.
- `src/processes/surface/surface_energy_balance.jl`: the fused `solve_surface_energy_balance_kernel!`, which solves `Ts`, recomputes evapotranspiration, and then recomputes all fluxes at the final `Ts`.
- `src/processes/surface/radiative_fluxes.jl`: upwelling shortwave and longwave from the blended `albedo` and `emissivity` at the single `Ts`.
- `src/processes/surface/turbulent_fluxes.jl`: sensible heat at `Ts`; latent heat partitioned as `(1 − f)·H_l,ground + f·H_l,snow`, both at `Ts`.
- `src/processes/surface/albedo.jl`: `DiagnosticAlbedo` blends ground, vegetation, and snow albedo and emissivity by `f_snow` and `f_veg`, with the ground albedo depending on top-layer soil moisture.
- `src/processes/snow/snow_interfaces.jl`: `compute_snow_surface_heat_flux` (`S`) and `compute_snow_interface_fluxes!`, which store `snow_surface_heat_flux = f·S`, `snow_basal_heat_flux = f·Q_base`, `soil_heat_flux = f·Q_base + (1 − f)·G`, and the sublimation `f·E_subl`.
- `src/processes/snow/mass/snow_mass.jl`: `compute_snow_sublimation_flux`, evaluated at `Ts` as a saturated surface.
- `src/processes/snow/energy/snow_energy.jl`: `dŪ_snow/dt = Q_base − Q_top + Q_precip + Q_subl`, with `Q_top = snow_surface_heat_flux`.
- `src/processes/surface/evapotranspiration/*.jl`: bare-ground evaporation and canopy evapotranspiration read `skin_temperature` as the snow-free surface (and canopy) temperature.
- `src/models/coupled/land_model.jl`: `compute_auxiliary!` solves the surface energy balance, then `compute_snow_interface_fluxes!`.
- SpeedyWeather's `SpeedyWeatherTerrariumExt` copies `skin_temperature` (as the land surface temperature used for SpeedyWeather's own longwave), `sensible_heat_flux`, `latent_heat_flux / L_vap` (as the humidity flux), `snow_water_equivalent`, and `albedo`.

Standard land models handle this in one of two ways.
Tiled models (for example JSBACH and ORCHIDEE in some configurations) solve a separate energy balance per surface type and aggregate the fluxes.
CLM keeps one ground temperature but uses a multi-layer snowpack with a thin top layer and resets layers above 0 °C to 0 °C, putting the excess into melt.
Terrarium's single bulk snow layer cannot use the CLM approach without becoming multi-layer, so tiling is the smaller structural change.

## Summary of changes

### Physics

Each grid cell is split into a snow tile (area fraction `f = f_snow`) and a snow-free tile (`1 − f`), each with its own skin temperature and energy balance.
Both tiles share the atmospheric forcing and the aerodynamic resistance.
As in the current albedo blend, snow covers the vegetation, so vegetation and canopy evapotranspiration belong to the snow-free tile only.

Snow-free tile, skin temperature `T_sf`, solved as now but without the snow terms:

    R_net,sf(T_sf) + H_s(T_sf) + H_l,sf(T_sf) = G(T_sf, T_g),   G = 2κ_s(T_g − T_sf)/Δz_g,

with the snow-free albedo and emissivity (ground and vegetation blend) and the evapotranspiration latent flux.

Snow tile, skin temperature `T_sn`:

    R_net,sn(T_sn) + H_s(T_sn) + H_l,sn(T_sn) = S(T_sn, T_snow),   S = 2κ_snow(T_snow − T_sn)/d_snow,

with the snow albedo and emissivity and the sublimation latent flux.
The unconstrained root `T_sn*` is limited to the melting point, `T_sn = min(T_sn*, 0 °C)`, written with `ifelse`.
The energy delivered to the snowpack top is the snow tile's demand evaluated at the limited temperature,

    Q_top,sn = R_net,sn(T_sn) + H_s(T_sn) + H_l,sn(T_sn),

which equals the conductive flux `S(T_sn)` whenever the limit is inactive (that is the root condition), and otherwise includes the surface-melt surplus that conduction alone could not carry.
`snow_surface_heat_flux` becomes `f·Q_top,sn`, so the existing snow energy tendency and `FreeWater` closure turn the surplus into liquid water and the existing drainage removes it.
Surface energy is conserved by construction: the atmosphere sees exactly the fluxes whose sum enters the snowpack.

Grid-cell outputs are area-weighted tile sums:

- `surface_shortwave_up`, `surface_longwave_up`, `surface_net_radiation`, `sensible_heat_flux`, and `latent_heat_flux` as `f·X_sn + (1 − f)·X_sf`;
- `albedo` and `emissivity` keep their current blended meaning;
- a grid-cell radiative skin temperature `T_rad` from the aggregated longwave, `ε σ T_rad⁴ = LW_up − (1 − ε)·LW_down` with the blended `ε`, so that the coupled atmosphere, which only sees one surface temperature, emits the same longwave as Terrarium.

### Code

1. `ImplicitSkinTemperature` with a snow component gains a second solved field for the snow tile.
   The no-snow method (`snow === nothing`) is unchanged.
   The `PrescribedSkinTemperature` path (external coupler owns the surface fluxes) is unchanged.
2. Split the flux kernel functions into per-tile variants that take the tile's skin temperature, albedo, and emissivity explicitly, instead of reading the single `skin_temperature` and the blended `albedo` inside:
   radiative fluxes (`compute_surface_upwelling_radiation`), sensible heat, the latent flux of each tile (evapotranspiration for the snow-free tile, `compute_snow_sublimation_flux` for the snow tile), and the ground heat flux demand.
3. `DiagnosticAlbedo` additionally stores the snow-free albedo and emissivity (ground and vegetation blend), since the snow-free tile needs them and the blended values cannot be un-blended when `f → 1`.
   The snow tile reads the snow albedo and emissivity directly from the snow component.
4. `solve_surface_energy_balance_kernel!` runs two scalar Newton solves per cell (snow-free and snow tile), applies the melting-point limit to the snow tile, recomputes evapotranspiration at `T_sf`, and stores the aggregated outputs and `T_rad`.
5. `compute_snow_interface_fluxes!` stores `snow_surface_heat_flux = f·Q_top,sn` and evaluates the sublimation at `T_sn`.
   `soil_heat_flux = f·Q_base + (1 − f)·G(T_sf)` is unchanged in form.
6. `diagnose_skin_temperature_residual` in `src/diagnostics/surface_fluxes.jl` reports both tile residuals.
7. Initialization seeds `T_sn` from `min(snow_temperature, 0 °C)` (or the ground temperature without snow) and `T_sf` from the ground temperature, as `skin_temperature` is seeded now.

All new code follows the kernel rules: `ifelse` for the limit, no throw paths, `NF`-typed constants, and no allocations.
The melting-point limit has a kink at 0 °C; Enzyme returns the one-sided derivative of the active branch, as for the existing phase-change closures.

### Open decision: field names

The plan needs three temperatures per cell: the snow-free tile skin, the snow tile skin, and the grid-cell radiative temperature.
Two naming options:

- **A (recommended):** `skin_temperature` stays the grid-cell radiative temperature `T_rad`, so outputs, diagnostics, and the SpeedyWeather coupling keep their meaning; new fields `snow_free_skin_temperature` and `snow_skin_temperature` hold the tiles.
  Evapotranspiration, bare-ground evaporation, and the ground heat flux switch to reading `snow_free_skin_temperature`.
- **B:** `skin_temperature` becomes the snow-free tile (no change in evapotranspiration code), plus new `snow_skin_temperature` and `radiative_skin_temperature`.
  Smaller diff, but every consumer of `skin_temperature` outside Terrarium (outputs, coupling) silently changes meaning.

Without snow, all three coincide in both options.

## Testing and verification

New and updated tests in `test/surface/` and `test/snow/`:

- **No-snow regression:** with `snow = nothing`, and with a snow component but zero SWE (`f = 0`), skin temperature and all fluxes match the base revision to round-off.
- **Melting-point limit:** for any forcing with `f > 0`, `T_sn ≤ 0 °C`; when the limit is inactive, `Q_top,sn` equals the conductive flux `S(T_sn)` to solver tolerance.
- **Energy closure:** per cell, `f·(R_net + H_s + H_l)_sn + (1 − f)·(R_net + H_s + H_l)_sf = f·Q_top,sn + (1 − f)·G`, and the aggregated outputs equal the tile-weighted sums.
- **Radiative temperature:** `ε σ T_rad⁴ + (1 − ε)·LW_down` reproduces the aggregated `surface_longwave_up`.
- **Column melt test:** the idealized column of the issue (15 °C air, 250 W/m² daily-mean shortwave, 0.3 m SWE, 30 days).
  The snow skin stays at or below 0 °C, and the melt rate is bounded below by the energy budget at 0 °C (a test threshold derived from the run, documented in the test).
  The snowpack energy change over the run equals the time-integrated `f·Q_top,sn + f·Q_base` plus precipitation and sublimation terms.
- **Cold conditions:** under sub-freezing forcing, the limit is inactive and results stay close to the base revision (only the tile split changes the fluxes).
- **Differentiability:** extend `test/differentiability/snow_model_diff.jl` (or `prescribed_seb_diff.jl`) with an Enzyme reverse-mode check through `solve_surface_energy_balance!` with snow, on both sides of the limit, checked against finite differences away from the kink.
  Run locally because it touches the differentiability tests.
- **Reactant:** the change touches kernels used under Reactant; leave `test/reactant/` to CI unless the Reactant tests fail there.
- **GPU:** run the snow and surface test files on a GPU node (the user submits the job).

Verification beyond the tests:

- Rerun the scratch column script (`snow_melt_column.jl`) with snow albedo 0.8 and 0.5 and record melt rates in the PR.
- Rerun the coupled `docs/joss` simulation for at least two years on the GPU and compare the September Northern Hemisphere snow-covered points and the SWE trend against `run_0002`.
- Run the full test suite and a draft doc build.

## Documentation changes

- `docs/src/processes/surface_energy/skin_temperature.md`: the tiled energy balance, the melting-point limit, and the radiative temperature, with the equations above.
- `docs/src/processes/surface_energy/surface_energy_balance.md`: aggregation of tile fluxes, and which fields belong to which tile.
- `docs/src/processes/surface_energy/albedo.md`: the snow-free albedo and emissivity outputs.
- `docs/src/processes/snow/snow_energy.md`: `Q_top` now includes the surface-melt surplus.
- Docstrings of all changed kernel functions, and the new fields in `variables`.

## Known limitations

- Both tiles share one aerodynamic resistance; snow and snow-free roughness lengths are not distinguished.
- Snow fully buries the vegetation in the snow tile; there is no canopy snow or snow under a canopy.
- The snow albedo stays constant (`ConstantSnowAlbedo`); albedo aging is separate future work and would further increase melt.
- The tile skin temperatures are diagnostic roots, not prognostic states, so a newly snow-covered area has no thermal memory of its own; this matches the current single skin temperature.
- SpeedyWeather converts the aggregated latent heat flux to a humidity flux with the latent heat of vaporization only, so the sublimation part is slightly misconverted (about 13 %).
  Passing the water vapor mass flux instead belongs in the SpeedyWeather extension (to be recorded in `docs/dev/upstream`).
- Snow at temperatures above 0 °C remains possible inside the bulk pack only through the energy closure, as now; the limit applies to the surface.

## Future work

- Snow albedo aging and a lower albedo for melting snow.
- Separate roughness lengths per tile once the atmosphere scheme supports them.
- Canopy snow interception and a vegetation tile that can protrude through the snowpack.
- Pass tile-aware fluxes (water vapor mass flux, radiative temperature) explicitly to SpeedyWeather.

## Revision log
