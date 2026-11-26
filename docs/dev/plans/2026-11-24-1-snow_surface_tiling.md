# Snow and snow-free surface tiles with a melting-point-limited snow skin

> Status: **planned**. Revision 1 awaiting review; no code has been changed.

Date of initial draft: 2026-11-24

Author: Claude Opus 5.5 (claude-opus-5-5), Claude Code

Reviewer:

Base revision: 7368cf643bcaf962668082124f1f1a367b4f2f83

## Originating prompt

> Create a new worktree for the snow issue and draft a plan

Approach chosen by the user: separate snow and snow-free tiles with a 0 °C cap on the snow skin temperature based on current `main`.

Revision 1 fixes the code design: the tiles are variable namespaces of the single `SurfaceEnergyBalance`, selected by dispatch on the snow argument (see the revision log).

## Problem description

Snow in a `LandModel` melts far too slowly, and in multi-year coupled runs it accumulates from year to year at high latitudes.
The issue draft `scratch/issues/13-snowmelt-limited-by-snow-conduction.md` records the evidence:

- A standalone column with `SingleLayerSnow` under warm forcing (15 °C air, 250 W/m² daily-mean shortwave) melts at about 1 mm/day, where the energy budget of a 0 °C snow surface under that forcing allows an order of magnitude more.
  The skin temperature settles at about +8 °C over a 97 % snow-covered surface.
  Lowering the snow albedo from 0.8 to 0.5 raises melt only to about 1.4 mm/day.
- In a 5-year coupled SpeedyWeather–Terrarium run (T64), SWE grows by about 0.16–0.21 m per year across Siberia, northern Canada, and Alaska, and about 1950 Northern Hemisphere land points keep snow through September.

The cause is structural.
`ImplicitSkinTemperature` solves one skin temperature `Ts` per grid cell from

```math
R_{\text{net}}(T_s) + H_s(T_s) + H_l(T_s) - (1 - f_{\text{snow}})\, G(T_s, T_g) - f_{\text{snow}}\, S(T_s, T_{\text{snow}}) = 0,
```

where the snow-top flux `S = 2κ_snow(T_snow − Ts)/d_snow` conducts through half the bulk snow depth.
Nothing limits `Ts` to the melting point over snow.
When the atmosphere supplies more energy than the conduction path can take, `Ts` rises above 0 °C, the surplus returns to the atmosphere through longwave emission and weaker sensible heating, and only the small conductive flux reaches the pack.
For a 1 m deep pack and an 8 K difference, `S` is about 5 W/m², which matches the simulated melt rates.

## Background

Relevant code at the base revision:

- `src/processes/surface/skin_temperature.jl`: `ImplicitSkinTemperature`, the ground-only inverse `compute_skin_temperature(..., seb)`, the blended inverse `compute_skin_temperature(..., seb, constants, snow::AbstractSnow)`, and the Newton solve `solve_skin_temperature!` on the residual `Ts_prev − Ts_implicit`.
- `src/processes/surface/surface_energy_balance.jl`: the fused `solve_surface_energy_balance_kernel!`, which solves `Ts`, recomputes evapotranspiration, and then recomputes all fluxes at the final `Ts`; and the standalone `compute_surface_energy_fluxes!` entry point, which does not take `snow`.
- `src/processes/surface/radiative_fluxes.jl`: upwelling shortwave and longwave from the blended `albedo` and `emissivity` at the single `Ts`; `DiagnosedRadiativeFluxes` also has its own standalone `compute_auxiliary!` launch.
- `src/processes/surface/turbulent_fluxes.jl`: sensible heat at `Ts`; latent heat partitioned as `(1 − f)·H_l,ground + f·H_l,snow`, both at `Ts`; `DiagnosedTurbulentFluxes` also has its own standalone `compute_auxiliary!` launch.
- `src/processes/surface/ground_heat_flux.jl`: `compute_ground_heat_flux` for `ImplicitSkinTemperature` reads `fields.skin_temperature` directly.
- `src/processes/surface/albedo.jl`: `DiagnosticAlbedo` blends ground, vegetation, and snow albedo and emissivity by `f_snow` and `f_veg`, with the ground albedo depending on top-layer soil moisture.
- `src/processes/surface/evapotranspiration/*.jl`: bare-ground evaporation and canopy evapotranspiration declare `input(:skin_temperature)`, read it as the snow-free surface (and canopy) temperature, and scale their stored fluxes by `1 − f_snow` inside `compute_evapotranspiration_fluxes!`.
- `src/processes/surface/surface_hydrology.jl`: `compute_auxiliary!` passes `snow` to the evapotranspiration scheme for that scaling and runs before the surface energy balance in `LandModel`.
- `src/processes/snow/snow_interfaces.jl`: `compute_snow_surface_heat_flux` (`S`) and `compute_snow_interface_fluxes!`, which store `snow_surface_heat_flux = f·S`, `snow_basal_heat_flux = f·Q_base`, `soil_heat_flux = f·Q_base + (1 − f)·G`, and the sublimation `f·E_subl`.
- `src/processes/snow/mass/snow_mass.jl`: `compute_snow_sublimation_flux`, evaluated at `Ts` as a saturated surface.
- `src/processes/snow/energy/snow_energy.jl`: `dŪ_snow/dt = Q_base − Q_top + Q_precip + Q_subl`, with `Q_top = snow_surface_heat_flux`, gated by `W > 0`.
- `src/processes/snow/energy/snow_energy_closures.jl`: the `FreeWater` closure clips `snow_temperature` at 0 °C and keeps any energy above the fully melted reference in `snow_energy`.
- `src/models/coupled/land_model.jl`: `compute_auxiliary!` runs surface hydrology (evapotranspiration), then the albedo, then `solve_surface_energy_balance!`, then `compute_snow_interface_fluxes!`.
- `src/abstract_model.jl`: `variables(model)` maps the one-argument `variables` over `processes(model)`.
- `src/abstract_variables.jl` and `src/state_variables.jl`: variable `Namespace`s; namespaces are merged by name, a namespace of the state is itself a `StateVariables`, explicit timesteppers and the Heun cache recurse into namespaces, IMEX does not route namespaced variables, and input sources, the Rasters output, and the Reactant correctness test address namespaced paths.
- `src/diagnostics/surface_fluxes.jl`: `diagnose_skin_temperature_residual` and `diagnose_evaporation_flux` assume the single root solve and the `1 − f_snow` scaling.
- SpeedyWeather's `SpeedyWeatherTerrariumExt` copies `skin_temperature` (as the land surface temperature used for SpeedyWeather's own longwave), `sensible_heat_flux`, `latent_heat_flux / L_vap` (as the humidity flux), `snow_water_equivalent`, and `albedo`.

Established land models treat the problem in two ways.
Noah-MP [niuCommunityNoahLand2011](@cite) solves separate energy balances for its vegetated and bare fractions, each by Newton iteration, and then resets a snow-covered ground temperature above freezing to the freezing point and recomputes the fluxes at that temperature, with the resulting imbalance melting snow in the phase change step.
JULES [bestJointUKLand2011](@cite) solves one surface temperature per tile with a single linearized implicit increment, resets a snow or ice tile above the melting point to the melting point, and turns the excess energy over the step into melt, limited by the available snow.
CLM keeps one ground temperature per column with a thin top snow layer and resets layers above 0 °C to 0 °C in its phase change step.
Terrarium's single bulk snow layer cannot use the CLM approach without becoming multi-layer, so tiling with the Noah-MP sequence (solve, clamp, recompute the fluxes at the clamped temperature) is the smaller structural change.
Single-temperature alternatives were considered during review (a post-solve melt correction, and a pinned snow part inside the blended solve); both need the same per-tile flux evaluation, and both either violate energy conservation or need a three-way branch with an approximation in patchy snow, so tiling was retained.

## Summary of changes

### Physics

Each grid cell is split into a snow tile (area fraction `f = f_snow`) and a snow-free tile (`1 − f`), each with its own skin temperature and energy balance.
Both tiles share the atmospheric forcing and the aerodynamic resistance.
As in the current albedo blend, snow covers the vegetation, so vegetation and canopy evapotranspiration belong to the snow-free tile only.

Snow-free tile, skin temperature `T_sf`, solved as now but without the snow terms:

```math
R_{\text{net},sf}(T_{sf}) + H_s(T_{sf}) + H_{l,sf}(T_{sf}) = G(T_{sf}, T_g), \qquad G = \frac{2 \kappa_s (T_g - T_{sf})}{\Delta z_g},
```

with the snow-free albedo and emissivity (ground and vegetation blend) and the evapotranspiration latent flux.

Snow tile, skin temperature `T_sn`:

```math
R_{\text{net},sn}(T_{sn}) + H_s(T_{sn}) + H_{l,sn}(T_{sn}) = S(T_{sn}, T_{\text{snow}}), \qquad S = \frac{2 \kappa_{\text{snow}} (T_{\text{snow}} - T_{sn})}{d_{\text{snow}}},
```

with the snow albedo and emissivity and the sublimation latent flux.
The unconstrained root `T_sn*` is limited to the melting point, `T_sn = min(T_sn*, 0 °C)`, written with `ifelse`.
The energy delivered to the snowpack top is the snow tile's demand evaluated at the limited temperature,

```math
Q_{\text{top},sn} = R_{\text{net},sn}(T_{sn}) + H_s(T_{sn}) + H_{l,sn}(T_{sn}),
```

which equals the conductive flux `S(T_sn)` whenever the limit is inactive (that is the root condition), and otherwise includes the surface-melt surplus that conduction alone could not carry.
The surplus

```math
Q_{\text{melt}} = Q_{\text{top},sn} - S(T_{sn}) \le 0
```

is stored as a diagnostic.
`snow_surface_heat_flux` becomes `f·Q_top,sn`, so the existing snow energy tendency and `FreeWater` closure turn the surplus into liquid water and the existing drainage removes it.
Surface energy is conserved by construction: the atmosphere sees exactly the fluxes whose sum enters the snowpack.

Evapotranspiration is split the same way.
The snow-free tile evaporates and transpires at `T_sf` with the existing schemes, unscaled.
The snow tile sublimates at `T_sn` with `compute_snow_sublimation_flux`, unscaled.
The grid-cell-mean water fluxes consumed by the soil, the canopy interception, and the snow mass balance are the area-weighted sums applied afterwards, `(1 − f)·E_sf` and `f·E_subl`.

Grid-cell outputs are area-weighted tile sums:

- `surface_shortwave_up`, `surface_longwave_up`, `surface_net_radiation`, `sensible_heat_flux`, and `latent_heat_flux` as `f·X_sn + (1 − f)·X_sf`;
- `evaporation_ground`, `transpiration`, and `evaporation_canopy` as `(1 − f)·X_sf`, and `sublimation` as `f·E_subl`;
- `albedo` and `emissivity` keep their current blended meaning;
- `ground_heat_flux` is the snow-free tile's `G(T_sf)`, as now the per-unit-area flux into the bare ground, and `soil_heat_flux = f·Q_base + (1 − f)·G` is unchanged in form;
- a grid-cell radiative skin temperature `T_rad` from the aggregated longwave with the blended emissivity `ε`,
  ```math
  \varepsilon \sigma T_{\text{rad}}^4 = L_{\uparrow} - (1 - \varepsilon)\, L_{\downarrow},
  ```
  so that the coupled atmosphere, which only sees one surface temperature, emits the same longwave as Terrarium.

Without snow, or with a snow component and `f = 0`, the snow-free tile and the grid cell coincide and every output reduces to the base revision.

### Code

The tiles are variable namespaces of the existing processes, not new process types.
Which tile a kernel function serves is carried by the `out` and `fields` the launch passes (the root state or a tile namespace) and by dispatch on the snow argument: the snow-free tile calls the existing no-snow methods with `snow = nothing`, and the snow tile calls the methods taking `snow::AbstractSnow`, which change meaning from "area-weighted blend" to "over snow".

1. **Namespaces from a two-argument `variables`.**
   `variables(seb::SurfaceEnergyBalance, ::Nothing) = variables(seb)` keeps every snow-free configuration unchanged.
   `variables(seb::SurfaceEnergyBalance, snow::AbstractSnow)` returns the root variables plus `namespace(:snow_free, tile_vars)` and `namespace(:snow, tile_vars)`.
   `tile_vars` are the variables the surface energy balance *owns*: the prognostic `skin_temperature` (the solve target, persisted as the next step's initial guess) and the auxiliary fluxes, `albedo`, and `emissivity`.
   Inputs such as `ground_temperature` are excluded by a kind filter and stay at root; tile kernels receive them through `merge(root_fields, tile_fields)`, with tile fields shadowing on name collisions such as `albedo`.
   The root variables with snow are the aggregated auxiliaries listed under Physics, the root `skin_temperature` as an auxiliary holding `T_rad`, the root `ground_heat_flux`, and the diagnostic `snow_melt_energy_flux`.
   `variables(model::LandModel)` is overridden to call `variables(model.surface_energy_balance, model.snow)` and `variables(model.surface_hydrology, model.snow)` in place of the one-argument defaults.
   `variables(seb::PrescribedSurfaceEnergyBalance, snow::AbstractSnow)` returns the plain root set: with the coupler owning the fluxes there is nothing to tile, and the snow surface flux keeps its current conductive form.
2. **Snow methods become snow-tile methods.**
   `compute_skin_temperature(..., seb, constants, snow)` inverts the snow conduction only, `Ts = T_snow − G₀·d_snow/(2κ_snow)`, and must never be evaluated with the `nothing` placeholder of `snow_thermal_interface`, whose zero conductivity would divide by zero.
   `compute_latent_heat_flux(..., snow)` returns the sublimation latent flux alone.
   `compute_albedo(..., vegetation, snow, soil)` returns the snow albedo and emissivity alone; the no-snow method keeps the ground and vegetation blend for the snow-free tile.
   The `f_snow` weights leave these three functions and move to the aggregation.
   `compute_ground_heat_flux` for `ImplicitSkinTemperature` keeps reading `fields.skin_temperature`, which in the snow-free namespace is `T_sf`.
3. **Two tile kernels and one aggregation kernel.**
   The existing `solve_surface_energy_balance_kernel!` becomes the snow-free tile kernel: it solves `T_sf` with the no-snow methods, recomputes the evapotranspiration fluxes at `T_sf` into the snow-free namespace (unscaled, so the recompute no longer takes `snow`), and recomputes the tile fluxes.
   A new `solve_snow_tile_kernel!` solves `T_sn*` with the snow methods, applies `T_sn = min(T_sn*, 0)` with `ifelse`, recomputes the sublimation at `T_sn` into the snow namespace, recomputes the tile fluxes at `T_sn`, and stores the tile demand `Q_top,sn` and the conductive flux `S(T_sn)`.
   The snow tile solve is seeded from `min(T_sn_prev, 0)`; the residual has a derivative kink at exactly 0 °C where the saturation humidity switches between ice and water, with both branches agreeing in value.
   A new `aggregate_surface_tiles_kernel!` (launched with `snow`, the only place that needs `f_snow`) writes the root area-weighted fluxes, the root `evaporation_ground`, `transpiration`, `evaporation_canopy`, and `sublimation` means, `snow_surface_heat_flux = f·Q_top,sn`, `snow_melt_energy_flux = f·(Q_top,sn − S(T_sn))`, and `T_rad` into the root `skin_temperature`.
   `solve_surface_energy_balance!(state, grid, seb, constants, atmos, hydrology, snow::AbstractSnow)` launches the three kernels in that order; the `nothing` method launches the first kernel alone on the root state, as now.
4. **Standalone flux launches.**
   The standalone `compute_auxiliary!` launches of `DiagnosedRadiativeFluxes` and `DiagnosedTurbulentFluxes` and the top-level `compute_surface_energy_fluxes!` operate on the root state and are therefore valid only without snow; with snow they are restricted at dispatch so they cannot silently disagree with the tiled solve.
5. **Evapotranspiration per tile.**
   `variables(evtr::AbstractEvapotranspiration, snow::AbstractSnow)` moves the evapotranspiration auxiliaries and conductances into the `snow_free` namespace, declares `sublimation` in the `snow` namespace, and keeps the root grid-cell means under their current names, so the soil hydrology forcing, the canopy interception, and the snow mass balance keep reading what they read now.
   The evapotranspiration `input(:skin_temperature)` resolves inside the `snow_free` namespace to `T_sf`, so the evapotranspiration schemes need no code change to read the right temperature.
   `compute_auxiliary!` of `SurfaceHydrology` with snow launches the evapotranspiration kernel on the `snow_free` namespace with merged root fields and drops the `1 − f_snow` scaling from `compute_evapotranspiration_fluxes!`; the scaling happens once, in the aggregation.
   Without snow the evapotranspiration variables stay at root and the launch is unchanged.
6. **Snow interface kernel.**
   `compute_snow_interface_fluxes!` no longer computes the sublimation or the snow-top conductive flux; both come from the snow tile and the aggregation.
   It keeps `Q_base`, `snow_basal_heat_flux = f·Q_base`, and `soil_heat_flux = f·Q_base + (1 − f)·G`.
   `compute_snow_surface_heat_flux` stays as the kernel function used by the snow tile and the tests.
7. **Initialization and diagnostics.**
   `initialize!(state, grid, seb, snow)` seeds the snow-free tile from the ground temperature, the snow tile from `min(snow_temperature, 0 °C)`, and the root `skin_temperature` from the ground temperature; the `nothing` method is unchanged.
   `diagnose_skin_temperature_residual` and `diagnose_evaporation_flux` in `src/diagnostics/surface_fluxes.jl` take a namespace argument and report per tile; the evaporation diagnostic reads the root mean instead of scaling.
8. **Coupling.**
   The SpeedyWeather extension needs no change: it keeps copying the root `skin_temperature` (now `T_rad`), the aggregated turbulent fluxes, and the blended `albedo`.

All new code follows the kernel rules: `ifelse` for the limit, no throw paths, `NF`-typed constants, and no allocations.
The melting-point limit has a kink at 0 °C; Enzyme returns the one-sided derivative of the active branch, as for the existing phase-change closures.
The tile flux at the snowpack top is continuous through the limit, because `Q_top,sn = S(T_sn)` at the root whenever the limit is inactive.
Two Newton solves per cell roughly double the cost of the surface energy balance kernel; the benchmark suite is rerun once at the end to record the change.

### Field names (resolved)

The root `skin_temperature` keeps its documented meaning, the longwave emission temperature of the land surface, and holds `T_rad` with snow.
The tile skin temperatures live at `snow_free => skin_temperature` and `snow => skin_temperature`, so no field is renamed and outputs, diagnostics, and the coupling keep their meaning.
Without snow all three coincide and only the root field exists.

## Testing and verification

New and updated tests in `test/surface/` and `test/snow/`:

- **No-snow regression:** with `snow = nothing`, and with a snow component but zero SWE (`f = 0`), the root skin temperature and all root fluxes match the base revision to round-off, and the state carries the tile namespaces only in the second case.
- **Namespaces:** `variables(seb, snow)` declares no input inside a tile namespace, the merged tile fields shadow the root on `albedo` and `emissivity`, and the tile `skin_temperature` fields persist across steps as initial guesses.
- **Melting-point limit:** for any forcing with `f > 0`, `T_sn ≤ 0 °C`; when the limit is inactive, `Q_top,sn` equals the conductive flux `S(T_sn)` to solver tolerance and `snow_melt_energy_flux` is zero.
- **Energy closure:** per cell, $f\,(R_{\text{net}} + H_s + H_l)_{sn} + (1 - f)\,(R_{\text{net}} + H_s + H_l)_{sf} = f\,Q_{\text{top},sn} + (1 - f)\,G$, and the aggregated outputs equal the tile-weighted sums.
- **Water closure:** the root `evaporation_ground`, `transpiration`, `evaporation_canopy`, and `sublimation` equal the tile values weighted by `1 − f` and `f`, and the latent heat flux equals $\rho_w \left[(1 - f)\, L_{lv}\, E_{sf} + f\, L_{sg}\, E_{\text{subl}}\right]$.
- **Radiative temperature:** $\varepsilon \sigma T_{\text{rad}}^4 + (1 - \varepsilon)\, L_{\downarrow}$ reproduces the aggregated `surface_longwave_up`.
- **Patchy snow:** with `f ≈ 0.5`, dark wet ground, and cold dry air, the snow-free tile settles above 0 °C while the snow tile settles below it, with no melt; this is the regime where a shared temperature fails.
- **Column melt test:** the idealized column of the issue (15 °C air, 250 W/m² daily-mean shortwave, 0.3 m SWE, 30 days), committed under `scratch/` so the run is reproducible.
  The snow skin stays at or below 0 °C.
  The melt is measured as liquid production plus drainage, not as SWE loss alone, because the first melt is held below the capillary retention; its rate is bounded below by a threshold derived from the energy budget at 0 °C and documented in the test.
  The snowpack energy change over the run equals the time-integrated `f·Q_top,sn + f·Q_base` plus precipitation and sublimation terms.
- **Melt-out:** a thin pack under the same forcing melts out completely; the test records the energy carried by `snow_energy` above the fully melted reference at the step the pack vanishes and checks it against the documented handling (see Known limitations).
- **Cold conditions:** under sub-freezing forcing, the limit is inactive and results stay close to the base revision (only the tile split changes the fluxes).
- **Differentiability:** extend `test/differentiability/snow_model_diff.jl` (or `prescribed_seb_diff.jl`) with an Enzyme reverse-mode check through `solve_surface_energy_balance!` with snow, on both sides of the limit, checked against finite differences away from the kink.
  Run locally because it touches the differentiability tests.
- **Reactant:** the change touches kernels used under Reactant; leave `test/reactant/` to CI unless the Reactant tests fail there.
- **GPU:** run the snow and surface test files on a GPU node (the user submits the job).

Verification beyond the tests:

- Rerun the scratch column script with snow albedo 0.8 and 0.5 and record melt rates in the PR.
- Rerun the coupled `docs/joss` simulation for at least two years on the GPU and compare the September Northern Hemisphere snow-covered points and the SWE trend against `run_0002`.
- Run the full test suite, a draft doc build, and the CPU benchmark once to record the cost of the second solve.

## Documentation changes

- `docs/src/processes/surface_energy/skin_temperature.md`: the tiled energy balance, the melting-point limit, and the radiative temperature, with the equations above.
- `docs/src/processes/surface_energy/surface_energy_balance.md`: the tile namespaces, the aggregation of tile fluxes, which fields belong to which tile, and the restriction of the standalone flux launches to the no-snow case.
- `docs/src/processes/surface_energy/albedo.md`: the snow-tile and snow-free albedo and emissivity.
- `docs/src/processes/surface_hydrology/evapotranspiration.md`: the per-tile evaporation and the aggregation.
- `docs/src/processes/snow/snow_energy.md`: `Q_top` now includes the surface-melt surplus, and `snow_melt_energy_flux`.
- `docs/src/running/`: the namespaced output paths `snow_free => skin_temperature` and `snow => skin_temperature`.
- Docstrings of all changed kernel functions, the two-argument `variables` methods, and the new fields.
- `docs/src/references.bib`: entries for Noah-MP [niuCommunityNoahLand2011](@cite) and JULES [bestJointUKLand2011](@cite).

## Known limitations

- Both tiles share one aerodynamic resistance; snow and snow-free roughness lengths are not distinguished.
- Snow fully buries the vegetation in the snow tile; there is no canopy snow or snow under a canopy.
- The snow albedo stays constant (`ConstantSnowAlbedo`); albedo aging is separate future work and would further increase melt.
- The tile skin temperatures are diagnostic roots, not prognostic states, so a newly snow-covered area has no thermal memory of its own; this matches the current single skin temperature.
- Energy at melt-out: when the surplus over a step exceeds the pack's remaining latent content, `snow_energy` rises above the fully melted reference, the closure clips the temperature at 0 °C, drainage leaves the energy untouched, and it is discarded once the pack is gone.
  This is pre-existing behavior, but the plan makes it happen at every melt-out with larger fluxes.
  JULES avoids it by capping melt at the snow available in the step, which is a discrete-time rule Terrarium does not allow.
  The fractional cover damps it (`f` is small when the pack is thin), and the melt-out test quantifies it; routing the surplus of a fully liquid pack into `soil_heat_flux` is left to a follow-up.
- The prescribed surface energy balance does not tile; the coupler owns the fluxes and the snow surface flux keeps its conductive form.
- The standalone flux launches of the diagnosed radiative and turbulent fluxes are not available with snow.
- SpeedyWeather converts the aggregated latent heat flux to a humidity flux with the latent heat of vaporization only, so the sublimation part is slightly misconverted (about 13 %).
  Passing the water vapor mass flux instead belongs in the SpeedyWeather extension (to be recorded in `docs/dev/upstream`).
- Snow at temperatures above 0 °C remains possible inside the bulk pack only through the energy closure, as now; the limit applies to the surface.

## Future work

- Snow albedo aging and a lower albedo for melting snow.
- Routing the melt-out energy surplus into the soil.
- Separate roughness lengths per tile once the atmosphere scheme supports them.
- Canopy snow interception and a vegetation tile that can protrude through the snowpack.
- Cover tiles (bare, vegetated, or per plant functional type) as sibling or enclosing namespaces, with the snow split applied inside each; the tile weights then become `f_cover · f_snow|cover`.
- Pass tile-aware fluxes (water vapor mass flux, radiative temperature) explicitly to SpeedyWeather.

## Revision log

- Rev 1 (2026-11-26, Claude Fable 5.1, Claude Code): code design revised after review; physics unchanged.
  The tiles are namespaces created by a two-argument `variables(seb, snow)` of the single `SurfaceEnergyBalance`, not a new process type, a tile tag, or renamed fields.
  Tile behavior is selected by dispatch on `snow = nothing` (snow-free tile, existing methods) versus `snow::AbstractSnow` (snow tile), with the `f_snow` blends moved from the kernel functions into one aggregation kernel.
  Evapotranspiration is split per tile with the fractional blending applied afterwards.
  The field-naming decision is resolved (root `skin_temperature` keeps its meaning and holds `T_rad`).
  Added: the Noah-MP and JULES precedents, the restriction of the standalone flux launches and of the prescribed path, the solver seeding and the 0 °C kink, the stored tile demand and the `snow_melt_energy_flux` diagnostic, the melt-out energy limitation and test, the patchy-snow and water-closure tests, and the committed column script.

  > Review it and make suggestions for revisions.

  > Should snow tiling be incorporated at the same level and in the same way as land cover tiling? e.g. vegetated vs. bare

  > Could we use namespaces for the snow SEB? i.e. set up the SEB twice in two namespaces, snow vs. bare

  > One alternative scheme I can think of that would avoid tiling would be to add a snowmelt term to the SEB [...]

  > To keep the fix minimal, I think I would prefer the approach of keeping everything blended rather than tiled, but clamping the skin temperature for snow in a post-solve correction [...]. Would that work?

  > so it sounds like tiling is still the most consistent approach

  > I would rather keep the single `SurfaceEnergyBalance` type but have an additional `variables` dispatch that accepts the snow scheme as a second argument and creates the namespaces.

  > rather than create a singelton tag, couldn't you just have two dispatch paths, one with the `AbstractSnow` process as an argument and one without?

  > Ok write this up as revision 1. However, for ET, I would suggest trying to rewrite it so that it is also separated between snow-covered and snow-free with the fractional blending applied afterwards.
