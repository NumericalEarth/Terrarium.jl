# Infiltration capacity and wetting-front conductivity for Richards soil hydrology

> Status: **in progress**. Plan approved; Phase 1 (infiltration capacity) is being implemented.

Date of initial draft: 2026-11-15

Author: Claude Opus 5.5, Claude Code (VS Code extension)

Reviewer: Brian Groenke (brian.groenke@pik-potsdam.de)

Base revision: cd885a538ddb90e10ff389dc34e93ee9749ead40

## Originating prompt

> In our Figure 3 for the JOSS paper, I am noticing that GPP seems to low over the amazon now that we have added Richards soil hydrology. Please investigate this issue.

> Create a new worktree and draft an implementation plan for the fix.

## Problem description

In the coupled Terrarium–SpeedyWeather simulation of the JOSS paper (`docs/joss/run_terrarium_speedy_simulation.jl`), switching the soil hydrology from `NoFlow` to `RichardsEq` dries out the land surface within weeks.
Over the Amazon (0–10°S, 295–310°E), the monthly means of the two runs (`run_0007` with `NoFlow`, `run_0008` with `RichardsEq`) are:

| | rain (mm/day) | ET (mm/day) | surface runoff (mm/day) | skin temperature |
| --- | --- | --- | --- | --- |
| `NoFlow`, month 1 | 4.3 | 2.8 | 3.8 | 26 °C |
| `RichardsEq`, month 1 | 1.9 | 1.3 | 1.6 | 32 °C |
| `RichardsEq`, month 12 | 0.26 | 0.06 | 0.22 | 39 °C |

With `RichardsEq`, about 85% of rainfall leaves as surface runoff from the first month on, so that only about 0.3 mm/day enters the soil.
Evapotranspiration, and with it gross primary production, collapses, and rainfall follows through the loss of moisture recycling.
Rainfall over the tropical Pacific is unchanged between the runs (3–4 mm/day), so the atmosphere is not the cause.
The same drying occurs over all land, which is why global mean ET in Figure 3 falls from about 3 to 0.5 mm/day and GPP from about 4 to 0.3 g C m⁻² day⁻¹.

## Background

### Infiltration is capped at the unsaturated conductivity

`compute_surface_runoff!` in `src/processes/surface/runoff/direct_surface_runoff.jl` limits infiltration to

```
I = min(P, K_face(Nz)) · (S_top < 1)
```

where `P` is the water reaching the ground and `K_face(Nz)` is the `hydraulic_conductivity` field at face `Nz`, which `compute_hydraulics!` (`src/processes/soil/hydrology/soil_hydrology_rre.jl`) sets to the minimum of the unsaturated conductivities of the two topmost cells.

With `UnsatKLinear` (`K = K_sat S`), this cap was close to `K_sat` and rarely binding.
Since the dry-soil fix of `docs/dev/plans/2026-11-06-1-dry_soil_richards_robustness.md`, the default is the consistent Brooks–Corey conductivity `K = K_sat S^(3 + 2/λ) = K_sat S¹³` (for `λ = 0.2`).
With `K_sat = 10⁻⁵ m/s` (864 mm/day) and the initial saturation of about 0.55 in the JOSS run, the cap drops from about 475 mm/day to `864 · 0.55¹³ ≈ 0.36 mm/day`, which matches the infiltration diagnosed from the run.
This produces a runaway feedback: drier soil has a lower conductivity, so it takes up less rain and dries further.

The cap is also physically inverted.
The infiltration capacity of a soil is at least `K_sat`: the matric suction gradient at a wetting front raises it above `K_sat` in dry soil, and it decays toward `K_sat` as the soil wets up (Green and Ampt 1911; Philip 1957).
Land surface models therefore bound infiltration by the saturated conductivity of the surface layer, reduced by an ice impedance factor for frozen soil; CLM5, for example, uses `q_infl,max = (1 − f_sat) Θ_ice k_sat` (Lawrence et al. 2018, Section 2.7.2.3, Eq. 2.7.34), with the ice impedance `Θ_ice = 10^(−Ω F_ice)` (Eq. 2.7.48).
The unsaturated conductivity of the current state is not the relevant bound.

### Face conductivity blocks wetting fronts

Even with a capacity of `K_sat`, the water has to move from the top cell into the drier cells below.
The Darcy flux uses two nested minima:

1. `compute_hydraulics!` stores at each face the minimum of the cell-centered conductivities of the two adjacent cells;
2. `darcy_conductivity` (`soil_hydrology_rre.jl`) then takes the minimum of two of those face values in the direction of flow.

The flux through a face is therefore limited by the driest of three cells, including one cell ahead of the wetting front.
For a saturated top cell over a cell at `S = 0.5`, the face conductivity is `K_sat · 0.5¹³ ≈ 1.2×10⁻⁹ m/s`, and with a head difference of about 0.3 m over a 3 cm spacing the downward flux is about 1 mm/day.
The top cell saturates within one rain event and rejects all further rain through the `S_top < 1` gate, so most of the rain still runs off.
For comparison, the conductivity at the mean saturation of the pair (`S̄ = 0.75`, as in CLM5) is about 200 times larger, and the flux is about 200 mm/day.

The 2026-11-06 plan kept the minimum deliberately, since it shuts off the flux out of a dry cell, which is what made the dry-soil regression test (issue #240) pass.
Any change here must keep that test passing.

### `NoFlow`

Under `NoFlow`, `compute_hydraulics!` is a no-op and the `hydraulic_conductivity` field stays zero, so infiltration is zero and all rain runs off.
This is consistent with the design of `NoFlow` (immobile, non-depleting soil water) and is kept as is.

### Water accounting

If an implicit step fills the top cell beyond saturation, `redistribute_saturation_profile!` routes the excess into the `surface_excess_water` pool, which drains with timescale `τ_r` back into infiltration or runoff.
A larger infiltration capacity is therefore water-conserving by construction.

## Summary of changes

### Phase 1: Infiltration capacity from the saturated conductivity

The infiltration capacity `I_max = K_sat · I_ice(f_top)` is declared as a lazily evaluated auxiliary variable, a `KernelFunctionOperation` built once at state initialization, rather than as a stored `Field` filled by its own kernel launch.
It is evaluated on the fly wherever the runoff kernel reads it, so it always reflects the current liquid water fraction and needs neither storage nor a `compute_auxiliary!` step.

- Declare `auxiliary(:infiltration_capacity, Ground(XY()), ...; units = u"m/s")` in `variables(::SoilHydrology{NF, RichardsEq})`, with a constructor that builds
  `KernelFunctionOperation{Center, Center, Nothing}(compute_infiltration_capacity, ground_grid, (; liquid_water_fraction), hydrology)`.
  - The constructor is a plain `Function` constructor (`auxiliary(name, loc, ctor, params)`), so that the operation captures only `liquid_water_fraction`.
    `kernel(compute_infiltration_capacity, hydrology)` would also work, but `KernelFunction` captures every field created before it, which bloats the operation's type and the GPU kernel arguments of every kernel that reads it.
  - `liquid_water_fraction` is an input variable and is therefore initialized before all auxiliary variables (see `initialize` in `src/state_variables.jl`).
- Add the kernel function `compute_infiltration_capacity(i, j, k, grid, fields, hydrology)`, which ignores `k` and evaluates the top cell (`Nz` of the ground grid):
  `K_sat = saturated_hydraulic_conductivity(props)` and `I_ice = ice_impedance_factor(props.unsat_hydraulic_cond, f_top)`.
  - Add `ice_impedance_factor(unsat_k::AbstractUnsatK, f)` methods: `10^(−Ω(1 − f))` for `UnsatKBrooksCorey` and `UnsatKVanGenuchten` (the existing `ice_impedance_factor(Ω, f)`), and `f` for `UnsatKLinear`, which matches `K_linear` at full saturation.
    Brooks–Corey and van Genuchten then share the ice impedance of the Darcy flux, so frozen soil keeps limiting infiltration consistently.
  - `K_sat` is taken without texture.
    Both existing implementations (`SoilHydraulicsSURFEX`, `ConstantSoilHydraulics`) ignore texture; texture lives in the stratigraphy namespaces, which are initialized after auxiliary variables and so cannot be captured (see Known limitations).
- Add the accessor `infiltration_capacity(i, j, grid, fields, hydrology)`, returning `fields.infiltration_capacity[i, j, 1]` for `RichardsEq` and zero for `NoFlow`, which preserves the current `NoFlow` behavior without declaring the variable there.
- In `compute_surface_runoff!`, replace `k_unsat` by `infiltration_capacity(...)`.
  `compute_infiltration` keeps its signature and the `S_top < 1` gate, and no extra arguments (`strat`, `bgc`) need to be passed to the runoff kernel.
- Using the top cell rather than the face field makes the capacity independent of the face weighting of Phase 2.
- Make `Base.fill!` and `Base.copyto!` on `StateVariables` skip auxiliary entries that are not `Field`s, by dispatch as `reset_field!` already does; both currently assume that every auxiliary entry is a `Field`.

This is the first process to use a lazily evaluated auxiliary variable, and thus the pilot conversion described under Future work in `docs/dev/plans/2026-11-14-1-kernel_operation_diagnostics.md` (worktree `kernel-operation-diagnostics`, not yet merged).
Its verification (single-step Enzyme adjoint and a Reactant run) is part of the testing below.
- Correct the docstring of `compute_infiltration` and the doc page, which describe the gate as a factor `(1 − f_sat)` while the code uses an indicator.

### Phase 2: Face conductivity at the mean saturation

- Replace the nested minima by a single face conductivity, evaluated at the thickness-weighted mean saturation and liquid fraction of the two adjacent cells, as in CLM5 (Lawrence et al. 2018, Section 2.7.3.1, Eq. 2.7.47: `k = Θ_ice k_sat [0.5(θ_i + θ_i+1) / 0.5(θ_sat,i + θ_sat,i+1)]^(2B + 3)`).
  `compute_hydraulics!` computes it once per face; `darcy_conductivity` uses the face value directly and drops the second minimum.
- For two dry cells the mean saturation is still small, so `K(S̄)` stays negligible (about 10⁻²² m/s at `S̄ = 0.05`), which should keep the dry-soil behavior of issue #240.
  For a wet cell over a dry one, `K(S̄)` lets the wetting front advance.
- Fallback, if the dry-soil or Celia et al. (1990) tests fail with the mean saturation: upstream weighting (conductivity of the cell the water flows out of), the standard choice for infiltration into dry soil (Forsyth et al. 1995).
  Switching to the fallback is a revision of this plan and needs approval before implementation.

### Phase 3: Rerun the JOSS simulation

- After merging into `main`, merge `main` into `joss-paper` and rerun `docs/joss/run_terrarium_speedy_simulation.jl` as a Slurm job (submitted by the user) and regenerate Figure 3.
- Plausibility checks (not pass/fail): Amazon ET of about 3 mm/day, surface runoff well below rainfall, and GPP closer to the `NoFlow` run.
- Separately, check on `joss-paper` whether the script's `vwc_source` climatology actually reaches `SaturationWaterTable`, which currently receives a constant `VolumetricWaterContent(0.25f0)`.
  This is a script issue, not part of the source change.

## Testing and verification

- New `LandModel` column regression test `LandModel: rain infiltrates into unsaturated soil` (written first, and checked to fail on the base revision): SURFEX hydraulics, steady rain of 10 mm/day for one day, `Δt = 450 s`, default `IMEX` time stepper, from a dry (initial saturation 0.1) and a dryish (0.3) column.
  - at least 95% of the rain infiltrates;
  - the column water balance (soil water, surface pool, rain, and surface runoff) closes to a relative tolerance of 10⁻³;
  - the rain is absorbed: the top cell stays unsaturated and the surface pool stays empty.
- Third case of the same test with rain above `K_sat` (100 mm/hour for one hour, initial saturation 0.3): infiltration-excess runoff occurs and infiltration is bounded by `K_sat`.
- Phase 2 regression test: a wet column (initial saturation about 0.6) under sustained heavy rain for several days, in which the nested minimum lets the top cell saturate and the rain run off, while the face conductivity of Phase 2 lets the water drain into the column.
  The rain rate and duration are chosen from the drainage capacities of the two face conductivities, and the test is checked to fail before Phase 2.
- Unit tests for the infiltration capacity: `state.infiltration_capacity` is a `KernelFunctionOperation`; it equals `K_sat` for an unfrozen top cell at any saturation, is reduced by `I_ice` for a frozen top cell, follows changes of `liquid_water_fraction` without any `compute_auxiliary!` call, and `infiltration_capacity` is zero for `NoFlow`; indexing it is type-stable (`@inferred`) for `Float32`.
- `fill!` and `copyto!` of a `StateVariables` holding the lazy variable run without error and leave the operation untouched.
- Lazy-variable pilot:
  - a single-step Enzyme adjoint of a `LandModel` column with `RichardsEq` and `DirectSurfaceRunoff`, added to `test/differentiability/` and checked against finite differences for the gradient of the column water with respect to rainfall;
  - the Reactant suite (`test/reactant/`) run locally once, since it compiles a `LandModel` whose state now holds a `KernelFunctionOperation`;
  - a GPU run of the new regression test, to check that the operation adapts to the device.
- Unit test for the face conductivity: equals `K(S̄)` and is symmetric in the two cells.
- Existing tests must keep passing, in particular:
  - `LandModel: dry soil under strong evaporation demand stays physical` (issue #240);
  - `Implicit Richards: infiltration into a dry column (Celia et al., 1990)` and the mass conservation tests in `test/soil/soil_hydrology_implicit_tests.jl`;
  - `test/surface/hydrology/surface_runoff_tests.jl` and the snowmelt and surface pool tests in `test/coupled_models/land_model_tests.jl`.
- Full test suite, then a draft doc build.
- Enzyme and Reactant: both new kernel functions are branch-free scalar functions without throw paths; the hydrology and land model differentiability tests run in CI in addition to the pilot checks above.
  Replacing the nested `min` by a smooth function of the mean saturation removes a nondifferentiable point.

## Documentation changes

- `docs/src/processes/surface_hydrology/surface_runoff.md`: rewrite "Infiltration capacity and soil saturation" with the saturated-conductivity capacity, the ice impedance, and references; add `compute_infiltration_capacity` under kernel functions.
- `docs/src/processes/soil/soil_hydrology.md`: list `infiltration_capacity` among the `RichardsEq` variables and note that it is lazily evaluated, so it cannot be written to output or accessed with `interior`.
- The general documentation of lazy auxiliary variables (`kernel` docstring, `docs/src/extending/state_variables.md`) is left to the kernel-operation-diagnostics plan.
- `docs/src/processes/soil/soil_hydrology.md`: document the face conductivity at the mean saturation.
- Docstrings for the new kernel function and the changed `darcy_conductivity` and `compute_hydraulics!`.
- Add Green and Ampt (1911), Philip (1957), and Lawrence et al. (2018) to `docs/src/references.bib`.

## Known limitations

- A capacity of `K_sat` neglects the sorptivity of dry soil (Green–Ampt, Philip), so it underestimates infiltration at the start of a storm on dry soil; this is conservative and adequate at the coupling time steps used here.
- Processes that reduce the infiltration of dry soil in reality, soil water repellency after drought or fire (Doerr et al. 2000) and surface sealing or crusting of bare soil (Assouline 2004), are not represented.
  Under the Richards equation alone, dry soil infiltrates faster rather than slower; the current cap reproduces a similar effect only as a discretization artifact, uniformly for all soils.
  After this change, infiltration-excess runoff occurs only when the water input exceeds `K_sat · I_ice`, which at grid scales of about 2° is rare, so flash flooding on dry or burned soils is not captured.
- There is no sub-grid variability of `K_sat` or of the saturated fraction (TOPMODEL, Arno), so saturation-excess runoff from partially saturated grid cells is not represented.
- `SoilHydraulicsSURFEX` uses a constant `K_sat` rather than the exponential decay with depth of SURFEX (see the existing TODO).
- The lazy `infiltration_capacity` cannot see the stratigraphy namespaces, which are initialized after auxiliary variables, so a texture-dependent `K_sat` would need either a change of the initialization order or a stored `Field`.
- Being a `KernelFunctionOperation`, `infiltration_capacity` cannot be written to output, read with `interior`, or checked with `checkfinite!`; the stored `infiltration` flux remains available for diagnostics.
- All `RichardsEq` simulations with a `LandModel` change: more water enters the soil and the wetting fronts are faster.

## Future work

- A Green–Ampt or Philip infiltration capacity, or a ponded pressure-head top boundary condition for the Richards equation.
- An explicit parameterization of soil water repellency and surface crusting, for example a factor that reduces the surface `K_sat` below a critical water content, or a crust layer with its own conductivity, grounded in the literature and kept separate from the unsaturated conductivity function.
  At coarse resolution, this would need a sub-grid distribution of rainfall intensity to have a meaningful effect.
- A sub-grid saturated fraction for saturation-excess runoff.
- Depth-dependent `K_sat` in `SoilHydraulicsSURFEX`.
- Texture-dependent `K_sat` (as in ISBA, Noilhan and Lacarrère 1995) for the lazy `infiltration_capacity`, by initializing the variable namespaces before the auxiliary variables in `StateVariables` (`src/state_variables.jl`) and passing their fields to auxiliary constructors as nested NamedTuples.
  The operation could then capture `get_fields(state, strat, bgc)` and evaluate the same texture-dependent `K_sat` as the Darcy flux.
  This needs the context dictionary of `initialize`, currently an `OrderedDict{Symbol, AbstractField}`, to accept namespaces, and it lifts the ordering restriction for all lazily evaluated auxiliary variables.
  Until then, an implementation that defines only the texture-dependent `saturated_hydraulic_conductivity(props, texture)` fails with a `MethodError` when the kernel is compiled, so the missing texture cannot be ignored silently.

## References

All entries were checked against the publisher or the official documentation on 2026-11-15.

- Assouline, S. (2004). Rainfall-induced soil surface sealing: a critical review of observations, conceptual models, and solutions. *Vadose Zone Journal* 3, 570–591. https://doi.org/10.2136/vzj2004.0570
- Doerr, S. H., Shakesby, R. A., and Walsh, R. P. D. (2000). Soil water repellency: its causes, characteristics and hydro-geomorphological significance. *Earth-Science Reviews* 51, 33–65. https://doi.org/10.1016/S0012-8252(00)00011-8
- Forsyth, P. A., Wu, Y. S., and Pruess, K. (1995). Robust numerical methods for saturated-unsaturated flow with dry initial conditions in heterogeneous media. *Advances in Water Resources* 18, 25–38.
- Green, W. H., and Ampt, G. A. (1911). Studies on soil physics. *The Journal of Agricultural Science* 4, 1–24. https://doi.org/10.1017/S0021859600001441
- Lawrence, D., Fisher, R., Koven, C., Oleson, K., Swenson, S., and Vertenstein, M. (2018). *Technical Description of Version 5.0 of the Community Land Model (CLM)*. NCAR. Chapter 2.7, Hydrology: https://escomp.github.io/CTSM/release-clm5.0/tech_note/Hydrology/CLM50_Tech_Note_Hydrology.html (infiltration capacity: Section 2.7.2.3, Eq. 2.7.34; Hortonian runoff: Eq. 2.7.35; interface conductivity: Section 2.7.3.1, Eq. 2.7.47; ice impedance: Eq. 2.7.48).
- Lawrence, D. M., et al. (2019). The Community Land Model Version 5: Description of new features, benchmarking, and impact of forcing uncertainty. *Journal of Advances in Modeling Earth Systems* 11, 4245–4287. https://doi.org/10.1029/2018MS001583 (journal overview of CLM5; the equations above are in the technical note).
- Noilhan, J., and Lacarrère, P. (1995). GCM grid-scale evaporation from mesoscale modeling. *Journal of Climate* 8, 206–223.
- Philip, J. R. (1957). The theory of infiltration: 4. Sorptivity and algebraic infiltration equations. *Soil Science* 84, 257–264.

## Revision log

> **Revision 1, 2026-11-15 (Claude Opus 5.5).** Initial draft.

> **Revision 2, 2026-11-15 (Claude Opus 5.5).** Added soil water repellency and surface crusting, the real-world causes of reduced infiltration into dry soil, as a known limitation and as future work.

> **Revision 3, 2026-11-15 (Claude Opus 5.5).** At the reviewer's request, declared the infiltration capacity `K_sat · I_ice` as a lazily evaluated auxiliary variable (a `KernelFunctionOperation` capturing only `liquid_water_fraction`) instead of computing it in the runoff kernel from the full soil composition.
> Added `ice_impedance_factor` methods per conductivity formulation, the `fill!`/`copyto!` fix for non-`Field` auxiliaries, the lazy-variable pilot checks (Enzyme, Reactant, GPU), and the resulting limitations (no texture-dependent `K_sat`, no output of the variable).

> **Revision 4, 2026-11-15 (Claude Opus 5.5).** Added a future work item for texture-dependent `K_sat` in the lazy `infiltration_capacity`, through initializing variable namespaces before auxiliary variables.

> **Revision 5, 2026-11-16 (Claude Opus 5.5).** Phase 1 showed that the wetting-front assertion of the regression test was not physical: 10 mm of rain fills only about 3 cm of soil from a saturation of 0.3, so it cannot reach below the top three cells, and the case cannot distinguish the face conductivities of Phase 2.
> Replaced the assertion by "the top cell stays unsaturated and the surface pool stays empty", renamed the test to "rain infiltrates into unsaturated soil" with an added dry case (initial saturation 0.1), and gave Phase 2 its own test with a wet column under sustained heavy rain.
> Added a References section with the sources checked against the publisher or official documentation, and cited the CLM5 technical note (with equation numbers) instead of the CLM5 journal paper for the CLM equations.

> **Revision 6, 2026-11-16 (Claude Opus 5.5).** At the reviewer's request, added the diagnostic `diagnose_total_ground_water(state, model)` in `src/diagnostics/ground_state.jl` (total water and ice in the soil column plus the surface pool), which replaces the column water balance helper duplicated across the `LandModel` regression tests.
> Changed `compute_infiltration_capacity` to the `(i, j, grid, fields, ...)` signature of 2D kernel functions, which Oceananigans supports for operations at reduced locations.
> Replaced the arbitrary nighttime forcing of the infiltration regression test with a physically consistent overcast, rainy day.
