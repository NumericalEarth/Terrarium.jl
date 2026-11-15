# Differentiability of `LandModel` with vegetation

> Status: **planned**. Awaiting review; no code has been changed.

Date of initial draft: 2026-11-15

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code (VS Code extension)

Reviewer:

Base revision: cd885a538ddb90e10ff389dc34e93ee9749ead40 (`main`)

## Originating prompt

> Create a new worktree with an implementation plan for investigating the error further and adding differentiability tests for `LandModel` with both dynamic and prescribed vegetation.

## Problem description

The reverse-mode adjoint of a `LandModel` time step does not compile when the model includes vegetation.
It came up in the JOSS paper's coupled SpeedyWeather–Terrarium sensitivity script (`docs/joss/run_coupled_sensitivity.jl` on `joss-paper`), which works with `vegetation = nothing` but fails after switching to `PrescribedVegetation`.
The failure reproduces with Terrarium alone, without SpeedyWeather:

```
ERROR: LoadError: IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.
 This usually indicates the use of a Union type, which is not fully supported with Enzyme.API.strictAliasing set to true [the default].
Failure within method:
adapt_structure(::CPU, ::@NamedTuple{albedo::Field{Center, Center, Nothing, …}, …})
   @ Adapt ~/.julia/packages/Adapt/hnGPi/src/base.jl:5
```

The adapted `NamedTuple` holds 36 fields, including `canopy_evaporation_conductance`, `transpiration_conductance`, `canopy_water_interception`, `saturation_canopy_water`, `leaf_area_index`, `stem_area_index`, `ground_evaporation_conductance`, and the surface energy balance and atmospheric input fields.
These are the arguments of the canopy evapotranspiration launch (`src/processes/surface/evapotranspiration/canopy_evapotranspiration.jl:126–127`).
None of the field types in the printed signature contains a `Union`, so the hint in the error message may be misleading.
With `Enzyme.API.strictAliasing!(false)`, the same adjoint was still compiling when it was killed after 40 minutes.

No test differentiates a `LandModel` with vegetation.
`test/differentiability/vegetation_model_diff.jl` covers `PrescribedVegetation` only through a standalone `VegetationModel`, and every `LandModel` adjoint test (`prescribed_seb_diff.jl`, `snow_model_diff.jl`) sets `vegetation = nothing`.
The default `LandModel` uses `PrognosticVegetation`, so the default land model has no adjoint coverage at all.

## Background

### Reproducer

Julia 1.10.12 (`+lts`), Enzyme 0.13.219, Adapt 4.7.4, Oceananigans 0.117.2, KernelAbstractions 0.9.46, CPU.
The failure took about 20 minutes of compilation to appear.

```julia
using Terrarium, Enzyme, Dates
import RingGrids
Enzyme.Compiler.RunAttributor[] = false
NF = Float32
grid = ColumnRingGrid(CPU(), NF, ExponentialSpacing(; N = 10, Δz_min = 0.05), RingGrids.OctahedralGaussianGrid(8))
soil = SoilEnergyWaterCarbon(NF; strat = HomogeneousSoilStratigraphy(NF))
initializer = SoilInitializer(NF; energy = QuasiThermalSteadyState(NF; T₀ = 15), hydrology = SaturationWaterTable(NF))
vegetation = PrescribedVegetation(NF)
model = LandModel(grid; soil, vegetation, snow = nothing, initializer, timestepper = ForwardEuler(NF; Δt = 270))
lai = RingGrids.Field(fill(NF(3), RingGrids.get_npoints(grid.rings)), grid.rings)
inputs = InputSources(InputSource(grid, lai; name = :leaf_area_index, domain = Terrarium.Canopy()))
function latent_heat_flux_after_step!(integrator, Δt)
    timestep!(integrator, Δt)
    return interior(integrator.state.latent_heat_flux)[1, 1, 1]
end
integrator = initialize(model; inputs)
dintegrator = make_zero(integrator)
autodiff(set_runtime_activity(Reverse), latent_heat_flux_after_step!, Active, Duplicated(integrator, dintegrator), Const(NF(270)))
```

The same script with `vegetation = nothing` compiles in about 3 minutes and gives a finite, nonzero gradient.
In the coupled model with `vegetation = nothing`, the adjoint agrees with finite differences to within 1.8 %.

### Related findings

- `docs/dev/notes/2026-09-24_NOTE_enzyme_landmodel_compile_time.md` records an earlier Enzyme failure in the same place: Oceananigans' CPU kernel-argument adaption (`convert_to_device(::CPU, args) = Adapt.adapt(CPU(), args)`) rebuilt the mutable `Clock` as a `NamedTuple`, and Enzyme failed with a `NoShadowError` in `fill_halo`.
  The note also recommends a `LandModel` `timestep!` case in `test/differentiability`, which was never added.
- The plan `2026-11-06-2-cpu_kernel_argument_adaption.md` (worktree `.worktrees/cpu-kernel-argument-adaption`, not implemented) shows that the size of kernel argument aggregates drives compile time superlinearly, and that vegetation enlarges exactly the canopy evapotranspiration launch.
  It proposes reducing every `Field` argument on CPU to its data array in `launch_kernel!`, as Oceananigans already does on GPU.
  That change would alter the code path that fails here.
- The canopy evapotranspiration kernel function rebinds its `fields` argument with `fields = merge(fields, conductances)` (`canopy_evapotranspiration.jl:298`) and indexes `fields.ground_evaporation_conductance[i, j]` with two indices (line 208).
  Both are conditions of a separate Julia 1.12 Enzyme failure (`cannot handle unknown binary operator: shl`) reproduced in a standalone MWE.
  They may be unrelated to this failure on Julia 1.10, but they are cheap to rule out.

## Summary of changes

The work has two phases.
Phase 1 is an investigation that ends in a revision of this plan with a concrete fix, which needs approval before any source change.
Phase 2 adds the tests and can start in parallel, with the failing cases marked `@test_broken` until the fix lands.

### Phase 1: locate the failure

All probes go in `scratch/landmodel_vegetation_ad/` (not tracked), run on Julia 1.10 with output written to log files.
Each probe that compiles the adjoint of a full time step should be run with a hard time limit (`timeout -s KILL`) of 60 minutes.

1. **Get the full error.**
   Rerun the reproducer with `Enzyme.Compiler.VERBOSE_ERRORS[] = true` set at the top of the script, catch the exception, and save `code_typed(err)` and the LLVM value that Enzyme rejects.
   In the original run the flag did not change the output, so check that it is set before the first `autodiff` call.
2. **Confirm the launch site.**
   Match the 36 field names to the `get_fields(state, evapotranspiration, interception, atmos, soil, vegetation; except = out)` call in `canopy_evapotranspiration.jl:126`, and check that no other launch with vegetation passes the same set.
3. **Reduce to one launch.**
   Differentiate only the canopy evapotranspiration `compute_auxiliary!` with `Duplicated(state, dstate)`, following the pattern of the field-based tests in `soil_hydrology_diff.jl`.
   If this reproduces the error, every following probe uses it, since it should compile in minutes rather than 20.
4. **Bisect the cause** with the reduced probe, one change at a time:
   - pass only the fields the kernel functions read, instead of the full `get_fields` result;
   - remove the `fields = merge(fields, conductances)` rebinding, by passing the conductances as a separate argument;
   - replace the two-index reads with three-index reads;
   - skip the CPU adaption, either with a temporary `Adapt.adapt_structure(::CPU, ::Field)` that returns the field's data array (as the CPU kernel-argument adaption plan proposes) or by launching through KernelAbstractions directly;
   - switch to `PrognosticVegetation` and to `Float64`.
5. **Decide on the fix** and record it as a revision of this plan.
   Possible outcomes, depending on what step 4 shows:
   - implementing the CPU kernel-argument adaption plan, possibly together with this one;
   - narrowing the arguments of the canopy evapotranspiration launch;
   - restructuring the kernel function (no rebinding, three-index reads);
   - if the failure is in Enzyme or Adapt and no reasonable Terrarium-side change avoids it, an upstream MWE drafted with the `draft-mwe` skill.

### Phase 2: tests

Add `test/differentiability/land_model_diff.jl` and include it in the Enzyme block of `test/runtests.jl`.
It follows the structure of `prescribed_seb_diff.jl` and `vegetation_model_diff.jl`: a builder function, a scalar objective after one `timestep!`, a reverse-mode adjoint with `set_runtime_activity(Reverse)` and `Duplicated(integrator, dintegrator)`, and a central finite-difference reference.

- **Grid and configuration:** a `ColumnGrid` with a few soil layers (`ExponentialSpacing(N = 5)`), `Float64`, `snow = nothing`, `ForwardEuler`, and a prescribed atmosphere set with `set!` (air temperature, humidity, pressure, wind speed, radiation, and precipitation) chosen so that both evaporation and transpiration are active.
  The default `IMEX` time stepper is left to a later test, to keep the compile time of this file down.
- **`PrescribedVegetation`:** the objective is the mean `latent_heat_flux` after one step, differentiated with respect to `leaf_area_index` and the soil `saturation_water_ice`.
  Check that the gradients are finite, that ∂LE/∂LAI is positive (more leaf area, more transpiration), and that both agree with central finite differences (rtol about 1e-4, as in `vegetation_model_diff.jl`).
- **`PrognosticVegetation`:** the same objective, differentiated with respect to `saturation_water_ice` and one prognostic vegetation variable, with the same checks.
  Which vegetation variable to use is decided while writing the test, by picking one that affects the latent heat flux within a single step.
- **Before the fix:** the adjoint calls that fail are wrapped so that the testsets record `@test_broken` instead of erroring, so the file can be merged first and the fix flips them to passing.
  If the failure takes about 20 minutes to appear, it may be better to keep the failing cases out of CI until the fix exists, and add them in the same PR as the fix.
  This choice should be confirmed by the reviewer.

## Testing and verification

- Run the Enzyme suite locally, `Pkg.test(; test_args = ["enzyme"])`, since the differentiability tests are changed.
  Record the compile time of each new testset, so that the cost added to CI is known.
- If Phase 1 leads to a source change, also run the full test suite and the draft doc build (`julia --project=docs docs/make.jl --local --draft`).
- For the fix: rerun the reproducer above and the coupled `run_coupled_sensitivity.jl` with `PrescribedVegetation`, and check the adjoint against the finite-difference check in that script.

## Documentation changes

- Update `docs/dev/notes/2026-09-24_NOTE_enzyme_landmodel_compile_time.md` with a short entry pointing to this plan, since it is a new failure in the same code path.
- No user-facing documentation changes, unless the fix changes how kernels receive their arguments (in which case the documentation change described in the CPU kernel-argument adaption plan applies).

## Known limitations

- The reproducer takes about 20 minutes per run, so Phase 1 depends on step 3 producing a faster reduced probe.
- `snow = nothing` in all new tests.
  The snow model's adjoint is covered separately in `snow_model_diff.jl`, and including it here would raise the compile time further.
- The tests cover one time step with `ForwardEuler` on CPU only.

## Future work

- A `LandModel` adjoint test with the default `IMEX` time stepper.
- Re-enabling vegetation in the JOSS coupled sensitivity script once the fix is in.
- A coupled SpeedyWeather–Terrarium adjoint test with vegetation in SpeedyWeather's differentiability tests (the current one, added with the `soil_moisture_capacity!` fix, uses a `LandModel` with `vegetation = nothing`).

## Revision log
