# Spatially variable plant traits and plant functional types

> Status: **planned**. Initial draft awaiting review; no code has been changed.

Date of initial draft: 2026-11-11

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code

Reviewer:

Base revision: `f171fa2923fa3cecdf0673c0764ecb63d210ab59` (`main`, branch `bg/plant-traits`)

## Originating prompt

> Draft an initial implementation plan for implementing variable plant traits and PFTs https://github.com/NumericalEarth/Terrarium.jl/issues/244

Issue [#244](https://github.com/NumericalEarth/Terrarium.jl/issues/244) ("Add support for variable plant traits including PFTs"):

> Currently we only have a subset of plant-dependent vegetation parameters refactored into `PlantTraits` with no mechanism for allowing plant types to vary spatially. This issue consists of two parts:
>
> 1. Refactor all plant-dependent parameters in vegetation process types into `PlantTraits`
> 2. Rename `PlantTraits` to `ConstantPlantTraits` subtyping `AbstractPlantTraits`
> 3. Add `FunctionalPlantTraits <: AbstractPlantTraits` with each trait parameter replaced by either `SVector`s or a custom `AbstractParam` type following a particular `AbstractPlantFunctionalType` scheme. Initially support only the PALADYN PFT scheme or a subset thereof.

Discussion on the issue:

> **mahabadri:** I guess `ConstantPlantTraits` would be used for single-PFT simulations, right? If that's the case, what would be the difference to using `FunctionalPlantTraits` with only 1 PFT?
>
> **bgroenks96:** I guess there is no meaningful difference. We could also represent it that way, would probably be cleaner.

## Problem description

All vegetation in Terrarium currently behaves as a single plant functional type (PFT), which is in most places a needleleaf tree.
Plant-dependent parameters are split between `PlantTraits` (`src/processes/vegetation/plant_traits.jl`) and the individual process structs, where many are annotated "PFT specific" or "TODO PFT-specific parameter".
There is no way to let these parameters vary in space, so a global simulation applies needleleaf tree parameters to tropical forests, grasslands, and tundra alike.

## Background

### Current `PlantTraits`

`PlantTraits{NF}` is a `@parameterized` struct with seven scalar parameters: `specific_leaf_area`, `awl`, `minimum_leaf_area_index`, `maximum_leaf_area_index`, `extinction_coefficient`, `albedo`, and `emissivity`.
It is a component of both `PrescribedVegetation` and `PrognosticVegetation` (field `traits`).
A few pointwise accessors already take the kernel-function signature (`specific_leaf_area(i, j, grid, fields, traits)` and so on), but most call sites read fields directly (`traits.specific_leaf_area`, `veg.traits.albedo`) inside scalar helper functions such as `compute_APAR(photo, traits, swdown, LAI)` or `compute_balanced_leaf_area_index(carbon_dynamics, traits, C_veg)`.
Signatures throughout the vegetation processes are annotated `traits::PlantTraits{NF}`.

### PFT-dependent parameters outside `PlantTraits`

| Process (file) | Parameters marked or known to be PFT-dependent |
|---|---|
| `PALADYNCarbonDynamics` (`dynamics/carbon_dynamics.jl`) | `γL`, `γR`, `γS` (leaf, root, and stem turnover) |
| `PALADYNAutotrophicRespiration` (`respiration/autotrophic_respiration.jl`) | `aws`; possibly `cn_sapwood`, `cn_root`, and `resp10` (tree vs. grass) |
| `MedlynStomatalConductance` (`stomatal_conductance/medlyn_stomatal_conductance.jl`) | `g₁` |
| `LUEPhotosynthesis` (`photosynthesis/lue_photosynthesis.jl`) | `T_CO2_high`, `T_CO2_low`, `T_photos_high`, `T_photos_low`; `Kc25` and `Ko25` are documented as needleleaf values; possibly `leaf_respiration_fraction` |
| `PALADYNPhenology` (`phenology/paladyn_phenology.jl`) | `f_deciduous`, `gdd_crit`, and the other placeholders flagged by the "PFT-specific values from PALADYN Table 5" TODO |
| `StaticExponentialRootDistribution` (`hydraulics/root_distribution.jl`) | `a`, `b` |
| `PALADYNCanopyInterception` (`surface/canopy_interception/canopy_interception.jl`) | `α_int` ("for tree PFTs"), `k_ext` (a duplicate of `extinction_coefficient`, marked `# TODO: Duplicated`) |

Which of the "possibly" entries actually vary by PFT in PALADYN has to be confirmed against Willeit and Ganopolski (2016) and the PALADYN source; this is one of the questions for the reviewers below.

### PALADYN PFTs

PALADYN distinguishes five PFTs: broadleaf trees, needleleaf trees, C3 grass, C4 grass, and shrubs.
`LUEPhotosynthesis` implements only the C3 pathway, so C4 grass cannot be represented faithfully until a C4 photosynthesis scheme exists.

### Spatial representation of vegetation

Vegetation state variables are 2D (`Canopy(XY())`) and describe a single vegetated tile per grid cell; there is no PFT dimension.
`PALADYNVegetationDynamics` evolves one `vegetation_area_fraction` for that single PFT.
Full PALADYN-style multi-PFT co-existence (one prognostic carbon pool, LAI, and cover fraction per PFT, with competition) would need a PFT dimension on every vegetation state variable and is out of scope here.

### Parameter handling (checked on the base revision)

A quick experiment with `SpeedyWeatherInternals.ParameterEditing` showed:

- An `@param` whose value is an `NTuple{N, NF}` is flattened into `N` scalar parameters named `1`, …, `N`, and `reconstruct` rebuilds the tuple correctly.
- An `@param` whose value is a `NamedTuple` (e.g. keyed by PFT name) is **silently dropped** from the parameter table.
- `StaticArrays` is only an indirect dependency (it is in `Manifest.toml` but not in `[deps]`), so using `SVector` in `src/` would require changing the root `Project.toml`.

## Summary of changes

The work is split into three phases, each of which can be reviewed and merged as its own PR.

### Phase 1: collect all PFT-dependent parameters into the traits type

1. Move the parameters in the table above (after reviewer confirmation) from the process structs into the traits struct, with the current needleleaf tree values as defaults so that results are unchanged.
   Parameter names should be descriptive (`leaf_turnover_rate` rather than `γL`, etc.), with the symbol given in the docstring.
2. Pass `traits` to every vegetation process that needs it.
   Most already receive it; the phenology and root distribution coupling functions do not yet.
3. Leave `PALADYNCanopyInterception` unchanged in this phase (see the open questions), apart from a comment pointing to this plan.
4. Introduce a single pointwise resolution step: each kernel function calls

   ```julia
   local_traits = plant_traits(i, j, grid, fields, traits)
   ```

   once and passes the resulting scalar traits object to the existing scalar helpers (`compute_APAR`, `compute_λ_NPP`, …).
   For the constant case `plant_traits` simply returns `traits`.
   Kernel-level signatures are widened from `traits::PlantTraits{NF}` to `traits::AbstractPlantTraits{NF}`, while scalar helpers keep a concrete `PlantTraits{NF}` argument.
   This keeps the diff in the physics code small and confines all spatial variability to one method.
5. Replace the direct accesses `veg.traits.albedo` and `veg.traits.emissivity` in `vegetation_base.jl` with the pointwise path, which also resolves the `# TODO: will need to change once PFTs are added`.

### Phase 2: type hierarchy

1. Add `abstract type AbstractPlantTraits{NF} end`.
2. Make the existing `PlantTraits{NF}` a subtype of `AbstractPlantTraits{NF}`, keeping its name.
   This departs from the issue, which proposes renaming it to `ConstantPlantTraits`; keeping the name avoids a breaking rename for the common single-PFT case.
3. Widen the `traits` field types in `PrescribedVegetation` and `PrognosticVegetation` to a type parameter `Traits <: AbstractPlantTraits{NF}`.
4. Add `variables(::AbstractPlantTraits)` (empty for `PlantTraits`) and include it in the vegetation `variables`, so that trait-specific inputs are allocated automatically.

### Phase 3: `FunctionalPlantTraits` and the PALADYN scheme

1. Add an abstract PFT scheme type and a first implementation:

   ```julia
   abstract type AbstractPlantFunctionalTypes end
   struct PALADYNPlantFunctionalTypes <: AbstractPlantFunctionalTypes end
   pft_names(::PALADYNPlantFunctionalTypes) = (:broadleaf_tree, :needleleaf_tree, :c3_grass, :shrub)
   ```

   C4 grass is left out until C4 photosynthesis is available (see Known limitations).
2. Add

   ```julia
   struct FunctionalPlantTraits{NF, N, PFTs <: AbstractPlantFunctionalTypes} <: AbstractPlantTraits{NF}
       pfts::PFTs
       specific_leaf_area::NTuple{N, NF}
       # ... one NTuple{N, NF} per trait in PlantTraits
   end
   ```

   using `NTuple` rather than `SVector` to avoid adding `StaticArrays` to `[deps]`.
   A constructor `FunctionalPlantTraits(NF, PALADYNPlantFunctionalTypes(); overrides...)` fills in the PALADYN table values, and a `FunctionalPlantTraits(NF, scheme, traits::PlantTraits...)` method builds it from one `PlantTraits` per PFT.
3. `variables(::FunctionalPlantTraits)` declares one 2D input, `plant_functional_type` (`Canopy(XY())`), holding the 1-based PFT index of each grid cell, which can be set with the existing input and initializer machinery.
4. `plant_traits(i, j, grid, fields, traits::FunctionalPlantTraits)` reads the index and builds a `PlantTraits{NF}` by selecting each trait.
   Selection must satisfy the kernel and Reactant rules, so it uses an unrolled, branch-free reduction rather than dynamic tuple indexing or `round(Int, …)`:

   ```julia
   @inline select_pft(values::NTuple{N, NF}, pft::NF) where {N, NF} =
       sum(ntuple(k -> ifelse(pft == NF(k), values[k], zero(NF)), Val(N)))
   ```

   The index field stays in the model's number format, so no integer conversion (and thus no `InexactError` path) is needed.
5. Validate the PFT map on the host side, in `initialize!` for `FunctionalPlantTraits`: every value must be an integer in `1:N`, since an invalid value would silently select all-zero traits.
6. Extend `ParameterEditing.parameters` (or `@parameterized` metadata) for `FunctionalPlantTraits` so each per-PFT entry is labeled with its PFT name (e.g. `specific_leaf_area.needleleaf_tree`) instead of `1`, `2`, …, and carries the bounds of the corresponding scalar trait.
   If this turns out to be awkward, the fallback is a small `PFTParam <: AbstractParam` wrapper as suggested in the issue.
7. Export `AbstractPlantTraits`, `PlantTraits`, `FunctionalPlantTraits`, `AbstractPlantFunctionalTypes`, and `PALADYNPlantFunctionalTypes`.

### Single type or two? (response to the issue discussion)

The issue thread suggests representing a single PFT as `FunctionalPlantTraits` with `N = 1` rather than keeping a separate constant type.
This plan recommends keeping a separate constant `PlantTraits` type anyway, for two reasons.
First, it is exactly the pointwise object that `plant_traits` returns, so the scalar physics helpers need it regardless.
Second, a single-PFT `FunctionalPlantTraits` would require a `plant_functional_type` input field filled with ones, and would show parameters as `specific_leaf_area.needleleaf_tree` in calibration tables where a plain `specific_leaf_area` is clearer.
If the reviewer prefers one public type, `PlantTraits` can become internal and the default can be `FunctionalPlantTraits` with one PFT and no input field; this changes phase 3 only.

## Questions for the reviewers

1. Which of the "possibly PFT-dependent" parameters (`cn_sapwood`, `cn_root`, `resp10`, `Kc25`, `Ko25`, `leaf_respiration_fraction`, the phenology parameters, `ν_seed`, `γv_min`, and `C_can`) should move into the traits? (@mahabadri, @matteo-willeit)
2. Where should the PALADYN per-PFT values come from: the paper tables, the PALADYN code (and which version), or both?
3. Canopy interception lives in `SurfaceHydrology`, which can run without a vegetation component (`SurfaceHydrologyModel`).
   Should `α_int` and `k_ext` move into the traits (making interception depend on the vegetation's traits in `LandModel`), or stay in the interception process for now with only the duplicated `k_ext` resolved?
   The recommendation is to defer this to a follow-up.
4. Is a dominant-PFT index map sufficient for the first version, or is a sub-grid PFT cover fraction needed from the start?

## Testing and verification

- **Regression**: with default `PlantTraits`, all existing vegetation and coupled-model tests must pass unchanged, which checks that the phase 1 move preserves the current defaults.
- **Equivalence**: a `FunctionalPlantTraits` model whose PFT map is uniform must reproduce, to round-off, a `PlantTraits` model built from that PFT's values.
- **Spatial variability**: on a small grid with a mixed PFT map, each column must match the corresponding single-PFT column run.
- **Selection**: unit tests for `select_pft`, including that it is allocation-free and type-stable (`@inferred`, `@allocated`) for `Float32` and `Float64`.
- **Validation**: `initialize!` rejects non-integer and out-of-range PFT indices.
- **Parameters**: `parameters(FunctionalPlantTraits(...))` lists named per-PFT entries, and `reconstruct` round-trips.
- **Float32**: extend `test/vegetation/float32_tests.jl` to `FunctionalPlantTraits`.
- **Differentiability**: extend `test/differentiability/vegetation_model_diff.jl` so that Enzyme gradients with respect to a per-PFT trait (e.g. `specific_leaf_area` of one PFT) match central differences.
- **Reactant**: add a `FunctionalPlantTraits` configuration to the `test/reactant/` registry.
  Following the repository rules, this suite is run locally only because the change touches it; otherwise CI covers it.
- Run the full suite and a draft doc build (`julia --project=docs docs/make.jl --local --draft`).

## Documentation changes

- Add a "Plant traits and functional types" page under `docs/src/processes/vegetation/` with Overview, Implementations (`PlantTraits`, `FunctionalPlantTraits`, `PALADYNPlantFunctionalTypes`), Methods, and Kernel functions sections, following the doc page rules.
  It should include a table of the PALADYN per-PFT values with citations.
- Update the vegetation, carbon dynamics, photosynthesis, stomatal conductance, and autotrophic respiration pages where parameters moved, and update the `traits` description in `vegetation.md`.
- Add an `@example` showing a global or multi-column run with a PFT map.
- Add any new references (e.g. PALADYN table sources) to `docs/src/references.bib`.

## Known limitations

- One PFT per grid cell; no sub-grid co-existence or competition between PFTs.
- C4 grass is excluded until a C4 photosynthesis scheme is implemented.
- The PFT map is a static input; gradients with respect to it are undefined (it is discrete), though gradients with respect to every per-PFT trait value are available.
- Moving parameters out of the process structs renames some public keyword arguments (e.g. `PALADYNCarbonDynamics(NF; γL = …)`), which is a breaking change for user scripts.

## Future work

- Sub-grid PFT tiles with a PFT dimension on the vegetation state and PALADYN competition dynamics.
- C4 photosynthesis and the C4 grass PFT.
- Directly prescribed spatial trait maps (e.g. from the TRY database) as a third `AbstractPlantTraits` implementation, reading each trait from an input field.
- Additional PFT schemes (e.g. JULES or CLM), each as a new `AbstractPlantFunctionalTypes` subtype.
- Moving canopy interception parameters into the traits (see question 3).

## Revision log

No revisions yet.
