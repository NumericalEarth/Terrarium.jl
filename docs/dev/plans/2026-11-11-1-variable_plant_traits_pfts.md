# Spatially variable plant traits and plant functional types

> Status: **in progress**. Phases 1 and 2 are being implemented in this PR; phase 3 will be reviewed and approved separately in a follow-up PR.

Date of initial draft: 2026-11-11

Author: Claude Opus 5.5 (`claude-opus-5-5`), Claude Code

Reviewer: Brian Groenke (brian.groenke@pik-potsdam.de); phases 1 and 2 only

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

The work is split into three phases.
Phases 1 and 2 are implemented together in the first PR.
Phase 3 follows in a separate PR and needs its own approval before any code is changed.

### Phase 1: collect all PFT-dependent parameters into the traits type

1. Move the parameters in the table above that are marked PFT-specific in the code from the process structs into the traits struct, with the current defaults so that results are unchanged.
   The borderline parameters (reviewer question 1) and the root distribution parameters stay where they are for now (see revision 1).
   Parameter names should be descriptive (`leaf_turnover_rate` rather than `γL`, etc.), with the symbol given in the docstring.
2. Pass `traits` to every vegetation process that needs it.
   Most already receive it; the phenology coupling functions do not yet.
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

### Phase 3: `FunctionalPlantTraits` and plant functional types

> Not yet approved; to be reviewed and implemented in a follow-up PR.

PFTs are standalone types that can be mixed and matched, each holding a complete set of plant traits.
A "scheme" such as PALADYN is only a preset: a fixed, ordered set of PFTs with default values from one source.

1. Add an abstract PFT type and one concrete type per PFT, each wrapping its traits:

   ```julia
   abstract type AbstractPlantFunctionalType{NF} end

   struct NeedleleafTree{NF, Traits <: AbstractPlantTraits{NF}} <: AbstractPlantFunctionalType{NF}
       traits::Traits
   end
   ```

   The first set is `BroadleafTree`, `NeedleleafTree`, `C3Grass`, and `Shrub`.
   C4 grass is left out until C4 photosynthesis is available (see Known limitations).
   The `Traits` type parameter lets a PFT hold traits that themselves vary in space (e.g. field-valued parameters), which are localized as described in item 5.
2. Keep the provenance of default values separate from PFT identity.
   Each PFT's keyword constructor takes a defaults source, e.g. `NeedleleafTree(NF; defaults = PALADYNDefaults(), kwargs...)`, and looks up its values with `default_traits(::Type{<:NeedleleafTree}, ::PALADYNDefaults)`.
   Keyword arguments override individual traits.
   Other sources (JULES, CLM, user tables) can be added later as new defaults types without new PFT types.
3. Provide the PALADYN preset as a function rather than a type, since it must construct values:

   ```julia
   PALADYNPlantFunctionalTypes(NF; defaults = PALADYNDefaults()) =
       (BroadleafTree(NF; defaults), NeedleleafTree(NF; defaults), C3Grass(NF; defaults), Shrub(NF; defaults))
   ```

   The tuple order defines the meaning of the indices in the PFT map.
4. Add

   ```julia
   struct FunctionalPlantTraits{NF, PFTs <: Tuple{Vararg{AbstractPlantFunctionalType{NF}}}} <: AbstractPlantTraits{NF}
       pfts::PFTs
   end
   ```

   and declare one 2D input in `variables(::FunctionalPlantTraits)`, `plant_functional_type` (`Canopy(XY())`), holding the 1-based PFT index of each grid cell.
   The index field stays in the model's number format, so no integer conversion (and thus no `InexactError` path) is needed in kernels.
5. Extend `plant_traits` (from phase 1) to PFTs, so that `plant_traits(i, j, grid, fields, pft)` returns the same PFT type with its traits localized at `(i, j)`.
   For `PlantTraits` with scalar parameters this is the identity, as in phase 1.
6. Evaluate trait-dependent kernel functions through a single helper that chooses the evaluation strategy by dispatch:

   ```julia
   value = map_plant_traits(i, j, grid, fields, traits) do local_traits
       compute_gpp(photosynthesis, local_traits, ...)
   end
   ```

   - For `PlantTraits`, the closure is called once with `plant_traits(i, j, grid, fields, traits)`.
   - For `FunctionalPlantTraits` whose PFTs all localize to the same concrete type, the code path is identical for every PFT, so the helper selects the local traits of the cell's PFT and calls the closure once.
   - Otherwise, the helper evaluates the closure for every PFT in the tuple (an unrolled loop over a tuple whose types are known at compile time, so it stays type-stable) and selects the result of the cell's PFT.
     This is what allows process methods to dispatch on the PFT type, or on trait-level types inside it, within a kernel.

   Selection uses an unrolled `ifelse` fold over the tuple, `ifelse(pft_index == NF(k), value_k, accumulated)`, rather than dynamic tuple indexing, so there is no reachable throw path under Reactant.
   It works for numbers, tuples of numbers (for functions with several outputs), and isbits structs such as `PlantTraits`.
   Kernel functions migrate from the phase 1 `local_traits = plant_traits(...)` call to this form; the scalar helpers are unchanged.
   Closures passed to the helper must capture only isbits values, following the Reactant rule for closures in kernels.
7. Prefer dispatching on trait-level types (e.g. a photosynthetic pathway or a life form stored as a type parameter of `PlantTraits`) over dispatching on PFT names.
   A user-defined PFT then gets the correct physics from the composition of its traits, and PFTs that share trait types keep the single-evaluation path from item 6.
   No such trait-level types are added in this phase; they come with the first structural difference that needs them (e.g. C4 photosynthesis).
8. Validate the PFT map on the host side, in `initialize!` for `FunctionalPlantTraits`: every value must be an integer in `1:N`.
9. Extend `ParameterEditing.parameters` for `FunctionalPlantTraits` so that entries are grouped and named by PFT (e.g. `needleleaf_tree.specific_leaf_area`) rather than by tuple position.
   If the same PFT type appears more than once, names receive a numeric suffix.
10. Export `AbstractPlantFunctionalType`, the four PFT types, `PALADYNDefaults`, `PALADYNPlantFunctionalTypes`, and `FunctionalPlantTraits`.

### Design rationale

**PFTs as standalone types rather than members of a scheme.**
What a scheme such as PALADYN really provides is a set of parameter values calibrated together with its process formulations, a fixed ordering for land-cover crosswalks, and (later) competition rules between PFTs.
The first is captured by the defaults source (item 2), the second by the preset tuple (item 3), and the third belongs on the collection of PFTs once tiling exists.
None of these require PFT identity to be tied to a scheme, and mixing PFTs from different sources remains possible, though it is the user's responsibility to judge whether such a mix is meaningful.

**Each PFT holds a full `PlantTraits`, rather than each parameter holding one value per PFT.**
With one value per PFT per parameter there is no PFT object to dispatch on, only an index, so all PFTs would be restricted to the same code path.
Holding the traits inside each PFT keeps dispatch possible, groups parameters by PFT for calibration, and subsumes spatial trait maps: a single "PFT" whose traits are field-valued is localized by the same `plant_traits` mechanism.
It also extends naturally to sub-grid tiling, where the selection in item 6 is replaced by a cover-fraction-weighted sum over PFTs.

**Keeping a separate constant `PlantTraits` type.**
The issue thread suggests representing a single PFT as `FunctionalPlantTraits` with one PFT.
`PlantTraits` is kept because it is the pointwise object that the scalar physics helpers need regardless, and because a single-PFT `FunctionalPlantTraits` would need a `plant_functional_type` input field filled with ones and would show parameters as `needleleaf_tree.specific_leaf_area` where a plain `specific_leaf_area` is clearer.

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
- **Selection**: unit tests for `map_plant_traits` covering all three paths (constant traits, PFTs with a shared trait type, and PFTs with different types), including that each is allocation-free and type-stable (`@inferred`, `@allocated`) for `Float32` and `Float64`.
- **Path consistency**: for PFTs that differ only in parameter values, the single-evaluation path and the evaluate-all path must give identical results.
- **Finite results**: every pointwise function must return finite values for every PFT in the PALADYN preset under each test forcing, since the evaluate-all path computes PFTs that do not apply to a cell.
- **Validation**: `initialize!` rejects non-integer and out-of-range PFT indices.
- **Parameters**: `parameters(FunctionalPlantTraits(...))` lists entries grouped and named by PFT, and `reconstruct` round-trips.
- **Float32**: extend `test/vegetation/float32_tests.jl` to `FunctionalPlantTraits`.
- **Differentiability**: extend `test/differentiability/vegetation_model_diff.jl` so that Enzyme gradients with respect to a per-PFT trait (e.g. `specific_leaf_area` of one PFT) match central differences.
- **Reactant**: add a `FunctionalPlantTraits` configuration to the `test/reactant/` registry.
  Following the repository rules, this suite is run locally only because the change touches it; otherwise CI covers it.
- **Performance**: benchmark a global configuration with the PALADYN preset on both paths to quantify the cost of evaluating every PFT, and check the effect on compile time.
- Run the full suite and a draft doc build (`julia --project=docs docs/make.jl --local --draft`).

## Documentation changes

- Add a "Plant traits and functional types" page under `docs/src/processes/vegetation/` with Overview, Implementations (`PlantTraits`, `FunctionalPlantTraits`, the PFT types, and `PALADYNPlantFunctionalTypes`), Methods, and Kernel functions sections, following the doc page rules.
  It should include a table of the PALADYN per-PFT values with citations.
- Update the vegetation, carbon dynamics, photosynthesis, stomatal conductance, and autotrophic respiration pages where parameters moved, and update the `traits` description in `vegetation.md`.
- Add an `@example` showing a global or multi-column run with a PFT map.
- Add any new references (e.g. PALADYN table sources) to `docs/src/references.bib`.

## Known limitations

- One PFT per grid cell; no sub-grid co-existence or competition between PFTs.
- C4 grass is excluded until a C4 photosynthesis scheme is implemented.
- When PFTs use different code paths, every PFT is evaluated in every cell, so vegetation kernels cost about N times as much and take longer to compile.
- Pointwise functions must return finite values for any valid trait set, because results for PFTs that do not apply to a cell are still computed; a NaN there can contaminate reverse-mode gradients even though it never reaches the forward result.
- The PFT map is a static input; gradients with respect to it are undefined (it is discrete), though gradients with respect to every per-PFT trait value are available.
- Moving parameters out of the process structs renames some public keyword arguments (e.g. `PALADYNCarbonDynamics(NF; γL = …)`), which is a breaking change for user scripts.

## Future work

- Sub-grid PFT tiles with a PFT dimension on the vegetation state and PALADYN competition dynamics.
- C4 photosynthesis and the C4 grass PFT.
- Directly prescribed spatial trait maps (e.g. from the TRY database) as field-valued parameters inside `PlantTraits`, localized by `plant_traits`.
- Additional default sources (e.g. JULES or CLM) as new defaults types, and trait-level types for structural differences such as photosynthetic pathway.
- Moving canopy interception parameters into the traits (see question 3).

## Revision log

**Revision 1** (2026-11-23, Claude Opus 5.5 `claude-opus-5-5`), approved before implementation:

- Phase 1 defers the root distribution parameters `a` and `b`. The root fraction is a lazily evaluated `FunctionField` built from `StaticExponentialRootDistribution` alone, so moving them requires a separate redesign of that field.
- Phase 1 moves all PALADYN Table 5 phenology placeholders (`f_deciduous`, `T_gdd_base`, `gdd_crit`, and `T_senescence_range`) into the traits; `gdd_relaxation_time` stays in `PALADYNPhenology`, since it has no PALADYN counterpart.
- The other borderline parameters (reviewer question 1) stay in their process structs until the reviewers answer.

> How should phase 1 handle the root distribution parameters `a` and `b`? Defer.
> Which phenology parameters should move into PlantTraits in phase 1? All Table 5 placeholders.
