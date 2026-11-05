# PR-A: ground/substrate abstraction (soil → ground)

> Status: **completed**. The medium-agnostic energy machinery was lifted into a top-level
> `thermodynamics/` module and reused by soil without behavior change; the final naming diverged from
> the original `soil → ground` proposal (see Revision log).

Date of initial draft: 2026-07-10

Base revision: 462c6422b22829b88e491457894264419704d691

## Originating prompt

> Draft separate design documents for PR-A and PR-B (see design doc for single-layer snow scheme in
> `scratch/design/`). Regarding the structure for PR-A, I still think the `soil` folder should
> probably be renmaed to `ground`. It's fine if there are still some soil-specific types in the ground
> folder since soil is ground. We can keep the existing process types but then they should subtype e.g.
> `AbstractGroundEnergyBalance` which should then have generic method implementations where
> appropriate. Alternatively we could consider a slightly more radical restructuring where the
> top-level folders correspond to process groups: e.g. energy, hydrology, carbon, vegetation, and soil,
> where soil here represents the truly soil-specific code like soil properties and stratigraphy. Please
> give feedback on this and include it in the design doc for PR-A. Make sure to include my original
> prompt in that document.

## Revision log

Naming and structure evolved during implementation:

- The energy-balance supertype was renamed across iterations (`AbstractGroundEnergyBalance` →
  `AbstractInternalEnergyBalance` → `AbstractEnergyBalance` → `AbstractThermodynamics`), and the
  concrete `SoilEnergyBalance` → `SoilThermodynamics`.
- The speculative `AbstractGround` coupled-process supertype was dropped (single subtype, no dispatch);
  `AbstractSoil` subtypes `AbstractCoupledProcesses` directly and the `soil/` directory was kept.
- The generic framework was lifted into a top-level `thermodynamics/` module (rather than `heat/` or a
  `ground/energy/` subtree). Merging `surface_energy/` + `surface_hydrology/` into `surface/` and the
  vegetation subfolder reorganization landed alongside as related structural cleanup.

## Purpose

Much of the soil **energy** machinery is medium-agnostic, and the single-layer snow scheme reuses it:
the 1D heat-conduction operator (`ExplicitTwoPhaseHeatConduction`), the `FreeWater` enthalpy maps
(`energy_to_temperature` / `liquid_water_fraction`), and the conductive-coupling interface
(`ground_temperature`). PR-A introduces an explicit ground/substrate abstraction so this machinery
has a medium-agnostic home that both soil and snow subtype, without a behavior change. PR-A is a pure
refactor: no new physics, no numerical change, verified by the existing test suite passing unchanged.

## Feedback on the two proposed structures

Two structures were proposed in the prompt. Both are larger than the "additive, no-rename" step
recommended in the snow design doc. The analysis below evaluates them against the actual coupling in
the current source.

### What is actually medium-agnostic vs. soil-specific

Verified against the current source:

- **Medium-agnostic (reused verbatim by snow):**
  - the scalar enthalpy maps `liquid_water_fraction(::FreeWater, U, Lθ, sat)` and
    `energy_to_temperature(::FreeWater, U, Lθ, C)`
    ([`soil_energy_closures.jl:135-163`](../../src/processes/soil/energy/soil_energy_closures.jl#L135-L163)) —
    these reference no soil types at all;
  - the Fourier divergence skeleton of the conduction operator
    (`compute_energy_tendency` / `diffusive_heat_flux` for `ExplicitTwoPhaseHeatConduction`,
    [`soil_energy.jl:111-149`](../../src/processes/soil/energy/soil_energy.jl#L111-L149)) — generic
    once the conductivity is supplied through a per-medium hook;
  - the `ground_temperature` conductive-coupling accessor consumed by the SEB.

- **Intrinsically soil-specific (stays as the soil *parameterization* of a ground medium):**
  - the field-level closure kernels `energy_to_temperature!` / `temperature_to_energy!`, which build a
    `SoilVolume` from porosity and saturation
    ([`soil_energy_closures.jl:64-130`](../../src/processes/soil/energy/soil_energy_closures.jl#L64-L130));
  - `compute_thermal_conductivity(i, j, k, …)`, which constructs `soil_composition(…)`
    ([`soil_energy.jl:128-137`](../../src/processes/soil/energy/soil_energy.jl#L128-L137));
  - all of stratigraphy, Richards hydrology and hydraulic closures, soil thermal *constituent*
    properties, and soil biogeochemistry.

The boundary is therefore *within* the energy subsystem: the **operator and the scalar enthalpy maps**
are ground-level; the **conductivity and the field-level closure** are soil-level hooks. Snow supplies
its own conductivity and its own field-level closure while reusing the operator skeleton and the scalar
maps. This is exactly what the snow doc's "Enthalpy closure (reusing `FreeWater`)" section requires.

### Option 1 — rename `soil/` → `ground/`, with a shared `AbstractInternalEnergyBalance` (recommended)

This is the author's leaning and it aligns cleanly with the boundary above.

- Rename the directory `processes/soil/` → `processes/ground/`.
- Introduce a separate **energy-balance** supertype `AbstractInternalEnergyBalance{NF} <:
  AbstractProcess{NF}` with `AbstractSoilEnergyBalance{NF} <: AbstractInternalEnergyBalance{NF}`. See
  "Naming of the energy-balance supertype" below for why this is *not* named `AbstractGround…`.
- Hang the medium-agnostic methods on the `AbstractInternalEnergyBalance` supertype; keep the
  soil-specific dispatches on `SoilEnergyBalance` / `AbstractSoil`.
- Soil-specific types keep their `Soil*` names inside the `ground/` tree (as the prompt notes, "soil is
  ground"); stratigraphy, hydrology, and biogeochem subtrees are unaffected except for the directory
  move.
- **No `AbstractGround` coupled-process supertype is introduced.** `AbstractSoil` subtypes
  `AbstractCoupledProcesses` directly. A shared coupled-medium supertype would have exactly one subtype
  and no dispatch today, so it is deferred until a second subsurface-earth medium (e.g. bedrock) with
  genuinely shared coupled-process behavior actually exists. The `ground/` directory name carries the
  grouping intent without needing a type, and the energy machinery snow reuses lives on
  `AbstractInternalEnergyBalance`, independent of any such supertype.

Assessment: moderate, well-bounded churn (a directory move, a thin new abstract layer, import/export
and doc-path updates). It honestly names the subsurface-medium machinery and gives a clean home for a
future multi-layer snow column (`MultiLayerSnowEnergyBalance <: AbstractInternalEnergyBalance`).
**Recommended.**

#### Naming of the energy-balance supertype

The shared energy abstraction is deliberately **not** named `AbstractGround…`. Two observations drove
this (resolved with the author):

1. **Snow is not "ground."** At the component level the snow scheme is its own coupled process
   (`AbstractSnow <: AbstractCoupledProcesses`), a sibling of soil — not a ground medium. So nothing
   claims "snow is ground."
2. **The energy-balance supertype is shared by media that aren't ground.** A future *multi-layer* snow
   column would legitimately reuse the same vertically resolved energy balance. Naming that supertype
   after the conserved quantity it evolves — the `internal_energy` prognostic — rather than after a
   medium ("ground"), a geometry ("column"), or a transport mechanism ("conductive") keeps it correct
   under later generalization (3D fluxes, advective–diffusive transport via a pluggable
   `AbstractHeatOperator`). It also parallels the existing `AbstractSurfaceEnergyBalance` (surface skin)
   vs. `AbstractInternalEnergyBalance` (medium internal energy).

Note the *single-layer* snow scheme is still **not** an `AbstractInternalEnergyBalance`: its
depth-integrated prognostic gives a flux-sum tendency rather than a transport divergence, so it reuses
only the medium-agnostic enthalpy maps. The distinction is the discretization of the energy, not the
medium.

### Option 2 — top-level folders by process group (energy, hydrology, carbon, vegetation, soil)

Assessment: **not recommended now.** Reasons:

1. **It cuts across a more fundamental axis than it serves.** The current top-level split is by
   *medium/component* (soil, surface_energy, surface_hydrology, vegetation, atmosphere), which tracks
   the surface (0D, `XY`) vs. subsurface (vertically resolved, `XYZ`) distinction. A process-group
   split would put the skin-temperature surface energy balance and the 1D soil heat column both under
   `energy/`, conflating two structurally different things. The same tension appears for hydrology
   (Richards column vs. surface hydrology) and carbon (soil biogeochem vs. vegetation carbon).
2. **It fragments tightly-coupled code.** `SoilEnergyWaterCarbon`
   ([`soil_coupled.jl`](../../src/processes/soil/soil_coupled.jl)) couples energy + water + carbon for
   soil and is constructed, initialized, and tested as a unit; a process-group split scatters its parts
   across three top-level folders.
3. **It does not serve snow better than Option 1.** Snow's only structural need is to reuse the energy
   operator and the enthalpy maps. Option 1 already exposes those. The additional hydrology/carbon/
   vegetation reorganization in Option 2 is orthogonal to the snow prerequisite and is pure scope creep
   relative to it (cf. `AGENTS.md`, "scope creep in PRs").
4. **It is reversible later, but the rename is a strict subset.** Option 1 does not preclude a future
   move to process-group folders if a concrete need arises; Option 2 cannot be partially adopted
   without the wide churn.

There is a legitimate kernel in Option 2: the *shared energy primitives* genuinely are process-domain
code that both soil and snow use. The recommendation below captures that kernel by giving the shared
primitives a clearly-named home in a **top-level `processes/heat/` directory** — rather than burying
the generic physics under a medium's tree — without the full cross-cutting reorganization (the surface
process trees are left in place).

### Recommendation

Adopt **Option 1** (rename `soil/` → `ground/`; no `AbstractGround` coupled-process supertype), and
lift the medium-agnostic primitives (operator + tendency skeleton, scalar enthalpy maps, and the
`AbstractInternalEnergyBalance` / `AbstractInternalEnergyClosure` framework types) into a top-level
`processes/heat/` directory so they dispatch on `AbstractInternalEnergyBalance` and reference no soil
types. Soil keeps only its *implementation* (`SoilEnergyBalance` and the soil closure) under
`ground/energy/`. This satisfies the author's stated preference, unlocks snow reuse, and keeps the diff
a behavior-neutral set of moves. Defer Option 2's process-group reorganization (including any
`surface/` merge) indefinitely.

## Target structure

Before:

```
src/processes/soil/
├── abstract_types.jl              # AbstractSoil; get_stratigraphy/energy/hydrology/biogeochemistry
├── soil_coupled.jl                # SoilEnergyWaterCarbon
├── energy/
│   ├── abstract_types.jl          # AbstractSoilEnergyBalance, AbstractHeatOperator,
│   │                              #   AbstractSoilEnergyClosure, AbstractBulkWeighting
│   ├── soil_energy.jl             # SoilEnergyBalance, ExplicitTwoPhaseHeatConduction, ground_temperature
│   ├── soil_energy_closures.jl    # SoilEnergyTemperatureClosure + FreeWater scalar maps
│   └── soil_thermal_properties.jl
├── hydrology/        (Richards + hydraulic closures/properties)
├── stratigraphy/     (texture, porosity, horizon, volume)
└── biogeochem/       (constant soil carbon)
```

After. The medium-agnostic energy framework is lifted to a **top-level `processes/heat/` directory**, so
the generic physics no longer lives under a medium's tree (`ground/`). Soil keeps only its
*implementation* of an internal energy balance.

```
src/processes/heat/                # NEW top-level: medium-agnostic energy transport framework
├── abstract_types.jl              # AbstractInternalEnergyBalance{NF}; AbstractHeatOperator;
│                                  #   AbstractInternalEnergyClosure; get_heat_operator,
│                                  #   compute_energy_tendency, compute_thermal_conductivity (interface)
├── energy_tendency.jl             # ExplicitTwoPhaseHeatConduction operator + generic
│                                  #   compute_energy_tendency / diffusive_heat_flux skeleton
│                                  #   (conductivity via the per-medium compute_thermal_conductivity hook)
└── enthalpy.jl                    # FreeWater scalar maps energy_to_temperature / liquid_water_fraction

src/processes/ground/
├── abstract_types.jl              # AbstractSoil{NF} <: AbstractCoupledProcesses{NF};
│                                  #   soil accessors (get_stratigraphy/energy/hydrology/biogeochemistry,
│                                  #   ground_temperature interface)
├── soil_coupled.jl                # SoilEnergyWaterCarbon (unchanged)
├── energy/
│   ├── abstract_types.jl          # AbstractSoilEnergyBalance{NF} <: AbstractInternalEnergyBalance{NF};
│   │                              #   get_thermal_properties; AbstractBulkWeighting (soil-specific)
│   ├── soil_energy.jl             # SoilEnergyBalance: soil conductivity (soil_composition) + soil hooks
│   ├── soil_energy_closures.jl    # SoilEnergyTemperatureClosure field-level closure kernels (SoilVolume)
│   └── soil_thermal_properties.jl
├── hydrology/        (unchanged; soil-specific, names kept)
├── stratigraphy/     (unchanged; soil-specific, names kept)
└── biogeochem/       (unchanged; soil-specific, names kept)
```

`energy_tendency.jl` and `enthalpy.jl` are extracted from the original `soil_energy.jl` /
`soil_energy_closures.jl`; the extraction is a code move, not a rewrite. The surface process trees
(`surface_energy/`, `surface_hydrology/`) are left unchanged in this PR.

## Abstract-type and method changes

### New supertypes

```julia
# AbstractSoil subtypes AbstractCoupledProcesses directly; no AbstractGround layer (see Option 1).
abstract type AbstractSoil{NF} <: AbstractCoupledProcesses{NF} end

# In processes/heat/ (medium-agnostic framework):
abstract type AbstractInternalEnergyBalance{NF} <: AbstractProcess{NF} end
abstract type AbstractInternalEnergyClosure <: AbstractClosureRelation end

# In processes/ground/energy/ (soil implementation). No intermediate AbstractSoilEnergyClosure layer:
# SoilEnergyTemperatureClosure subtypes AbstractInternalEnergyClosure directly.
abstract type AbstractSoilEnergyBalance{NF} <: AbstractInternalEnergyBalance{NF} end
```

### Methods on `AbstractInternalEnergyBalance` (generic implementations)

- the conduction-operator tendency skeleton (`compute_energy_tendencies!` and the
  `diffusive_heat_flux` divergence for `ExplicitTwoPhaseHeatConduction`), with `compute_thermal_conductivity`
  remaining an abstract hook each medium dispatches;
- the scalar enthalpy maps in `enthalpy.jl` (already type-generic; no dispatch change needed beyond
  their new location and a `using`/`include` update).

### Methods kept soil-specific (dispatch on `SoilEnergyBalance` / `AbstractSoil`)

- `compute_thermal_conductivity(i, j, k, grid, fields, energy::SoilEnergyBalance, …)` building a
  `SoilVolume`;
- the field-level closure kernels `energy_to_temperature!` / `temperature_to_energy!`;
- `ground_temperature` field constructor returning the view of the uppermost soil layer
  ([`soil_energy.jl:51-57`](../../src/processes/soil/energy/soil_energy.jl#L51-L57)) — the *accessor
  interface* is ground-level, the soil *implementation* stays.

## Naming policy (what gets renamed vs. kept)

- **New / renamed:** the directory `soil/` → `ground/`; the new abstract supertypes
  (`AbstractInternalEnergyBalance`, `AbstractInternalEnergyClosure`); the extracted primitive files.
- **Kept as `Soil`:** all public concrete types and constructors — `SoilModel`, `SoilEnergyWaterCarbon`,
  `SoilEnergyBalance`, `SoilEnergyTemperatureClosure`, `SoilThermalProperties`, `SoilHydrology`,
  `SoilStratigraphy`, `SoilTexture`, `SoilVolume`, etc. No public API or export name changes.
- **Out of scope:** `models/soil/` (`SoilModel` and its BC/init helpers) is *not* renamed in PR-A;
  renaming public model types is wider API churn with no snow-prerequisite benefit. The snow model
  lands in a new `models/snow/` tree regardless. Revisit a `models/soil` → `models/ground` move only if
  a concrete need appears.

Because exported names are unchanged, user scripts, examples, and docs that rely on `using Terrarium`
are unaffected.

## Migration steps

1. **Move the directory** `src/processes/soil/` → `src/processes/ground/` (git mv to preserve history).
2. **Lift the framework** into a new top-level `src/processes/heat/`: `heat/energy_tendency.jl`
   (operator + generic divergence skeleton) and `heat/enthalpy.jl` (scalar `FreeWater` maps), moving the
   corresponding code out of `soil_energy.jl` / `soil_energy_closures.jl`.
3. **Add the abstract layer:** `heat/abstract_types.jl` holds the framework types
   (`AbstractInternalEnergyBalance`, `AbstractHeatOperator`, `AbstractInternalEnergyClosure`) and the
   `get_heat_operator` / `compute_energy_tendency` / `compute_thermal_conductivity` interface;
   `ground/energy/abstract_types.jl` keeps the soil-specific `AbstractSoilEnergyBalance`,
   `get_thermal_properties`, and `AbstractBulkWeighting`.
4. **Update includes/exports** in [`processes.jl`](../../src/processes/processes.jl): add
   `include("heat/abstract_types.jl")` to the abstract-types block (before `ground/`), add a `# Heat`
   section including `heat/energy_tendency.jl` and `heat/enthalpy.jl`, and repoint the `# Ground` block
   to `ground/…`. Export the new abstract supertypes if downstream (snow) code needs them.
5. **Fix imports** flagged by the ExplicitImports tests; update any explicit `using`/`import` of moved
   names within `src/`.
6. **Update doc paths** under `docs/src/processes/soil/…` and any `@autodocs`/`include` references to
   the moved files; update `AGENTS.md` only if the module-structure listing references `soil/`
   explicitly (it currently uses a generic `processes/` entry, so likely no change).
7. **Run the full suite and doctests**; confirm zero behavior change.

## Testing

- **No behavior change:** the existing soil energy, hydrology, and coupled `LandModel`/`SoilModel`
  tests must pass **unchanged**; PR-A introduces no new numerics.
- **ExplicitImports:** the source-import test must remain green after the moves (common failure mode for
  this kind of refactor — see `AGENTS.md` pitfall 2/4).
- **Doctests:** any moved docstrings with `jldoctest` blocks still build.
- **Differentiability:** no Enzyme changes expected; the existing `test/differentiability` suite guards
  against accidental type-stability regressions from the dispatch-layer changes.

## Risks and mitigations

- **Import breakage** from the directory move and extraction — mitigated by the ExplicitImports test and
  by keeping the extraction a pure code move.
- **Over-abstraction:** resist generalizing hydrology/stratigraphy/biogeochem in this PR; they stay
  soil-specific. The abstraction is confined to the energy subsystem (`AbstractInternalEnergyBalance`).
  No `AbstractGround` coupled-process supertype is introduced — a single-subtype, zero-dispatch layer
  would be speculative; it is deferred until a second subsurface medium genuinely needs it.
- **Scope creep:** the `models/soil` rename and any process-group reorganization (Option 2) are
  explicitly excluded.

## Relationship to PR-B and the snow scheme

- **PR-B** ([`pr_b_surface_ground_heat_flux_split.md`](pr_b_surface_ground_heat_flux_split.md)) splits
  the dual-role `ground_heat_flux` into `surface_heat_flux` (SEB closure flux) and `ground_heat_flux`
  (soil-top BC). It is largely independent of PR-A; the snow doc suggests **PR-A then PR-B** because
  PR-B's `ground_heat_flux` naming and the shared conductive-coupling interface read more naturally on
  top of the ground abstraction. Either order is workable.
- The **snow scheme** depends on both. The single-layer scheme reuses only the medium-agnostic
  `enthalpy.jl` scalar maps introduced here; it does *not* subtype `AbstractInternalEnergyBalance`,
  since its depth-integrated energy is a flux-sum balance with no transport divergence. A future
  multi-layer snow column would subtype `AbstractInternalEnergyBalance` and additionally reuse the
  conduction operator skeleton.
