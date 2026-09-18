# Oceananigans AbstractGrid Integration for Terrarium

> Status: **in progress**. Phases 1-4 are implemented (Phase 3 without the `VarDomain` work, see Revision 6): `ColumnGrid` is now a type alias for a `RectilinearGrid` and `ColumnRingGrid` an ordinary `AbstractGrid`, a single `LandGrid <: AbstractLandGrid` wraps the per-domain (ground, snow, canopy) grids, and model constructors build it from a user-supplied `AbstractGrid` via `create_land_grid`. The remaining work is the `VarDomain` tagging that a vertically resolved snow or canopy domain will need.

Date of initial draft: 2026-08-25

Base revision: 4f3841955af84ac6cb4a6f88b04538fc0dd7d658

## Originating prompt

> Please review the Oceananigans `AbstractGrid` interface and grid implementations. Then review the current Terrarium `AbstractLandGrid`s and draft a plan for how to integrate the two interfaces. The general idea is that `AbstractLandGrid` should be a valid subtype of `AbstractGrid`, but implementations of `AbstractLandGrid` should allow for separate vertical discretizations in three domains: Ground, Snow, and Canopy. The existing column grids should still be based on `RectilinearGrid`, but there will need to be a new more generic `LandGrid` that wraps an underlying Oceananigans grid and creates three instances, one for each domain. Review the instructions in AGENTS.md for drafting plans and stop to ask any clarifying questions that are necessary.
>
> **Clarifications**:
> - Domain boundaries: **Fixed (pre-computed offsets)** for GPU compatibility
> - Horizontal grid: `LandGrid` wraps **any Oceananigans grid** (not just RectilinearGrid); `ColumnGrid` becomes `ColumnLandGrid` as a sibling implementation *(superseded by Revision 4: no rename; the column grids become plain `AbstractGrid`s)*
> - Backward compatibility: **None** — breaking changes are acceptable
> - Priority: **Ground domain first**, then Snow and Canopy incrementally
> - GPU testing: Available for validation

## Revision log

> Revision 0 (2026-08-25): Initial draft created with clarifications from user feedback. Key decisions: fixed domain boundaries, no backward compatibility, Ground-first priority, LandGrid as generic Oceananigans grid wrapper.
>
> Revision 1 (2026-08-26): Simplified plan. Phase 1 is now just making existing grids implement the `AbstractGrid` interface by forwarding to the underlying `RectilinearGrid` — no new types. `DomainGrid` removed; per-domain grids in `LandGrid` are simply separate `RectilinearGrid` instances.
>
> Revision 3 (2026-08-26): Further simplified to 3 phases. Phase 2 combines type renaming with multi-domain support and `VarDomain` additions. Phase 3 introduces generic `LandGrid`. Phase 4 removed — no Oceananigans upstream changes.
>
> Revision 4 (2026-09-17): Restructured around a clean separation between *spatial discretizations* and *land grids*. (i) `ColumnGrid` and `ColumnRingGrid` keep their current names and behaviour but are no longer `AbstractLandGrid`s — they subtype `Oceananigans.Grids.AbstractGrid` only, alongside `RectilinearGrid`, `LatitudeLongitudeGrid`, etc. No renaming to `ColumnLandGrid`. (ii) There is exactly one `AbstractLandGrid` implementation, `LandGrid`, which wraps separate ground, snow, and canopy grids all derived from the *same* underlying `AbstractGrid`. (iii) Model constructors take an `AbstractGrid` positionally and build the `LandGrid` internally through a standard interface method `create_land_grid(grid, soil, snow, vegetation)`, so the per-domain vertical discretizations follow from the process components the user selected rather than being specified twice.
>
> Revision 8 (2026-09-18): Removed `get_field_grid` entirely. With `ColumnGrid` now an ordinary `RectilinearGrid` and `grid` always an `AbstractGrid`, the "unwrap to the grid `Field`s live on" accessor no longer earns its place. Call sites which resolve the vertical dimension — soil energy and hydrology (`Nz`, `Δz`, `∂z`, `znodes`), snow interfaces, root distribution, diffusion timescales, kernel operations — now use [`ground_domain`](@ref), which names the domain they mean and is the identity on grids that are not land grids. `ColumnRingGrid`'s own interface forwarding reads its wrapped grid via `getfield(grid, :grid)`, and `ground_domain` replaces `get_field_grid` as the one method an `AbstractLandGrid` implementation must define.
>
> Revision 8 (2026-09-18): Removed `get_field_grid` and replaced Terrarium's vertical discretization types with Oceananigans built-ins. (i) `get_field_grid` is gone: every grid is now an `AbstractGrid`, so code either uses the grid it was given or, where the vertical dimension is genuinely ambiguous (the soil energy/hydrology kernels, root distribution, plant-available water), asks for the ground domain explicitly. (ii) The per-domain accessors are renamed to the codebase's `get_*` convention: `ground_domain`, `snow_domain`, `canopy_domain`, and `create_grid` becomes `create_land_grid`. (iii) `AbstractVerticalSpacing`, `UniformSpacing`, `PrescribedSpacing` and `get_spacing` are deleted. Uniformly spaced and prescribed columns are given directly as a range or vector of cell interfaces, which is what the grid constructors always built internally. (iv) `ExponentialSpacing` survives as a *function* returning an Oceananigans `ExponentialDiscretization`: both describe the same geometric family, so the (`Δz_min`, `Δz_max`, `N`) parameterization land models want is converted in closed form to Oceananigans' (`N`, `left`, `right`, `scale`) one via `ρ = (Δz_max/Δz_min)^(1/(N-1))`, `D = Δz_min (ρᴺ-1)/(ρ-1)`, `scale = D/(N log ρ)`. The layer thicknesses agree to floating point; the only visible change is the loss of the old `sig` significant-digit rounding of layer thicknesses, worth ~0.4% on a default profile. (v) The grid constructors take the vertical coordinate directly, inferring `NF` from its `eltype` for an array of interfaces and defaulting to `Float64` for a discretization (the format Oceananigans computes interfaces in).
>
> Revision 7 (2026-09-18): Simplified the grid type hierarchy following review. (i) `Oceananigans.Grids.AbstractGrid` is imported into `Terrarium` so it needs no qualification. (ii) `ColumnGrid` is no longer a wrapper struct at all: it is a *type alias* for `RectilinearGrid{NF, Periodic, Flat, Bounded, …}` plus constructors which build one from an `AbstractVerticalSpacing`, so a column grid simply *is* an Oceananigans grid. (iii) `AbstractColumnGrid` and the `WrappedGrid` union are therefore gone; `ColumnRingGrid` remains a wrapper struct (it carries the ring grid and mask) and subtypes `AbstractGrid` directly, defining its own copy of the interface forwarding separately from `AbstractLandGrid`. (iv) All Terrarium methods which dispatch on a grid now take `AbstractLandGrid`, with two deliberate exceptions recorded in Revision 6(v) below, replaced by: the `AbstractModel` constructor (which accepts any `AbstractGrid` and calls `create_land_grid` — the whole point of Phase 4) and the `Field`/`FieldTimeSeries` constructors taking `VarDims` (a purely geometric mapping from `VarDims` to an Oceananigans location, carrying no land model semantics, and needed by `ColumnRingGrid`'s own field conversions). Tests which pass a grid to a process function now build a `LandGrid` explicitly. (v) `root_fraction` dispatches on `AbstractLandGrid{<:Any, <:Any, Flat}`, which expresses the "column-like discretization" requirement of its two-argument `FunctionField` more precisely than the old `AbstractColumnGrid` bound did.
>
> Revision 6 (2026-09-17, *implementation notes — pending approval*): Deviations made while implementing Phases 2-4.
> (i) The shared wrapper forwarding is anchored on a union alias `WrappedGrid = Union{AbstractLandGrid, AbstractColumnGrid}`, with `AbstractColumnGrid` moved from `column_grid.jl` into `grids.jl` alongside `AbstractLandGrid`; both column grids keep `AbstractColumnGrid` as their common supertype and it is that type which carries the forwarded dispatches.
> (ii) `LandGrid`'s keyword constructor takes ready-made per-domain grids (`LandGrid(ground; snow, canopy)`) rather than per-domain `AbstractVerticalSpacing`s. Building a domain grid from a spacing requires a grid-type-specific "same horizontal discretization, new vertical discretization" operation; that is deferred until a vertically resolved snow or canopy scheme actually needs it.
> (iii) Phase 3 step 4 (the `VarDomain` tag and domain-aware `Field`/`launch!` routing) is deferred for the same reason: with every non-ground domain unresolved, there is no variable for it to route. It remains the next piece of work.
> (iv) `create_land_grid` is implemented with the component-aware signature defaulting to a ground-only land grid, which is correct for every currently implemented (0D) snow and vegetation scheme. The per-component `ground_discretization`/`snow_discretization`/`canopy_discretization` methods are deferred with (ii) rather than added as no-ops.
> (v) Dispatches that only construct `Field`s or launch kernels were rebased from `AbstractLandGrid` to `WrappedGrid` (timesteppers, boundary conditions, input initialization), and the *converting* entry points — the `AbstractModel` constructor, `StateVariables`, and `InputSource` — accept any `AbstractGrid` and normalize it via `create_land_grid`. This keeps unit tests (and users) able to work directly on a spatial discretization without a model.
> (vi) Small forwarding methods were added so a `LandGrid` delegates to its ground discretization where a concrete spatial grid type is required: `root_fraction`, the `RingGrids`↔`Oceananigans` field conversions, and the `Rasters` extension's `InputSource`.
>
> Revision 5 (2026-09-17): Clarified Phase 4. Model struct definitions do not change at all: `grid` stays the first field and keeps its `AbstractLandGrid` annotation, so the inner `@kwdef` constructor still takes a land grid. Accepting a bare `AbstractGrid` is handled entirely by external (outer) constructors, including a generic fallback for `AbstractModel` types that wraps the grid in a ground-only `LandGrid` (no snow, no canopy).

## Problem description

Currently, Terrarium's `AbstractLandGrid` interface is loosely coupled to Oceananigans:
- `AbstractLandGrid{NF, Arch}` is a standalone abstract type (not a subtype of `Oceananigans.AbstractGrid`)
- Current implementations (`ColumnGrid`, `ColumnRingGrid`) wrap a single `Oceananigans.RectilinearGrid`
- The TODO comment in `grids.jl` explicitly acknowledges this is a prototype: "These grid types should be replaced with proper implementations of Oceananigans `AbstractGrid` at some point"
- All three domains (Ground, Snow, Canopy) currently share the same vertical discretization via a single underlying grid

The desired architecture requires:
1. `AbstractLandGrid` to be a proper subtype of `Oceananigans.AbstractGrid`
2. Support for **three separate vertical discretizations** (Ground, Snow, Canopy) within a single land grid
3. **No backward compatibility** — breaking changes are acceptable for this refactoring
4. Seamless integration with Oceananigans' field operations, kernel launching, and node/spacing APIs
5. **Fixed domain boundaries** (pre-computed offsets) for GPU/Reactant compatibility
6. A clear split of responsibilities between two layers of grid type:
   - **Spatial discretizations** are plain `Oceananigans.Grids.AbstractGrid`s. `ColumnGrid` and
     `ColumnRingGrid` belong to this layer, as land-specific siblings of `RectilinearGrid` and
     `LatitudeLongitudeGrid`. They know nothing about ground, snow, or canopy.
   - **Land grids** are `AbstractLandGrid`s, with a single implementation, `LandGrid`, holding one
     grid per domain, each built from the same user-supplied `AbstractGrid`.
7. Users construct models from an `AbstractGrid`; the `LandGrid` is an internal detail assembled by
   the model constructor via `create_land_grid(grid, soil, snow, vegetation)`

## Background

### Oceananigans AbstractGrid interface

Key characteristics of Oceananigans grids:
- **Type parameters**: `AbstractGrid{FT, TX, TY, TZ, Arch}` where topology `{TX, TY, TZ}` and architecture `Arch`
- **Required fields/methods**: `Nx, Ny, Nz, Hx, Hy, Hz, architecture`, plus topology-dependent coordinate arrays
- **Core API**: `size()`, `topology()`, `nodes()`, `halo_size()`, `architecture()`
- **Location system**: Fields defined at `Center`, `Face`, or `Nothing` in each dimension
- **Underlying grid concept**: `AbstractUnderlyingGrid` for primary grids, with curvilinear extensions

### Current Terrarium AbstractLandGrid

```julia
abstract type AbstractLandGrid{NF, Arch} end

# Current wrapper pattern
ground_domain(grid::AbstractLandGrid) :: Oceananigans.AbstractGrid
Base.size(grid::AbstractLandGrid) = size(ground_domain(grid))
Architectures.architecture(grid::AbstractLandGrid) = architecture(ground_domain(grid))
```

Current implementations:
- `ColumnGrid`: Wraps a single `RectilinearGrid` with 1D horizontal (column index) + vertical
- `ColumnRingGrid`: Wraps a `RectilinearGrid` with lateral discretization from `RingGrids.AbstractGrid`

As of Revision 4 these two types are *not* land grids: they are ordinary spatial discretizations
that happen to be defined in Terrarium, and they move out from under `AbstractLandGrid` (Phase 2).
Revision 7 takes this further: `ColumnGrid` becomes a type alias for a `RectilinearGrid` with a
`Flat` lateral dimension rather than a wrapper struct, so it needs no forwarding at all, while
`ColumnRingGrid` stays a struct — it carries the ring grid and mask — and subtypes `AbstractGrid`
directly, keeping its own forwarding.

### Multi-domain requirements

Land models need three vertically-stacked domains:
1. **Ground**: Soil profile, typically 50-100+ layers, exponentially spaced
2. **Snow**: Seasonal snowpack, 1-20+ layers, variable thickness
3. **Canopy**: Vegetation layers, 1-10 layers, typically near-surface

Each domain has:
- Independent vertical discretization (different `Nz`, different spacing)
- Shared horizontal discretization (same columns/ring grid)
- Coupled boundary conditions at interfaces (ground-snow, snow-canopy, canopy-atmosphere)

## Summary of changes

### Phase 1 (complete): Make existing grids implement `AbstractGrid`

Add topology type parameters to `AbstractLandGrid` and forward interface methods to the underlying grid. No new types needed.

```julia
# Before
abstract type AbstractLandGrid{NF, Arch} end

# After
abstract type AbstractLandGrid{NF, TX, TY, TZ, Arch} <: Oceananigans.AbstractGrid{NF, TX, TY, TZ, Arch} end
```

Forwarded methods (delegate to `ground_domain(grid)`): `size`, `halo_size`, `nodes`, `architecture`, `isrectilinear`.

### Phase 2: Detach the column grids from `AbstractLandGrid`

`ColumnGrid` and `ColumnRingGrid` are spatial discretizations, not land grids. They keep their
current names, fields, constructors, and behaviour, but their common supertype moves:

```julia
# Before
abstract type AbstractColumnGrid{NF, Arch} <: AbstractLandGrid{NF, Periodic, Flat, Bounded, Arch} end

# After
abstract type AbstractColumnGrid{NF, Arch} <: Oceananigans.Grids.AbstractGrid{NF, Periodic, Flat, Bounded, Arch, Nothing} end
```

The `AbstractGrid` forwarding currently defined on `AbstractLandGrid` (`size`, `halo_size`, `nodes`,
`architecture`, `isrectilinear`, property forwarding via `getproperty`/`propertynames`) is defined
separately for `AbstractLandGrid` and for `ColumnRingGrid`, the two remaining grid wrappers. As of
Revision 7 `ColumnGrid` is not a wrapper and needs none of it.

Nothing else about `ColumnRingGrid` changes: `@adapt_structure`, `show`, and its `RingGrids` mask
handling are untouched.

### Phase 3: Introduce `LandGrid`, the single `AbstractLandGrid` implementation

`LandGrid` wraps one grid per domain, all derived from the same underlying `AbstractGrid`:

```julia
struct LandGrid{NF, TX, TY, TZ, Arch, G <: Oceananigans.Grids.AbstractGrid, S, C} <: AbstractLandGrid{NF, TX, TY, TZ, Arch}
    "Grid for the ground (soil) domain; also the grid `Field`s default to."
    ground::G

    "Grid for the snow domain, or `nothing` when the model has no snow."
    snow::S

    "Grid for the canopy domain, or `nothing` when the model has no explicit canopy layers."
    canopy::C
end
```

- `G` may be **any** Oceananigans grid — `RectilinearGrid`, `LatitudeLongitudeGrid`, or Terrarium's
  own `ColumnGrid`/`ColumnRingGrid`, which are now valid `AbstractGrid`s thanks to Phase 2.
- `S` and `C` are either `typeof(ground)`-like grids of the same horizontal discretization or
  `Nothing`. Absent domains are represented by `Nothing` rather than a zero-layer grid so that no
  per-domain branch is reachable from a kernel.
- The three domain grids share the horizontal discretization exactly and differ only in their
  vertical discretization, which is built from a per-domain `AbstractVerticalSpacing`.
- Domain access is by type-stable accessor: `ground_domain(grid)`, `snow_domain(grid)`,
  `canopy_domain(grid)`, with the ground domain serving as the default grid for `Field`s.
- `Field` and `launch!` dispatch on the domain grid selected for the variable being allocated or the
  kernel being launched; the `VarDomain` tag on variables (deferred from the old Phase 2 wording)
  selects which domain grid a variable lives on.

### Phase 4: `create_land_grid` interface and model constructors

Model constructors accept an `AbstractGrid` positionally and build the `LandGrid` internally:

```julia
model = SoilModel(ColumnGrid(vert))             # or RectilinearGrid, LatitudeLongitudeGrid, ...
model = LandModel(grid; soil, snow, vegetation)
```

The assembly goes through one standard interface method:

```julia
"""
    create_land_grid(grid::AbstractGrid, soil, snow, vegetation)::AbstractLandGrid

Construct the `LandGrid` for a model discretized on `grid` whose ground, snow, and canopy
domains are determined by the `soil`, `snow`, and `vegetation` process components.
"""
function create_land_grid end
```

- The default method builds a `LandGrid` from `grid` plus the vertical discretization each component
  requests: `ground_discretization(grid, soil)`, `snow_discretization(grid, snow)`, and
  `canopy_discretization(grid, vegetation)`. Components that need no vertical domain (a
  single-layer snow scheme, a big-leaf canopy) return `nothing`, giving `Nothing` for that field.
- A `LandGrid` passed where an `AbstractGrid` is expected is accepted unchanged
  (`create_land_grid(grid::AbstractLandGrid, args...) = grid`), so a pre-built land grid can be supplied
  directly and the method is idempotent.
- `create_land_grid` is a host-side constructor, so argument validation (mismatched horizontal
  discretizations, incompatible component/domain combinations) belongs here rather than in any
  kernel-reachable code.
- **Model structs are untouched.** Field ordering stays exactly as it is — `grid` remains the first
  field and keeps its `GridType <: AbstractLandGrid{NF}` annotation, so the inner `@kwdef`
  constructor still requires a land grid. Every `get_grid(model)` call therefore still returns an
  `AbstractLandGrid`, and downstream process code is unaffected.
- **All widening happens in external (outer) constructors.** Each model type gets an outer
  constructor taking an `AbstractGrid`; it resolves the components, calls `create_land_grid`, and forwards
  the resulting `LandGrid` plus the resolved components to the `@kwdef` constructor. This is also
  what makes the ordering work: the component defaults are themselves functions of `grid` (e.g.
  `default_soil(grid, vegetation)`), so `create_land_grid` cannot be called until they are resolved —
  something the inner `@kwdef` constructor cannot express. Component defaults that only need
  `eltype(grid)` are unaffected, since `eltype` agrees between the spatial grid and the land grid
  built from it.
- **Generic default for `AbstractModel` types.** A single fallback outer constructor covers every
  model that has no domain beyond the ground:

  ```julia
  """
      $SIGNATURES

  Construct a model on the spatial discretization `grid` with a ground-only `LandGrid`
  (no snow or canopy domain).
  """
  function (::Type{M})(grid::Oceananigans.Grids.AbstractGrid, args...; kwargs...) where {M <: AbstractModel}
      return M(LandGrid(grid), args...; kwargs...)
  end
  ```

  `LandGrid(grid)` defaults `snow` and `canopy` to `nothing`, giving a land grid whose only domain is
  the ground. `SoilModel`, the surface models, and any future ground-only model therefore need no
  constructor of their own. Models with additional domains — `LandModel`, `SnowModel`,
  `VegetationModel` — override this fallback with a constructor that resolves their components and
  goes through `create_land_grid(grid, soil, snow, vegetation)`.

## Testing and verification

### Phase 1 unit tests

1. **AbstractGrid interface** (`test/grids/abstract_grid_interface.jl`):
   - `AbstractLandGrid <: Oceananigans.AbstractGrid`
   - `size`, `halo_size`, `topology`, `architecture`, `eltype` on `ColumnGrid` and `ColumnRingGrid`
   - `nodes`, `xnodes`, `ynodes`, `znodes` forward correctly
   - `isrectilinear` returns correct value
   - CPU and GPU

2. **No regression**: all existing model tests pass unchanged

### Phase 2 unit tests

3. **Column grids as plain `AbstractGrid`s** (`test/grids/abstract_grid_interface.jl`):
   - `ColumnGrid <: Oceananigans.Grids.AbstractGrid` and `!(ColumnGrid <: AbstractLandGrid)`
   - All forwarded interface methods still behave as in Phase 1 (no functional change)

### Phase 3 unit tests

4. **LandGrid construction** (`test/grids/create_land_grid.jl`):
   - Construct with ground-only, ground+snow, and all three domains; absent domains are `Nothing`
   - `ground_domain`/`snow_domain`/`canopy_domain` return the expected grid per domain
   - Construction over each supported underlying grid: `ColumnGrid`, `ColumnRingGrid`,
     `RectilinearGrid`, `LatitudeLongitudeGrid`
   - Horizontal discretization is identical across domains
   - `Field` allocated for a snow-domain variable is sized by the snow grid
   - Accessors are type stable (`@inferred`) and `adapt`/`on_architecture` round-trip on GPU

### Phase 4 unit tests

5. **`create_land_grid` interface** (`test/grids/create_land_grid.jl`):
   - `create_land_grid(grid, soil, snow, vegetation)` returns a `LandGrid` with the domains implied by the
     components; snowless and canopy-less configurations give `Nothing` for those domains
   - Idempotence: `create_land_grid(::AbstractLandGrid, ...)` returns its argument unchanged
   - `SoilModel(grid)` / `LandModel(grid; ...)` accept a bare `AbstractGrid` and store a `LandGrid`
   - The generic `AbstractModel` fallback yields a ground-only land grid: `snow_domain` and
     `canopy_domain` are `nothing`, and `ground_domain` matches the grid that was passed in
   - Passing a `LandGrid` directly still works and is stored as-is
   - Invalid combinations raise a clear error from the host-side constructor

### Integration tests

6. **Multi-domain simulation** (`test/models/multi_domain_land_model.jl`):
   - `LandModel` with Ground + Snow domains active
   - Energy conservation across domain interface

### Differentiability tests

7. Reactant/Enzyme: differentiate through multi-domain initialization; verify no throw paths in domain dispatch

## Documentation changes

### API documentation (`docs/src/grids.md`)

- Restructure the page around the two layers: **spatial discretizations** (`ColumnGrid`,
  `ColumnRingGrid`, and the Oceananigans grids) and the **land grid** (`LandGrid`)
- Make clear that the common path is to pass a spatial discretization to a model constructor and let
  it build the `LandGrid`; direct `LandGrid` construction is for advanced use
- Code examples:
  ```julia
  # The usual path: hand a spatial discretization to the model
  grid = ColumnGrid(ExponentialSpacing(0.05, 100.0, 50), 10)
  model = LandModel(grid)

  # The model holds a LandGrid built via `create_land_grid`
  land = get_grid(model) 
  size(ground_domain(land))  # (10, 1, 50)
  snow_domain(land)          # grid for the snow domain, or nothing

  # Advanced: build the land grid explicitly
  land = LandGrid(grid; snow = UniformSpacing(0.1, 20), canopy = UniformSpacing(1.0, 3))
  model = LandModel(land)
  ```
- Document `create_land_grid`, `ground_domain`, `snow_domain`, and `canopy_domain` as the public interface

### Model documentation updates

- Update `LandModel` docstring to explain domain keyword arguments
- Add examples of multi-domain simulations
- Document domain-specific boundary conditions

### Migration guide

**Breaking changes documentation**:
- `ColumnGrid` and `ColumnRingGrid` are unchanged in name and construction, but are no longer
  `AbstractLandGrid`s — code dispatching on `AbstractLandGrid` to catch them must dispatch on
  `AbstractGrid` (or on `AbstractColumnGrid`) instead
- `get_grid(model)` now returns a `LandGrid` wrapping the grid that was passed in, not the grid
  itself; use `ground_domain(get_grid(model))` to recover the previous object
- Code transformation examples (old → new) and rationale for the two-layer split

## Known limitations

1. **Performance overhead**: Domain offset calculations add indirection; may impact GPU performance if not inlined properly
2. **Memory usage**: Three separate grid instances increase memory footprint
3. **Indirection in user-facing objects**: `get_grid(model)` no longer returns the object the user
   passed in, which is a small but real loss of transparency; mitigated by the `ground_domain`
   accessor and by `create_land_grid` being idempotent on `LandGrid`s
4. **Complexity**: Multi-domain indexing is more error-prone than single-domain
5. **Breaking changes**: All existing user code must be updated — no backward compatibility

## Implementation steps (phased approach)

### Phase 1 (complete): Make existing grids implement `AbstractGrid`
1. Add topology type parameters to `AbstractLandGrid{NF, TX, TY, TZ, Arch}`
2. Update `ColumnGrid` and `ColumnRingGrid` to extract topology from their underlying grids
3. Forward `AbstractGrid` interface methods (`size`, `halo_size`, `nodes`, `architecture`, etc.) to the underlying grid
4. Unit tests verifying `AbstractLandGrid <: AbstractGrid` and all forwarded methods
5. GPU tests on existing models with no functional change

### Phase 2: Detach the column grids from `AbstractLandGrid`
1. Factor the shared wrapper forwarding (`size`, `halo_size`, `nodes`, `architecture`,
   `isrectilinear`, `getproperty`/`propertynames`) out of `AbstractLandGrid` so it can be reused by
   any thin `AbstractGrid` wrapper
2. Reparent `AbstractColumnGrid` to `Oceananigans.Grids.AbstractGrid`; leave `ColumnGrid` and
   `ColumnRingGrid` otherwise untouched
3. Update the (now few) `AbstractLandGrid` dispatches in `src/` that were really meant for "any
   grid" to dispatch on `AbstractGrid`
4. Tests: column grids satisfy the `AbstractGrid` interface and are no longer `AbstractLandGrid`s

### Phase 3: Implement `LandGrid`
1. Define `LandGrid{NF, TX, TY, TZ, Arch, G, S, C} <: AbstractLandGrid{NF, TX, TY, TZ, Arch}` with
   `ground`, `snow`, and `canopy` fields; absent domains are `Nothing`
2. Keyword constructor `LandGrid(grid; snow = nothing, canopy = nothing)` builds each domain grid
   from the shared horizontal discretization of `grid` plus a per-domain `AbstractVerticalSpacing`;
   all validation lives here
3. `ground_domain`/`snow_domain`/`canopy_domain` accessors,
   `@adapt_structure`, and a `show` method
4. Introduce the `VarDomain` tag in the variable system and route `Field` allocation and `launch!`
   through the domain grid it selects
5. Tests for construction over each underlying grid type, type-stable accessors, and
   domain-aware field creation; GPU adapt round-trip

### Phase 4: `create_land_grid` interface and model constructors
1. Define `create_land_grid(grid, soil, snow, vegetation)` plus the per-component
   `ground_discretization`/`snow_discretization`/`canopy_discretization` methods
2. Add the idempotent `create_land_grid(grid::AbstractLandGrid, args...) = grid` method
3. Add the generic outer constructor for `AbstractModel` types that wraps a bare `AbstractGrid` in
   a ground-only `LandGrid`; leave all model struct definitions (field order, `grid` first,
   `GridType <: AbstractLandGrid{NF}`) unchanged
4. Add per-model outer constructors for the models with additional domains (`LandModel`,
   `SnowModel`, `VegetationModel`) that resolve their components and go through `create_land_grid`
5. Update examples, docs, and tests to construct models from plain spatial discretizations
6. Integration tests with full multi-domain simulations
7. Reactant/Enzyme differentiability tests

## Clarifying questions (resolved)

| Question | Answer |
|----------|--------|
| Domain coupling strategy | **Fixed boundaries** (pre-computed offsets) for GPU compatibility |
| Horizontal discretization | `LandGrid` wraps **any Oceananigans grid**; `ColumnGrid`/`ColumnRingGrid` are spatial discretizations, not land grids (Rev. 4) |
| Number of `AbstractLandGrid` implementations | Exactly one: `LandGrid` (Rev. 4) |
| How models get a land grid | Constructors take an `AbstractGrid` and call `create_land_grid(grid, soil, snow, vegetation)` (Rev. 4) |
| Field location semantics | TBD — can fields exist at domain interfaces? |
| Backward compatibility scope | **None** — breaking changes acceptable |
| Testing infrastructure | GPU resources available for testing |
| Reactant compatibility | Must avoid throw paths; test AD early |
| Priority domains | **Ground first**, then Snow, then Canopy incrementally |
| Timeline constraints | Paper submission in 1-2 months |

### Remaining questions

1. **Field location semantics**: Can fields exist at domain interfaces (e.g., snow-ground boundary)? Should we support `Face` locations that span domains?
2. **Snow depth variation**: With fixed boundaries, how do we handle seasonal snow accumulation/melt? Pre-allocate max snow layers and use masking?

## Dependencies and prerequisites

- Oceananigans.jl: Current stable release (verify `AbstractGrid` API stability)
- RingGrids.jl: For `ColumnRingGrid` integration
- KernelAbstractions.jl: For GPU kernel launching
- Documenter.jl: For documentation updates
- TestEnv.jl: For isolated test environments

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing user code | High | Medium | Acceptable — no backward compat; Phase 2 keeps grid names stable and Phase 4 keeps constructor calls source-compatible, so most user scripts need no change |
| GPU performance degradation | Medium | Medium | Profile early, inline offset calculations, benchmark vs single-domain |
| Reactant compilation failures | Medium | High | Avoid throw paths, test AD early, hoist domain logic out of kernels |
| Complexity overwhelm | High | Medium | Phased approach (Ground→Snow→Canopy), rigorous code review, keep core types simple |
| Oceananigans API changes | Low | Medium | Pin Oceananigans version, track upstream changes |

---

*This plan document should be reviewed and signed off before implementation begins. Revise based on feedback from maintainers and potential users.*
