# Oceananigans AbstractGrid Integration for Terrarium

> Status: **planned**. Draft integration plan for making `AbstractLandGrid` a subtype of Oceananigans `AbstractGrid` with support for multi-domain vertical discretizations (Ground, Snow, Canopy).

Date of initial draft: 2026-08-25

Base revision: 4f3841955af84ac6cb4a6f88b04538fc0dd7d658

## Originating prompt

> Please review the Oceananigans `AbstractGrid` interface and grid implementations. Then review the current Terrarium `AbstractLandGrid`s and draft a plan for how to integrate the two interfaces. The general idea is that `AbstractLandGrid` should be a valid subtype of `AbstractGrid`, but implementations of `AbstractLandGrid` should allow for separate vertical discretizations in three domains: Ground, Snow, and Canopy. The existing column grids should still be based on `RectilinearGrid`, but there will need to be a new more generic `LandGrid` that wraps an underlying Oceananigans grid and creates three instances, one for each domain. Review the instructions in AGENTS.md for drafting plans and stop to ask any clarifying questions that are necessary.
>
> **Clarifications**:
> - Domain boundaries: **Fixed (pre-computed offsets)** for GPU compatibility
> - Horizontal grid: `LandGrid` wraps **any Oceananigans grid** (not just RectilinearGrid); `ColumnGrid` becomes `ColumnLandGrid` as a sibling implementation
> - Backward compatibility: **None** — breaking changes are acceptable
> - Priority: **Ground domain first**, then Snow and Canopy incrementally
> - GPU testing: Available for validation

## Revision log

> Revision 0 (2026-08-25): Initial draft created with clarifications from user feedback. Key decisions: fixed domain boundaries, no backward compatibility, Ground-first priority, LandGrid as generic Oceananigans grid wrapper.
>
> Revision 1 (2026-08-26): Simplified plan. Phase 1 is now just making existing grids implement the `AbstractGrid` interface by forwarding to the underlying `RectilinearGrid` — no new types. `DomainGrid` removed; per-domain grids in `LandGrid` are simply separate `RectilinearGrid` instances.
>
> Revision 3 (2026-08-26): Further simplified to 3 phases. Phase 2 combines type renaming with multi-domain support and `VarDomain` additions. Phase 3 introduces generic `LandGrid`. Phase 4 removed — no Oceananigans upstream changes.

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
6. `LandGrid` as a generic wrapper for any Oceananigans grid type; `ColumnLandGrid` as sibling to existing column patterns

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
get_field_grid(grid::AbstractLandGrid) :: Oceananigans.AbstractGrid
Base.size(grid::AbstractLandGrid) = size(get_field_grid(grid))
Architectures.architecture(grid::AbstractLandGrid) = architecture(get_field_grid(grid))
```

Current implementations:
- `ColumnGrid`: Wraps a single `RectilinearGrid` with 1D horizontal (column index) + vertical
- `ColumnRingGrid`: Wraps a `RectilinearGrid` with lateral discretization from `RingGrids.AbstractGrid`

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

### Phase 1: Make existing grids implement `AbstractGrid`

Add topology type parameters to `AbstractLandGrid` and forward interface methods to the underlying grid. No new types needed.

```julia
# Before
abstract type AbstractLandGrid{NF, Arch} end

# After
abstract type AbstractLandGrid{NF, TX, TY, TZ, Arch} <: Oceananigans.AbstractGrid{NF, TX, TY, TZ, Arch} end
```

Forwarded methods (delegate to `get_field_grid(grid)`): `size`, `halo_size`, `nodes`, `architecture`, `isrectilinear`.

### Phase 2: Rename grid types and add multi-domain support with `VarDomain`

Rename existing grid types (e.g., `ColumnGrid` → `ColumnLandGrid`) and add separate domain grids for Ground, Snow, and Canopy. Introduce `VarDomain` type for the variable system to tag variables with their domain. Detailed design to be specified separately.

### Phase 3: Implement generic `LandGrid` supporting all Oceananigans grids

`LandGrid{NF, TX, TY, TZ, Arch, G<:AbstractGrid}` holds three domain grids of type `G`, where `G` can be any Oceananigans grid type. Constructor builds each domain grid from shared horizontal discretization + per-domain `AbstractVerticalSpacing`. Domain access via `get_domain_grid(grid, Val(:ground))`. Update `Field` and `launch!` to dispatch on domain grid.

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

3. **LandGrid construction** (`test/grids/land_grid.jl`):
   - Construct with ground-only, then ground+snow, then all three domains
   - `get_domain_grid` returns the correct `RectilinearGrid` per domain
   - `Field(grid, Val(:snow), dims)` creates field on the snow grid

### Integration tests

4. **Multi-domain simulation** (`test/models/multi_domain_land_model.jl`):
   - `LandModel` with Ground + Snow domains active
   - Energy conservation across domain interface

### Differentiability tests

5. Reactant/Enzyme: differentiate through multi-domain initialization; verify no throw paths in domain dispatch

## Documentation changes

### API documentation (`docs/src/grids.md`)

- New section: "Multi-domain Land Grids"
- Explain `LandGrid` constructor and domain concepts
- Code examples:
  ```julia
  # Create multi-domain land grid
  ground_vert = ExponentialSpacing(0.05, 100.0, 50)
  snow_vert = UniformSpacing(0.1, 20)
  canopy_vert = UniformSpacing(1.0, 3)
  
  grid = LandGrid(CPU(), Float32, ground_vert, snow_vert, canopy_vert; num_columns=10)
  
  # Query domain properties
  size(grid, :ground)  # (10, 1, 50)
  size(grid, :snow)    # (10, 1, 20)
  size(grid, :canopy)  # (10, 1, 3)
  
  # Create domain-specific field
  soil_temp = Field(grid, :ground, (Center(), Center(), Center()))
  ```

### Model documentation updates

- Update `LandModel` docstring to explain domain keyword arguments
- Add examples of multi-domain simulations
- Document domain-specific boundary conditions

### Migration guide

**Breaking changes documentation**:
- "Migrating from ColumnGrid to ColumnLandGrid"
- Code transformation examples (old → new)
- API differences and rationale

## Known limitations

1. **Performance overhead**: Domain offset calculations add indirection; may impact GPU performance if not inlined properly
2. **Memory usage**: Three separate grid instances increase memory footprint
3. **Complexity**: Multi-domain indexing is more error-prone than single-domain
4. **Breaking changes**: All existing user code must be updated — no backward compatibility

## Implementation steps (phased approach)

### Phase 1: Make existing grids implement `AbstractGrid`
1. Add topology type parameters to `AbstractLandGrid{NF, TX, TY, TZ, Arch}`
2. Update `ColumnGrid` and `ColumnRingGrid` to extract topology from their underlying grids
3. Forward `AbstractGrid` interface methods (`size`, `halo_size`, `nodes`, `architecture`, etc.) to the underlying grid
4. Unit tests verifying `AbstractLandGrid <: AbstractGrid` and all forwarded methods
5. GPU tests on existing models with no functional change

### Phase 2: Rename grid types and add multi-domain support with `VarDomain`
1. Rename `ColumnGrid` → `ColumnLandGrid` (or similar clearer name)
2. Add separate domain grids for Ground, Snow, and Canopy within the renamed grid type
3. Introduce `VarDomain` type for the variable system to tag variables with their domain
4. Update state variables, initializers, and tendency kernels to dispatch on domain
5. Tests for multi-domain grid construction and domain-aware field creation
6. **Note**: Detailed design of `VarDomain` and domain dispatch to be specified in a separate plan revision

### Phase 3: Implement generic `LandGrid` supporting all Oceananigans grids
1. `LandGrid{NF, TX, TY, TZ, Arch, G<:AbstractGrid}` holds three domain grids of type `G`
2. `G` can be any Oceananigans grid type (`RectilinearGrid`, `LatitudeLongitudeGrid`, `CubedSphereGrid`, etc.)
3. Constructor builds each domain grid from shared horizontal discretization + per-domain `AbstractVerticalSpacing`
4. Domain access via `get_domain_grid(grid, Val(:ground))`
5. Update `Field` and `launch!` to dispatch on domain grid
6. Integration tests with full multi-domain simulations
7. Reactant/Enzyme differentiability tests

## Clarifying questions (resolved)

| Question | Answer |
|----------|--------|
| Domain coupling strategy | **Fixed boundaries** (pre-computed offsets) for GPU compatibility |
| Horizontal discretization | `LandGrid` wraps **any Oceananigans grid**; `ColumnLandGrid` is sibling implementation |
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
| Breaking existing user code | High | Medium | Acceptable — no backward compat, update all tests/examples in Phase 1 |
| GPU performance degradation | Medium | Medium | Profile early, inline offset calculations, benchmark vs single-domain |
| Reactant compilation failures | Medium | High | Avoid throw paths, test AD early, hoist domain logic out of kernels |
| Complexity overwhelm | High | Medium | Phased approach (Ground→Snow→Canopy), rigorous code review, keep core types simple |
| Oceananigans API changes | Low | Medium | Pin Oceananigans version, track upstream changes |
| Timeline pressure (paper) | Medium | High | Ground-only implementation sufficient for paper; multi-domain is future work |

---

*This plan document should be reviewed and signed off before implementation begins. Revise based on feedback from maintainers and potential users.*
