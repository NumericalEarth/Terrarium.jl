# Oceananigans `getbc` indexes boundary conditions in 2D, which breaks operation-valued BCs under Reactant

> Status: **open**, upstream. Terrarium reproduces it; a Terrarium-free MWE is included and verified.

Date of initial draft: 2026-08-04

Base revision: e015a29db9d97090edfd112a5eea165640134404 (Oceananigans 0.110.9, Reactant 0.2.272, Julia 1.12.6)

MWE: `docs/dev/2026-08/2026-08-04-oceananigans_getbc_mwe.jl`

## Problem description

Oceananigans evaluates array-valued boundary conditions inside kernels through `getbc`
(`src/BoundaryConditions/boundary_condition.jl`):

```julia
@inline getbc(condition::AbstractArray, i::Integer, j::Integer, grid::AbstractGrid, args...) =
    @inbounds condition[i, j]
```

The condition is indexed with **two** indices, but every Oceananigans field-like object is
three-dimensional. That works today only by accident, and only for one concrete type:

- **`Field`** defines a catch-all
  ```julia
  @propagate_inbounds Base.getindex(f::Field, inds...) = getindex(f.data, inds...)
  ```
  (`src/Fields/field.jl:424`), which forwards any number of indices straight to the underlying
  `OffsetArray`, where a 2D index into a 3D array is padded cheaply and statically.

- **`AbstractOperation`** (e.g. the `UnaryOperation` produced by `-some_field`) defines only the
  three-index form. `op[i, j]` therefore falls back to `Base`'s generic `AbstractArray` path:

  ```
  _getindex -> _to_subscript_indices -> axes(::AbstractField) -> size(::Field) -> size(::AbstractGrid, loc, indices)
  ```

  `axes(::Abstract3DField)` (`src/Fields/abstract_field.jl:62`) destructures `indices(f)`, and inside a
  compiled kernel that whole chain becomes a dynamic dispatch.

On the CPU the fallback is harmless. Under Reactant the kernel fails to compile:

```
InvalidIRError: compiling MethodInstance for Oceananigans.BoundaryConditions.gpu__compute_z_bcs!(...)
Reason: unsupported call to an unknown function (call to jl_f_throw_methoderror)
Stacktrace:
 [1] indexed_iterate       @ ./tuple.jl:165
 [2] axes                  @ Oceananigans/src/Fields/abstract_field.jl:62
 [3] _to_subscript_indices @ ./abstractarray.jl:1400
 [4] _to_subscript_indices @ ./abstractarray.jl:1398
 [5] _getindex             @ ./abstractarray.jl:1383
 [6] getindex              @ ./abstractarray.jl:1342
 [7] getbc                 @ Oceananigans/src/BoundaryConditions/boundary_condition.jl:182
 [8] getbc                 @ Oceananigans/src/BoundaryConditions/boundary_condition.jl:174
 [9] compute_z_top_bc!     @ Oceananigans/src/BoundaryConditions/compute_flux_bcs.jl:161
```

## Reproducing

`2026-08-04-oceananigans_getbc_mwe.jl` — Oceananigans + Reactant + CUDA only, no Terrarium. It builds
one `CenterField` whose top flux BC is a `Field` and one whose top flux BC is `-field`, and compiles
`compute_z_bcs!` for each:

```
(1) Field-valued top flux BC
    OK
(2) AbstractOperation-valued top flux BC (`-surface_flux`)
    FAILED:
    InvalidIRError: compiling MethodInstance for Oceananigans.BoundaryConditions.gpu__compute_z_bcs!(...)
    Reason: unsupported call to an unknown function (call to jl_f_throw_methoderror)
    ...
```

Two things are needed to reproduce, both of which cost time if you guess wrong:

- **The compiled path.** An *eager* `compute_z_bcs!` (or `fill_halo_regions!`) on a `ReactantState` grid
  succeeds. Only `@compile raise=true raise_first=true sync=true` surfaces it.
- **`compute_z_bcs!`, not `fill_halo_regions!`.** Halo filling with an operation-valued BC compiles fine
  in isolation; it is the flux-BC-into-tendency kernel that fails.

Neither the grid topology (`Flat` vs `Periodic` horizontal) nor the vertical coordinate type
(uniform `StepRangeLen` vs array-valued z) makes any difference — all four combinations behave the same.

## Where Terrarium hits it

`LandModel` builds the `saturation_water_ice` top boundary condition by negating the infiltration rate,
because the hydrology module computes infiltration positive *downwards* while Terrarium's flux
convention is positive *upwards* (`src/models/coupled/land_model.jl`):

```julia
# Note that the hydrology module computes infiltration as positive so we need to negate it here
# since fluxes are by convention positive upwards
infiltration_bc = InfiltrationFlux(-infiltration)
```

`-infiltration` is a lazy `UnaryOperation`. Every other Terrarium BC condition is a plain `Field` —
including the sibling energy BC `SoilHeatFlux(soil_heat_flux)`, which compiles without trouble. This is
the last blocker for the coupled `LandModel` under Reactant: with it worked around locally, the
`:land_soil_snow` CPU-vs-Reactant correctness test passes 59/59 fields after 100 steps.

## Suggested fix

Index the condition in three dimensions inside `getbc`, matching how every other kernel touches a field:

```julia
@inline getbc(condition::AbstractArray, i::Integer, j::Integer, grid::AbstractGrid, args...) =
    @inbounds condition[i, j, 1]
```

The `1` is correct for the surface-normal boundaries: an `XY` boundary condition field has a singleton
third dimension. The same reasoning applies to the `x` and `y` variants of `getbc`, which index
`condition[j, k]` / `condition[i, k]` and would need the singleton in the corresponding slot instead.
A location-aware helper is probably cleaner than three hand-written index tuples.

Alternatives, both worse:

- Give `AbstractOperation` a catch-all `getindex(op, inds...)` mirroring `Field`'s. This keeps the 2D
  indexing convention alive and leaves the same trap for the next field-like type.
- Require BC conditions to be `Field`s and materialize operations at construction. That is what
  Terrarium can do as a workaround, but it costs an extra field plus an explicit refresh step, and it
  silently changes the semantics for anyone relying on a BC that tracks its operand lazily.

## Related

- The same 2D-indexing convention bites Terrarium's own nonlinear solvers, which wrote a `Field` with
  `field[i, j]`; `Field` defines `setindex!` for exactly three indices and has no catch-all, so that
  fails even for plain `Field`s. Fixed on the Terrarium side with a `pad_indices` helper — see
  `2026-08-04-PLAN_reactant_coupled_land_model.md`.
