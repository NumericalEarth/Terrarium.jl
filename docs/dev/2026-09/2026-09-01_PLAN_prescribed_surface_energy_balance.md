# `PrescribedSurfaceEnergyBalance`: ground heat flux as a first-class SEB sub-process

> Status: **completed** (Terrarium.jl side). Extracts the ground heat flux into its own surface-energy-balance
> sub-process, so that it (like the radiative and turbulent fluxes) has `Diagnosed` and
> `Prescribed` implementations. `PrescribedSurfaceEnergyBalance` then becomes a type alias for a
> `SurfaceEnergyBalance` whose four flux sub-processes are all prescribed, rather than a separate
> no-op driver. Implemented on `bg/ground-heat-flux`; the NumericalEarth extension follow-up (change 6)
> and the coupled end-to-end run (verification 9) live in `NumericalEarth.jl` and remain outstanding.

Date of initial draft: 2026-09-01

Base revision: `1f3d1f3e` (tip of `bg/var-domains`). This work is stacked on `bg/prescribed-seb`,
which is `bg/var-domains` plus the plan commit. See "Choice of base branch" below.

## Originating prompt

> [...] let's try to make Option A work with the `BulkTemperature` scheme and `apply_air_land_radiative_fluxes`
> using the `land_surface_energy_flux` method I added, which can be overridden in the extension. Let's write
> the ground heat flux directly into Terrarium, ignoring its side for now. We'll update it in a second step to
> have a `PrescribedSurfaceEnergyBalance` scheme that accepts all fluxes as prescribed, nothing calculated
> internally. Will that work?

Revised after:

> I want to propose an alternative refactoring here: we separate `ground_heat_flux` into its own SEB
> sub-process and then allow this to be prescribed. there would then be a dispatch on
> `PrescribedSurfaceEnergyBalance` as an alias for a `SurfaceEnergyBalance` type with all four fluxes
> set to their prescribed types.

## Revision log

- **Rev 1 (2026-09-01):** Initial draft. Proposed a new `PrescribedSurfaceEnergyBalance` *type* whose
  SEB driver methods were all no-ops, plus a special-case reclassification of `ground_heat_flux` from
  `auxiliary` to `input`. Energy only; vapor deferred.
- **Rev 2 (2026-09-25):** Replaces the no-op-driver design. The ground heat flux becomes its own
  sub-process (`AbstractGroundHeatFlux`) with `Diagnosed` and `Prescribed` implementations, mirroring
  the existing `Prescribed`/`Diagnosed` pairs for the radiative and turbulent fluxes.
  `PrescribedSurfaceEnergyBalance` becomes a type alias plus a convenience constructor. Re-baselined
  from `cc6b24090` onto `bg/var-domains` (`5b8bda663`), whose variable-domain API this design depends
  on. *Approved by the author on 2026-09-25.*
- **Rev 3 (2026-09-25):** Implementation notes, recorded during development on `bg/ground-heat-flux`
  (branched from `bg/prescribed-seb` at `26c86d20e`). Three deviations from Rev 2:
  1. **The demand dispatches on `AbstractGroundHeatFlux`, not only on `DiagnosedGroundHeatFlux`.**
     Change 2 assigned the residual closure to `DiagnosedGroundHeatFlux` alone, but the implicit skin
     temperature solve also needs the demand in order to invert its conduction relation. Putting an
     indexed `compute_ground_heat_flux_demand(i, j, grid, fields, ghf)` on the abstract type, with the
     prescribed method returning the supplied field, is what actually makes the `Implicit` +
     `Prescribed` row of the change 5 table well posed; without it that row would silently invert
     against `R_net + H_s + H_l` while storing an unrelated `G`. `compute_skin_temperature` therefore
     gained a `ghf` argument after `skinT` (propagated to `compute_skin_temperature_residual!` and to
     `diagnose_skin_temperature_residual`).
  2. **`DiagnosedGroundHeatFlux` still dispatches on the skin temperature scheme.** Change 1 described
     it as reproducing `G = R_net + H_s + H_l`, which is only the `PrescribedSkinTemperature` case;
     `ImplicitSkinTemperature` stores the explicit conductive flux `2κ_g(T_g − T_s)/Δz_g`. Both
     methods moved across unchanged, so behavior is preserved (verified bit for bit; see below).
  3. **The documentation split into two pages.** `skin_temperature.md` is retitled "Skin temperature"
     (its inbound `@ref`s updated) and the new `ground_heat_flux.md` carries the sub-process, its two
     implementations, and the supported-combination table.

## Problem description

Two problems, one of which is a coupling bug and one of which is a modeling limitation. The second is
the cause of the first.

**The modeling limitation.** The ground heat flux `G` has no process of its own. It is computed by a
family of functions that all dispatch on `AbstractSkinTemperature`, whose own docstring describes it
as the "base type for skin temperature *and ground heat flux* schemes". Concretely:

- the accessor `ground_heat_flux(i, j, grid, fields, ::AbstractSkinTemperature)` (`abstract_types.jl:63`);
- `compute_ground_heat_flux_demand(::AbstractSkinTemperature, R_net, H_s, H_l) = R_net + H_s + H_l`
  (`skin_temperature.jl:118`), which hardcodes the residual closure;
- roughly eight further `compute_ground_heat_flux*` methods and a kernel in `skin_temperature.jl`;
- the field itself, declared **twice**, by both `variables(::PrescribedSkinTemperature)` and
  `variables(::ImplicitSkinTemperature)`, each as `auxiliary(:ground_heat_flux, Ground(Top()), …)`.

Consequences: `G` cannot be prescribed without also replacing the skin temperature scheme, it cannot
be given an alternative closure (for example a conductive law), and one sub-process slot carries two
responsibilities. Every other flux in the SEB already has the `Prescribed`/`Diagnosed` pair; `G` is
the odd one out.

**The coupling bug.** In the coupled `EarthSystemModel`, NumericalEarth's `InterfaceComputations`
owns the atmosphere-land interface fluxes. Step 1 of this work (already implemented in
`NumericalEarthTerrariumExt`) assembles the net surface energy flux and writes it directly into
Terrarium's `ground_heat_flux`, which without snow *is* the soil-top boundary condition
(`soil_heat_flux === ground_heat_flux`). That write does not survive: Terrarium's own SEB recomputes
`ground_heat_flux` as the residual before the soil tendency reads it. The result is a double surface
energy balance that is redundant, mutually inconsistent, and empirically unstable (the cold-column
runaway documented in `NumericalEarth.jl/scratch/terrarium_ne_interface_design_discussion.md`).

**Goal.** Make it possible to configure a `SurfaceEnergyBalance` in which every flux, including `G`,
is supplied externally, so the coupler-assembled value is authoritative; and do so by *composition*
of existing sub-process slots rather than by adding a parallel driver.

## Background

### Choice of base branch

`bg/var-domains` rewrites the variable declaration API that this design is built on. Declarations
move from a bare `XY()` domain to domain-aware forms, for example:

```julia
# main
auxiliary(:ground_heat_flux, XY(), units = u"W/m^2", desc = "Ground heat flux")

# bg/var-domains
auxiliary(:ground_heat_flux, Ground(Top()), units = u"W/m^2", desc = "Ground heat flux")
```

It also changes indexing in this same code (`out.ground_heat_flux[i, j, 1]` became
`[i, j, end]`) and grid access (`get_field_grid(grid)` became `ground_domain(grid)`), and it touches
every SEB sub-process file. Since this design is almost entirely about variable declarations, basing
it on `main` would mean writing every declaration twice and conflicting on exactly the files
`bg/var-domains` rewrites.

The new API also helps: `Ground(Top())` says the ground heat flux is a top-face quantity, distinct
from `Ground(Top(z = Center()))` for the cell-centered ground temperature. Giving `G` its own
sub-process lets that distinction sit where it belongs.

*Risk:* `bg/var-domains` is unmerged and well ahead of `main`, so this work is stacked behind it. If
`bg/var-domains` is reworked in review, this rebases on top of that churn. All branches involved are
also a few commits behind `main`, so a merge from `main` is due regardless.

### The current SEB structure

```julia
struct SurfaceEnergyBalance{
        NF,
        SkinTemperature <: AbstractSkinTemperature{NF},
        TurbulentFluxes <: AbstractTurbulentFluxes{NF},
        RadiativeFluxes <: AbstractRadiativeFluxes{NF},
        Albedo <: AbstractAlbedo{NF},
    } <: AbstractSurfaceEnergyBalance{NF}
    skin_temperature :: SkinTemperature
    radiative_fluxes :: RadiativeFluxes
    turbulent_fluxes :: TurbulentFluxes
    albedo           :: Albedo
end
```

Note that the type parameters are ordered `(NF, SkinTemperature, TurbulentFluxes, RadiativeFluxes,
Albedo)` while the fields are ordered `(skin_temperature, radiative_fluxes, turbulent_fluxes,
albedo)`. The radiative and turbulent entries are **swapped** between the two orders. This is a
latent hazard for anyone writing a type alias by reading the field list, and adding a fifth parameter
makes it worse. See change 3.

### The pattern to mirror

`PrescribedTurbulentFluxes` and `DiagnosedTurbulentFluxes` declare the *same* variable names and
differ only in classification:

```julia
variables(::PrescribedTurbulentFluxes) = (
    input(:sensible_heat_flux, Surface(XY()), …),
    input(:latent_heat_flux,   Surface(XY()), …),
)

variables(::DiagnosedTurbulentFluxes) = (
    auxiliary(:sensible_heat_flux, Surface(XY()), …),
    auxiliary(:latent_heat_flux,   Surface(XY()), …),
)
```

This is exactly the mechanism needed for `G`, and it removes Rev 1's awkward special case: the
`auxiliary`-to-`input` reclassification is no longer a driver-level exception but simply which
sub-process is installed.

### Sign convention (verified, unchanged from Rev 1)

Terrarium's `ground_heat_flux = R_net + H_s + H_l`, all fluxes **positive upward**, equals
NumericalEarth's `surface_energy_flux` term for term (`𝒬ᵀ + 𝒬ᵛ − ΣQ_rad`, also positive upward).
**No sign flip** is required when the coupler writes into `ground_heat_flux`.

### Field wiring (unchanged from Rev 1)

Without snow, `soil_heat_flux === ground_heat_flux` (the same `Field` object), and the soil energy
boundary condition is `SoilHeatFlux(soil_heat_flux)`. Making `ground_heat_flux` a prescribed input
therefore routes the coupler's value straight to the soil column's Neumann top boundary condition.

## Summary of changes

### 1. New sub-process: `AbstractGroundHeatFlux`

```julia
abstract type AbstractGroundHeatFlux{NF} <: AbstractProcess{NF} end

"Ground heat flux diagnosed as the residual that closes the surface energy balance."
struct DiagnosedGroundHeatFlux{NF} <: AbstractGroundHeatFlux{NF} end

"Ground heat flux supplied externally, for example by a coupler."
struct PrescribedGroundHeatFlux{NF} <: AbstractGroundHeatFlux{NF} end

variables(::DiagnosedGroundHeatFlux) = (
    auxiliary(:ground_heat_flux, Ground(Top()), units = u"W/m^2", desc = "Ground heat flux"),
)

variables(::PrescribedGroundHeatFlux) = (
    input(:ground_heat_flux, Ground(Top()), units = u"W/m^2", desc = "Ground heat flux"),
)
```

`DiagnosedGroundHeatFlux` reproduces today's behavior exactly (`G = R_net + H_s + H_l`).
`PrescribedGroundHeatFlux` computes nothing: its `compute_ground_heat_flux!` is a no-op, so the field
retains whatever was written into it.

### 2. Move the ground heat flux machinery

Create `src/processes/surface/ground_heat_flux.jl` and move the `compute_ground_heat_flux*` family
out of `skin_temperature.jl`, re-dispatching on `AbstractGroundHeatFlux` instead of
`AbstractSkinTemperature`. The residual closure `compute_ground_heat_flux_demand(…, R_net, H_s, H_l)`
becomes a method on `DiagnosedGroundHeatFlux`. The accessor
`ground_heat_flux(i, j, grid, fields, ::AbstractSkinTemperature)` in `abstract_types.jl` moves to
dispatch on the new abstract type. Remove the duplicated `ground_heat_flux` declarations from both
skin temperature variants; the field is now declared once, by the ground heat flux sub-process.

Note the existing "demand" versus realized distinction (`G₀` is adjusted downstream, for example by
snow). Preserve it; the new sub-process owns both the demand closure and whatever adjusts it.

### 3. Extend and reorder `SurfaceEnergyBalance`

```julia
struct SurfaceEnergyBalance{
        NF,
        SkinTemperature <: AbstractSkinTemperature{NF},
        RadiativeFluxes <: AbstractRadiativeFluxes{NF},
        TurbulentFluxes <: AbstractTurbulentFluxes{NF},
        GroundHeatFlux  <: AbstractGroundHeatFlux{NF},
        Albedo          <: AbstractAlbedo{NF},
    } <: AbstractSurfaceEnergyBalance{NF}
    skin_temperature :: SkinTemperature
    radiative_fluxes :: RadiativeFluxes
    turbulent_fluxes :: TurbulentFluxes
    ground_heat_flux :: GroundHeatFlux
    albedo           :: Albedo
end
```

Type parameters are reordered to match field order, fixing the existing radiative/turbulent swap.
Add `get_ground_heat_flux(seb)` alongside the existing getters, extend `variables(seb)` to include
the new sub-process, and update the keyword constructor with
`ground_heat_flux::AbstractGroundHeatFlux = DiagnosedGroundHeatFlux(NF)` so current behavior is the
default. Update the call site in `surface_energy_balance.jl:129` to pass `seb.ground_heat_flux`.

### 4. The alias and its constructor

```julia
const PrescribedSurfaceEnergyBalance{NF, Albedo} = SurfaceEnergyBalance{
    NF,
    <:PrescribedSkinTemperature{NF},
    <:PrescribedRadiativeFluxes{NF},
    <:PrescribedTurbulentFluxes{NF},
    <:PrescribedGroundHeatFlux{NF},
    Albedo,
}

PrescribedSurfaceEnergyBalance(::Type{NF}; albedo = ConstantAlbedo(NF)) where {NF} =
    SurfaceEnergyBalance(NF;
        skin_temperature = PrescribedSkinTemperature(NF),
        radiative_fluxes = PrescribedRadiativeFluxes(NF),
        turbulent_fluxes = PrescribedTurbulentFluxes(NF),
        ground_heat_flux = PrescribedGroundHeatFlux(NF),
        albedo)
```

Albedo is deliberately left free. With all four fluxes prescribed it only feeds the radiative flux
computation, which is itself prescribed, so it is inert.

### 5. Supported combinations

The point of the refactor is that the coupling designs become configurations rather than types:

| skin temperature | ground heat flux | meaning |
|---|---|---|
| `Implicit` | `Diagnosed` | standalone Terrarium (today's default) |
| `Prescribed` | `Diagnosed` | today's coupled hybrid: `G` as residual at the coupler's `Tₛ` |
| `Prescribed` | `Prescribed` | **Option A**, the coupled target for this plan |
| `Implicit` | `Prescribed` | back out `Tₛ` from a supplied `G` |
| `Implicit` | `Conductive` | future: true implicit Option A (see Future work) |

Solve sequencing differs between rows, so the driver must state which pairs it supports and in what
order it evaluates them. Not every combination is well posed, and the type system alone will not say
so; validate in the host-side keyword constructor, never in a kernel (per `AGENTS.md`).

### 6. NumericalEarth extension follow-up

Step 1 already assembles the net flux into `ground_heat_flux` and overrides `land_surface_energy_flux`.
Once this lands, the extension's `update_net_fluxes!` pushes of `skin_temperature`,
`sensible_heat_flux`, and `latent_heat_flux` feed prescribed inputs that the SEB no longer consumes
internally; trim to what the prescribed SEB and the hydrology actually read.

## Testing and verification

> Outcome (2026-09-25): items 1-8 and 10 done and passing; items 6 and 9 are cross-repository and
> remain outstanding. The full suite (`Pkg.test()`) and the Enzyme suite pass, and the draft doc build
> reports no `docs_block` errors.

1. **Default behavior is unchanged.** `SurfaceEnergyBalance(NF)` with the new
   `DiagnosedGroundHeatFlux` default must reproduce pre-refactor `ground_heat_flux` bit for bit on an
   existing standalone case. This is the main regression guard for the code move.
2. **No overwrite.** With `PrescribedGroundHeatFlux`, set `ground_heat_flux` to a known field, run one
   `timestep!`, and assert the soil tendency saw exactly that flux and that `compute_auxiliary!` left
   the field unchanged.
3. **Energy conservation.** With a prescribed constant `G`, the soil column's integrated
   internal-energy change over `N` steps equals `G · area · Δt · N` within tolerance.
4. **Variable classification.** Assert `ground_heat_flux` is declared exactly once, and that it
   resolves as an input under `PrescribedGroundHeatFlux` and as an auxiliary under
   `DiagnosedGroundHeatFlux`.
5. **Alias and constructor.** Pin the alias with a test that `PrescribedSurfaceEnergyBalance(NF)`
   constructs a value which `isa PrescribedSurfaceEnergyBalance`. This guards the type-parameter
   ordering, which is the most likely thing to get silently wrong.
6. **Over-determination.** With all four fluxes prescribed, `G = R_net + H_s + H_l` is not enforced by
   construction. Document the invariant and add a host-side consistency check or test.
7. **Type stability and allocations.** `@code_warntype` and `@allocated` on `compute_auxiliary!` and a
   timestep for each supported combination.
8. **Differentiability.** Enzyme adjoint through a `timestep!` with respect to the prescribed
   `ground_heat_flux`, the clean boundary-condition seam this design is for.
9. **Coupled end to end.** Re-run `examples/era5_forced_terrarium_land.jl` with the prescribed SEB. The
   cold-column minimum should track the delivered forcing instead of collapsing, and the stored
   `ground_heat_flux` should equal `R_net + H + LE` by construction.
10. **Draft doc build** (`julia --project=docs docs/make.jl --local --draft`).

### Verification results

1. **Default behavior is unchanged — confirmed bit for bit.** A 30-step deterministic `LandModel` run
   with the default SEB (`ImplicitSkinTemperature` + all fluxes diagnosed), dumping `ground_heat_flux`,
   `skin_temperature`, `surface_net_radiation`, the turbulent fluxes, and the final `internal_energy`
   profile, produces byte-identical output on `bg/ground-heat-flux` and on the `26c86d20e` baseline
   (max absolute difference exactly `0.0` over 180 values).
2. **No overwrite — confirmed.** `test/surface/ground_heat_flux.jl` sets `G = 25` W/m² alongside
   deliberately inconsistent turbulent fluxes (`H_s = 111`, `H_l = 222`) and asserts the field is
   unchanged through `compute_auxiliary!`, through `compute_boundary_conditions!`, and across a
   `timestep!`, and that the soil-top energy BC is that very field.
3. **Energy conservation — confirmed.** Over 20 steps at `Δt = 60` s with a prescribed constant
   `G = 25` W/m², the column's integrated internal-energy change equals `−G·Δt·N` to within `1e-6`
   relative tolerance (negative because all fluxes are positive upward).
4. **Variable classification — confirmed.** `ground_heat_flux` is declared exactly once by each SEB
   configuration, as an `AuxiliaryVariable` under `DiagnosedGroundHeatFlux` and an `InputVariable`
   under `PrescribedGroundHeatFlux`; neither skin temperature scheme declares it any more.
5. **Alias and constructor — confirmed**, together with each sub-process getter, so the test fails
   loudly if the type parameter order and the field order drift apart again.
6. *(NumericalEarth extension trim)* **Not done.** `NumericalEarthTerrariumExt` lives in
   `NumericalEarth.jl`; this is a separate follow-up in that repository.
7. **Type stability and allocations — confirmed** for all four combinations of change 5's table:
   `solve_surface_energy_balance!` infers `Nothing` and allocates an identical 3504 bytes in every
   case, and `compute_auxiliary!` allocates nothing. All four integrate to finite state.
8. **Differentiability — confirmed.** `test/differentiability/prescribed_seb_diff.jl` takes a reverse
   adjoint of a `timestep!` through a `PrescribedSurfaceEnergyBalance` with respect to the prescribed
   `ground_heat_flux` and asserts the sensitivity is finite and strictly negative (a positive, upward
   flux cools the column).
9. *(Coupled end to end)* **Not done.** Requires the coupled `EarthSystemModel` and the extension
   changes of item 6, both in `NumericalEarth.jl`.
10. **Draft doc build — confirmed clean.** No `docs_block` errors remain. The build still terminates
    on `:external_cross_references`, but only because the sandbox cannot reach the upstream
    `objects.inv` inventories (Oceananigans, KernelAbstractions, SpeedyWeather, FreezeCurves,
    Thermodynamics); every such failure is in a page this work does not touch.

## Documentation changes

- Docstrings for `AbstractGroundHeatFlux`, `DiagnosedGroundHeatFlux`, and `PrescribedGroundHeatFlux`,
  stating the positive-upward convention and the residual closure `G = R_net + H_s + H_l`.
- Docstring for the `PrescribedSurfaceEnergyBalance` alias and its constructor.
- A "Ground heat flux" implementations subsection on the surface energy balance doc page, with
  `canonical = false` docstrings, following the page structure required by `AGENTS.md`.
- The supported-combination table, so the configuration space is documented rather than implied.
- Update the `AbstractSkinTemperature` docstring, which currently claims responsibility for the ground
  heat flux.

## Known limitations

- **Energy only.** Moisture is not prescribed here. If NumericalEarth owns `LE` while Terrarium's
  hydrology still computes its own evapotranspiration, moisture is double counted the same way energy
  was. The vapor seam is the immediate follow-up.
- **Option A-lite.** Paired with the coupler's `BulkTemperature`, the coupling remains explicit-lag
  (fluxes evaluated at the previous step's soil-top `T`). Stability rests on the top soil cell's heat
  capacity, which is ample at `Δt = 5 min` and `Δz = 0.1 m` but is not a substitute for an implicit
  skin solve at larger `Δt`.
- **Snow.** With snow, `soil_heat_flux !== ground_heat_flux` and a blended conductive flux intervenes.
  The prescribed sub-process's interaction with the snow energy balance is out of scope and needs its
  own design. The `demand` versus realized distinction noted in change 2 is entangled with this.
- **Stacked on an unmerged branch.** See "Choice of base branch".

## Future work

- `ConductiveGroundHeatFlux`, implementing `G = −κ/δ · (Tₛ − T_soil)`. This is the closure that
  NumericalEarth's `SkinTemperature(internal_flux)` needs from Terrarium, and it is the reason the
  sub-process split is worth doing independently of the coupling: it is simply not expressible today.
  With it, `Implicit` skin plus `Conductive` `G` gives the true implicit Option A and removes the
  residual explicit lag.
- Prescribed vapor and moisture flux, to close the moisture budget under coupler ownership.
- The broader `InterfaceComputations` delegation question. Note that Rev 1's `surface_interface_inputs`
  sketch is superseded: the delegation mechanism largely already exists in NumericalEarth, the contract
  home is `InterfaceComputations` itself, and the genuine gap is the hardcoded component-state
  assembly. See `NumericalEarth.jl/scratch/terrarium_ne_interface_design_discussion.md`.
