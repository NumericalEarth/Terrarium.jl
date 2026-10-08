# Spatially varying soil initializers via input variables

> Status: **planned**. Draft awaiting human review; nothing implemented yet.

Date of initial draft: 2026-10-08

Base revision: `ce9ae606ae39063f52bb0ef23a3328a8b1904055` (tip of `main`, branch `bg/improved-init`)

## Originating prompt

> We need to generalize the existing soil initializer to allow for spatially varying inputs. The idea
> would be to declare corresponding input variables for T0 and Qgeo and then pass the struct values as
> default value initializers. These initializers could also be functions or callable structs that contain
> additional parameters, e.g. the periodic initialization used for soil temp and currently duplicated
> across many examples. Draft an implementation plan and summarize your findings. Ask questions if needed.

## Revision log

- **Rev 1 (2026-10-08):** Initial draft.
- **Rev 2 (2026-10-08):** Open questions resolved by the author: variable names `initial_surface_temperature` and `geothermal_heat_flux` confirmed; hydrology initializers (change 5) are in scope; `LatitudinalClimatology(T_equator, ΔT)` is the only concrete field initializer and no generic function wrapper is added, since a plain function can be assigned directly to the initializer fields; `AbstractFieldInitializer` lives in core `src/initializers.jl`.
  Implementation not yet approved.

## Problem description

The soil energy initializers in `src/models/soil/soil_model_init.jl` are column-uniform:

- `ConstantSoilTemperature{NF}` stores a scalar `T₀`.
- `QuasiThermalSteadyState{NF}` stores scalars `T₀`, `Qgeo`, and `k_eff` and sets the temperature profile with a host-side closure, `set!(state.temperature, (x, z) -> T₀ - Qgeo / k_eff * z)`.

Neither can express a surface temperature or geothermal heat flux that varies between columns.
As a result, every global example re-implements the same thing by hand: a latitude-dependent climatology `mean_annual_temperature(lat) = 20 - abs(40 * sin(lat))` and a closure `(x, z) -> T₀[round(Int, x)] - 0.05 * z` that indexes a masked latitude (or regridded air temperature) vector by the column index `x`.
This pattern appears, with small variations, in

- `examples/simulations/soil_heat_global.jl` (which explicitly notes that `QuasiThermalSteadyState` "does not (yet) support spatially variable parameters"),
- `examples/simulations/soil_heat_global_soilgrids.jl`,
- `examples/simulations/speedy_wet_land.jl`,
- `examples/simulations/land_global_era5.jl` (surface temperature from the ERA5 `t2m` mean),
- `examples/autodiff/global_sensitivity.jl` (surface temperature from the first ERA5 `t2m` slice), and
- `examples/autodiff/differentiating_terrarium_reactant.jl` (a plain closure).

These closures capture host arrays, run through Oceananigans' `set_to_function!` (which detours via the CPU on GPU and Reactant grids), and hardcode the geothermal gradient as a number instead of deriving it from the geothermal heat flux and thermal conductivity.
The geothermal heat flux used for initialization is also disconnected from the `geothermal_heat_flux` variable that the `GeothermalHeatFlux()` boundary condition reads, so the initial profile and the bottom boundary condition can silently disagree.

## Background

Three pieces of existing infrastructure shape the design.

**Input variables with defaults.** `input(name, loc; default, units, ...)` declares an `InputVariable` whose `default` is applied once, at `Field` construction time, in `initialize(var, grid, clock, fields, bcs)` (`src/state_variables.jl`) via `set!(field, var.default)`.
The `Def` type parameter is currently restricted to `Union{Nothing, Number, Function}`.
`reset!(state)` zeroes prognostic, auxiliary, and tendency fields but leaves inputs alone, so defaults survive `initialize!(integrator)`.
Existing processes already use this (e.g. `sand_fraction` in `soil_horizon.jl`, `ground_temperature` in `autotrophic_respiration.jl`).

**Input sources override declared inputs by name.** `StateVariables(model; input_variables)` builds `Variables(tuplejoin(variables(model), input_variables))`.
For duplicate names, `register!` keeps the first declaration (the model's) provided the metadata (`name`, `dims`, `units`, `eltype`) match and the domains are compatible; otherwise it throws.
At `initialize!(integrator)`, `InputSource`s are applied *before* the user `initializers` NamedTuple and the model initializer, so an `InputSource` named like a declared input simply overwrites its default field.
This is exactly the mechanism needed to let users supply a spatially varying `T₀` or `Qgeo` from data.

**Initializers are invisible to `variables`.** `variables(::AbstractModel)` collects variables from `processes(model)`, a generated function that only picks fields whose type is `<: AbstractProcess`.
The `initializer` field is an `AbstractInitializer` and is skipped, so an initializer currently has no way to declare state variables.
`variables(::Any) = ()` already provides a safe fallback for `DefaultInitializer`.

Two further facts matter for the design:

- Initializer structs are not `@parameterized`, and parameters are opt-in via `@param`, so an initializer may hold non-scalar fields (functions, callable structs, `Field`s) without affecting `parameters(model)` or `ParameterEditing.reconstruct`.
- On a `ColumnRingGrid`, the horizontal coordinate `x` passed to `set!(field, f)` is the (1-based, float-valued) index of the active land column; latitude and longitude must be looked up through `grid.rings` and `grid.mask`.
  Any reusable "latitude-dependent" initializer therefore needs access to the grid, which the `initialize(var, grid, ...)` hook already has.

## Summary of changes

### 1. Let initializers declare variables

- In `src/abstract_model.jl`, split the shared default so that models also collect their initializer's variables:

  ```julia
  variables(obj::AbstractCoupledProcesses) = tuplejoin(fastmap(variables, processes(obj))...)
  variables(model::AbstractModel) = tuplejoin(fastmap(variables, processes(model))..., variables(get_initializer(model)))
  ```

  `variables(::Any) = ()` keeps `DefaultInitializer` and all third-party initializers working unchanged.
- In `src/models/soil/soil_model_init.jl`, add `variables(init::SoilInitializer) = tuplejoin(variables(init.energy), variables(init.hydrology), variables(init.biogeochem))`.
- Update the `AbstractInitializer` docstring in `src/initializers.jl` to document the optional `variables(::AbstractInitializer)` extension point.

`LandModel` has its own `StateVariables` method but it also goes through `variables(model)`, so it picks the change up automatically.

### 2. Generalize `InputVariable` defaults

- Widen the `Def` type parameter of `InputVariable` (`src/abstract_variables.jl`) from `Union{Nothing, Number, Function}` to an unconstrained parameter.
  It remains concrete per instance.
  Document the accepted forms: `nothing`, a number, any argument accepted by `Oceananigans.set!` (functions of the node coordinates, arrays, `Field`s), or an `AbstractFieldInitializer` (below).
- Introduce a small hook in `initialize(var, grid, clock, fields, bcs)` so that the default is applied through a single overloadable method instead of a bare `set!`:

  ```julia
  set_default!(field, grid, default) = set!(field, default)
  set_default!(field, grid, ::Nothing) = nothing
  ```

### 3. `AbstractFieldInitializer`: reusable, grid-aware, parameterized field initializers

Add to `src/initializers.jl`:

```julia
"""
Base type for reusable initializers of a single `Field`. Implementations are callable structs that
provide `initialize!(field, grid, init::AbstractFieldInitializer)`; the default evaluates `init` at
the node coordinates via `set!`. They can be passed as `default` of an `input` variable, as an entry of
the `initializers` NamedTuple of `initialize`, or as a parameter of a model initializer.
"""
abstract type AbstractFieldInitializer end

initialize!(field::AbstractField, grid::AbstractGrid, init::AbstractFieldInitializer) = set!(field, (coords...) -> init(coords...))
set_default!(field, grid, init::AbstractFieldInitializer) = initialize!(field, grid, init)
```

and make the user-facing `initialize!(state, inits::NamedTuple)` path accept them as well (it currently calls `set!` directly, which would not know what to do with a callable struct).
Only one concrete implementation is in scope here, to replace the duplicated example code:

```julia
"""
Latitude-dependent surface temperature climatology on a `ColumnRingGrid`:
T(φ) = T_equator - |ΔT sin φ|. Evaluated per active land column.
"""
@kwdef struct LatitudinalClimatology{NF} <: AbstractFieldInitializer
    T_equator::NF = 20
    ΔT::NF = 40
end
(init::LatitudinalClimatology)(lat) = init.T_equator - abs(init.ΔT * sin(lat))
function initialize!(field, grid::ColumnRingGrid, init::LatitudinalClimatology)
    # host-side: gather latitudes of active columns, evaluate, then set!(field, values)
end
```

No generic wrapper around a user function is provided: a plain function of the node coordinates can be assigned directly to `T₀`/`Qgeo` (or to any `input` default), and the implementation must keep that path working.
The `periodic_bc` *boundary* condition in the same examples (diurnal cycle shifted by longitude) is a boundary-condition concern and is left out of this plan; see "Future work".

### 4. Generalize the soil energy initializers

Rewrite `ConstantSoilTemperature` and `QuasiThermalSteadyState` in `src/models/soil/soil_model_init.jl`:

```julia
struct QuasiThermalSteadyState{NF, T0, QG} <: AbstractInitializer{NF}
    "Initial surface temperature (°C): number, function of column coordinate, `Field`, or `AbstractFieldInitializer`"
    T₀::T0
    "Geothermal heat flux (W/m²), same accepted forms"
    Qgeo::QG
    "Bulk thermal conductivity (W/m/K)"
    k_eff::NF
end
QuasiThermalSteadyState(::Type{NF}; T₀ = zero(NF), Qgeo = NF(0.02), k_eff = one(NF)) where {NF}

variables(init::QuasiThermalSteadyState) = (
    input(:initial_surface_temperature, Ground(Top()); default = init.T₀, units = u"°C", desc = "..."),
    input(:geothermal_heat_flux, Ground(Bottom()); default = init.Qgeo, units = u"W/m^2", desc = "..."),
)
```

`initialize!(state, model, init::QuasiThermalSteadyState)` then reads the two input fields and fills `temperature` with a kernel launched via `launch!`, `T[i, j, k] = T₀[i, j] - Qgeo[i, j] / k_eff * z_k`, instead of a host-side closure.
This is GPU- and Reactant-native (no `set_to_function!` detour) and keeps `k_eff` as a scalar parameter.
`ConstantSoilTemperature` gets the same treatment with only the `initial_surface_temperature` input.

Reusing the name `geothermal_heat_flux` means that a model with `GeothermalHeatFlux()` as bottom boundary condition and `QuasiThermalSteadyState` as initializer shares one field for both, and that an `InputSource(grid, Qgeo_field; name = :geothermal_heat_flux, domain = Ground(Bottom()), units = u"W/m^2")` drives both consistently.
The units in such an `InputSource` must match the declaration, otherwise `Variables` reports a conflict; this is documented.

Numeric literals (`0.0`, `0.02`, `1.0`) in the current `@kwdef` structs are replaced by `NF`-typed defaults as required by the project rules.

### 5. Hydrology initializers (same pattern, smaller scope)

`SaturationWaterTable` (`water_table_depth`, `vadose_zone_saturation`) and `ConstantSaturation` (`sat`) are generalized the same way (`input(:water_table_depth, Ground(XY()); default = ...)` etc.) so the whole `SoilInitializer` is consistent (confirmed in scope, Rev 2).

### 6. Examples

Replace the hand-rolled closures with the new initializers:

- `soil_heat_global.jl`, `soil_heat_global_soilgrids.jl`, `speedy_wet_land.jl`: `SoilInitializer(NF; energy = QuasiThermalSteadyState(NF; T₀ = LatitudinalClimatology(NF), Qgeo = ..., k_eff = ...))`, removing the duplicated `mean_annual_temperature`/`initial_soil_temperature` definitions and the "does not (yet) support spatially variable parameters" remark.
- `land_global_era5.jl`, `global_sensitivity.jl`: pass the regridded air temperature `Field` as `T₀`, or as an `InputSource` named `initial_surface_temperature`.
- `differentiating_terrarium_reactant.jl`: the plain closure stays valid through the `initializers` NamedTuple; optionally switch to the initializer to exercise it under Reactant.

### 7. Exports

Export `AbstractFieldInitializer` and `LatitudinalClimatology` from `src/models/models.jl` or `src/Terrarium.jl` next to the existing initializer exports.

## Testing and verification

- `test/soil/soil_energy_tests.jl`: extend the existing `SoilInitializer` tests to
  - verify that `QuasiThermalSteadyState` with scalar `T₀`/`Qgeo` reproduces the current profile bit for bit (regression),
  - verify a column-varying `T₀` given as a function and as a `Field`,
  - verify that an `InputSource` named `geothermal_heat_flux` overrides the default and that the profile uses the overridden values, and
  - verify that `initialize!(integrator)` (re-initialization) leaves the result unchanged.
- `test/state_variables.jl`: `variables(model)` includes the initializer's inputs; `DefaultInitializer` contributes none; mismatched units in an `InputSource` raise the existing conflict error.
- `test/grids.jl` or a new `test/initializers.jl`: `LatitudinalClimatology` on a masked `ColumnRingGrid` assigns each active column the value at its latitude.
- `test/differentiability/soil_energy_diff.jl`: run the Enzyme energy test with the new initializer to confirm the kernel-based `initialize!` is differentiable.
- `test/reactant/`: add the kernel-based initializer to one of the existing model configurations in `setup.jl` so it is exercised on the Reactant backend.
- Full suite via `Pkg.test()`, then the draft doc build.

## Documentation changes

- `docs/src/models/soil_model.md` ("Initializers" section): document the accepted forms of `T₀`/`Qgeo` and the declared input variables, with an `@example` using `LatitudinalClimatology`.
- `docs/src/running/initialization.md`: add a "Field initializers" subsection for `AbstractFieldInitializer`, and note that model initializers may declare input variables.
- `docs/src/extending/core_interfaces.md`: mention `variables(::AbstractInitializer)`.
- Docstrings for all new types and methods with `jldoctest` blocks where a short verifiable output exists.

## Known limitations

- `k_eff` stays a scalar.
  A true thermal steady state would need the stratigraphy's conductivity; this is the same caveat the current docstring already states.
- `AbstractFieldInitializer` defaults are applied at `Field` construction (as all input defaults are today), i.e. before any `InputSource` runs, and only once.
  That is the desired order, but it means a default that depends on other inputs is not supported.
- The `Variables` conflict rule requires matching units between an `InputSource` and the declared input; users must pass `units = u"W/m^2"` (resp. `u"°C"`) to the source.
- `LandModel` with a `SoilInitializer` is not addressed here; whichever way `initialize!` dispatches for that combination today is unchanged.

## Future work

- A reusable periodic *boundary condition* (`T₀(lat) + A sin(2π t / P - lon)`) to remove the matching `get_temperature_bc` duplication in the examples.
- Applying auxiliary-variable initializers in `reset!` (existing TODO in `state_variables.jl`).

## Resolved questions (Rev 2)

1. Variable names: `initial_surface_temperature` and `geothermal_heat_flux` (shared with the bottom BC).
2. Hydrology initializers are included.
3. `LatitudinalClimatology(T_equator, ΔT)` with the fixed form `T_equator - |ΔT sin φ|`; no generic function wrapper.
   Users assign functions directly to initializer fields.
4. `AbstractFieldInitializer` is defined in core `src/initializers.jl`.
