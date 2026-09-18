# Tests that still require an explicit `LandGrid` wrapper

Scratch notes, generated from the working tree. 12 files, 27 wrapper sites.

## The actual blocker

Most of the process functions these tests call take an **unannotated** `grid` argument —
`compute_auxiliary!(state, grid, process, args...)` and friends impose no type at all. They fail
on a bare `ColumnGrid` only because they call `launch!` internally:

```julia
# src/grids/grid_utils.jl:2
function Oceananigans.launch!(grid::AbstractLandGrid, workspec, kernel!::Function, first_arg, other_args...; kwargs...)
```

With a plain grid this convenience method doesn't apply, and the call falls through to
Oceananigans' own `launch!(arch, grid, workspec, ...)`, which reads the arguments in different
positions — the observed symptom is
`MethodError: no method matching work_layout(::Type{XY}, ::typeof(compute_auxiliary_kernel!), ::Tuple{})`.

So the lever is **one method**, not a family of them. Three entry points are separately annotated
and would need widening too:

- `initialize!` — src/state_variables.jl:129 (the `InputSources` method)
- `update_inputs!` — src/state_variables.jl:144
- `explicit_step!` — src/timesteppers/abstract_timestepper.jl:147

Also annotated, though not called directly by these tests: `compute_z_bcs!`
(`src/boundary_conditions.jl:46`), reached from tendency computation, and
`root_fraction` (`src/processes/vegetation/hydraulics/root_distribution.jl:55`), whose bound is
`AbstractLandGrid{<:Any, <:Any, Flat}` and deliberately encodes a column-like discretization.

## Per-file detail

| File | Wrapper lines | Calls taking the grid | Blocked via |
|---|---|---|---|
| `test/inputs/namespaced_inputs.jl` | 6, 71 | `initialize!`&nbsp;(31,41), `update_inputs!`&nbsp;(48,59,62) | annotated entry point |
| `test/snow/snow_energy_tests.jl` | 6, 97 | `closure!`&nbsp;(26,34,43,55,63,76), `invclosure!`&nbsp;(52), `initialize!`&nbsp;(86) | both |
| `test/snow/snow_properties_tests.jl` | 76 | `compute_auxiliary!`&nbsp;(81) | `launch!` |
| `test/soil/soil_energy_tests.jl` | 32, 55, 67, 103, 152 | `initialize!`&nbsp;(39,44,49,61), `compute_tendencies!`&nbsp;(62), `closure!`&nbsp;(73) | both |
| `test/soil/soil_hydrology_tests.jl` | 94, 135, 204, 225, 255 | `adjust_saturation_profile!`&nbsp;(103,113,123,130), `closure!`&nbsp;(237), `compute_boundary_conditions!`&nbsp;(239) | `launch!` |
| `test/surface/diagnostic_albedo_tests.jl` | 6 | `compute_auxiliary!`&nbsp;(12,22,28,33) | `launch!` |
| `test/surface/hydrology/surface_runoff_tests.jl` | 61 | `compute_tendencies!`&nbsp;(69,79,85) | `launch!` |
| `test/surface/radiative_fluxes.jl` | 5, 21 | `compute_auxiliary!`&nbsp;(16,35) | `launch!` |
| `test/surface/skin_temperature.jl` | 130, 146, 188, 271 | `test_skin_temperature_solve!`&nbsp;(7), `compute_auxiliary!`&nbsp;(137) | `launch!` |
| `test/surface/turbulent_fluxes.jl` | 5, 19 | `compute_auxiliary!`&nbsp;(30,36) | `launch!` |
| `test/timestepping/explicit_step.jl` | 23 | `explicit_step!`&nbsp;(54) | annotated entry point |
| `test/vegetation/plant_available_water_tests.jl` | 14 | `compute_auxiliary!`&nbsp;(43,54,64,74) | `launch!` |

`test_skin_temperature_solve!` in `test/surface/skin_temperature.jl` is a test-local helper whose
own signature is annotated `Terrarium.AbstractLandGrid{NF}`; it would just need its annotation
relaxed alongside.

## If you widen `launch!`

The change is one line:

```julia
-function Oceananigans.launch!(grid::AbstractLandGrid, workspec, ...)
+function Oceananigans.launch!(grid::AbstractGrid, workspec, ...)
```

It is safe in the sense that the body only needs `architecture` and the grid itself, both of which
any `AbstractGrid` provides; and process code resolves its domain through `ground_domain(grid)`,
which is the identity on a non-land grid. The cost is that the model's grid type stops being
visible in the kernel-launching interface, which is what the `AbstractLandGrid` bound was buying.

Expected effect: 10 of the 12 files unblocked by `launch!` alone; `namespaced_inputs.jl` and
`explicit_step.jl` additionally need the three annotated entry points above. This split is from
static inspection of the signatures, not from a test run.
