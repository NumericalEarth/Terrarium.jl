# Accelerating simulations with Reactant

```@meta
CurrentModule = Terrarium
```

Terrarium models can be compiled and executed with
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), which traces the model time step into
MLIR/StableHLO and compiles it with XLA for high-performance execution on CPUs, GPUs, and TPUs.
The only user-facing change is the architecture passed to the grid: with `ReactantState()`, the
model state is allocated and initialized on the device, and the first call to `timestep!` or `run!`
compiles the stepping loop transparently (subsequent calls with the same arguments reuse the compiled
program).

!!! warning "Experimental"
    Reactant support is experimental. Correctness against the CPU implementation is tested
    continuously in `test/reactant/` for `SoilModel` (heat conduction and Richards-equation
    hydrology), `SnowModel`, `VegetationModel` with constant and time-varying (`FieldTimeSeries`)
    inputs, and the coupled `LandModel` with soil, snow and vegetation, on `ColumnGrid` and
    `ColumnRingGrid` with uniform or stretched (`ExponentialSpacing`) vertical grids. Two things to
    keep in mind: the default skin-temperature solver (`RootSolver`) iterates to a tolerance and cannot
    be compiled, so a `LandModel` needs `ImplicitSkinTemperature(NF; solver = NewtonSolver(NF))`; and
    only in-memory `FieldTimeSeries` input sources can be updated inside the compiled loop, whereas
    raster-backed and on-disk sources read from the host on every step and cannot.

## Example

The following mirrors the [soil heat conduction example](@ref soil_heat_column), changing only
the architecture. Note that this code is not executed during the documentation build since
Reactant compilation takes several minutes.

```julia
using Terrarium
using Reactant, CUDA  # CUDA is required by Reactant's kernel integration, even on CPU

# ReactantState() instead of CPU() or GPU() — the only change!
grid = ColumnGrid(ReactantState(), Float32, ExponentialSpacing(N = 10))
model = SoilModel(grid)

boundary_conditions = PrescribedSurfaceTemperature(:T_ub, 1.0f0)
initializers = (temperature = (x, z) -> -1.0f0 - 0.05f0 * z,)
integrator = initialize(model; boundary_conditions, initializers)

# The first step compiles the model with XLA (takes a moment); further steps are fast.
run!(integrator, steps = 144, Δt = 600.0f0)

# Materialize device results on the CPU for analysis/plotting
T = Array(interior(integrator.state.temperature))
```

The target device is selected by Reactant, e.g. `Reactant.set_default_backend("gpu")` before
constructing the model.

In the example folder you can also find examples that demonstrate how to use the Reactant model to take derivatives of the model and integrate and train neural networks. 