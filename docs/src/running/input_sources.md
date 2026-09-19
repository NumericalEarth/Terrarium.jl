# Input sources

```@meta
CurrentModule = Terrarium
```

Input sources supply both static and time-varying input data for the [`InputVariable`](@ref)s in a Terrarium simulation. Recall that input variables are distinct from model/process *parameters*: parameters are specified as properties of process/parameterization `struct`s and are spatially invariant, whereas inputs are defined over the model `grid`; from the perspective of a process implementation, input variables look and behave exactly like any other `Field` in the `state`.

## The `InputSource` interface

```@docs; canonical=false
InputSource
```

All [`InputSource`](@ref)s implement the following interface:

```@docs; canonical = false
variables(::InputSource)
initialize!(inputs, grid, clock, fields, input::InputSource)
update_inputs!(inputs, grid, clock, fields, input::InputSource)
```

## Built-in input source types

### Static input `Field`s

A [`FieldInputSource`](@ref) holds a single `Field` that is copied into the state once at initialization and is thereafter unchanged. This is the appropriate input source for spatially-varying but time-constant forcings (e.g. maps of soil properties or prescribed climatology).

```@docs; canonical = false
InputSource(grid::AbstractLandGrid{NF}, field::FS; domain = Ground()) where {NF, FS <: AnyField{NF}}
```

```julia
using Oceananigans: Field
using Terrarium: Ground, Snow, Surface

# Existing Field or array on the model grid
albedo_field = Field(grid_2d)
set!(albedo_field, 0.3)

source = InputSource(grid, albedo_field; name = :albedo, domain = Ground())
```

For a [`ColumnRingGrid`](@ref), a [`RingGrids.Field`](@extref SpeedyWeather RingGrids.Field) can be passed directly and will be converted automatically:

```julia
albedo_ring = RingGrids.Field(albedo_data, global_grid)
source = InputSource(snow_domain, albedo_ring; name = :albedo)
```

### Time-varying `Field` inputs

A [`FieldTimeSeriesInputSource`](@ref) wraps an Oceananigans [`FieldTimeSeries`](@extref Oceananigans.OutputReaders.FieldTimeSeries-Tuple{JLD2.JLDFile, String}). At each time step the input field is set to the snapshot in the time series that is closest to `clock.time` (using `FieldTimeSeries[Time(t)]`).

```julia
using Oceananigans.Units: hours

# Allocate and populate a FieldTimeSeries
times = 0.0:3600.0:86400.0 # hourly for one day (seconds)
fts = FieldTimeSeries(grid, Ground(XY()), times)
fts.data .= randn(size(fts)) # fill with data
source = InputSource(fts; name = :air_temperature, units = u"°C")
```

```@docs; canonical = false
InputSource(grid::AbstractLandGrid{NF}, field::FS; domain = Ground()) where {NF, FS <: AnyFieldTimeSeries{NF}}
```

The `FieldTimeSeries` can also be loaded from a file using the relevant constructors provided by Oceananigans.

### Inputs from raster data

Alternatively, Terrarium provides an extension module for [`Rasters.jl`](https://github.com/rafaqz/Rasters.jl) with a `RasterInputSource` that reads data directly from any format supported by Rasters.jl (NetCDF, GeoTIFF, Zarr, etc.) and supports both static and time-varying inputs.

!!! note
    Load `Rasters` together  with `Terrarium` to activate the extension:
    ```julia
    using Terrarium, Rasters
    ```

**Static raster input**

If the `Raster` has no time dimension, `initialize!` copies the data once and `update_inputs!` is a no-op:

```julia
using Rasters

raster = Raster("path/to/temperature.nc"; name = :temperature)
source = InputSource(grid, raster; domain = Ground())   # Defaults to using name of Raster
```

**Time-varying raster input**

If the `Raster` has a `Ti` (time) dimension, values are linearly interpolated between the two nearest time points at each call to `update_inputs!`. Outside the bounds of the time axis the nearest available snapshot is used (flat extrapolation).

```julia
raster = Raster("path/to/forcing_timeseries.nc"; name = :air_temperature)
source = InputSource(grid, raster; reftime = DateTime(2000, 1, 1), domain = Ground())
```

The `reftime` keyword maps the simulation's numeric `clock.time` (seconds) to wall-clock `DateTime`. If `reftime` is `nothing` (the default), the first time-axis value is used as the reference point. Pass `reftime` explicitly when the simulation clock does not start at the first record of the dataset:

```julia
# Simulation starts at t=0 s, but data begins on Jan 1 2000
source = InputSource(grid, raster; reftime = DateTime(2000, 1, 1), domain = Ground())
```

## Multiple input sources

One or more `InputSource`s can be collected in an `InputSources` container:

```julia
inputs = InputSources(source1, source2, source3)
integrator = initialize(model, Heun(Δt = 3600.0); inputs)
```

Internally, `InputSources` will iterate over each source in order when calling `initialize!` and `update_inputs!`.

## Namespaced input variables

Model components may declare their input variables inside a variable [`Namespace`](@ref). For example, a
[`SoilStratigraphy`](@ref) wraps the variables of each soil horizon in a namespace named after that horizon,
and every [`PrescribedSoilHorizon`](@ref) declares separate `sand_fraction`, `silt_fraction`, `clay_fraction`,
and `thickness` inputs. Input sources can target such variables by passing a namespaced path as the `name`:

```julia
inputs = InputSources(
    ## targets the `sand_fraction` input variable in the `organic` horizon namespace
    InputSource(grid, sand_field; name = :organic => :sand_fraction, domain = Ground()),
    ## nested namespaces chain as pairs: ns1 => ns2 => varname
    InputSource(grid, other_field; name = :ns1 => :ns2 => :x, domain = Ground()),
)
```

A plain `Symbol` name always refers to the root namespace; a source named `:sand` will *not* write
into `organic.sand` and vice versa. Internally, names are normalized to a tuple of `Symbol`s via
[`varpath`](@ref):

```@docs; canonical = false
varpath
```

## Using inputs inside process kernels

Input variables are stored in `state.inputs` and are also accessible through the top-level `state` shorthand, just like prognostic or auxiliary variables. Inside a kernel function, inputs appear as named fields and are retrieved the same way as any other field:

```julia
@propagate_inbounds function compute_snow_flux_tendency(i, j, grid, fields, snow_melt::DegreeDaySnow)
    T = fields.air_temperature[i, j, 1]   # input variable
    P = fields.snow_fall[i, j, 1]         # input variable
    k = snow_melt.k
    T_melt = snow_melt.T_melt
    return ifelse(T > T_melt, P - k * (T - T_melt), P)
end
```

The minimum set of input fields needed by a process is declared by including `input` variables in the `variables` method of the relevant `AbstractProcess`:

```julia
Terrarium.variables(snow::DegreeDaySnow{NF}) where {NF} = (
    input(:air_temperature, Surface(XY()), units = u"°C"),
    input(:snow_fall,       Snow(XY()), units = u"m/s"),
    prognostic(:snow_storage, Snow(XY())),
)
```

Terrarium's [`get_fields`](@ref) utility then automatically collects only the fields named in `variables` when assembling the argument list for kernel launch.

## Implementing a custom input source

To add a new input source backend:

1. Define a `struct` subtyping `InputSource{NF, name}` where `NF` is the numeric float
   type and `name` is a `Symbol` identifying the variable.
2. Implement `variables(source::MySource)` returning a tuple of `input(...)` variable
   descriptors.
3. Implement `initialize!(fields, source::MySource, clock)` for any one-time setup.
4. Implement `update_inputs!(fields, source::MySource, clock::Clock)` to update the
   input field at each time step.
5. Optionally provide a convenience `InputSource(grid, ...; name, units, domain = Ground())` constructor
   dispatch so users do not need to reference the concrete type name.

```julia
struct MyInputSource{NF} <: InputSource{NF, :my_var}
    data::Vector{NF}
end

Terrarium.variables(::MyInputSource{NF}) where {NF} = (input(:my_var, Ground(XY())),)

function Terrarium.update_inputs!(fields, source::MyInputSource, clock::Clock)
    # populate fields.my_var from source.data at clock.time
    ...
end
```

