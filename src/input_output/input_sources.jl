"""
    $TYPEDEF

Base type for input data sources. Implementations of `InputSource` are free to load data
from any arbitrary backend. They expect an `initialize!(inputs, grid, clock, fields, ::InputSource)` that
is called once at model initialization and an `update_inputs!(inputs, grid, clock, fields, ::InputSource)`
method that is called at every time step, where `inputs` are the input `Field`s to be written, `grid` is
the model grid, `clock` the simulation clock, and `fields` the full (read-only) model state. Both default
to doing nothing. Implementations should additionally provide a constructor as a dispatch of `InputSource`.
Namespace routing (`scope`) is handled by the enclosing [`InputSources`] container.

The type argument `NF` corresponds to the numeric type of the input data, `name` to its name that's also used in its `variables` definition. 
"""
abstract type InputSource{NF, name} end

# Default kwarg constructor for convenience
InputSource(; kwargs...) = InputSource(kwargs...)

"""
    $SIGNATURES

Retrieve the [`VarLocation`](@ref) of the variable this input source provides, and its constituent
[`VarDims`](@ref) and [`VarDomain`](@ref). The dimensions are inferred from the source data; the
domain is given when the source is constructed and must match the variable it feeds.
"""
@inline varloc(source::InputSource) = source.loc
@inline vardims(source::InputSource) = vardims(varloc(source))
@inline vardomain(source::InputSource) = vardomain(varloc(source))

"""
    $TYPEDSIGNATURES

Returns a tuple of `Symbol`s corresponding to variable names supported by this `InputSource`.
"""
variables(::InputSource) = ()

varpath(::InputSource{NF, name}) where {NF, name} = varpath(name)

"""
    $TYPEDSIGNATURES

Returns the name of the input variable provided by this source, i.e. the last entry of its [`varpath`](@ref).
"""
varname(source::InputSource) = last(varpath(source))

"""
    $TYPEDSIGNATURES

Determine whether the given `source` provides an input variable for the namespace at the given
`scope`, where `scope` is the path of namespace names from the root namespace, i.e. `()` for the
root namespace itself, `(:ns1,)` for its child namespace `ns1`, and so on.
"""
matches_scope(source::InputSource, scope::VarPath) = Base.front(varpath(source)) == scope

"""
    $TYPEDSIGNATURES

Initializes the input `source`, writing into the input `inputs` fields at model start. The `grid` and
the full model state `fields` (read-only) are provided so that sources may compute their inputs from the
grid geometry or other state variables. Namespace routing is handled by the caller (see [`InputSources`]).
Default implementation does nothing.
"""
initialize!(inputs, grid, clock, fields, ::InputSource) = nothing

"""
    $TYPEDSIGNATURES

Updates the values of the input variables stored in `inputs` from the given input `source`, called at
every time step. The `grid` and the full model state `fields` (read-only) are provided so that sources
may compute their inputs from the grid geometry or other state variables. Namespace routing is handled by
the caller (see [`InputSources`]). Default implementation returns `nothing`.
"""
update_inputs!(inputs, grid, clock, fields, ::InputSource) = nothing

"""
Type alias for an `AbstractField` with any X, Y, Z location or grid.
"""
const AnyField{NF} = AbstractField{LX, LY, LZ, G, NF} where {LX, LY, LZ, G}

"""
Container type for wrapping multiple `InputSource`s.
"""
struct InputSources{NF, name, Sources <: Tuple{Vararg{InputSource{NF}}}} <: InputSource{NF, name}
    sources::Sources
end

InputSources(::Type{NF}, name = nothing) where {NF} = InputSources{NF, name, Tuple{}}(())
InputSources(input::InputSources) = input
InputSources(input::InputSource{NF}, others::InputSource{NF}...) where {NF} = InputSources(nothing, input, others...)
function InputSources(name, input::InputSource{NF}, others::InputSource{NF}...) where {NF}
    sources = tuple(input, others...)
    return InputSources{NF, name, typeof(sources)}(sources)
end

variables(sources::InputSources) = tuplejoin(map(variables, sources.sources)...)

varname(::InputSources) = nothing

matches_scope(sources::InputSources, scope) = any(map(src -> matches_scope(src, scope), sources.sources))

function initialize!(inputs, grid, clock, fields, input::InputSources, scope::VarPath = ())
    for source in input.sources
        matches_scope(source, scope) && initialize!(inputs, grid, clock, fields, source)
    end
    return nothing
end

function update_inputs!(inputs, grid, clock, fields, input::InputSources, scope::VarPath = ())
    for source in input.sources
        matches_scope(source, scope) && update_inputs!(inputs, grid, clock, fields, source)
    end
    return nothing
end

"""
    $TYPEDEF

Input source that defines `input` state variables with the given names which
can then be directly modified by the user.
"""
struct FieldInputSource{NF, name, VL <: VarLocation, FS <: AnyField{NF}, UT} <: InputSource{NF, name}
    "Variable location: spatial dimensions and the model domain the variable lives on"
    loc::VL

    "Physical units"
    units::UT

    "Input field"
    field::FS
end

function initialize!(inputs, grid, clock, fields, source::FieldInputSource)
    name = varname(source)
    if hasproperty(inputs, name)
        field = getproperty(inputs, name)
        set!(field, source.field)
    end
    return nothing
end

"""
    $TYPEDSIGNATURES

Create a `FieldInputSource` with the given grid and input variable `fields`. Use it for static input fields.
The `name` can either be a plain `Symbol` or a namespaced path; see [`varpath`](@ref).

The spatial dimensions are inferred from `field`. The `domain` defaults to [`Surface`](@ref), which
is where most forcing data lives; give it explicitly for a source feeding a variable on another
domain. It must match the [`VarDomain`](@ref) that variable is declared with, since a variable
declared on two different domains is a conflict rather than a merge, and is reported as one.
"""
function InputSource(grid::AbstractGrid{NF}, field::FS; name, domain::VarDomain = Surface(), units = NoUnits) where {NF, FS <: AnyField{NF}}
    # ensure fields are on the same architecture as the grid
    field = on_architecture(architecture(grid), field)

    # Check that fields match grid
    @assert field.grid == ground_domain(grid) "Field must have the same grid as the input grid"

    # infer the VarDims and subsequently the Field location from the data dimensions
    loc = VarLocation(Terrarium.vardims(field), domain)

    path = varpath(name)
    return FieldInputSource{NF, path, typeof(loc), typeof(field), typeof(units)}(loc, units, field)
end

"""
    $TYPEDSIGNATURES

Convenience function to create a `FieldInputSource` from a `RingGrids.Field`.
Converts the RingGrids field to an Oceananigans field and then creates the input source.
"""
function InputSource(grid::ColumnRingGrid{NF}, ring_field::RingGrids.AbstractField; name, domain::VarDomain = Surface(), units = NoUnits) where {NF}
    oceananigans_field = Field(ring_field, grid)
    loc = VarLocation(Terrarium.vardims(oceananigans_field), domain)
    path = varpath(name)
    return FieldInputSource{NF, path, typeof(loc), typeof(oceananigans_field), typeof(units)}(loc, units, oceananigans_field)
end

# Land grids delegate to the discretization of the ground domain, which carries the ring grid.
InputSource(grid::AbstractLandGrid, ring_field::RingGrids.AbstractField; kwargs...) = InputSource(ground_domain(grid), ring_field; kwargs...)

variables(source::FieldInputSource) = tuple(with_scope(Base.front(varpath(source)), input(varname(source), source.loc; units = source.units)))

"""
Type alias for a `FieldTimeSeries` with any X, Y, Z location or grid.
"""
const AnyFieldTimeSeries{NF} = FieldTimeSeries{LX, LY, LZ, TI, K, I, D, G, NF} where {LX, LY, LZ, TI, K, I, D, G}

"""
    $TYPEDEF

Input source that reads input fields from pre-specified Oceananigans `FieldTimeSeries`.
"""
struct FieldTimeSeriesInputSource{NF, name, VL <: VarLocation, FTS <: AnyFieldTimeSeries{NF}, TT, UT} <: InputSource{NF, name}
    "Variable location: spatial dimensions and the model domain the variable lives on"
    loc::VL

    "Physical units"
    units::UT

    "Reference time"
    reftime::TT

    "Field time series data"
    fts::FTS
end

function InputSource(fts::AnyFieldTimeSeries{NF}; name, domain::VarDomain = Surface(), reftime = first(fts.times), units = NoUnits) where {NF}
    loc = VarLocation(vardims(fts), domain)
    path = varpath(name)
    return FieldTimeSeriesInputSource{NF, path, typeof(loc), typeof(fts), typeof(reftime), typeof(units)}(loc, units, reftime, fts)
end

variables(source::FieldTimeSeriesInputSource) = tuple(with_scope(Base.front(varpath(source)), input(varname(source), source.loc; units = source.units)))

# to initialize just update the state once at the start time
function initialize!(inputs, grid, clock, fields, source::FieldTimeSeriesInputSource)
    return update_inputs!(inputs, grid, clock, fields, source)
end

function update_inputs!(inputs, grid, clock, fields, source::FieldTimeSeriesInputSource)
    name = varname(source)
    t = timestamp(eltype(source.fts.times), source.reftime, clock.time)
    if hasproperty(inputs, name)
        field_t = getproperty(inputs, name)
        set!(field_t, source.fts[Time(t)])
    end
    return nothing
end

# Internal helper method to check that all Field dimensions match
function checkdims(; named_fields...)
    @assert length(named_fields) >= 1 "at least one input field must be provided"
    # infer dimensions of all provided fields
    field_dims = map(vardims, values(named_fields))
    @assert length(field_dims) == 1 || foldl(==, field_dims) "all fields must have matching dimensions"
    return first(field_dims)
end
