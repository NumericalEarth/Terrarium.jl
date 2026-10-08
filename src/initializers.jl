# Initializer interface

"""
Base type for model initializers. Implementations should provide a dispatch of the `initialize!(state, model::M, init::I)` method where
`M` corresponds to the model type and `I` to the initializer. Initializers may additionally implement
[`variables`](@ref) to declare `input` variables for their (possibly spatially varying) parameters;
these are collected by `variables(model)` alongside the process variables.
"""
abstract type AbstractInitializer{NF} end

# Default implementations of initialize!
initialize!(state, model::AbstractModel, init::AbstractInitializer) = nothing
initialize!(state, model::AbstractModel) = initialize!(state, model, get_initializer(model))

# Fallback dispatch for initialize! on process types
initialize!(state, grid, process::AbstractProcess, args...) = nothing
initialize!(state, grid, ::Nothing, args...) = nothing

"""
    $TYPEDSIGNATURES

Initialize the `state` with `Field` initializers (any valid argument to `set!`) in `inits`.
"""
function initialize!(state, inits::NamedTuple{names}) where {names}
    return fastiterate(names) do name
        set!(getproperty(state, name), inits[name])
    end
end

"""
Marker type for a no-op initializer that leaves all `Field`s set to their default values.
"""
struct DefaultInitializer{NF} <: AbstractInitializer{NF} end

DefaultInitializer(::Type{NF}) where {NF} = DefaultInitializer{NF}()

# Field initializers

"""
Base type for reusable initializers of a single `Field`. Implementations should provide a dispatch of
`initialize!(field, grid, init::AbstractFieldInitializer)`. The default evaluates `init` as a callable
at the node coordinates of `field`, i.e. `init(coords...)`, via `set!`.

A field initializer is accepted everywhere a `set!`-compatible value is: as the `default` of an `input`
variable, as an entry of the `initializers` keyword argument of [`initialize`](@ref), or as a parameter
of a model [`AbstractInitializer`](@ref). In the latter case it may declare `@param` fields, which are
then exposed as model parameters, and the model initializer re-evaluates it at every `initialize!`.
"""
abstract type AbstractFieldInitializer end

"""
    $TYPEDSIGNATURES

Initialize `field` on `grid` with the given field initializer. The default implementation evaluates
`init(coords...)` at the node coordinates of `field`.
"""
initialize!(field::AbstractField, grid::AbstractGrid, init::AbstractFieldInitializer) = set!(field, (coords...) -> init(coords...))

Oceananigans.Fields.set!(field::Field, init::AbstractFieldInitializer) = initialize!(field, field.grid, init)

"""
    $TYPEDSIGNATURES

Re-evaluate `init` into `field` if it is an [`AbstractFieldInitializer`](@ref); all other initial values
(numbers, functions, arrays, `Field`s) are applied only once as construction-time defaults and are left
untouched here so that they remain overridable by [`InputSource`](@ref)s.
"""
reinitialize!(field::AbstractField, init::AbstractFieldInitializer) = set!(field, init)
reinitialize!(::AbstractField, ::Any) = nothing

"""
    $TYPEDEF

Latitude-dependent surface temperature climatology (°C):

    T(φ) = T_equator - |ΔT sin φ|

where `φ` is the latitude in degrees. Supported on a [`ColumnRingGrid`](@ref), where it is evaluated
on the host at the latitude of each active column, and on a `LatitudeLongitudeGrid`, where it is
evaluated at the latitude of each node.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct LatitudinalClimatology{NF} <: AbstractFieldInitializer
    "Mean annual surface temperature at the equator (°C)"
    @param T_equator::NF = 20.0
    "Temperature difference between the equator and the poles (°C)"
    @param ΔT::NF = 40.0
end

LatitudinalClimatology(::Type{NF}; kwargs...) where {NF} = LatitudinalClimatology{NF}(; kwargs...)

(init::LatitudinalClimatology)(latitude) = init.T_equator - abs(init.ΔT * sind(latitude))

function initialize!(field::AbstractField, grid::ColumnRingGrid, init::LatitudinalClimatology)
    column_values = reshape(init.(φnodes(grid)), :, 1, 1)
    # broadcast the per-column values over the remaining (vertical) extent of the field, if any
    set!(field, repeat(column_values, 1, size(field, 2), size(field, 3)))
    return nothing
end

initialize!(field::AbstractField, ::LatitudeLongitudeGrid, init::LatitudinalClimatology) = set!(field, (λ, φ, args...) -> init(φ))

initialize!(field::AbstractField, grid::AbstractLandGrid, init::LatitudinalClimatology) = initialize!(field, ground_domain(grid), init)

# Parameters of initializers are collected from their fields, like for processes. Numbers become
# parameters, parameterized field initializers contribute theirs, and functions contribute none.
function ParameterEditing.parameters(::Type{PT}, init::AbstractInitializer; kwargs...) where {PT <: ParameterEditing.AbstractParam}
    init_params = map(fieldnames(typeof(init))) do name
        name => ParameterEditing.parameters(PT, getproperty(init, name); kwargs...)
    end
    nonempty_params = filter(p -> length(p[2]) > 0, init_params)
    return ParameterEditing.ParameterTable((; nonempty_params...))
end

# `Field`s used as initial values are state, not parameters.
ParameterEditing.parameters(::Type{PT}, ::AbstractField; kwargs...) where {PT <: ParameterEditing.AbstractParam} = (;)
