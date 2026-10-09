"""
    $TYPEDEF

Base type for time stepper caches. Each [`AbstractTimeStepper`](@ref) allocates a corresponding
`AbstractTimeStepperCache` subtype (via [`initialize`](@ref)) to hold any working state it needs between
stages/steps.
"""
abstract type AbstractTimeStepperCache{NF} end

"""
    $TYPEDEF

Trivial cache for time steppers that require no working state (e.g. [`ForwardEuler`](@ref)).
"""
struct EmptyCache{NF} <: AbstractTimeStepperCache{NF} end

# AbstractTimeStepper

"""
Base type for time steppers.
"""
abstract type AbstractTimeStepper{NF} end

"""
    $TYPEDEF

Base type for implicit-explicit (IMEX) time steppers. An `AbstractIMEX` combines an explicit and an implicit
sub-stepper. The explicit sub-stepper applies the explicitly treated tendency terms of *all* prognostic variables;
variables of [`timestepping`](@ref) class [`Implicit`](@ref) are subsequently advanced by the implicit sub-stepper
(see [`implicit_step!`](@ref)), while those of class [`Explicit`](@ref) are fully integrated by the explicit stage.

Concrete subtypes (e.g. [`IMEX`](@ref)) must provide the explicit and implicit sub-steppers via
[`explicit_timestepper`](@ref) and [`implicit_timestepper`](@ref); most other behavior is defined here against
`AbstractIMEX`.
"""
abstract type AbstractIMEX{NF} <: AbstractTimeStepper{NF} end

"""
    $TYPEDEF

Trait supertype classifying how a timestepper or prognostic variable is integrated in time, either
[`Explicit`](@ref) or [`Implicit`](@ref). See [`timestepping`](@ref).
"""
abstract type Timestepping end

"""
    $TYPEDEF

[`Timestepping`](@ref) trait marking *explicit* integration. It is the trait of explicit timesteppers
(e.g. [`ForwardEuler`](@ref), [`Heun`](@ref)) and the default class of every prognostic variable — i.e. the
sub-stepper an [`AbstractIMEX`](@ref) routes the variable to.
"""
struct Explicit <: Timestepping end

"""
    $TYPEDEF

[`Timestepping`](@ref) trait marking *implicit* integration. It is the trait of implicit timesteppers and,
under an [`AbstractIMEX`](@ref), of prognostic variables routed to the implicit sub-stepper.
"""
struct Implicit <: Timestepping end

"""
    timestepping(timestepper::AbstractTimeStepper)::Timestepping

Return the [`Timestepping`](@ref) trait — [`Explicit`](@ref) or [`Implicit`](@ref) — of the given
`timestepper`. Every concrete timestepper must define this trait (e.g. `timestepping(::ForwardEuler) =
Explicit()`); there is no default so that a new scheme declares its class explicitly. It is used, among other
things, to route each sub-stepper of an [`AbstractIMEX`](@ref) to its slice of the [`IMEXCache`](@ref).

    timestepping(var::AbstractVariable, model::AbstractModel, timestepper::AbstractTimeStepper)::Timestepping

Return the [`Timestepping`](@ref) class with which the prognostic variable `var` of `model` is integrated
under `timestepper`. Defaults to `Explicit()` for all variables; specialize this method (typically on an
[`AbstractIMEX`](@ref) timestepper together with particular variable and/or model types) to route selected
variables to the implicit sub-stepper.
"""
timestepping(::AbstractVariable, model, ::AbstractTimeStepper) = Explicit()

"""
    default_dt(timestepper::AbstractTimeStepper)

Get the current timestep size for the time stepper.
"""
function default_dt end

"""
    $SIGNATURES

Default `timestepper` for models: a single `explicit` [`ForwardEuler`](@ref) time stepper.
"""
default_timestepper(::Type{NF}) where {NF} = ForwardEuler(NF)

"""
    get_timestepper(model::AbstractModel)::AbstractTimeStepper

Return the `timestepper` associated with the given `model`. All `AbstractModel`s are required to
define a `timestepper` field holding an [`AbstractTimeStepper`](@ref) (e.g. [`ForwardEuler`](@ref),
[`Heun`](@ref), or [`IMEX`](@ref)).
"""
@inline get_timestepper(model::AbstractModel) = model.timestepper

"""
    initialize(timestepper::AbstractTimeStepper, state, progvars, model)

Allocate the time stepper cache for `timestepper` against the given `state`. `progvars` is the named tuple
of prognostic variable metadata and `model` the owning [`AbstractModel`](@ref); both are needed e.g. by
[`AbstractIMEX`](@ref) timesteppers to resolve each variable's [`timestepping`](@ref) class.
"""
initialize(timestepper::AbstractTimeStepper, state, progvars, model) = initialize(timestepper, state)

"""
    timestep!(integrator::ModelIntegrator, timestepper::AbstractTimeStepper, Δt)

Advance prognostic variables of the `integrator` model by one time step based on the current state, or by `Δt` units of time.
"""
function timestep! end

"""
    timestep!(state, model::AbstractModel, timestepper::AbstractTimeStepper, Δt)

Apply any necessary corrections or model-specific time stepping logic after applying `timestepper` to the prognostic state
variables defined by `model`.
"""
timestep!(state, model::AbstractModel, timestepper::AbstractTimeStepper, Δt) = nothing

"""
    initialize(::AbstractTimeStepper, state)

Initialize and return the [`AbstractTimeStepperCache`](@ref) holding any intermediate fields/state required by the
given `timestepper`. Time steppers that need no working state fall back to the default implementation,
which returns an [`EmptyCache`](@ref).
"""
initialize(timestepper::AbstractTimeStepper{NF}, state) where {NF} = EmptyCache{NF}()

"""
    get_cache(cache::AbstractTimeStepperCache, timestepper::AbstractTimeStepper)

Return the working cache for `timestepper` given the model's `state.timestepper_cache`. For a single
timestepper this is the cache itself; an [`IMEX`](@ref) cache returns the sub-cache matching the
timestepper's class.
"""
get_cache(cache::AbstractTimeStepperCache, ::AbstractTimeStepper) = cache

"""
    timestepping(state::StateVariables, ::Val{name})
    timestepping(state::StateVariables, var::AbstractVariable{name})

Return the resolved [`Timestepping`](@ref) class ([`Explicit`](@ref) or [`Implicit`](@ref)) of the
prognostic variable `name` in `state`. Under an [`AbstractIMEX`](@ref) timestepper the class is read
from the type parameter of the [`IMEXCache`](@ref) held in `state.timestepper_cache`; for any other
cache (i.e. a single timestepper) every variable is `Explicit()`. Names that are not top-level
prognostic variables of `state` (including variables in namespaces, which an IMEX does not route)
are also `Explicit()`.

The lookup is resolved at compile time, so this method is type stable and can be used to pass the
trait into process methods (e.g. `compute_tendencies!`) which dispatch on it.
"""
@inline timestepping(state::StateVariables, ::AbstractVariable{name}) where {name} = timestepping(state, Val(name))
@inline timestepping(state::StateVariables, name::Val) = resolved_timestepping(getfield(state, :timestepper_cache), state, name)

# A cache that does not carry per-variable classes (i.e. a single timestepper) integrates every variable explicitly.
@inline resolved_timestepping(::AbstractTimeStepperCache, ::StateVariables, ::Val) = Explicit()

"""
    timestepping(timestepper::Optional{AbstractTimeStepper}, state::StateVariables, ::Val{name})

Return the [`Timestepping`](@ref) class with which the prognostic variable `name` of `state` is
integrated by `timestepper`. This is the form intended for process methods which receive the model's
timestepper and extract the traits of their own variables. A single timestepper integrates all
variables with its own class (see `timestepping(timestepper)`), an [`AbstractIMEX`](@ref) returns the
per-variable class resolved in its [`IMEXCache`](@ref) (see `timestepping(state, ::Val)`), and
`nothing` (no timestepper, e.g. standalone process evaluation) gives `Explicit()`. Type stable.
"""
@inline timestepping(timestepper::AbstractTimeStepper, ::StateVariables, ::Val) = timestepping(timestepper)
@inline timestepping(::Nothing, ::StateVariables, ::Val) = Explicit()

"""
    explicit_stage!(integrator::ModelIntegrator, timestepper::AbstractTimeStepper, Δt, names::Tuple)

Apply a single explicit stage of `timestepper` to the prognostic variables in `names`, using the
tendencies already stored in `integrator.state`. Unlike `timestep!(integrator, timestepper, Δt, names)`,
this neither calls `update_state!` nor applies the model hook or `closure!`; it is the building block
used by an [`AbstractIMEX`](@ref) timestepper, which evaluates the tendencies once and orchestrates
the explicit and implicit parts of the step itself. Only single-stage schemes (currently
[`ForwardEuler`](@ref)) implement this method.
"""
function explicit_stage! end

"""
    validate_timestepper(timestepper::AbstractTimeStepper)
    validate_timestepper(timestepper::AbstractTimeStepper, imex::AbstractIMEX)

Host-side validation of a timestepper configuration, called from [`initialize`](@ref) before any cache
is allocated. The one-argument form validates `timestepper` on its own and is a no-op by default; the
two-argument form validates `timestepper` in its role as a sub-stepper of `imex` and throws an
`ArgumentError` for schemes not supported in that role (e.g. multi-stage schemes such as [`Heun`](@ref)
as the explicit partner). Not exported.
"""
validate_timestepper(::AbstractTimeStepper) = nothing
validate_timestepper(::AbstractTimeStepper, ::AbstractIMEX) = nothing

"""
    update_cache!(cache::AbstractTimeStepperCache, integrator::ModelIntegrator, Δt, names::Tuple)

Update the timestepper `cache` from the current state of `integrator` for the prognostic variables in
`names`, at the beginning of a step (after `update_state!`, while the prognostic fields still hold
`uⁿ`). Timesteppers that need to retain part of the beginning-of-step state implement this method on
their cache type (e.g. the [`BackwardEulerCache`](@ref) of [`BackwardEuler`](@ref) stores a copy of
`uⁿ`). The default is a no-op.
"""
update_cache!(::AbstractTimeStepperCache, integrator, Δt, names) = nothing

"""
    implicit_step!(integrator::ModelIntegrator, timestepper::AbstractTimeStepper, Δt, names::Tuple)

Apply the implicit part of a time step of size `Δt` with the implicit `timestepper` to the
prognostic variables in `names`. This is called by an [`AbstractIMEX`](@ref) timestepper after
`update_state!` and the explicit stage, so for each variable in `names` the prognostic field already
holds `u* = uⁿ + Δt G_exp`, where the tendency `G_exp` contains only the explicitly treated terms.
`u*` is the right-hand side of the implicit system that this method solves in place.

This is Terrarium's own function. It is unrelated to (and does not extend)
`Oceananigans.TimeSteppers.implicit_step!`, which Terrarium does not import.

Implicit timesteppers must implement this method; the fallback throws an `ArgumentError`, and
`implicit_step!(integrator, ::Nothing, Δt, names)` is a no-op.
"""
function implicit_step!(integrator, timestepper::AbstractTimeStepper, Δt, names)
    throw(ArgumentError("$(nameof(typeof(timestepper))) does not implement `implicit_step!` and cannot be used as an implicit timestepper"))
end

implicit_step!(integrator, ::Nothing, Δt, names) = nothing

"""
    $SIGNATURES

Evaluate an explicit update `u += ∂u∂t*Δt` for the prognostic fields of `state` listed in `names` and
their corresponding tendencies. By default, this is implemented as a simple Euler update `u += dudt*Δt`
which can serve as a building block for more complex, multi-stage timesteppers. Where necessary,
additional dispatches of `explicit_step_kernel!(field, tendency, ::AbstractGrid, ::TimeStepper, Δt)`
can be defined to implement more specialized time-stepping schemes.
"""
function explicit_step!(state, grid::AbstractGrid, timestepper::AbstractTimeStepper, Δt, names::Tuple{Vararg{Symbol}})
    # step only this namespace's prognostic variables that are also selected in `names`
    fastiterate(prognostic_names(state)) do name
        if name ∈ names
            # update prognostic state variable
            explicit_step!(state.prognostic[name], state.tendencies[name], grid, timestepper, Δt)
            # debug site post-step
            debugsite!(explicit_step!, state.prognostic[name], name)
        end
    end
    # recurse into child namespaces
    # steps exactly prognostic_names(ns) ∩ names
    fastiterate(state.namespaces) do ns
        explicit_step!(ns, grid, timestepper, Δt, names)
    end
    return nothing
end

"""
Accumulate `tendency*Δt` in the given prognostic `field`. This method can be overridden by specialized
timestepping schemes as needed.
"""
function explicit_step!(
        field::AbstractField{LX, LY, LZ},
        tendency::AbstractField{LX, LY, LZ},
        grid::AbstractGrid{NF},
        timestepper::AbstractTimeStepper,
        Δt,
        args...
    ) where {LX, LY, LZ, NF}
    Δt = convert_dt(NF, Δt)
    launch!(
        grid, XYZ, explicit_step_xyz_kernel!,
        field, tendency, timestepper, Δt, args...
    )
    return nothing
end

"""
Alias for a `Field` restricted to a single vertical index, i.e. one declared at the [`Top`](@ref) or
[`Bottom`](@ref) of a domain. Such a field carries a `Center` or `Face` vertical location, so it is
not distinguishable from a fully resolved (`XYZ`) field by its location parameters alone; the
`indices` type parameter is what separates the two.
"""
const VerticallySlicedField{LX, LY, LZ} = Field{LX, LY, LZ, <:Any, <:Any, Tuple{Colon, Colon, UnitRange{Int}}}

# A vertically sliced field occupies exactly one `k`, so it is stepped by a 2D kernel which resolves
# that index with `end`. Launching the 3D kernel over the full column would index outside the slice.
function explicit_step!(
        field::VerticallySlicedField{LX, LY, LZ},
        tendency::VerticallySlicedField{LX, LY, LZ},
        grid::AbstractGrid{NF},
        timestepper::AbstractTimeStepper,
        Δt,
        args...
    ) where {LX, LY, LZ, NF}
    Δt = convert_dt(NF, Δt)
    launch!(
        grid, XY, explicit_step_slab_kernel!,
        field, tendency, timestepper, Δt, args...
    )
    return nothing
end

function explicit_step!(
        field::AbstractField{LX, LY, Nothing},
        tendency::AbstractField{LX, LY, Nothing},
        grid::AbstractGrid,
        timestepper::AbstractTimeStepper,
        Δt,
        args...
    ) where {LX, LY}
    launch!(
        grid, XY, explicit_step_xy_kernel!,
        field, tendency, timestepper, Δt, args...
    )
    return nothing
end

@kernel function explicit_step_xyz_kernel!(
        field,
        grid,
        tendency,
        ::AbstractTimeStepper,
        Δt
    )
    i, j, k = @index(Global, NTuple)
    u = field
    ∂u∂t = tendency
    @inbounds let Δt = convert(eltype(eltype(tendency)), Δt) # eltype of a scalar type is itself, so this also handles e.g. `SVector` element types
        u[i, j, k] += ∂u∂t[i, j, k] * Δt
    end
end

@kernel function explicit_step_xy_kernel!(
        field,
        grid,
        tendency,
        ::AbstractTimeStepper,
        Δt
    )
    i, j = @index(Global, NTuple)
    u = field
    ∂u∂t = tendency
    @inbounds let Δt = convert(eltype(eltype(tendency)), Δt) # eltype of a scalar type is itself, so this also handles e.g. `SVector` element types
        u[i, j, 1] += ∂u∂t[i, j, 1] * Δt
    end
end

@kernel function explicit_step_slab_kernel!(
        field,
        grid,
        tendency,
        ::AbstractTimeStepper,
        Δt
    )
    i, j = @index(Global, NTuple)
    u = field
    ∂u∂t = tendency
    @inbounds let Δt = convert(eltype(eltype(tendency)), Δt) # eltype of a scalar type is itself, so this also handles e.g. `SVector` element types
        u[i, j, end] += ∂u∂t[i, j, end] * Δt
    end
end

# Default debug hooks
@inline debughook!(::typeof(explicit_step!), field, name) = checkfinite!(field, name)
