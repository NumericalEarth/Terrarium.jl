"""
    explicit_timestepper(imex::AbstractIMEX)

Return the explicit sub-stepper of the [`AbstractIMEX`](@ref) timestepper `imex`.
"""
@inline explicit_timestepper(imex::AbstractIMEX) = imex.explicit

"""
    implicit_timestepper(imex::AbstractIMEX)

Return the implicit sub-stepper of the [`AbstractIMEX`](@ref) timestepper `imex`.
"""
@inline implicit_timestepper(imex::AbstractIMEX) = imex.implicit

"""
    $TYPEDEF

Implicit-explicit (IMEX) time stepper that integrates each prognostic variable according to its
[`timestepping`](@ref) class: variables of class [`Explicit`](@ref) are stepped by `explicit` alone, and those
of class [`Implicit`](@ref) receive the explicit stage for their explicitly treated terms followed by an
[`implicit_step!`](@ref) of `implicit`. Only single-stage explicit schemes (currently [`ForwardEuler`](@ref))
are supported as the explicit partner.

Each variable's class is resolved from `timestepping(var, model, imex)`, which defaults to `Explicit()` for
all variables. To integrate selected variables implicitly, specialize [`timestepping`](@ref) on the relevant
variable and/or model types together with the IMEX timestepper. The resolved per-variable classes are stored
in the [`IMEXCache`](@ref) type parameter and used to route variables at each step.

Properties:
$(TYPEDFIELDS)
"""
struct IMEX{
        NF,
        E <: AbstractTimeStepper{NF},
        I <: AbstractTimeStepper{NF},
    } <: AbstractIMEX{NF}
    "Sub-stepper for prognostic variables of class `Explicit` (must have `timestepping(explicit) === Explicit()`)"
    explicit::E

    "Sub-stepper for prognostic variables of class `Implicit` (must have `timestepping(implicit) === Implicit()`)"
    implicit::I
end

"""
    IMEX(explicit, implicit)
    IMEX(; explicit, implicit)

Construct an [`IMEX`](@ref) time stepper from an `explicit` and an `implicit` sub-stepper (which must share
the same numerical type `NF`). Which prognostic variables are integrated implicitly is controlled by
specializing [`timestepping`](@ref).
"""
IMEX(; explicit, implicit) = IMEX(explicit, implicit)

default_dt(imex::AbstractIMEX) = default_dt(explicit_timestepper(imex))

"""
    $TYPEDEF

Cache for an [`AbstractIMEX`](@ref) time stepper. Holds each sub-stepper's own cache; the resolved per-variable
timestepping classes are stored as the leading type parameter `classes` (a tuple of [`Explicit`](@ref)/[`Implicit`](@ref)
instances, in prognostic-variable order) so that routing each variable to its sub-stepper is type stable.

Properties:
$(TYPEDFIELDS)
"""
struct IMEXCache{
        classes,
        NF,
        EC <: AbstractTimeStepperCache{NF},
        IC <: AbstractTimeStepperCache{NF},
    } <: AbstractTimeStepperCache{NF}
    "Cache for the explicit sub-stepper"
    explicit::EC

    "Cache for the implicit sub-stepper"
    implicit::IC
end

# Construct an IMEXCache with the resolved `classes` given as the leading type parameter.
function IMEXCache{classes}(explicit::EC, implicit::IC) where {classes, NF, EC <: AbstractTimeStepperCache{NF}, IC <: AbstractTimeStepperCache{NF}}
    return IMEXCache{classes, NF, EC, IC}(explicit, implicit)
end

function Adapt.adapt_structure(to, cache::IMEXCache{classes}) where {classes}
    return IMEXCache{classes}(Adapt.adapt_structure(to, cache.explicit), Adapt.adapt_structure(to, cache.implicit))
end

"""
    timestepping(cache::IMEXCache)

Return the resolved timestepping class of every prognostic variable (a tuple of [`Explicit`](@ref)/[`Implicit`](@ref)
instances, in prognostic-variable order) held in the [`IMEXCache`](@ref) type parameter.
"""
timestepping(::IMEXCache{classes}) where {classes} = classes

# Build the IMEX cache: resolve every prognostic variable's class (via `timestepping(var, model, imex)`),
# stored as the cache's type parameter, and allocate each sub-stepper's cache for the variables routed to it.
function initialize(imex::AbstractIMEX, state, progvars, model)
    validate_timestepper(imex)
    classes = resolve_timestepping(imex, progvars, model)
    explicit_progvars = select_prognostic_variables(progvars, classes, Explicit())
    implicit_progvars = select_prognostic_variables(progvars, classes, Implicit())
    explicit_cache = initialize(explicit_timestepper(imex), state, explicit_progvars, model)
    implicit_cache = initialize(implicit_timestepper(imex), state, implicit_progvars, model)
    return IMEXCache{classes}(explicit_cache, implicit_cache)
end

"""
    validate_timestepper(imex::AbstractIMEX)

Host-side validation of an [`AbstractIMEX`](@ref) timestepper: the sub-steppers must be given in the
order `IMEX(explicit, implicit)` (checked through their [`timestepping`](@ref) traits), and each
sub-stepper must accept its role via `validate_timestepper(sub_stepper, imex)`.
"""
function validate_timestepper(imex::AbstractIMEX)
    explicit = explicit_timestepper(imex)
    implicit = implicit_timestepper(imex)
    timestepping(explicit) === Explicit() || throw(
        ArgumentError(
            "the explicit sub-stepper of an `IMEX` must be an explicit timestepper, got $(nameof(typeof(explicit))); the sub-steppers are given in the order `IMEX(explicit, implicit)`"
        )
    )
    timestepping(implicit) === Implicit() || throw(
        ArgumentError(
            "the implicit sub-stepper of an `IMEX` must be an implicit timestepper, got $(nameof(typeof(implicit))); the sub-steppers are given in the order `IMEX(explicit, implicit)`"
        )
    )
    validate_timestepper(explicit, imex)
    validate_timestepper(implicit, imex)
    return nothing
end

# Host-side selection of the prognostic variables in `progvars` whose resolved class matches `class`,
# returned as a `NamedTuple` in prognostic-variable order.
function select_prognostic_variables(progvars, classes::Tuple, class::Timestepping)
    selected = Tuple(name for (name, c) in zip(keys(progvars), classes) if c == class)
    return NamedTuple{selected}(map(name -> progvars[name], selected))
end

# A sub-stepper of an IMEX reads its own slice of the cache, selected by its `timestepping` trait.
get_cache(cache::IMEXCache, timestepper::AbstractTimeStepper) = get_cache(cache, timestepping(timestepper))
get_cache(cache::IMEXCache, ::Explicit) = cache.explicit
get_cache(cache::IMEXCache, ::Implicit) = cache.implicit

"""
    $SIGNATURES

Resolve the timestepping class of each prognostic variable in `progvars` (an OrderedDict or
`NamedTuple` of [`PrognosticVariable`](@ref)s) for the IMEX timestepper `imex` and owning `model`, returned as a tuple of
[`Explicit`](@ref)/[`Implicit`](@ref) instances in the same order as `progvars`. Each variable's class is
given by [`timestepping`](@ref)`(var, model, imex)`.
"""
function resolve_timestepping(imex::AbstractIMEX, progvars, model)
    return map(var -> timestepping(var, model, imex), values(progvars))
end

"""
    timestep!(integrator::ModelIntegrator, timestepper::AbstractIMEX, Δt)

Advance the model forward by one IMEX Euler step of size `Δt` using an [`AbstractIMEX`](@ref)
timestepper. The step consists of:

1. a single `update_state!`, in which the tendencies of implicitly routed variables hold only their
   explicitly treated terms (see [`timestepping(state, ::Val)`](@ref timestepping));
2. [`update_cache!`](@ref) on the implicit sub-stepper's cache for the variables whose resolved class
   in the [`IMEXCache`](@ref) is [`Implicit`](@ref), while their prognostic fields still hold `uⁿ`;
3. the explicit sub-stepper's [`explicit_stage!`](@ref) applied to *all* prognostic variables, giving
   `u* = uⁿ + Δt G_exp`, which for implicitly routed variables is the right-hand side of the implicit
   system;
4. [`implicit_step!`](@ref) of the implicit sub-stepper for these variables;
5. the model hook `timestep!(state, model, timestepper, Δt)` followed by `closure!`;
6. a single clock tick.

Explicit terms are therefore evaluated at `tⁿ`.
"""
function timestep!(integrator::ModelIntegrator, timestepper::AbstractIMEX, Δt)
    (; state, model, inputs) = integrator
    # type-stable selection of the implicitly routed prognostic variables
    implicit_names = prognostic_names(state, state.timestepper_cache, Implicit())
    # evaluate auxiliaries and (explicit) tendencies once for the whole step
    update_state!(state, model, inputs; compute_tendencies = true)
    # let the implicit sub-stepper store what it needs from uⁿ before the explicit stage overwrites it
    implicit_cache = get_cache(state.timestepper_cache, implicit_timestepper(timestepper))
    isempty(implicit_names) || update_cache!(implicit_cache, integrator, Δt, implicit_names)
    # explicit stage for all prognostic variables: u* = uⁿ + Δt G_exp
    explicit_stage!(integrator, explicit_timestepper(timestepper), Δt, prognostic_names(state))
    # implicit solve for the implicitly routed variables, starting from u*
    isempty(implicit_names) || implicit_step!(integrator, implicit_timestepper(timestepper), Δt, implicit_names)
    # model-specific post-step corrections and closure relations
    timestep!(state, model, timestepper, Δt)
    closure!(state, model)
    # advance the clock once for the entire step
    tick!(state.clock, Δt)
    return nothing
end

# Select, at compile time, the prognostic variable names whose resolved class matches `class`, by zipping
# the state's prognostic names with the IMEX cache's resolved class list (both in prognostic-variable order).
@generated function prognostic_names(::StateVariables{NF, prognames}, ::IMEXCache{classes}, ::Class) where {NF, prognames, classes, Class}
    selected = Symbol[name for (name, c) in zip(prognames, classes) if c isa Class]
    return Expr(:tuple, map(QuoteNode, selected)...)
end

# Compile-time lookup of the resolved class of the prognostic variable `name`; names that are not
# top-level prognostic variables of `state` are not routed by the IMEX and hence `Explicit()`.
# Under an IMEX, each variable's class is the one resolved in the IMEX cache held by `state`.
@inline timestepping(::AbstractIMEX, state::StateVariables, name::Val) = timestepping(state, name)

@generated function resolved_timestepping(::IMEXCache{classes}, ::StateVariables{NF, prognames}, ::Val{name}) where {classes, NF, prognames, name}
    idx = findfirst(==(name), prognames)
    class = isnothing(idx) ? Explicit() : classes[idx]
    return :($class)
end
