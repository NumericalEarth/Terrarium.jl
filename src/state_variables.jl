abstract type AbstractStateVariables end

"""
    $TYPEDEF

Container type for all `Field`s corresponding to state variables defined by a model.
`StateVariables` partitions the fields into three categories: prognostic, tendencies, and
auxiliary. Prognostic variables are those which characterize the state of the system and
are assigned tendencies to be integrated by the timestepper. Auxiliary fields are additional
state variables derived from the prognostic state variables but which are conditionally
independent of their values at the previous time step given the current prognostic state.
It is worth noting that tendencies are also treated internally as auxiliary variables;
however, they are assigned their own category here since they need to be handled separately
by the timestepping scheme.
"""
struct StateVariables{
        NF,
        prognames, closurenames, auxnames, inputnames, nsnames,
        ProgFields, TendFields, AuxFields, InputFields, Namespaces,
        Cache,
        ClockType,
    } <: AbstractStateVariables
    prognostic::NamedTuple{prognames, ProgFields}
    tendencies::NamedTuple{prognames, TendFields}
    auxiliary::NamedTuple{auxnames, AuxFields}
    inputs::NamedTuple{inputnames, InputFields}
    namespaces::NamedTuple{nsnames, Namespaces}
    timestepper_cache::Cache
    clock::ClockType

    function StateVariables(
            ::Type{NF},
            closurenames::Tuple{Vararg{Symbol}},
            prognostic::NamedTuple{prognames, ProgFields},
            tendencies::NamedTuple{prognames, TendFields},
            auxiliary::NamedTuple{auxnames, AuxFields},
            inputs::NamedTuple{inputnames, InputFields},
            namespaces::NamedTuple{nsnames, Namespaces},
            timestepper_cache::Cache,
            clock::ClockType,
        ) where {
            NF, prognames, auxnames, inputnames, nsnames,
            ProgFields, TendFields, AuxFields, InputFields, Namespaces, Cache, ClockType,
        }
        return new{
            NF, prognames, closurenames, auxnames, inputnames, nsnames,
            ProgFields, TendFields, AuxFields, InputFields, Namespaces, Cache, ClockType,
        }(
            prognostic,
            tendencies,
            auxiliary,
            inputs,
            namespaces,
            timestepper_cache,
            clock,
        )
    end

end

# Name getters (always type-stable, inlined constant propagation)
@inline prognostic_names(state::StateVariables) = keys(getfield(state, :prognostic))
@inline auxiliary_names(state::StateVariables) = keys(getfield(state, :auxiliary))
@inline input_names(state::StateVariables) = keys(getfield(state, :inputs))
@inline namespace_names(state::StateVariables) = keys(getfield(state, :namespaces))
@inline closure_names(::StateVariables{NF, pnames, cnames}) where {NF, pnames, cnames} = cnames

# Allow reconstruction from properties
ConstructionBase.constructorof(::Type{<:StateVariables{NF, pnames, cnames}}) where {NF, pnames, cnames} = (args...) -> StateVariables(NF, cnames, args...)

"""
    update_state!(state::StateVariables, model::AbstractModel, inputs::InputSources; compute_tendencies = true)

Update the `state` for the given `model` and `inputs`; this includes calling `update_inputs!` and
`fill_halo_regions!` followed by `compute_auxiliary!` and `compute_tendencies!`, if `compute_tendencies = true`.
"""
function Oceananigans.TimeSteppers.update_state!(state::StateVariables, model::AbstractModel, inputs::InputSources; compute_tendencies = true)
    reset_tendencies!(state)
    update_inputs!(state, get_grid(model), inputs)
    compute_auxiliary!(state, model)
    compute_boundary_conditions!(state, model)
    if compute_tendencies
        compute_tendencies!(state, model)
    end
    return nothing
end

"""
Reset all `Field`s in `state` to zero.
"""
function Oceananigans.TimeSteppers.reset!(state::StateVariables)
    # reset all prognostic fields
    fastiterate(state.prognostic) do field
        set!(field, zero(eltype(field)))
    end
    fastiterate(state.auxiliary) do field
        # TODO: technically we should apply auxiliary variable initializers here
        isa(field, Field) && set!(field, zero(eltype(field)))
    end
    # reset all tendency fields
    fastiterate(state.tendencies) do field
        set!(field, zero(eltype(field)))
    end
    # recurse over namespaces
    return fastiterate(state.namespaces) do ns
        reset!(ns)
    end
end

"""
Reset all tendencies in `state` to zero.
"""
function reset_tendencies!(state::StateVariables)
    # reset all tendency fields
    fastiterate(state.tendencies) do field
        set!(field, zero(eltype(field)))
    end
    # recurse over namespaces
    return fastiterate(state.namespaces) do ns
        reset_tendencies!(ns)
    end
end

"""
Initialize input variables from the given input `sources`. The `scope` corresponds to the
path of namespace names from the root namespace to `state` and is used to match namespaced
input sources to their target variables; see [`varpath`](@ref).
"""
function initialize!(state::StateVariables, grid::AbstractGrid, sources::InputSources, scope::Tuple{Vararg{Symbol}} = ())
    # initialize inputs in current namespace, passing the full state as read-only `fields`
    initialize!(state.inputs, grid, state.clock, state, sources, scope)
    # recursively initialize namespaces
    fastiterate(namespace_names(state)) do nsname
        initialize!(getproperty(getfield(state, :namespaces), nsname), grid, sources, (scope..., nsname))
    end
    return nothing
end

"""
Update input variables from the given input `sources`. The `scope` corresponds to the
path of namespace names from the root namespace to `state` and is used to match namespaced
input sources to their target variables; see [`varpath`](@ref).
"""
function update_inputs!(state::StateVariables, grid::AbstractGrid, sources::InputSources, scope::Tuple{Vararg{Symbol}} = ())
    # update inputs in current namespace, passing the full state as read-only `fields`
    update_inputs!(state.inputs, grid, state.clock, state, sources, scope)
    # debug: check all inputs are finite
    debugsite!(state.inputs)
    # recursively update namespaces
    fastiterate(namespace_names(state)) do nsname
        update_inputs!(getproperty(getfield(state, :namespaces), nsname), grid, sources, (scope..., nsname))
    end
    return nothing
end

"""
    get_fields(state, queries::Union{Symbol, Pair}...)

Retrieves fields with names given in `queries` and returns them in a `NamedTuple`. Each argument
in `queries` can either be a `Symbol` corresponding to a field/variable defined in the namespace
of `state` or a `Pair{Symbol, Tuple}` where the key is the child namespace and the value is a
tuple of queries from that namespace.

!!! warning
    This method relies on runtime dispatch and thus should not be used in performance-critical code.
    If you need to query fields for specific sets of variables or components, use one of the
    type-stable variants instead.

```julia
# initialize model state
state = StateVariables(model)
# get the temperature and saturation_water_ice fields
fields = get_fields(state, :temperature, :saturation_water_ice)
# extract temperature as well as variables from a namespace
nested_fields = get_fields(state, :temperature, :namespace => (:subvar1, :subvar2))
```
"""
function get_fields(state, queries::Union{Symbol, Pair}...)
    fields = map(queries) do query
        if isa(query, Symbol)
            query => getproperty(state, query)
        else
            isa(query, Pair)
            key, value = query
            @assert isa(value, Tuple) "namespaces queries must be given as tuples"
            key => get_fields(getproperty(state, key), value...)
        end
    end
    return (; fields...)
end

"""
    $TYPEDSIGNATURES

Retrieves the `Field` from `state` matching the `name` of the given variable.
"""
@inline get_field(state, ::Union{AbstractVariable{name}, Namespace{name}}) where {name} = getproperty(state, name)

# NOTE: use `$SIGNATURES` (not `$TYPEDSIGNATURES`) here. `$TYPEDSIGNATURES` performs return-type
# inference via `Base.return_types`, which crashes with a `BoundsError` in `may_invoke_generator`
# when applied to this `@generated` method on Julia 1.10, breaking the docs build.
"""
    $SIGNATURES

Retrieves all `Field`s from `state` matching the names of the given variables. Any `Namespace`s
in `vars` are resolved recursively and their fields are merged into the returned `NamedTuple`
keyed by namespace name, with the namespace's own fields collected into a nested `NamedTuple`.
"""
@generated get_fields(state, vars::Tuple{Vararg{Union{AbstractVariable, Namespace}}}) = _get_fields_expr(:state, :vars, vars)

"""
    $TYPEDSIGNATURES

Retrieves all `Field`s declared by the given `Namespace` from `state`, where `state` is assumed
to correspond to the (nested) `StateVariables` of the namespace itself.
"""
@inline get_fields(state, ns::Namespace) = get_fields(state, ns.vars)

"""
    $SIGNATURES

Retrieves all non-tendency `Field`s from `state` defined on the given `components`.
"""
@inline function get_fields(state, components...; except = (;))
    component_vars = fastmap(components) do component
        allvars = variables(component)
        closurevars = closure_variables(component)
        tuplejoin(allvars, closurevars)
    end
    vars = tuplejoin(component_vars...)
    component_fields = get_fields(state, vars)
    return ntdiff(component_fields, except)
end

"""
    $SIGNATURES

Retrieves all `Field`s from `state` corresponding to prognostic variables defined on the given `components`.
"""
@inline function prognostic_fields(state, components...)
    component_progvars = fastmap(prognostic_variables, components)
    progvars = tuplejoin(component_progvars...)
    return get_fields(state, progvars)
end

"""
    $SIGNATURES

Retrieves all `Field`s from `state` corresponding to tendencies defined on the given `components`.
"""
@inline function tendency_fields(state, components...)
    component_progvars = fastmap(prognostic_variables, components)
    progvars = tuplejoin(component_progvars...)
    return get_fields(state.tendencies, progvars)
end

"""
    $SIGNATURES

Retrieves all `Field`s from `state` corresponding to auxiliary variables defined on the given `components`.
"""
@inline function auxiliary_fields(state, components...)
    component_auxvars = fastmap(auxiliary_variables, components)
    auxvars = tuplejoin(component_auxvars...)
    return get_fields(state, auxvars)
end

"""
    $SIGNATURES

Retrieves all `Field`s from `state` corresponding to closure variables defined on the given `components`.
"""
@inline function closure_fields(state, components...)
    component_closurevars = fastmap(closure_variables, components)
    closurevars = tuplejoin(component_closurevars...)
    return get_fields(state, closurevars)
end

"""
    $SIGNATURES

Retrieves all `Field`s from `state` corresponding to input variables defined on the given `components`.
"""
@inline function input_fields(state, components...)
    component_inputvars = fastmap(input_variables, components)
    inputvars = tuplejoin(component_inputvars...)
    return get_fields(state, inputvars)
end

# Initialization of StateVariables from models and processes
"""
    $TYPEDSIGNATURES

Initialize a `StateVariables` data structure containing `Field`s for all variables defined by `model` defined on its
associated `grid`. The `clock` specifies the initial simulation time and is mutated on each time step. User-specified
`boundary_conditions` and `initializers` can be provided as `NamedTuple`s with keys corresponding to the names of state
variables to which they should be applied. If the state variables are defined within namespaces, the given `NamedTuple`
must follow the same structure. The `fields` argument allows for manual preconstruction of `Field`s for the named state
variables. The time stepper cache is allocated from the model's `timestepper`.
"""
function StateVariables(
        model::AbstractModel{NF},
        params = nothing;
        clock = Clock(time = zero(NF)),
        input_variables = (),
        boundary_conditions = (;),
        initializers = (;),
        fields = (;)
    ) where {NF}
    model_rec = isnothing(params) ? model : ParameterEditing.reconstruct(model, params)
    vars = Variables(tuplejoin(variables(model_rec), input_variables))
    state = StateVariables(vars, model_rec.grid; clock, timestepper = get_timestepper(model), model = model_rec, boundary_conditions, initializers, fields)
    return state
end

"""
    $TYPEDSIGNATURES

Initialize a `StateVariables` data structure containing `Field`s defined on the given `grid`
for all variables defined by `process`. Any predefined `boundary_conditions` and `fields` will
be passed through to `initialize` for each variable.

The `grid` may be either a land grid or an ordinary spatial discretization, which is converted to
a land grid via [`create_land_grid`](@ref).
"""
function StateVariables(
        process::AbstractProcess{NF},
        grid::AbstractGrid{NF},
        params = nothing;
        clock = Clock(time = zero(NF)),
        input_variables = (),
        timestepper = default_timestepper(NF),
        boundary_conditions = (;),
        initializers = (;),
        fields = (;)
    ) where {NF}
    process_rec = isnothing(params) ? process : ParameterEditing.reconstruct(process, params)
    vars = Variables(tuplejoin(variables(process_rec), input_variables))
    state = StateVariables(vars, grid; clock, timestepper, boundary_conditions, initializers, fields)
    return state
end

# Initialization from variable metadata

"""
    $TYPEDSIGNATURES

Initialize a `StateVariables` data structure containing `Field`s defined on the given `grid`
for all variables in `vars`. Any predefined `boundary_conditions` and `fields` will be passed
through to `initialize` for each variable. The `timestepper`'s cache is allocated via
`initialize(timestepper, state, progvars)`.

The `grid` may be either a land grid or an ordinary spatial discretization, which is converted to
a land grid via [`create_land_grid`](@ref); state variables always live on a land grid.
"""
function StateVariables(
        vars::Variables,
        grid::AbstractGrid{NF};
        clock::Clock = Clock(time = 0.0),
        timestepper = default_timestepper(NF),
        model = nothing,
        boundary_conditions = (;),
        initializers = (;),
        fields = (;)
    ) where {NF}
    # State variables always live on a land grid, so an ordinary spatial discretization is converted
    # to one here exactly as the model constructors do; this is a no-op for a land grid.
    land_grid = create_land_grid(grid)
    # Initialize Fields for each variable group, if they are not already given in the user defined `fields`.
    fields_dict = OrderedDict{Symbol, AbstractField}(pairs(fields))
    input_fields_dict = initialize(vars.inputs, land_grid, clock, fields_dict, boundary_conditions)
    tendency_fields_dict = initialize(vars.tendencies, land_grid, clock, fields_dict, boundary_conditions)
    prognostic_fields_dict = initialize(vars.prognostic, land_grid, clock, merge(fields_dict, input_fields_dict), boundary_conditions)
    auxiliary_fields_dict = initialize(vars.auxiliary, land_grid, clock, merge(fields_dict, input_fields_dict, prognostic_fields_dict), boundary_conditions)
    # recursively initialize state variables for each namespace
    namespaces = map(values(vars.namespaces)) do ns
        ns_bcs = get(boundary_conditions, varname(ns), (;))
        ns_fields = get(fields, varname(ns), (;))
        varname(ns) => StateVariables(variables(ns), land_grid; clock, boundary_conditions = ns_bcs, fields = ns_fields)
    end
    # get closure variable names
    closurenames = map(varname, closure_variables(values(vars.prognostic)))
    # Convert OrderedDicts to NamedTuples for the final StateVariables construction.
    # This single conversion per group is negligible compared to the per-iteration merge() cost.
    prognostic_fields = NamedTuple{Tuple(keys(prognostic_fields_dict))}(values(prognostic_fields_dict))
    tendency_fields = NamedTuple{Tuple(keys(tendency_fields_dict))}(values(tendency_fields_dict))
    auxiliary_fields = NamedTuple{Tuple(keys(auxiliary_fields_dict))}(values(auxiliary_fields_dict))
    input_fields = NamedTuple{Tuple(keys(input_fields_dict))}(values(input_fields_dict))
    namespaces_nt = NamedTuple(namespaces)
    # construct StateVariables with an empty cache; the timestepper-specific cache
    # is allocated below now that all other state variables have been initialized
    initial_state = StateVariables(
        NF,
        closurenames,
        prognostic_fields,
        tendency_fields,
        auxiliary_fields,
        input_fields,
        namespaces_nt,
        EmptyCache{NF}(),
        clock,
    )
    # allocate the timestepper's cache
    cache = initialize(timestepper, initial_state, NamedTuple(vars.prognostic), model)
    state = StateVariables(
        NF,
        closurenames,
        prognostic_fields,
        tendency_fields,
        auxiliary_fields,
        input_fields,
        namespaces_nt,
        cache,
        clock,
    )
    # Apply Field initializers
    initialize!(state, initializers)
    return state
end

"""
    $TYPEDSIGNATURES

Initialize `Field`s on `grid` for each of the variables in the given OrderedDict `vars`.
Any predefined `boundary_conditions` and `fields` will be passed through to `initialize`
for each variable.
"""
function initialize(
        vars::OrderedDict{Symbol, <:AbstractVariable},
        grid::AbstractLandGrid,
        clock::Clock,
        fields::OrderedDict{Symbol, AbstractField},
        boundary_conditions::NamedTuple,
    )
    # Initialize or retrieve Fields for each variable in `var`, accumulating the newly created Fields in a named tuple;
    # Note that one major caveat to this approach is that the Fields visible to each constructor are dependent on the order
    # in which the variables were declared :/
    new_fields = OrderedDict{Symbol, AbstractField}()
    context_fields = OrderedDict{Symbol, AbstractField}(pairs(fields))
    for (name, var) in vars
        field = initialize(var, grid, clock, context_fields, boundary_conditions)
        context_fields[name] = field
        new_fields[name] = field
    end
    return new_fields
end

# Convenience dispatch that accepts `fields` as a NamedTuple and converts to OrderedDict
initialize(
    var::AbstractVariable,
    grid::AbstractLandGrid,
    clock::Clock,
    fields::NamedTuple,
    boundary_conditions::NamedTuple,
) = initialize(var, grid, clock, OrderedDict{Symbol, AbstractField}(pairs(fields)), boundary_conditions)

"""
    $TYPEDSIGNATURES    

Initialize a `Field` on `grid` based on the given `var` metadata. The named tuple of `boundary_conditions` should follow the standard convention of
`(var1 = (; top, bottom, ...), var2 = (; top, bottom, ...))`. If `user_fields` contains a `Field` matching the name of `var`, this field
will be directly returned. Otherwise check the `accumulated` dict for fields from previous groups.
Otherwise, the new `Field` is constructed using the given `boundary_conditions`.
"""
function initialize(
        var::AbstractVariable,
        grid::AbstractLandGrid,
        clock::Clock,
        fields::OrderedDict{Symbol, AbstractField},
        boundary_conditions::NamedTuple,
    )
    name = varname(var)
    if haskey(fields, name)
        return fields[name]
    else
        bcs = get(boundary_conditions, name, nothing)
        field = Field(grid, vardims(var), bcs)
        # if field is an input variable and has a default value/initializer, call set! on it
        if isa(var, InputVariable) && !isnothing(var.default)
            set!(field, var.default)
        end
        return field
    end
end

# Intialization for auxiliary variables that may define custom Field constructors
"""
    $TYPEDSIGNATURES

Initialize a `Field` on `grid` for the given [`AuxiliaryVariable`](@ref).
"""
function initialize(
        var::AuxiliaryVariable,
        grid::AbstractLandGrid,
        clock::Clock,
        fields::OrderedDict{Symbol, AbstractField},
        boundary_conditions::NamedTuple
    )
    name = varname(var)
    if haskey(fields, name)
        return fields[name]
    elseif isnothing(var.ctor)
        # retrieve boundary condition (if any) and create Field
        bcs = get(boundary_conditions, name, nothing)
        return Field(grid, vardims(var), bcs)
    else
        # invoke field constructor if specified
        return var.ctor(var, grid, clock, NamedTuple(fields))
    end
end

# Base overrides

function Adapt.adapt_structure(to, state::StateVariables{NF}) where {NF}
    return StateVariables(
        NF,
        closure_names(state),
        Adapt.adapt_structure(to, state.prognostic),
        Adapt.adapt_structure(to, state.tendencies),
        Adapt.adapt_structure(to, state.auxiliary),
        Adapt.adapt_structure(to, state.inputs),
        Adapt.adapt_structure(to, state.namespaces),
        Adapt.adapt_structure(to, state.timestepper_cache),
        Adapt.adapt_structure(to, state.clock),
    )
end

Base.eltype(::StateVariables{NF}) where {NF} = NF

Base.propertynames(state::StateVariables) = tuplejoin(
    prognostic_names(state),
    auxiliary_names(state),
    input_names(state),
    namespace_names(state),
    fieldnames(typeof(state)),
)

function Base.getproperty(state::StateVariables, name::Symbol)
    # forward getproperty calls to variable groups
    if name ∈ prognostic_names(state)
        return getproperty(getfield(state, :prognostic), name)
    elseif name ∈ auxiliary_names(state)
        return getproperty(getfield(state, :auxiliary), name)
    elseif name ∈ input_names(state)
        return getproperty(getfield(state, :inputs), name)
    elseif name ∈ namespace_names(state)
        return getproperty(getfield(state, :namespaces), name)
    else
        return getfield(state, name)
    end
end

# helper function e.g. for usage with Enzyme
function Base.fill!(state::StateVariables{NF}, value) where {NF}
    for progname in prognostic_names(state)
        fill!(getproperty(state.prognostic, progname), NF(value))
    end
    for tendname in keys(state.tendencies)
        fill!(getproperty(state.tendencies, tendname), NF(value))
    end
    for auxname in keys(state.auxiliary)
        fill!(getproperty(state.auxiliary, auxname), NF(value))
    end
    return nothing
end

function Base.copyto!(state::SV, other::SV) where {SV <: StateVariables}
    for progname in prognostic_names(state)
        copyto!(getproperty(state.prognostic, progname), getproperty(other.prognostic, progname))
    end
    for tendname in prognostic_names(state)
        copyto!(getproperty(state.tendencies, tendname), getproperty(other.tendencies, tendname))
    end
    for auxname in auxiliary_names(state)
        copyto!(getproperty(state.auxiliary, auxname), getproperty(other.auxiliary, auxname))
    end
    for inputname in input_names(state)
        copyto!(getproperty(state.inputs, inputname), getproperty(other.inputs, inputname))
    end
    for nsname in namespace_names(state)
        copyto!(getproperty(state.namespaces, nsname), getproperty(other.namespaces, nsname))
    end
    set!(state.clock, other.clock)
    return nothing
end

function Base.summary(state::StateVariables{NF}) where {NF}
    clockstr = summary(state.clock)
    str = "StateVariables{$NF}(clock = $clockstr, prognostic = $(keys(state.prognostic)), auxiliary = $(keys(state.auxiliary)), inputs = $(keys(state.inputs)), namespaces = $(keys(state.namespaces)), timestepper_cache = $(nameof(typeof(state.timestepper_cache))))"
    return str
end

function Base.show(io::IO, state::StateVariables{NF}) where {NF}
    println(io, "StateVariables{$NF}")
    print(io, "├─ Clock: $(state.clock)")
    println(io)
    print(io, "├─ Prognostic: ")
    show(io, state.prognostic)
    println(io)
    print(io, "├─ Auxiliary: ")
    show(io, state.auxiliary)
    println(io)
    print(io, "├─ Inputs: ")
    show(io, state.inputs)
    println(io)
    print(io, "├─ Namespaces: $(keys(state.namespaces))")
    println(io)
    return print(io, "└─ Timestepper cache: $(nameof(typeof(state.timestepper_cache)))")
end

"""
Generation-time helper for [`get_fields`](@ref). Given the type `Vars` of a tuple of
`AbstractVariable`s and `Namespace`s, builds an expression that retrieves all matching fields
from `state_ex` (an expression evaluating to the state container) and `vars_ex` (an expression
evaluating to the variable tuple). The recursion over namespaces is performed here, at expansion
time, so that the generated body for `get_fields` contains no self-call. This is what makes the
method type stable: a runtime self-recursive `get_fields` would otherwise trigger inference's
recursion limiting and widen the return type of the nested namespace lookups to an abstract
`NamedTuple`.
"""
function _get_fields_expr(state_ex, vars_ex, ::Type{Vars}) where {Vars}
    types = collect(Vars.parameters)
    names = map(varname, types)
    # deduplicate by name (keep first occurrence), as in `deduplicate_vars`
    unique_idx = unique(i -> names[i], eachindex(types))
    plain_idx = filter(i -> types[i] <: AbstractVariable, unique_idx)
    ns_idx = filter(i -> types[i] <: Namespace, unique_idx)
    plain_names = Tuple(names[i] for i in plain_idx)
    plain_fields = map(i -> :(get_field($state_ex, $vars_ex[$i])), plain_idx)
    fields = :(NamedTuple{$plain_names}(tuple($(plain_fields...))))
    isempty(ns_idx) && return fields
    ns_names = Tuple(names[i] for i in ns_idx)
    ns_fields = map(ns_idx) do i
        substate, subvars = gensym(:state), gensym(:vars)
        # `Namespace{name, Vars}` => recurse on the nested variable tuple type `Vars`
        inner = _get_fields_expr(substate, subvars, Vars.parameters[i].parameters[2])
        quote
            let $substate = getproperty($state_ex, $(QuoteNode(names[i]))), $subvars = $vars_ex[$i].vars
                $inner
            end
        end
    end
    return :(merge($fields, NamedTuple{$ns_names}(tuple($(ns_fields...)))))
end
