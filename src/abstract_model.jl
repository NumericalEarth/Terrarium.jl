"""
    $TYPEDEF

Base type for all "processes". Implementations of `AbstractProcess` define equations,
state variables, and parameterizations which characterize the dynamics of a system at
for any given transient state. Note that processes should be largely agnostic to the
details regarding spatial and temporal discretization of the model; i.e. they should
not require specification of a specific grid or time stepping scheme but rather should
be able to operate on any given set of `Field`s and parameters representing the state of
a model at any point in time. Note that process types may also wrap/orchestrate one or
more other process types.
"""
abstract type AbstractProcess{NF} end

"""
    $TYPEDEF

Base type for `AbstractProces` implementations that define a coupling interface for
(typically two or more) sub-processes.
"""
abstract type AbstractCoupledProcesses{NF} <: AbstractProcess{NF} end

"""
    $TYPEDEF

Base type for all Terrarium "models". Models are standalone representations of a system
that consist of

(i) a spatial `grid` characterizing the model domain,
(ii) zero or more `AbstractProcess`es defining the dynamics, and
(iii) an `AbstractInitializer` responsible for defining the initial state of the model.

Implementations of `AbstractModel` are required to implement, at minimum, three methods:
- [`variables`](@ref) which declares the state variables requried by the model,
- [`compute_auxiliary!`](@ref) which is responsible for computing all auxiliary (non-prognostic) variables,
- [`compute_tendencies!`](@ref) which is responsible for computing the tendencies of all prognostic variables.

Note that a default implementation of `variables` is provided which automatically collects all
variables declared by `AbstractProcess`es defined as fields (properties) of `struct`s that subtype `AbstractModel`.
"""
abstract type AbstractModel{NF, Grid <: AbstractLandGrid{NF}}  end

# Method interface for AbstractModel and AbstractProcess

"""
    variables(model::AbstractModel)
    variables(process::AbstractProcess)

Return a `Tuple` of `AbstractVariable`s (i.e. `PrognosticVariable`, `AuxiliaryVariable`, etc.)
defined by the model or process.
"""
function variables end

"""
    initialize!(state, model::AbstractModel)

Initialize all variables defined in `state` which are defined by `model`. This defaults to simply
calling `initialize!(state, model, get_initializer(model))`.

    initialize!(state, model::AbstractModel, initializer::AbstractInitializer)

Initialize the model state variables using the corresponding `initializer`. This method only needs to be
implemented if initialization routines are necessary in addition to direct field/variable initializers.

    initialize!(state, grid, process::AbstractProcess, args...)

Initialize all state variables associated with the given `process` on `grid`. Implementations of
`AbstractProcess` may define additional `args` that correspond to different process coupling interfaces.
"""
function initialize! end

"""
    compute_auxiliary!(state, model::AbstractModel)

Compute updates to all auxiliary variables based on the current prognostic state of the `model`.

    compute_auxiliary!(state, grid, process::AbstractProcess, args...)

Compute all auxiliary state variables for the given `process` on `grid`. Implementations of
`AbstractProcess` may define additional `args` that correspond to different process coupling interfaces.
"""
function compute_auxiliary! end

"""
    compute_boundary_conditions!(state, model::AbstractModel)

Compute all internal and external boundary conditions for spatially explicit prognostic and auxiliary variables
defined on the model. This should typically be invoked *after* `compute_auxiliary!` but *before* `compute_tendencies!`.

    compute_boundary_conditions!(state, grid, process::AbstractProcess, args...)

Compute boundary conditions for all spatially explicit prognostic and auxiliary variables defined by
the given process on `grid`.
"""
function compute_boundary_conditions! end

"""
    compute_tendencies!(state, model::AbstractModel)

Compute tendencies for all prognostic state variables for `model` stored in the given `state`.
This method should be called after `compute_auxiliary!`.

    compute_tendencies!(state, grid, process::AbstractProcess, args...)

Compute the tendencies of all prognostic state variables for the given `process` on `grid`. Implementations of
`AbstractProcess` may define additional `args` that correspond to different process coupling interfaces.
"""
function compute_tendencies! end

# Default method implementations

# Allow variables to be defined on any type, defaulting to an empty tuple
variables(::Any) = ()

"""
    variables(obj::Union{AbstractCoupledProcesses, AbstractModel})

Default implementation of [`variables`](@ref) for composite [`AbstractModel`](@ref) and
[`AbstractCoupledProcesses`](@ref) types that automatically collects all variables from all processes defined
as properties/fields on the given `obj`.
"""
variables(obj::Union{AbstractCoupledProcesses, AbstractModel}) = tuplejoin(fastmap(variables, processes(obj))...)

# Fallback dispatches to make implementing compute_boundary_conditions! optional
compute_boundary_conditions!(state, ::AbstractModel) = nothing
compute_boundary_conditions!(state, grid, ::AbstractProcess, args...) = nothing

# Allow dispatch on nothing for process types
@inline compute_auxiliary!(state, grid, ::Nothing, args...) = nothing
@inline compute_boundary_conditions!(state, grid, ::Nothing, args...) = nothing
@inline compute_tendencies!(state, grid, ::Nothing, args...) = nothing

"""
    processes(obj::Union{AbstractCoupledProcesses, AbstractModel})

Return a tuple of `AbstractProces`es contained in the given model or coupled processes type.
Note that this is a type-stable, `@generated` function that is compiled for each argument type.
"""
@generated function processes(obj::Union{AbstractCoupledProcesses, AbstractModel})
    names = fieldnames(obj)
    types = fieldtypes(obj)
    proc_fields = filter(Tuple(zip(names, types))) do (name, type)
        type <: AbstractProcess
    end
    proc_fieldnames = map(first, proc_fields)
    accessors = map(name -> :(obj.$name), proc_fieldnames)
    return :(tuple($(accessors...)))
end

"""
    closures(proc::AbstractProcess)

Return a tuple of `AbstractClosureRelation`s defined by the given processes type.
Note that this is a type-stable, `@generated` function that is compiled for each argument type.
"""
@generated function closures(proc::AbstractProcess)
    names = fieldnames(proc)
    types = fieldtypes(proc)
    closure_fields = filter(Tuple(zip(names, types))) do (name, type)
        type <: AbstractClosureRelation
    end
    closure_fieldnames = map(first, closure_fields)
    accessors = map(name -> :(obj.$name), closure_fieldnames)
    return :(tuple($(accessors...)))
end

"""
    get_grid(model::AbstractModel)::AbstractLandGrid

Return the spatial grid associated with the given `model`.
"""
@inline get_grid(model::AbstractModel) = model.grid

"""
    get_initializer(model::AbstractModel)::AbstractInitializer

Return the initializer associated with the given `model`.
"""
@inline get_initializer(model::AbstractModel) = model.initializer

"""
    get_constants(model::AbstractModel)::PhysicalConstants

Return the `PhysicalConstants` associated with the given `model`.
"""
@inline get_constants(model::AbstractModel) = model.constants

"""
    closure!(state, model::AbstractModel)

Apply all closure relations defined for the given `model`.
"""
closure!(state, model::AbstractModel) = nothing

"""
    closure!(state, grid, proc::AbstractProcess, args...)

Apply the forward closure mappings for the process `proc` on the given `grid` with additional implementation-specific `args`
defined by the coupling interface for the process type. The default implementation calls `invclosure!` with `args` for each
closure returned by [`closures`](@ref).
"""
closure!(state, grid, proc::AbstractProcess, args...) = fastiterate!(closure -> closure!(state, grid, closure, proc, args...), closures(proc))

# Allow dispatch on nothing for absent process components
closure!(state, grid, ::Nothing, args...) = nothing

"""
    closure!(state, grid, closure::AbstractClosureRelation, process, args...)

Apply `closure` for `process` on the given `grid` with additional implementation-specific `args`.
"""
closure!(state, grid, closure::AbstractClosureRelation, ::AbstractProcess, args...) = nothing

"""
    invclosure!(state, model::AbstractModel)

Apply the inverse of all closure relations defined for the given `model`.
"""
invclosure!(state, model::AbstractModel) = nothing

"""
    invclosure!(state, grid, proc::AbstractProcess, args...)    

Apply the inverse closure mappings for the process `proc` on the given `grid` with additional implementation-specific `args`
defined by the coupling interface for the process type. The default implementation calls `invclosure!` with `args` for each
closure returned by [`closures`](@ref).
"""
invclosure!(state, grid, proc::AbstractProcess, args...) = fastiterate!(closure -> invclosure!(state, grid, closure, proc, args...), closures(proc))

# Allow dispatch on nothing for absent process components
invclosure!(state, grid, ::Nothing, args...) = nothing

"""
    invclosure!(state, grid, closure::AbstractClosureRelation, process::AbstractProcess, args...)

Apply the inverse of `closure` for the process `proc` on the given `grid` with additional implementation-specific `args`
defined by the coupling interface for the process type.
"""
invclosure!(state, grid, closure::AbstractClosureRelation, ::AbstractProcess, args...) = nothing

"""
    (::Type{Model})(grid::AbstractLandGrid; kwargs...) where {Model <: AbstractModel}

Convenience constructor for all `AbstractModel` types that accepts `grid` as a positional argument.
"""
(::Type{Model})(grid::AbstractLandGrid; kwargs...) where {Model <: AbstractModel} = Model(; grid, kwargs...)

"""
    (::Type{Model})(grid::AbstractGrid; kwargs...) where {Model <: AbstractModel}

Default constructor for all `AbstractModel` types that accepts an ordinary spatial discretization
`grid` and builds the model's [`LandGrid`](@ref) from it via [`create_land_grid`](@ref). Models which
resolve additional vertical domains should define their own constructor which passes the relevant
process components to `create_land_grid`.
"""
(::Type{Model})(grid::AbstractGrid; kwargs...) where {Model <: AbstractModel} = Model(create_land_grid(grid); kwargs...)

# Default parameters collection for processes
function ParameterEditing.parameters(proc::AbstractProcess; kwargs...)
    proc_params = map(fieldnames(typeof(proc))) do name
        name => ParameterEditing.parameters(getproperty(proc, name))
    end
    nonempty_params = filter(p -> length(p[2]) > 0, proc_params)
    return ParameterEditing.ParameterTable((; nonempty_params...))
end

function Base.show(io::IO, model::AbstractModel{NF}) where {NF}
    println(io, "$(nameof(typeof(model))){$NF} on $(architecture(get_grid(model)))")
    for name in propertynames(model)
        print(io, "├── $name:  $(summary(getproperty(model, name)))\n")
    end
    return nothing
end

function Base.show(io::IO, processes::AbstractCoupledProcesses{NF}) where {NF}
    println(io, "$(nameof(typeof(processes))){$NF}")
    props = getproperties(processes)
    println(io, "├── Processes:")
    for name in keys(props)
        obj = getproperty(props, name)
        if isa(obj, AbstractProcess)
            print(io, "├──── $name:  $(summary(getproperty(props, name)))\n")
        end
    end
    if any(obj -> !isa(obj, AbstractProcess), values(props))
        println(io, "├── Properties:")
        for name in keys(props)
            obj = getproperty(props, name)
            if !isa(obj, AbstractProcess)
                print(io, "├──── $name:  $(summary(getproperty(props, name)))\n")
            end
        end
    end
    return nothing
end

function Base.show(io::IO, process::AbstractProcess{NF}) where {NF}
    println(io, "$(nameof(typeof(process))){$NF}")
    for name in propertynames(process)
        print(io, "├── $name:  $(summary(getproperty(process, name)))\n")
    end
    return nothing
end
