"""
    $TYPEDEF

Helper type used for specifying a single point or cross-section in space along the X, Y, or Z axis.
If `val = nothing`, the coordinate is undefined or integrated over the extent of the axis.
"""
@kwdef struct Coordinate{V, L}
    val::V = nothing
    loc::L = Face()
end

# Dispatch for Oceananigans `location` method
Oceananigans.location(dims::Coordinate) = dims.loc

# A `Coordinate` resolves to a single index along its axis. An integer selects that index directly;
# a function (`firstindex`/`lastindex`) is applied to the range of valid indices along the axis, so
# that the same `Coordinate` means the top face (`Nz + 1`) or the top cell center (`Nz`) according to
# the location it carries.
Oceananigans.Fields.indices(axis, dims::Coordinate{<:Integer}) = dims.val
Oceananigans.Fields.indices(axis, dims::Coordinate{<:Function}) = dims.val(axis)

# Abstract state variable types

"""
    $TYPEDEF

Indicator type describing the location on which a variable should be instantiated on an Oceananigans grid.
The fields `x`, `y`, and `z` should be one of: `Center()` or `Face()` for variables that should be discretized
on grid cell centers or faces along each axis, `Coordinate` for variables that are defined at a single point along an axis,
or `nothing` for variables that represented quantities integrated over a domain. 
"""
@kwdef struct VarDims{LX, LY, LZ}
    x::LX = Center()
    y::LY = Center()
    z::LZ = Center()
end

# Resolve one axis of a `VarDims` to its Oceananigans location. `nothing` means the variable has no
# extent along that axis; a `Coordinate` contributes the location it is defined at.
@inline axis_location(::Nothing) = nothing
@inline axis_location(loc::CenterOrFace) = loc
@inline axis_location(coord::Coordinate) = location(coord)

# The range of valid indices of `grid` along dimension `dim` at location `loc`. This is what the
# `firstindex`/`lastindex` of a `Coordinate` are resolved against, and it is why a `Coordinate` must
# carry its location: along a bounded axis there are `N` cell centers but `N + 1` faces.
@inline axis_range(grid::AbstractGrid, dim::Int, loc) =
    Base.OneTo(Oceananigans.Grids.total_length(loc, topology(grid, dim)(), size(grid, dim)))

# Resolve one axis of a `VarDims` to the corresponding entry of the `indices` argument of the `Field`
# constructor. Anything which is not a `Coordinate` spans the whole axis.
@inline axis_indices(_, ::Union{Nothing, CenterOrFace}) = Colon()
@inline axis_indices(axis, coord::Coordinate) = indices(axis, coord)

Oceananigans.location(dims::VarDims) = (axis_location(dims.x), axis_location(dims.y), axis_location(dims.z))

Oceananigans.Fields.indices(grid::AbstractGrid, dims::VarDims) = (
    axis_indices(axis_range(grid, 1, axis_location(dims.x)), dims.x),
    axis_indices(axis_range(grid, 2, axis_location(dims.y)), dims.y),
    axis_indices(axis_range(grid, 3, axis_location(dims.z)), dims.z),
)

# VarDims aliases

"""
    XY(x = Center(), y = Center(), z = nothing)

Dimensions for a variable with no vertical extent, i.e. one assigned a 2D (lateral only) field on
its associated grid. This covers both genuinely two-dimensional quantities and those integrated or
averaged over a domain's vertical extent.
"""
const XY = VarDims{LX, LY, LZ} where {LX <: CenterOrFace, LY <: CenterOrFace, LZ <: Union{Nothing, Coordinate}}

XY(x::CenterOrFace, y::CenterOrFace = Center(), z::Union{Nothing, Coordinate} = nothing) = VarDims(x, y, z)
XY(; x::CenterOrFace = Center(), y::CenterOrFace = Center(), z::Union{Nothing, Coordinate} = nothing) = VarDims(x, y, z)

"""
    Top(x = Center(), y = Center(), z = Face())

Dimensions for a variable defined at a single point at the *top* of a domain's vertical axis, such
as a surface flux. The vertical index is resolved with `lastindex`, so a field declared this way is
restricted to one `k`: `Nz + 1` at `Face` (the default, appropriate for fluxes across the interface)
and `Nz` at `Center` (the uppermost cell).

See also [`Bottom`](@ref).
"""
const Top{TZ} = VarDims{LX, LY, LZ} where {LX <: CenterOrFace, LY <: CenterOrFace, TZ <: CenterOrFace, LZ <: Coordinate{typeof(lastindex), TZ}}

Top(x::CenterOrFace, y::CenterOrFace = Center(), z::CenterOrFace = Face()) = VarDims(x, y, Coordinate(lastindex, z))
Top(; x::CenterOrFace = Center(), y::CenterOrFace = Center(), z::CenterOrFace = Face()) = VarDims(x, y, Coordinate(lastindex, z))

"""
    Bottom(x = Center(), y = Center(), z = Face())

Dimensions for a variable defined at a single point at the *bottom* of a domain's vertical axis,
such as a basal flux. The vertical index is resolved with `firstindex`, so a field declared this way
is restricted to `k = 1`.

See also [`Top`](@ref).
"""
const Bottom{TZ} = VarDims{LX, LY, LZ} where {LX <: CenterOrFace, LY <: CenterOrFace, TZ <: CenterOrFace, LZ <: Coordinate{typeof(firstindex), TZ}}

Bottom(x::CenterOrFace, y::CenterOrFace = Center(), z::CenterOrFace = Face()) = VarDims(x, y, Coordinate(firstindex, z))
Bottom(; x::CenterOrFace = Center(), y::CenterOrFace = Center(), z::CenterOrFace = Face()) = VarDims(x, y, Coordinate(firstindex, z))

"""
    XYZ(x = Center(), y = Center(), z = Center())

Dimensions for a variable which is resolved over a domain's full vertical extent, i.e. one assigned
a 3D field on its associated grid.
"""
const XYZ = VarDims{LX, LY, LZ} where {LX <: CenterOrFace, LY <: CenterOrFace, LZ <: CenterOrFace}

XYZ(x::CenterOrFace, y::CenterOrFace = Center(), z::CenterOrFace = Center()) = VarDims(x, y, z)
XYZ(; x::CenterOrFace = Center(), y::CenterOrFace = Center(), z::CenterOrFace = Center()) = VarDims(x, y, z)

"""
    $SIGNATURES

Infer the appropriate `VarDims` from the given `Field`.

This infers the dimensions of an *externally supplied* field, e.g. one wrapped by an
[`InputSource`](@ref), and returns only `XY` or `XYZ`. It is deliberately not the inverse of the
`Field` constructor: a field created from a [`Top`](@ref) or [`Bottom`](@ref) variable reports `XY`,
because recovering the `Coordinate` would mean inferring intent from the field's indices. Where the
domain or the coordinate matters, read the variable's declared [`VarLocation`](@ref) instead.
"""
vardims(::AbstractField{LX, LY, Nothing}) where {LX, LY} = XY(LX(), LY())
vardims(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = XYZ(LX(), LY(), LZ())

"""
    $TYPEDEF

Represents the "location" of an abstract variable, i.e. both the spatial domain and
its dimensionality.

The `domain` may be `nothing`, which declares a variable as domain-agnostic: it makes no claim about
where the variable lives, so it is compatible with any domain and adopts whichever one another
declaration of the same variable states. This is the default for an [`InputSource`](@ref), which
generally cannot know the domain of the variable it feeds, and it is also the natural choice for a
model discretized on a plain `AbstractGrid`, which has only one discretization.

On an [`AbstractLandGrid`](@ref) a variable which is still domainless once all declarations are
merged is allocated on the ground domain. That is unambiguous for a 2D variable, since every domain
shares the same horizontal discretization, but a variable with a vertical extent or position also
warns; see `domain_matters`.
"""
struct VarLocation{Dims, Domain}
    dims::Dims
    domain::Domain
end

VarLocation(dims::VarDims) = VarLocation(dims, nothing)

vardims(var::VarLocation) = var.dims
vardomain(var::VarLocation) = var.domain

Base.summary(::Ground) = "ground"
Base.summary(::Snow) = "snow"
Base.summary(::Canopy) = "canopy"
Base.summary(::Surface) = "surface"
Base.summary(::Atmosphere) = "atmosphere"

# Variables may be declared without a domain, so the places which name a variable's domain need a
# word for its absence. Defining `Base.summary(::Nothing)` would be type piracy.
@inline domain_summary(domain::VarDomain) = summary(domain)
@inline domain_summary(::Nothing) = "unspecified"

"""
    $SIGNATURES

Whether two declarations of the same variable agree on its domain. A declaration without a domain
makes no claim and is compatible with any: this is what lets an [`InputSource`](@ref), which
generally cannot know which domain the variable it feeds belongs to, stay domain-agnostic and adopt
whatever the declaring process states. Two *different* stated domains remain a conflict.
"""
@inline domains_compatible(d1::VarDomain, d2::VarDomain) = d1 == d2
@inline domains_compatible(::Nothing, ::VarDomain) = true
@inline domains_compatible(::VarDomain, ::Nothing) = true
@inline domains_compatible(::Nothing, ::Nothing) = true

# Aliased constructors for VarLocation on the three domains
Ground(dims::VarDims) = VarLocation(dims, Ground())
Snow(dims::VarDims) = VarLocation(dims, Snow())
Canopy(dims::VarDims) = VarLocation(dims, Canopy())
Surface(dims::XY) = VarLocation(dims, Surface())
Surface(::XYZ) = error("surface variables must be 2D (XY)")
Atmosphere(dims::XY) = VarLocation(dims, Atmosphere())
Atmosphere(::XYZ) = error("atmospheric forcing variables must be 2D (XY)")

"""
Base type for state variable placeholder types.
"""
abstract type AbstractVariable{name, VL, UT} end

"""
    $SIGNATURES

Retrieve the name of the given variable or closure. For closure relations, `varname`
should return the name of the variable returned by the closure relation.
"""
@inline varname(::AbstractVariable{name}) where {name} = name
@inline varname(::Type{<:AbstractVariable{name}}) where {name} = name
@inline varname(namespace::Pair{Symbol}) = first(namespace)

"""
    $SIGNATURES

Retrieve the [`VarLocation`](@ref) of this variable, i.e. both its grid dimensions and the model
domain it is defined on.
"""
@inline varloc(var::AbstractVariable) = var.loc
@inline varloc(::Type{<:AbstractVariable{name, VL}}) where {name, VL} = VL

"""
    $SIGNATURES

Retrieve the grid dimensions on which this variable is defined.
"""
@inline vardims(var::AbstractVariable) = vardims(varloc(var))
@inline vardims(::Type{<:AbstractVariable{name, <:VarLocation{Dims}}}) where {name, Dims} = Dims

"""
    $SIGNATURES

Retrieve the grid domain on which this variable is defined.
"""
@inline vardomain(var::AbstractVariable) = vardomain(varloc(var))
@inline vardomain(::Type{<:AbstractVariable{name, <:VarLocation{Dims, Domain}}}) where {name, Dims, Domain} = Domain

"""
    $SIGNATURES

Retrieve the physical units for the given variable.
"""
@inline varunits(var::AbstractVariable) = var.units
@inline varunits(::Type{<:AbstractVariable{name, VL, UT}}) where {name, VL, UT} = UT

# Test equality between variables by their names, dimensions, domains, and physical units
Base.:(==)(var1::AbstractVariable, var2::AbstractVariable) =
    varname(var1) == varname(var2) &&
    vardims(var1) == vardims(var2) &&
    vardomain(var1) == vardomain(var2) &&
    varunits(var1) == varunits(var2)

function Base.summary(var::AbstractVariable)
    unitstr = varunits(var) == NoUnits ? "-" : varunits(var)
    text = "$(string(varname(var))) [$(unitstr)] on $(typeof(vardims(var))) of the $(domain_summary(vardomain(var))) domain"
    return text
end

"""
    $TYPEDEF

Represents metadata for a generic state variable with the given `name` and spatial `loc`.
"""
struct Variable{name, VL, UT} <: AbstractVariable{name, VL, UT}
    "Variable location"
    loc::VL

    "Physical units"
    units::UT

    Variable(name::Symbol, loc::VarLocation, units::Units = NoUnits) = new{name, typeof(loc), typeof(units)}(loc, units)
end

"""
    $TYPEDEF

Base type for prognostic variable closure relations for differential equations of the form:

```math
\\frac{\\partial g(u)}{\\partial t} = F(u)
```
where `F` represents the RHS tendency as a function of the state variable `u`, and `g(u)` is a closure or constitutive
relation that maps `u` to the physical units matching the tendency. Common examples in soil hydrothermal modeling
are temperature-enthalpy and saturation-pressure relations.
"""
abstract type AbstractClosureRelation end

"""
Baste type for process state variables with specific intents, e.g. `prognostic`, `auxiliary`, or `input`.
"""
abstract type AbstractProcessVariable{name, VL, UT} <: AbstractVariable{name, VL, UT} end

@inline varloc(pv::AbstractProcessVariable) = varloc(pv.var)
@inline vardims(pv::AbstractProcessVariable) = vardims(pv.var)
@inline vardomain(pv::AbstractProcessVariable) = vardomain(pv.var)
@inline varunits(pv::AbstractProcessVariable) = varunits(pv.var)

function Base.show(io::IO, ::MIME"text/plain", var::AbstractVariable)
    units = varunits(var)
    domain = domain_summary(vardomain(var))
    return if units != NoUnits
        println(io, "$(nameof(typeof(var))) $(varname(var)) on the $domain domain with dimensions $(typeof(vardims(var))) and units $(string(varunits(var)))")
    else
        println(io, "$(nameof(typeof(var))) $(varname(var)) on the $domain domain with dimensions $(typeof(vardims(var)))")
    end
end

"""
    $TYPEDEF

Represents an auxiliary (a.k.a "diagnostic") state variable with the given `name`
and spatial `dims`. Auxiliary variables are those which are diagnosed directly or
indirectly from the values of one or more prognostic variables.
"""
struct AuxiliaryVariable{
        name,
        VL <: VarLocation,
        UT <: Units,
        Var <: Variable{name, VL, UT},
        BT <: DomainSets.AbstractInterval,
        FC,
    } <: AbstractProcessVariable{name, VL, UT}
    "State variable"
    var::Var

    "Field constructor"
    ctor::FC

    "Bounds for numerical bounds of the variable"
    bounds::BT

    "Variable description"
    desc::String
end

"""
    $TYPEDEF

Represents a spatially varying input (e.g. forcing) variable with the given `name` and spatial `dims`.
Input variables can also be made to vary in time through the use of [`InputSource`](@ref)s.
"""
struct InputVariable{
        name,
        VL <: VarLocation,
        UT <: Units,
        Var <: Variable{name, VL, UT},
        BT <: DomainSets.AbstractInterval,
        Def <: Union{Nothing, Number, Function},
    } <: AbstractProcessVariable{name, VL, UT}
    "State variable"
    var::Var

    "Default value or function initializer"
    default::Def

    "Variable bounds"
    bounds::BT

    "Variable description"
    desc::String
end

Adapt.adapt_structure(to, var::InputVariable) = var.var

"""
    $TYPEDEF

Represents a prognostic state variable with the given `name` and spatial `dims`. Prognostic variables
are those which are integrated by the timestepper and fully define the state of the system at any given
point in (simulation) time. From a computational perspective, they can be seen as the "roots" of the
computational graph for `update_state!`/`timestep!`. Prognostic variables generally should not be modified
by any code not belonging to the timestepper or user. They automatically define a `tendency` (auxiliary)
variable which is used to hold the value of their instantaneous time derivative computed by `compute_tendencies!`.
"""
struct PrognosticVariable{
        name,
        VL <: VarLocation,
        UT <: Units,
        Var <: Variable{name, VL, UT},
        CL <: Union{Nothing, AbstractClosureRelation},
        TV <: Union{Nothing, AuxiliaryVariable},
        BT <: DomainSets.AbstractInterval,
    } <: AbstractProcessVariable{name, VL, UT}
    "State variable"
    var::Var

    "Closure relation for the tendency of the prognostic variable"
    closure::CL

    "Variable corresponding to the tendency for prognostic variables"
    tendency::TV

    "Variable bounds"
    bounds::BT

    "Variable description"
    desc::String
end

hasclosure(var::PrognosticVariable) = !isnothing(var.closure)

"""
    $TYPEDEF

Represents a new variable namespace, typically from a subcomponent of the model.
"""
struct Namespace{name, Vars}
    vars::Vars

    Namespace(name::Symbol, vars) = new{name, typeof(vars)}(vars)
end

@inline varname(ns::Namespace{name}) where {name} = varname(typeof(ns))
@inline varname(::Type{<:Namespace{name}}) where {name} = name

variables(ns::Namespace) = getfield(ns, :vars)

Base.propertynames(ns::Namespace) = (:vars, propertynames(getfield(ns, :vars))...)
Base.getproperty(ns::Namespace, name::Symbol) = name == :vars ? getfield(ns, :vars) : getproperty(getfield(ns, :vars), name)

# Variable container

"""
    $TYPEDEF

Container for abstract state variable definitions. Automatically collates and merges all variables
and namespaces passed into the constructor. Uses OrderedDicts internally to avoid NamedTuple type
explosion during initialization (each merge in a foldl creates a new distinct type), converting to
NamedTuples only at the final StateVariables construction step.
"""
struct Variables
    prognostic::OrderedDict{Symbol, AbstractVariable}
    tendencies::OrderedDict{Symbol, AuxiliaryVariable}
    auxiliary::OrderedDict{Symbol, AbstractVariable}
    inputs::OrderedDict{Symbol, AbstractVariable}
    namespaces::OrderedDict{Symbol, Namespace}
end

"""
    VarPath

Type alias for namespaced variable paths of the form `(namespace_1, ..., namespace_N, varname)`.
Used to specify the location of variables in nested namespaces.
"""
const VarPath = Tuple{Vararg{Symbol}}

"""
    varpath(name::Symbol)
    varpath(path::Pair)
    varpath(path::Tuple{Vararg{Symbol}})

Normalize the given variable name into a path of the form `(namespace_1, ..., namespace_N, varname)`.
Plain `Symbol` names correspond to variables in the root namespace, i.e. the path `(varname,)`. Namespaced
variables can be specified either as `Pair`s, e.g. `:ns1 => :ns2 => :varname`, or directly as a tuple of
`Symbol`s, e.g. `(:ns1, :ns2, :varname)`.
"""
varpath(name::Symbol) = (name,)
varpath(path::VarPath) = path
varpath(path::Pair) = (Symbol(first(path)), varpath(last(path))...)

"""
    $SIGNATURES

Wrap the given variable `var` in nested `Namespace`s according to the `path`, where `path` is the
namespace scope (i.e. the sequence of enclosing namespace names, *excluding* the variable's own name).
An empty `path` returns `var` unwrapped.
"""
with_scope(path::VarPath, var::AbstractVariable) =
    isempty(path) ? var : namespace(first(path), (with_scope(Base.tail(path), var),))

Variables(obj) = Variables(variables(obj))
Variables(vars::Variables) = vars
Variables(vars::Union{AbstractProcessVariable, Namespace}...) = Variables(vars)
"""
    $SIGNATURES

Describe how two declarations of the same variable disagree. Used to report incompatible duplicates
in terms of the attribute which differs rather than by printing both variables and leaving the
reader to spot it.
"""
function describe_conflict(var1::AbstractVariable, var2::AbstractVariable)
    differences = String[]
    vardims(var1) != vardims(var2) && push!(differences, "dimensions $(typeof(vardims(var1))) vs $(typeof(vardims(var2)))")
    !domains_compatible(vardomain(var1), vardomain(var2)) && push!(differences, "domain $(domain_summary(vardomain(var1))) vs $(domain_summary(vardomain(var2)))")
    varunits(var1) != varunits(var2) && push!(differences, "units $(varunits(var1)) vs $(varunits(var2))")
    return isempty(differences) ? "differing declarations" : join(differences, ", ")
end

function Variables(vars::Tuple{Vararg{Union{AbstractProcessVariable, Namespace}}})
    # partition variables into prognostic, auxiliary, input, and namespace groups;
    # duplicates within each group are automatically merged
    varmeta(var::AbstractVariable) = (varname(var), vardims(var), varunits(var))
    varmeta(ns::Namespace) = varname(ns)
    # The domain is compared separately from the rest of the metadata because a domainless
    # declaration is compatible with any domain rather than equal to it.
    compatible(v1, v2) = varmeta(v1) == varmeta(v2) && domains_compatible(vardomain(v1), vardomain(v2))
    function register!(vardict::AbstractDict, var::AbstractVariable)
        name = varname(var)
        if !haskey(vardict, name)
            vardict[name] = var
        elseif !compatible(vardict[name], var)
            error("Found incompatible duplicates of variable $name: $(describe_conflict(var, vardict[name]))")
        elseif isnothing(vardomain(vardict[name])) && !isnothing(vardomain(var))
            # the stated domain wins over the agnostic one
            vardict[name] = var
        else
            vardict[name] = first(merge(vardict[name], var))
        end
        return nothing
    end
    # Namespaces carry no domain, so they are merged on their name alone.
    function register!(nsdict::AbstractDict, ns::Namespace)
        name = varname(ns)
        nsdict[name] = haskey(nsdict, name) ? first(merge(nsdict[name], ns)) : ns
        return nothing
    end
    # create OrderedDicts for each variable type
    prognostic_vars = OrderedDict{Symbol, AbstractVariable}()
    tendency_vars = OrderedDict{Symbol, AuxiliaryVariable}()
    auxiliary_vars = OrderedDict{Symbol, AbstractVariable}()
    input_vars = OrderedDict{Symbol, InputVariable}()
    namespaces = OrderedDict{Symbol, Namespace}()
    # register prognostic variables variables
    for var in filter(var -> isa(var, PrognosticVariable), vars)
        register!(prognostic_vars, var)
    end
    # recursively collect all closure variables; this needs to come
    # before auxiliary variable registration so that closure variable
    # Fields are available to auxiliary variable Field constructors
    for var in filter(var -> isa(var, PrognosticVariable) && hasclosure(var), vars)
        closure_vars = Variables(variables(var.closure))
        @assert isempty(closure_vars.prognostic) "Closures are not allowed to declare prognostic variables"
        @assert isempty(closure_vars.namespaces) "Closures are not allowed to declare namespaces"
        merge!(auxiliary_vars, closure_vars.auxiliary)
        merge!(input_vars, closure_vars.inputs)
    end
    # register auxiliary variables
    for var in filter(var -> isa(var, AuxiliaryVariable), vars)
        register!(auxiliary_vars, var)
    end
    # register input variables
    for var in filter(var -> isa(var, InputVariable), vars)
        name = varname(var)
        # only register input variables if they are not already declared as prognostic or auxiliary
        haskey(prognostic_vars, name) || haskey(auxiliary_vars, name) || register!(input_vars, var)
    end
    # register namespaces recursively
    for ns in filter(var -> isa(var, Namespace), vars)
        name = varname(ns)
        ns_vars = variables(ns)
        inner = Namespace(name, Variables(ns_vars))
        register!(namespaces, inner)
    end
    # register tendencies for prognostic variables
    for var in values(prognostic_vars)
        tendency_vars[varname(var)] = var.tendency
    end
    # check for illegal duplicates across variable groups
    check_duplicates(
        values(prognostic_vars)...,
        values(auxiliary_vars)...,
        values(input_vars)...,
        values(namespaces)...
    )
    return Variables(
        prognostic_vars,
        tendency_vars,
        auxiliary_vars,
        input_vars,
        namespaces,
    )
end

"""
Check for variables/namespaces with duplicate names and raise an error if duplicates are detected. Not type stable.
"""
function check_duplicates(vars::Union{AbstractVariable, Namespace}...)
    names = unique(map(varname, vars))
    groups = Dict(map(n -> n => filter(==(n) ∘ varname, vars), names)...)
    for key in keys(groups)
        if length(groups[key]) > 1
            error("Found conflicting variable/namespace definitions for $key:\n$(join(groups[key], "\n"))")
        end
    end
    return
end

"""
    deduplicate_vars(vars::Tuple{Vararg{Union{AbstractVariable, Namespace}}})

Type-stable equivalent of [`deduplicate`](@ref) for tuples of `AbstractVariable`s and `Namespace`s.
"""
@generated function deduplicate_vars(vars::Tuple{Vararg{Union{AbstractVariable, Namespace}}})
    names = map(varname, vars.parameters)
    unique_idx = unique(i -> names[i], eachindex(vars.parameters))
    accessors = map(i -> :(vars[$i]), unique_idx)
    return :(tuple($(accessors...)))
end

"""
Merges all of the given `Variables` containers into a single container.
"""
function Base.merge(varss::Variables...)
    allvars = map(varss) do vars
        tuplejoin(
            values(vars.prognostic),
            values(vars.auxiliary),
            values(vars.inputs),
            values(vars.namespaces)
        )
    end
    return Variables(reduce(tuplejoin, allvars))
end

function Base.merge(varss::Tuple{Vararg{Union{AbstractVariable, Namespace}}}...)
    return tuplejoin(varss...)
end

function Base.merge(vars::AbstractVariable...)
    unique_vars = unique(vars)
    return Tuple(unique_vars)
end

function Base.merge(namespaces::Namespace...)
    names = unique(map(varname, namespaces))
    merged = map(names) do name
        group = filter(ns -> varname(ns) == name, namespaces)
        length(group) == 1 ? group[1] : Namespace(name, merge(map(ns -> Variables(variables(ns)), group)...))
    end
    return Tuple(merged)
end

function Base.propertynames(vars::Variables)
    fieldnames = (:prognostic, :tendencies, :auxiliary, :inputs, :namespaces)
    prognames = keys(getfield(vars, :prognostic))
    auxnames = keys(getfield(vars, :auxiliary))
    inputnames = keys(getfield(vars, :inputs))
    nsnames = keys(getfield(vars, :namespaces))
    return tuplejoin(fieldnames, prognames, auxnames, inputnames, nsnames)
end

function Base.getproperty(vars::Variables, name::Symbol)
    # forward getproperty calls to variable groups
    if name ∈ keys(getfield(vars, :prognostic))
        return getfield(vars, :prognostic)[name]
    elseif name ∈ keys(getfield(vars, :auxiliary))
        return getfield(vars, :auxiliary)[name]
    elseif name ∈ keys(getfield(vars, :inputs))
        return getfield(vars, :inputs)[name]
    elseif name ∈ keys(getfield(vars, :namespaces))
        return getfield(vars, :namespaces)[name]
    else
        return getfield(vars, name)
    end
end

function Base.summary(vars::Variables)
    str = "Variables(prognostic = $(keys(vars.prognostic)), auxiliary = $(keys(vars.auxiliary)), inputs = $(keys(vars.inputs)), namespaces = $(keys(vars.namespaces)))"
    return str
end

function Base.show(io::IO, vars::Variables)
    println(io, "Variables")
    println(io, "├─ Prognostic: ")
    for var in values(vars.prognostic)
        println(io, "├── $(summary(var))")
    end
    println(io, "├─ Auxiliary: ")
    for var in values(vars.auxiliary)
        println(io, "├── $(summary(var))")
    end
    println(io, "├─ Inputs: ")
    for var in values(vars.inputs)
        println(io, "├── $(summary(var))")
    end
    println(io, "├─ Namespaces:")
    for ns in values(vars.namespaces)
        println(io, "├── $(summary(ns))")
    end
    return nothing
end

# Automatically forward dispatches for `show` on tuples of variables to Variables;
# This is for the convenience of the user such that `variables(model)` pretty prints
function Base.show(
        io::IO,
        vartup::Tuple{Union{AbstractVariable, Namespace}, Vararg{Union{AbstractVariable, Namespace}}}
    )
    vars = Variables(vartup)
    show(io, vars)
    return nothing
end

"""
    $SIGNATURES

Convenience constructor for `Variable`. A variable normally states the domain it lives on by
wrapping its [`VarDims`](@ref) in a [`VarDomain`](@ref), e.g. `var(:temperature, Ground(XYZ()))` or
`var(:snow_temperature, Snow(XY()))`.

Bare dimensions, e.g. `var(:u, XY())`, declare the variable without a domain. This is for models
discretized on a plain `AbstractGrid`, where there is only one discretization and naming a domain
would say nothing. Prefer stating the domain in any model which runs on an
[`AbstractLandGrid`](@ref): there the domain is a real choice, and a reader of the declaration
should not have to infer it.
"""
@inline var(name::Symbol, loc::VarLocation, units::Units = NoUnits) = Variable(name, loc, units)

@inline var(name::Symbol, dims::VarDims, units::Units = NoUnits) = Variable(name, VarLocation(dims), units)

"""
    $SIGNATURES

Convenience constructors for `PrognosticVariable`.
"""
@inline prognostic(name::Symbol, loc::Union{VarDims, VarLocation}; units = NoUnits, closure = nothing, bounds = Unbounded, desc = "") = prognostic(var(name, loc, units); closure, bounds, desc)
@inline prognostic(var::Variable; closure = nothing, bounds = Unbounded, desc = "") = PrognosticVariable(var, closure, tendency(var), bounds, desc)

"""
    $SIGNATURES

Convenience constructor method for `AuxiliaryVariable`.
"""
@inline auxiliary(name::Symbol, loc::Union{VarDims, VarLocation}, ctor = nothing, params = nothing; units = NoUnits, bounds = Unbounded, desc = "") = auxiliary(var(name, loc, units), ctor, params; bounds, desc)
@inline auxiliary(var::Variable, ::Nothing, ::Nothing; bounds = Unbounded, desc = "") = AuxiliaryVariable(var, nothing, bounds, desc)
@inline auxiliary(var::Variable, ctor::Function, params; bounds = Unbounded, desc = "") = AuxiliaryVariable(var, (_, grid, clock, fields) -> ctor(grid, clock, fields, params), bounds, desc)
# `KernelFunction` constructors (from `kernel`) are callable structs, not `Function`s; they define
# their own `(var, grid, clock, fields)` call convention, so store them directly as the ctor.
@inline auxiliary(var::Variable, ctor::KernelFunction, ::Nothing; bounds = Unbounded, desc = "") = AuxiliaryVariable(var, ctor, bounds, desc)

"""
    $SIGNATURES

Convenience constructor method for `InputVariable`.
"""
@inline input(name::Symbol, loc::Union{VarDims, VarLocation}; default = nothing, units = NoUnits, bounds = Unbounded, desc = "") = input(var(name, loc, units); default, bounds, desc)
@inline input(var::Variable; default = nothing, bounds = Unbounded, desc = "") = InputVariable(var, default, bounds, desc)

"""
    $SIGNATURES

Creates an `AuxiliaryVariable` for the tendency of a prognostic variable with the given name, dimensions, and physical units.
This constructor is primarily used internally by other constructors and does not usually need to be called by implementations of `variables`.
"""
@inline tendency(var::Variable) = auxiliary(varname(var), varloc(var), units = upreferred(varunits(var)) / u"s")

"""
    $SIGNATURES

Convenience constructor method for variable `Namespace`s.
"""
@inline namespace(name::Symbol, vars::Union{Tuple, Variables}) = Namespace(name, vars)

"""
    $SIGNATURES

Convert the given `NamedTuple` of variables into a tuple of `Namespace`s.
"""
@inline namespaces(nt::NamedTuple{names}) where {names} = map((nm, vars) -> namespace(nm, vars), names, values(nt))

"""
Alias for `Variables(vars...)`
"""
@inline variables(vars::Union{AbstractVariable, Namespace}...) = Variables(vars)

"""
Helper method that selects only prognostic variables declared on `obj`.
"""
@inline prognostic_variables(obj) = prognostic_variables(variables(obj))
@inline prognostic_variables(vars::Variables) = vars.prognostic
@inline prognostic_variables(vars::Tuple) = deduplicate_vars(Tuple(filter(var -> isa(var, PrognosticVariable), vars)))

"""
Helper method that selects only auxiliary variables declared on `obj`.
"""
@inline auxiliary_variables(obj) = auxiliary_variables(variables(obj))
@inline auxiliary_variables(vars::Variables) = vars.auxiliary
@inline auxiliary_variables(vars::Tuple) = deduplicate_vars(Tuple(filter(var -> isa(var, AuxiliaryVariable), vars)))

"""
Helper method that selects only input variables declared on `obj`.
"""
@inline input_variables(obj) = input_variables(variables(obj))
@inline input_variables(vars::Variables) = vars.inputs
@inline input_variables(vars::Tuple) = deduplicate_vars(Tuple(filter(var -> isa(var, InputVariable), vars)))

"""
Helper method that selects only closure (auxiliary) variables declared on `obj`.
"""
@inline closure_variables(obj) = closure_variables(variables(obj))
@inline function closure_variables(vars::Tuple)
    progvars = prognostic_variables(vars)
    all_closure_vars = fastmap(var -> variables(var.closure), progvars)
    return deduplicate_vars(tuplejoin(all_closure_vars...))
end

function Base.NamedTuple(vars::Tuple{Vararg{Union{AbstractVariable, Namespace}}})
    keys = map(varname, vars)
    return NamedTuple{keys}(vars)
end
