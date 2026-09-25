"""
    $SIGNATURES

Launch `kernel!` over `grid` with the work layout given by `workspec`, passing `grid` itself as the
kernel's second argument so that kernels can be written in terms of the model's grid. The work specification
muist be given as a Terrarium [`VarDims`](@ref), i.e. `XY` or `XYZ`.
"""
@inline Oceananigans.launch!(grid::AbstractGrid, ::Union{XY, Type{<:XY}}, kernel!::Function, first_arg, other_args...; kwargs...) =
    launch_kernel!(grid, Val{:xy}(), kernel!, first_arg, other_args...; kwargs...)

@inline Oceananigans.launch!(grid::AbstractGrid, ::Union{XYZ, Type{<:XYZ}}, kernel!::Function, first_arg, other_args...; kwargs...) =
    launch_kernel!(grid, Val{:xyz}(), kernel!, first_arg, other_args...; kwargs...)

"""
    $SIGNATURES

Launch `kernel!` over `grid` with the Oceananigans `workspec`. `grid` is passed to the kernel as
its second argument, after `first_arg`.
"""
@inline function launch_kernel!(grid::AbstractGrid, workspec::Val, kernel!::Function, first_arg, other_args...; kwargs...)
    launch!(architecture(grid), grid, workspec, kernel!, first_arg, grid, other_args...; kwargs...)
    debugsite!(kernel!, first_arg, grid, other_args...)
    return nothing
end

# Helper functions for checking if a `RingGrids` or `Oceananigans` `Field` matches the given grid
field_matches_grid(field, grid) = field.grid == grid

function assert_field_matches_grid(field::Union{RingGrids.AbstractField, AbstractField}, grid)
    return @assert field_matches_grid(field, grid) "Field grid $(typeof(field.grid)) does not match $(typeof(grid))"
end

const RingGridOrField = Union{RingGrids.AbstractGrid, RingGrids.AbstractField}

Architectures.on_architecture(::GPU, obj::RingGridOrField) = RingGrids.Architectures.on_architecture(RingGrids.Architectures.GPU(), obj)
Architectures.on_architecture(::CPU, obj::RingGridOrField) = RingGrids.Architectures.on_architecture(RingGrids.Architectures.CPU(), obj)

RingGrids.Architectures.architecture(::GPU) = RingGrids.Architectures.GPU()
RingGrids.Architectures.architecture(::CPU) = RingGrids.Architectures.CPU()

# Field construction

"""
    Field(
        grid::AbstractGrid,
        loc::VarLocation,
        boundary_conditions = nothing,
        args...;
        kwargs...
    )

Auxiliary constructor for an Oceananigans `Field` on `grid` at the given Terrarium variable location
and boundary conditions. Additional arguments are passed directly to the `Field` constructor.

[`VarLocation`](@ref) determines both *where* the field is allocated and *what shape* it has: its
[`VarDomain`](@ref) selects the domain discretization via [`variable_grid`](@ref), and its
[`VarDims`](@ref) give the Oceananigans location and, for variables declared at a single point such
as [`Top`](@ref) or [`Bottom`](@ref), the `indices` which restrict the field to that point.

Note that the `Field` is allocated on the domain discretization rather than on the land grid itself, so
that `field.grid` names the domain the field lives on and Oceananigans' own grid methods, many of
which dispatch on concrete grid types, apply to it unchanged.
"""
function Oceananigans.Field(
        grid::AbstractLandGrid,
        loc::VarLocation,
        boundary_conditions = nothing,
        args...;
        kwargs...
    )
    return create_field(variable_grid(grid, loc), loc, boundary_conditions, args...; kwargs...)
end

# A plain grid has a single discretization, so there is no domain to resolve and `loc`'s domain (if
# any) is ignored; only its dimensions matter.
function Oceananigans.Field(
        grid::AbstractGrid,
        loc::VarLocation,
        boundary_conditions = nothing,
        args...;
        kwargs...
    )
    return create_field(grid, loc, boundary_conditions, args...; kwargs...)
end

# Bare dimensions declare a domainless variable; see `var`.
Oceananigans.Field(grid::AbstractGrid, dims::VarDims, args...; kwargs...) =
    Field(grid, VarLocation(dims), args...; kwargs...)

# Allocate the field on an already-resolved `domain` discretization.
function create_field(
        domain::AbstractGrid,
        loc::VarLocation,
        boundary_conditions = nothing,
        args...;
        kwargs...
    )
    dims = vardims(loc)
    # infer the location and index restriction of the Field on the Oceananigans grid from `dims`
    field_loc = location(dims)
    FT = Field{map(typeof, field_loc)...}
    field_indices = indices(domain, dims)
    # Specify BCs if defined
    field = if isa(boundary_conditions, FieldBoundaryConditions)
        FT(domain, args...; indices = field_indices, boundary_conditions, kwargs...)
    elseif isa(boundary_conditions, NamedTuple)
        # assume that named tuple corresponds to FieldBoundaryConditions positions
        field_bcs = FieldBoundaryConditions(domain, (Center(), Center(), nothing); boundary_conditions...)
        FT(domain, args...; indices = field_indices, boundary_conditions = field_bcs, kwargs...)
    else
        FT(domain, args...; indices = field_indices, kwargs...)
    end
    return field
end

"""
    FieldTimeSeries(
        grid::AbstractGrid,
        loc::VarLocation,
        times=eltype(grid)[]
    )

Construct a `FieldTimeSeries` on the domain discretization of `grid` selected by `loc`, with the
given `times`.
"""
function Oceananigans.FieldTimeSeries(
        grid::AbstractLandGrid,
        loc::VarLocation,
        times = eltype(grid)[]
    )
    return create_field_time_series(variable_grid(grid, loc), loc, architecture(grid), times)
end

# As for `Field`: a plain grid resolves no domain.
function Oceananigans.FieldTimeSeries(
        grid::AbstractGrid,
        loc::VarLocation,
        times = eltype(grid)[]
    )
    return create_field_time_series(grid, loc, architecture(grid), times)
end

Oceananigans.FieldTimeSeries(grid::AbstractGrid, dims::VarDims, times = eltype(grid)[]) =
    FieldTimeSeries(grid, VarLocation(dims), times)

function create_field_time_series(domain::AbstractGrid, loc::VarLocation, arch, times)
    dims = vardims(loc)
    return FieldTimeSeries(location(dims), domain, on_architecture(arch, times); indices = indices(domain, dims))
end
