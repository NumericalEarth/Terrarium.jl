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
        dims::VarDims,
        boundary_conditions = nothing,
        args...;
        kwargs...
    )

Auxiliary constructor for an Oceananigans `Field` on `grid` with the given Terrarium variable `dims` and boundary conditions.
Additional arguments are passed direclty to the `Field` constructor. The location of the `Field`
is determined by `VarDims` defined on `var`.

Note that the `Field` is allocated on the discretization of the ground domain rather than on a land
grid itself, so that `field.grid` names the domain the field lives on and Oceananigans' own grid
methods — many of which dispatch on concrete grid types — apply to it unchanged.
"""
function Oceananigans.Field(
        grid::AbstractGrid,
        dims::VarDims,
        boundary_conditions = nothing,
        args...;
        kwargs...
    )
    # infer the location of the Field on the Oceananigans grid from `dims`
    loc = location(dims)
    FT = Field{map(typeof, loc)...}
    # Specify BCs if defined
    field = if isa(boundary_conditions, FieldBoundaryConditions)
        FT(ground_domain(grid), args...; boundary_conditions, kwargs...)
    elseif isa(boundary_conditions, NamedTuple)
        # assume that named tuple corresponds to FieldBoundaryConditions positions
        field_bcs = FieldBoundaryConditions(ground_domain(grid), (Center(), Center(), nothing); boundary_conditions...)
        FT(ground_domain(grid), args...; boundary_conditions = field_bcs, kwargs...)
    else
        FT(ground_domain(grid), args...; kwargs...)
    end
    return field
end

"""
    FieldTimeSeries(
        grid::AbstractGrid,
        dims::VarDims,
        times=eltype(grid)[]
    )

Construct a `FieldTimeSeries` on the given `grid` with the given `dims` and `times`.
"""
function Oceananigans.FieldTimeSeries(
        grid::AbstractGrid,
        dims::VarDims,
        times = eltype(grid)[]
    )
    loc = location(dims)
    arch = architecture(grid)
    return FieldTimeSeries(loc, ground_domain(grid), on_architecture(arch, times))
end
