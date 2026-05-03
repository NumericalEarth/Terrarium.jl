# Convenience dispatches for Oceananigans.launch!
function Oceananigans.launch!(grid::AbstractLandGrid, workspec, kernel!::Function, first_arg, other_args...; kwargs...)
    fgrid = get_field_grid(grid)
    launch!(fgrid.architecture, fgrid, get_workspec(workspec), kernel!, first_arg, grid, other_args...; kwargs...)
    debugsite!(kernel!, first_arg, grid, other_args...)
    return nothing
end

"""
Returns the appropriate workspec for the given `AbstractField` or based on the given
field locations.
"""
@inline get_workspec(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = get_workspec(LX(), LY(), LZ())
@inline get_workspec(dims::VarDims) = get_workspec(typeof(dims))
@inline get_workspec(::Type{<:XY}) = Val{:xy}()
@inline get_workspec(::Type{<:XYZ}) = Val{:xyz}()
@inline get_workspec(::Any, ::Any, ::Nothing) = Val{:xy}()
@inline get_workspec(::Any, ::Any, ::Union{Center, Face}) = Val{:xyz}()
@inline get_workspec(::ValType{workspec}) where {workspec} = Val{workspec}()
@inline get_workspec(workspec) = workspec # fallback; pass directly to launch!

# Helper functions for checking if a `RingGrids` or `Oceananigans` `Field` matches the given grid
field_matches_grid(field, grid) = field.grid == grid

function assert_field_matches_grid(field::Union{RingGrids.AbstractField, AbstractField}, grid)
    return @assert field_matches_grid(field, grid) "Field grid $(typeof(field.grid)) does not match $(typeof(grid))"
end

const RingGridOrField = Union{RingGrids.AbstractGrid, RingGrids.AbstractField}

Architectures.on_architecture(::GPU, obj::RingGridOrField) = RingGrids.Architectures.on_architecture(RingGrids.Architectures.GPU(), obj)
Architectures.on_architecture(::CPU, obj::RingGridOrField) = RingGrids.Architectures.on_architecture(RingGrids.Architectures.CPU(), obj)

# Field construction

"""
    Field(
        grid::AbstractLandGrid,
        dims::VarDims,
        boundary_conditions = nothing,
        args...;
        kwargs...
    )

Auxiliary constructor for an Oceananigans `Field` on `grid` with the given Terrarium variable `dims` and boundary conditions.
Additional arguments are passed direclty to the `Field` constructor. The location of the `Field`
is determined by `VarDims` defined on `var`.
"""
function Oceananigans.Field(
        grid::AbstractLandGrid,
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
        FT(get_field_grid(grid), args...; boundary_conditions, kwargs...)
    elseif isa(boundary_conditions, NamedTuple)
        # assume that named tuple corresponds to FieldBoundaryConditions positions
        field_bcs = FieldBoundaryConditions(get_field_grid(grid), (Center(), Center(), nothing); boundary_conditions...)
        FT(get_field_grid(grid), args...; boundary_conditions = field_bcs, kwargs...)
    else
        FT(get_field_grid(grid), args...; kwargs...)
    end
    return field
end

"""
    FieldTimeSeries(
        grid::AbstractLandGrid,
        dims::VarDims,
        times=eltype(grid)[]
    )

Construct a `FieldTimeSeries` on the given land `grid` with the given `dims` and `times`.
"""
function Oceananigans.FieldTimeSeries(
        grid::AbstractLandGrid,
        dims::VarDims,
        times = eltype(grid)[]
    )
    loc = location(dims)
    return FieldTimeSeries(loc, get_field_grid(grid), times)
end
