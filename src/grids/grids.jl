abstract type AbstractLandGrid{NF, TX, TY, TZ, Arch} <: Oceananigans.Grids.AbstractGrid{NF, TX, TY, TZ, Arch, Nothing} end

Base.summary(grid::AbstractLandGrid) = "$(nameof(typeof(grid))) with dimensions $(size(grid))"
Base.size(grid::AbstractLandGrid) = size(get_field_grid(grid))

"""
Return the number of vertical layers defined by the given `grid`.
"""
num_layers(grid::AbstractLandGrid) = size(get_field_grid(grid), 3)

"""
    get_field_grid(grid::AbstractLandGrid)::Oceananigans.AbstractGrid

Returns the underlying `Oceananigans` grid type for `Field`s defined on the given land `grid`.
"""
function get_field_grid end

@inline get_field_grid(grid::Oceananigans.AbstractGrid) = grid

Architectures.architecture(grid::AbstractLandGrid) = architecture(get_field_grid(grid))
Architectures.on_architecture(arch, grid::AbstractLandGrid) = adapt(array_type(arch), grid)

Oceananigans.Grids.halo_size(grid::AbstractLandGrid) = Oceananigans.Grids.halo_size(get_field_grid(grid))
Oceananigans.Grids.isrectilinear(grid::AbstractLandGrid) = Oceananigans.Grids.isrectilinear(get_field_grid(grid))

@inline Oceananigans.Grids.nodes(grid::AbstractLandGrid, args...; kwargs...) = nodes(get_field_grid(grid), args...; kwargs...)
@inline Oceananigans.Grids.xnodes(grid::AbstractLandGrid, args...; kwargs...) = xnodes(get_field_grid(grid), args...; kwargs...)
@inline Oceananigans.Grids.ynodes(grid::AbstractLandGrid, args...; kwargs...) = ynodes(get_field_grid(grid), args...; kwargs...)
@inline Oceananigans.Grids.znodes(grid::AbstractLandGrid, args...; kwargs...) = znodes(get_field_grid(grid), args...; kwargs...)

# Oceananigans (and downstream packages such as NumericalEarth) access grid dimensions and
# discretization data as struct fields (`grid.Nx`, `grid.Ny`, `grid.Nz`, `grid.Hx`, coordinate
# arrays, etc.) rather than through accessor methods. Land grids do not store these directly, so
# forward any property that is not one of the wrapper's own fields to the underlying field grid.
@inline Base.getproperty(grid::AbstractLandGrid, name::Symbol) = hasfield(typeof(grid), name) ? getfield(grid, name) : getproperty(get_field_grid(grid), name)

Base.propertynames(grid::AbstractLandGrid) = (fieldnames(typeof(grid))..., propertynames(get_field_grid(grid))...)

include("grid_utils.jl")

include("column_grid.jl")

include("column_ring_grid.jl")
