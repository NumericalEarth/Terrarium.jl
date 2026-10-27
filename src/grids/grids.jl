"""
Base type for *land grids*, i.e. grids which collect the spatial discretizations of all vertical
domains (ground, snow, canopy) of a land model. `AbstractLandGrid`s are the grid types stored by
`AbstractModel`s; they are constructed from an ordinary Oceananigans `AbstractGrid` via
[`create_land_grid`](@ref). [`LandGrid`](@ref) is the only implementation.

Land grids wrap the per-domain spatial discretizations and forward the `AbstractGrid` interface to
the ground domain; implementations need only define [`ground_domain`](@ref).
"""
abstract type AbstractLandGrid{NF, TX, TY, TZ, Arch} <: AbstractGrid{NF, TX, TY, TZ, Arch, Nothing} end

"""
Return the number of vertical layers defined by the given `grid`.
"""
num_layers(grid::AbstractGrid) = size(grid, 3)

# Forwarding of the Oceananigans `AbstractGrid` interface to the ground domain grid.

Base.summary(grid::AbstractLandGrid) = "$(nameof(typeof(grid))) with dimensions $(size(grid))"
Base.size(grid::AbstractLandGrid) = size(ground_domain(grid))

Architectures.architecture(grid::AbstractLandGrid) = architecture(ground_domain(grid))
Architectures.on_architecture(arch, grid::AbstractLandGrid) = adapt(array_type(arch), grid)

Oceananigans.Grids.halo_size(grid::AbstractLandGrid) = halo_size(ground_domain(grid))
Oceananigans.Grids.isrectilinear(grid::AbstractLandGrid) = isrectilinear(ground_domain(grid))

@inline Oceananigans.Grids.nodes(grid::AbstractLandGrid, args...; kwargs...) = nodes(ground_domain(grid), args...; kwargs...)
@inline Oceananigans.Grids.xnodes(grid::AbstractLandGrid, args...; kwargs...) = xnodes(ground_domain(grid), args...; kwargs...)
@inline Oceananigans.Grids.ynodes(grid::AbstractLandGrid, args...; kwargs...) = ynodes(ground_domain(grid), args...; kwargs...)
@inline Oceananigans.Grids.znodes(grid::AbstractLandGrid, args...; kwargs...) = znodes(ground_domain(grid), args...; kwargs...)

# Oceananigans (and downstream packages such as NumericalEarth) access grid dimensions and
# discretization data as struct fields (`grid.Nx`, `grid.Ny`, `grid.Nz`, `grid.Hx`, coordinate
# arrays, etc.) rather than through accessor methods. Land grids do not store these directly, so
# forward any property that is not one of the wrapper's own fields to the ground domain grid.
@inline Base.getproperty(grid::AbstractLandGrid, name::Symbol) = hasfield(typeof(grid), name) ? getfield(grid, name) : getproperty(ground_domain(grid), name)

Base.propertynames(grid::AbstractLandGrid) = (fieldnames(typeof(grid))..., propertynames(ground_domain(grid))...)

include("grid_utils.jl")

include("land_grid.jl")

include("column_grid.jl")

include("column_ring_grid.jl")
