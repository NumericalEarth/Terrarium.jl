"""
Alias for an Oceananigans `RectilinearGrid` representing a set of laterally independent vertical
columns with dimensions (x, y, z) where `x` is the column dimension, `y = 1` is constant (`Flat`),
and `z` is the vertical axis.

Note that this is a plain spatial discretization, not a land grid; see [`LandGrid`](@ref) for the
land model counterpart which collects the discretizations of all vertical domains.
"""
const ColumnGrid{NF, Arch} = RectilinearGrid{NF, Periodic, Flat, Bounded, CZ, FX, FY, VX, VY, Arch, SZ} where {CZ, FX, FY, VX, VY, SZ}

"""
    $SIGNATURES

Creates a `ColumnGrid` from the given `architecture`, numeric type `NF`, and
`vertical_coordinate`, which may be any [`VerticalCoordinate`](@ref): a vector of cell interfaces,
such as the one returned by [`UniformSpacing`](@ref), or a discretization such as the one returned
by [`ExponentialSpacing`](@ref). The `num_columns` determines the number of laterally independent
columns initialized on the grid.
"""
function ColumnGrid(
        arch::AbstractArchitecture,
        ::Type{NF},
        vertical_coordinate::VerticalCoordinate,
        num_columns::Int = 1,
    ) where {NF <: AbstractFloat}
    # TODO: Need to eventually consider ordering of array dimensions;
    # using the z-axis here probably results in inefficient memory access patterns
    # since most or all land computations will be along this axis
    return RectilinearGrid(arch, NF, size = (num_columns, num_layers(vertical_coordinate)), x = (0, 1), z = vertical_coordinate, topology = (Periodic, Flat, Bounded))
end

# Default constructors; the number format is inferred from the vertical coordinate where possible.
ColumnGrid(vertical_coordinate::VerticalCoordinate, num_columns::Int = 1) = ColumnGrid(CPU(), coordinate_eltype(vertical_coordinate), vertical_coordinate, num_columns)
ColumnGrid(arch::AbstractArchitecture, vertical_coordinate::VerticalCoordinate, num_columns::Int = 1) = ColumnGrid(arch, coordinate_eltype(vertical_coordinate), vertical_coordinate, num_columns)

"""
    $SIGNATURES

Return the number format to use for a grid with vertical coordinate `z` when none is given
explicitly. Oceananigans discretizations compute their cell interfaces in `Float64`, so only a
coordinate given directly as an array of interfaces carries a number format of its own.
"""
coordinate_eltype(z::AbstractVector{NF}) where {NF <: AbstractFloat} = NF
# Oceananigans computes the interfaces of its discretizations in double precision, so that is the
# format a grid built from one defaults to; pass `NF` explicitly for anything else.
coordinate_eltype(z) = Float64
