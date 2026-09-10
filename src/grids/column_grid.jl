abstract type AbstractColumnGrid{NF, Arch} <: AbstractLandGrid{NF, Periodic, Flat, Bounded, Arch} end

"""
    $TYPEDEF

Represents a set of laterally independent vertical columns with dimensions (x, y, z)
where `x` is the column dimension, `y=1` is constant, and `z` is the vertical axis.
"""
struct ColumnGrid{NF, Arch, RectGrid <: Oceananigans.Grids.RectilinearGrid} <: AbstractColumnGrid{NF, Arch}
    "Underlying Oceananigans rectilinear grid on which `Field`s are defined."
    grid::RectGrid

    """
        $SIGNATURES

    Creates a `ColumnGrid` from the given `architecture`, numeric type `NF`, and vertical
    discretization `vert`. The `num_columns` determines the number of laterally independent
    columns initialized on the grid.
    """
    function ColumnGrid(
            arch::AbstractArchitecture,
            ::Type{NF},
            vert::AbstractVerticalSpacing,
            num_columns::Int = 1,
        ) where {NF <: AbstractFloat}
        Nz = num_layers(vert)
        # TODO: Need to eventually consider ordering of array dimensions;
        # using the z-axis here probably results in inefficient memory access patterns
        # since most or all land computations will be along this axis
        z_thick = get_spacing(vert)
        z_coords = convert.(NF, vcat(-reverse(cumsum(z_thick)), zero(eltype(z_thick))))
        grid = Oceananigans.Grids.RectilinearGrid(arch, NF, size = (num_columns, Nz), x = (0, 1), z = z_coords, topology = (Periodic, Flat, Bounded))
        return new{NF, typeof(arch), typeof(grid)}(grid)
    end
    # Default constructors
    ColumnGrid(vert::AbstractVerticalSpacing{NF}, num_columns::Int = 1) where {NF} = ColumnGrid(CPU(), NF, vert, num_columns)
    ColumnGrid(arch::AbstractArchitecture, vert::AbstractVerticalSpacing{NF}, num_columns::Int = 1) where {NF} = ColumnGrid(arch, NF, vert, num_columns)
    ColumnGrid(grid::Oceananigans.Grids.RectilinearGrid{NF}) where {NF} = new{NF, typeof(architecture(grid)), typeof(grid)}(grid)
end

@adapt_structure ColumnGrid

"""
    get_field_grid(grid)

Return the underlying Oceananigans `grid` stored in `ColumnGrid`.
"""
@inline get_field_grid(grid::ColumnGrid) = getfield(grid, :grid)

Architectures.architecture(grid::ColumnGrid) = architecture(grid.grid)

function Base.show(io::IO, mime::MIME"text/plain", grid::ColumnGrid{NF}) where {NF}
    println(io, "ColumnGrid{$NF} on $(architecture(grid)) with")
    return show(io, mime, grid.grid)
end
