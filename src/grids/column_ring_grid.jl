"""
    $TYPEDEF

Represents a global (spherical) grid of independent, vertical columns where the
spatial discretization in the horizontal direction is defined by a `RingGrids.AbstractGrid`.
"""
struct ColumnRingGrid{
        NF,
        Arch,
        RingGrid <: RingGrids.AbstractGrid,
        RectGrid <: Oceananigans.Grids.RectilinearGrid,
        Mask <: Union{AbstractArray, RingGrids.AbstractField},
    } <: AbstractColumnGrid{NF, Arch}
    "RingGrid specfying the lateral spatial discretization of the globe"
    rings::RingGrid

    "`RingGrids.Field` (or GPU-adapted array) representing a boolean-valued mask over `rings`"
    mask::Mask

    "Underlying `Oceananigans` `RectilinearGrid` type on which `Field`s are defined"
    grid::RectGrid

    function ColumnRingGrid(
            rings::RingGrids.AbstractGrid,
            mask::AbstractArray,
            grid::Oceananigans.Grids.RectilinearGrid{NF}
        ) where {NF}
        arch = architecture(grid)
        return new{NF, typeof(arch), typeof(rings), typeof(grid), typeof(mask)}(rings, mask, grid)
    end

    """
        $SIGNATURES

    Constructs a `ColumnRingGrid` over the given `rings`.
    """
    function ColumnRingGrid(
            arch::AbstractArchitecture,
            ::Type{NF},
            vert::AbstractVerticalSpacing,
            rings::RingGrids.AbstractGrid,
            mask::RingGrids.AbstractField{Bool} = convert.(Bool, ones(rings))
        ) where {NF <: AbstractFloat}
        assert_field_matches_grid(mask, rings)
        # get number of horizontal grid points by summing over mask
        Nh = sum(mask)
        # get number of vertical grid points
        Nz = num_layers(vert)
        # TODO: Need to consider ordering of array dimensions;
        # using the z-axis here probably results in inefficient memory access patterns
        # since most or all land computations will be along this axis
        z_thick = get_spacing(vert)
        z_coords = convert.(NF, vcat(-reverse(cumsum(z_thick)), zero(eltype(z_thick))))
        grid = Oceananigans.Grids.RectilinearGrid(arch, NF, size = (Nh, Nz), x = (1, Nh), z = z_coords, topology = (Periodic, Flat, Bounded))
        # adapt ring grid and mask
        rings = on_architecture(arch, rings)
        mask = on_architecture(arch, mask)
        return new{NF, typeof(arch), typeof(rings), typeof(grid), typeof(mask)}(rings, mask, grid)
    end

    ColumnRingGrid(
        arch::AbstractArchitecture,
        vert::AbstractVerticalSpacing,
        rings::RingGrids.AbstractGrid,
        mask::RingGrids.AbstractField{Bool} = convert.(Bool, ones(rings))
    ) = ColumnRingGrid(arch, Float32, vert, rings, mask)

    ColumnRingGrid(
        arch::AbstractArchitecture,
        ::Type{NF},
        vert::AbstractVerticalSpacing,
        mask::RingGrids.AbstractField{Bool},
    ) where {NF} = ColumnRingGrid(arch, NF, vert, mask.grid, mask)

    ColumnRingGrid(
        arch::AbstractArchitecture,
        vert::AbstractVerticalSpacing,
        mask::RingGrids.AbstractField{Bool}
    ) = ColumnRingGrid(arch, Float32, vert, mask.grid, mask)

    ColumnRingGrid(
        vert::AbstractVerticalSpacing,
        rings::RingGrids.AbstractGrid,
        mask::RingGrids.AbstractField{Bool} = convert.(Bool, ones(rings))
    ) = ColumnRingGrid(CPU(), Float32, vert, rings, mask)

    ColumnRingGrid(
        vert::AbstractVerticalSpacing,
        mask::RingGrids.AbstractField{Bool}
    ) = ColumnRingGrid(CPU(), Float32, vert, mask.grid, mask)
end

@adapt_structure ColumnRingGrid

get_field_grid(grid::ColumnRingGrid) = getfield(grid, :grid)

"""
    $SIGNATURES

Converts the given Oceananigans `Field` to a `RingGrids.Field` with a ring grid matching that of the given `ColumnRingGrid`.
"""
RingGrids.Field(field::Field{LX, LY, LZ}, grid::ColumnRingGrid; fill_value = NaN) where {LX, LY, LZ} = RingGrids.Field(architecture(field), dropdims(interior(field), dims = 2), grid; fill_value)
RingGrids.Field(arch::AbstractArchitecture, field::Field{LX, LY, LZ}, grid::ColumnRingGrid; fill_value = NaN) where {LX, LY, LZ} = RingGrids.Field(arch, dropdims(interior(field), dims = 2), grid; fill_value)
RingGrids.Field(field::AbstractArray, grid::ColumnRingGrid; fill_value = NaN) = RingGrids.Field(architecture(grid), field, grid; fill_value)
function RingGrids.Field(arch::AbstractArchitecture, field::AbstractArray, grid::ColumnRingGrid; fill_value = NaN)
    # ensure that grid and field are both on the device specified by `arch`
    grid = on_architecture(arch, grid)
    field = on_architecture(arch, field)
    # create new RingGrids field initialized with fill_value
    ring_field = RingGrids.Field(grid.rings, size(field)[2:end]...)
    fill!(ring_field, fill_value)
    # need to access underlying data arrays directly to avoid scalar indexing
    colons = (Colon() for _ in size(field)[2:end])
    ring_field.data[grid.mask.data, colons...] .= field
    return ring_field
end

"""
    $SIGNATURES

Converts a `RingGrids.Field` to an Oceananigans `Field`
using the given `ColumnRingGrid`. Only masked grid points are copied to the Oceananigans field.
For 2D RingGrids fields, returns a 2D Oceananigans field. For 3D fields, returns a 3D field.
"""
function Oceananigans.Field(ring_field::RingGrids.AbstractField, grid::ColumnRingGrid; default_value = zero(eltype(ring_field)))
    if ndims(ring_field) == 1
        # 2D field (horizontal only): treat the data as a single-column matrix so one masked gather
        # (`data[mask, :]`) serves both the 1D and 2D cases. There's a related Reactant bug that makes this necessary: https://github.com/EnzymeAD/Reactant.jl/issues/3087
        dims = XY()
        data = reshape(on_architecture(architecture(grid), ring_field.data), :, 1)
    elseif ndims(ring_field) == 2
        # 3D field (horizontal + vertical or other dimensions)
        @assert size(grid.grid, 3) == size(ring_field, 2) "Vertical dimension mismatch: grid has $(size(grid.grid, 3)) layers, but field has $(size(ring_field, 2)) layers"
        dims = XYZ()
        data = on_architecture(architecture(grid), ring_field.data)
    else
        error("Unsupported number of dimensions for RingGrids.Field: $(ndims(ring_field))")
    end

    mask = grid.mask.data   # host boolean mask (see note above)
    gathered = data[mask, :]
    values = reshape(gathered, size(gathered, 1), 1, size(gathered, 2))
    oceananigans_field = Field(grid, dims)
    set!(oceananigans_field, values)
    return oceananigans_field
end

function Oceananigans.FieldTimeSeries(ring_field::RingGrids.AbstractField, grid::ColumnRingGrid, times::AbstractVector; default_value = zero(eltype(ring_field)))
    @assert last(size(ring_field)) == length(times) "Last dimension of RingGrids Field must match the length of `times`"
    arch = architecture(grid)

    if ndims(ring_field) == 2
        # 2D field (horizontal only): treat the data as a single-column matrix so one masked gather
        # (`data[mask, :]`) serves both the 1D and 2D cases. There's a related Reactant bug that makes this necessary: https://github.com/EnzymeAD/Reactant.jl/issues/3087
        dims = XY()
        data = reshape(on_architecture(arch, ring_field.data), :, 1)
    elseif ndims(ring_field) == 3
        # 3D field (horizontal + vertical or other dimensions)
        @assert size(grid.grid, 3) == size(ring_field, 2) "Vertical dimension mismatch: grid has $(size(grid.grid, 3)) layers, but field has $(size(ring_field, 2)) layers"
        dims = XYZ()
        data = on_architecture(arch, ring_field.data)
    else
        error("Unsupported number of dimensions for RingGrids.Field: $(ndims(ring_field))")
    end

    mask = grid.mask.data   # host boolean mask (see note above)
    gathered = data[mask, :, :]
    values = reshape(gathered, size(gathered, 1), 1, size(gathered)[2:end]...)
    oceananigans_fts = FieldTimeSeries(grid, dims, times)
    copyto!(interior(oceananigans_fts), values)
    return oceananigans_fts
end

function Architectures.on_architecture(arch::AbstractArchitecture, grid::ColumnRingGrid)
    return ColumnRingGrid(
        on_architecture(arch, grid.rings),
        on_architecture(arch, grid.mask),
        on_architecture(arch, grid.grid)
    )
end

function Base.show(io::IO, mime::MIME"text/plain", grid::ColumnRingGrid{NF}) where {NF}
    println(io, "ColumnRingGrid{$NF} on $(architecture(grid)) with")
    show(io, mime, grid.rings)
    println(io)
    return show(io, mime, grid.grid)
end
