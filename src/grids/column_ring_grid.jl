"""
    $TYPEDEF

Represents a global (spherical) grid of independent, vertical columns where the
spatial discretization in the horizontal direction is defined by a `RingGrids.AbstractGrid`.
"""
struct ColumnRingGrid{
        NF,
        Arch,
        RingGrid <: RingGrids.AbstractGrid,
        RectGrid <: RectilinearGrid,
        Mask <: Union{AbstractArray, RingGrids.AbstractField},
    } <: AbstractGrid{NF, Periodic, Flat, Bounded, Arch, Nothing}
    "RingGrid specfying the lateral spatial discretization of the globe"
    rings::RingGrid

    "`RingGrids.Field` (or GPU-adapted array) representing a boolean-valued mask over `rings`"
    mask::Mask

    "Underlying `Oceananigans` `RectilinearGrid` type on which `Field`s are defined"
    grid::RectGrid

    function ColumnRingGrid(
            rings::RingGrids.AbstractGrid,
            mask::AbstractArray,
            grid::RectilinearGrid{NF}
        ) where {NF}
        arch = architecture(grid)
        return new{NF, typeof(arch), typeof(rings), typeof(grid), typeof(mask)}(rings, mask, grid)
    end

    """
        $SIGNATURES

    Constructs a `ColumnRingGrid` over the given `rings` with the given `vertical_coordinate`,
    which may be any [`VerticalCoordinate`](@ref).
    """
    function ColumnRingGrid(
            arch::AbstractArchitecture,
            ::Type{NF},
            vertical_coordinate::VerticalCoordinate,
            rings::RingGrids.AbstractGrid,
            mask::RingGrids.AbstractField{Bool} = convert.(Bool, ones(rings))
        ) where {NF <: AbstractFloat}
        assert_field_matches_grid(mask, rings)
        # get number of horizontal grid points by summing over mask
        Nh = sum(mask)
        # TODO: Need to consider ordering of array dimensions;
        # using the z-axis here probably results in inefficient memory access patterns
        # since most or all land computations will be along this axis
        grid = RectilinearGrid(arch, NF, size = (Nh, num_layers(vertical_coordinate)), x = (1, Nh), z = vertical_coordinate, topology = (Periodic, Flat, Bounded))
        # adapt ring grid and mask
        rings = on_architecture(arch, rings)
        mask = on_architecture(arch, mask)
        return new{NF, typeof(arch), typeof(rings), typeof(grid), typeof(mask)}(rings, mask, grid)
    end

    ColumnRingGrid(
        arch::AbstractArchitecture,
        vertical_coordinate::VerticalCoordinate,
        rings::RingGrids.AbstractGrid,
        mask::RingGrids.AbstractField{Bool} = convert.(Bool, ones(rings))
    ) = ColumnRingGrid(arch, Float32, vertical_coordinate, rings, mask)

    ColumnRingGrid(
        arch::AbstractArchitecture,
        ::Type{NF},
        vertical_coordinate::VerticalCoordinate,
        mask::RingGrids.AbstractField{Bool},
    ) where {NF} = ColumnRingGrid(arch, NF, vertical_coordinate, mask.grid, mask)

    ColumnRingGrid(
        arch::AbstractArchitecture,
        vertical_coordinate::VerticalCoordinate,
        mask::RingGrids.AbstractField{Bool}
    ) = ColumnRingGrid(arch, Float32, vertical_coordinate, mask.grid, mask)

    ColumnRingGrid(
        vertical_coordinate::VerticalCoordinate,
        rings::RingGrids.AbstractGrid,
        mask::RingGrids.AbstractField{Bool} = convert.(Bool, ones(rings))
    ) = ColumnRingGrid(CPU(), Float32, vertical_coordinate, rings, mask)

    ColumnRingGrid(
        vertical_coordinate::VerticalCoordinate,
        mask::RingGrids.AbstractField{Bool}
    ) = ColumnRingGrid(CPU(), Float32, vertical_coordinate, mask.grid, mask)
end

@adapt_structure ColumnRingGrid


# Forwarding of the Oceananigans `AbstractGrid` interface to the wrapped `RectilinearGrid`, which
# is the grid `Field`s are defined on. `ColumnRingGrid` is a grid wrapper like `AbstractLandGrid`,
# but the two are unrelated in the type hierarchy, so each defines its own forwarding.

Base.summary(grid::ColumnRingGrid) = "ColumnRingGrid with dimensions $(size(grid))"
Base.size(grid::ColumnRingGrid) = size(getfield(grid, :grid))

Architectures.architecture(grid::ColumnRingGrid) = architecture(getfield(grid, :grid))

Oceananigans.Grids.halo_size(grid::ColumnRingGrid) = halo_size(getfield(grid, :grid))
Oceananigans.Grids.isrectilinear(grid::ColumnRingGrid) = isrectilinear(getfield(grid, :grid))

@inline Oceananigans.Grids.nodes(grid::ColumnRingGrid, args...; kwargs...) = nodes(getfield(grid, :grid), args...; kwargs...)
@inline Oceananigans.Grids.xnodes(grid::ColumnRingGrid, args...; kwargs...) = xnodes(getfield(grid, :grid), args...; kwargs...)
@inline Oceananigans.Grids.ynodes(grid::ColumnRingGrid, args...; kwargs...) = ynodes(getfield(grid, :grid), args...; kwargs...)
@inline Oceananigans.Grids.znodes(grid::ColumnRingGrid, args...; kwargs...) = znodes(getfield(grid, :grid), args...; kwargs...)

# Generalized coordinate names and nodes. These are not part of the documented `AbstractGrid`
# interface and `ξname`/`ηname`/`rname` are not even exported by `Oceananigans.Grids`, but
# `set!(::Field, ::Function)` needs them to name and evaluate the coordinates of each node.
Oceananigans.Grids.ξname(grid::ColumnRingGrid) = Oceananigans.Grids.ξname(getfield(grid, :grid))
Oceananigans.Grids.ηname(grid::ColumnRingGrid) = Oceananigans.Grids.ηname(getfield(grid, :grid))
Oceananigans.Grids.rname(grid::ColumnRingGrid) = Oceananigans.Grids.rname(getfield(grid, :grid))

@inline Oceananigans.Grids.ξnode(i, j, k, grid::ColumnRingGrid, ℓx, ℓy, ℓz) = ξnode(i, j, k, getfield(grid, :grid), ℓx, ℓy, ℓz)
@inline Oceananigans.Grids.ηnode(i, j, k, grid::ColumnRingGrid, ℓx, ℓy, ℓz) = ηnode(i, j, k, getfield(grid, :grid), ℓx, ℓy, ℓz)
@inline Oceananigans.Grids.rnode(i, j, k, grid::ColumnRingGrid, ℓx, ℓy, ℓz) = rnode(i, j, k, getfield(grid, :grid), ℓx, ℓy, ℓz)

# See the corresponding note for land grids in `grids.jl`: Oceananigans accesses grid dimensions and
# discretization data as struct fields, so forward anything that is not one of our own fields.
@inline Base.getproperty(grid::ColumnRingGrid, name::Symbol) = hasfield(typeof(grid), name) ? getfield(grid, name) : getproperty(getfield(grid, :grid), name)

Base.propertynames(grid::ColumnRingGrid) = (fieldnames(typeof(grid))..., propertynames(getfield(grid, :grid))...)

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

# Land grids delegate the ring grid conversions to the discretization of their ground domain,
# which is the domain that carries the `rings` and `mask`.
RingGrids.Field(field::Union{AbstractField, AbstractArray}, grid::AbstractLandGrid; kwargs...) = RingGrids.Field(field, ground_domain(grid); kwargs...)
RingGrids.Field(arch::AbstractArchitecture, field::Union{AbstractField, AbstractArray}, grid::AbstractLandGrid; kwargs...) = RingGrids.Field(arch, field, ground_domain(grid); kwargs...)
Oceananigans.Field(ring_field::RingGrids.AbstractField, grid::AbstractLandGrid; kwargs...) = Oceananigans.Field(ring_field, ground_domain(grid); kwargs...)
Oceananigans.FieldTimeSeries(ring_field::RingGrids.AbstractField, grid::AbstractLandGrid, times::AbstractVector; kwargs...) = Oceananigans.FieldTimeSeries(ring_field, ground_domain(grid), times; kwargs...)

"""
    $SIGNATURES

Serialize a `ColumnRingGrid` as the `RectilinearGrid` it wraps. The `rings` and `mask` fields are
intentionally discarded, so `Field`s will be read back as plain `RectilinearGrid`s.
"""
Oceananigans.OutputWriters.serializeproperty!(file, address, grid::ColumnRingGrid) =
    Oceananigans.OutputWriters.serializeproperty!(file, address, getfield(grid, :grid))
