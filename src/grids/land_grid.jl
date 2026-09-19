"""
    $TYPEDEF

The spatial discretization of a land model, consisting of one grid per vertical domain: `ground`
(soil), `snow`, and `canopy`. All three domains share the same horizontal discretization and differ
only in their vertical discretization. Domains which are not resolved vertically by the model
(e.g. a single-layer snow scheme or a big-leaf canopy) are represented by `nothing`.

`LandGrid` is the only implementation of [`AbstractLandGrid`](@ref) and is normally constructed
implicitly by the model constructors via [`create_land_grid`](@ref) rather than directly by the user.

Properties:
$(TYPEDFIELDS)
"""
struct LandGrid{NF, TX, TY, TZ, Arch, GroundGrid, SnowGrid, CanopyGrid} <: AbstractLandGrid{NF, TX, TY, TZ, Arch}
    "Spatial discretization of the ground (soil) domain; also the grid on which `Field`s are defined by default."
    ground::GroundGrid

    "Spatial discretization of the snow domain, or `nothing` when the snowpack is not vertically resolved."
    snow::SnowGrid

    "Spatial discretization of the canopy domain, or `nothing` when the canopy is not vertically resolved."
    canopy::CanopyGrid

    """
        $SIGNATURES

    Construct a `LandGrid` from the given per-domain grids without checking their compatibility.
    This is the constructor invoked by `adapt`; use the keyword constructor
    `LandGrid(ground; snow, canopy)` to construct a land grid on the host.
    """
    function LandGrid(
            ground::AbstractGrid,
            snow::Optional{AbstractGrid},
            canopy::Optional{AbstractGrid},
        )
        NF = eltype(ground)
        TX, TY, TZ = topology(ground)
        arch = architecture(ground)
        return new{NF, TX, TY, TZ, typeof(arch), typeof(ground), typeof(snow), typeof(canopy)}(ground, snow, canopy)
    end
end

"""
    $SIGNATURES

Construct a `LandGrid` on the spatial discretization `ground` with optional `snow` and `canopy`
domain grids. The domain grids must all share the same horizontal discretization and architecture;
pass `nothing` (the default) for domains which are not vertically resolved.
"""
function LandGrid(
        ground::AbstractGrid;
        snow::Optional{AbstractGrid} = nothing,
        canopy::Optional{AbstractGrid} = nothing,
    )
    check_domain_compatibility(ground, snow, :snow)
    check_domain_compatibility(ground, canopy, :canopy)
    return LandGrid(ground, snow, canopy)
end

"""
    $SIGNATURES

Check that the `domain` grid named `name` is compatible with the `ground` domain grid, i.e. that
both share the same horizontal discretization, numeric type, and architecture. Throws an
`ArgumentError` if not. Note that this is a host-side check which must never be invoked from a
kernel.
"""
function check_domain_compatibility(ground::AbstractGrid, domain::AbstractGrid, name::Symbol)
    Nx, Ny, _ = size(ground)
    Mx, My, _ = size(domain)
    if (Nx, Ny) != (Mx, My)
        throw(ArgumentError("horizontal dimensions of the $name domain grid $((Mx, My)) do not match those of the ground domain grid $((Nx, Ny))"))
    end
    if eltype(ground) != eltype(domain)
        throw(ArgumentError("number format of the $name domain grid $(eltype(domain)) does not match that of the ground domain grid $(eltype(ground))"))
    end
    if architecture(ground) != architecture(domain)
        throw(ArgumentError("architecture of the $name domain grid $(architecture(domain)) does not match that of the ground domain grid $(architecture(ground))"))
    end
    return nothing
end

check_domain_compatibility(::AbstractGrid, ::Nothing, ::Symbol) = nothing

"""
    $SIGNATURES

Return the spatial discretization of the ground (soil) domain of `grid`. Grids which are not land
grids are their own ground discretization.
"""
@inline ground_domain(grid::AbstractGrid) = grid
@inline ground_domain(grid::LandGrid) = getfield(grid, :ground)

"""
    $SIGNATURES

Return the spatial discretization of the snow domain of `grid`, or `nothing` if the snowpack is not
vertically resolved. Grids which are not land grids never resolve a snow domain.
"""
@inline snow_domain(::AbstractGrid) = nothing
@inline snow_domain(grid::LandGrid) = getfield(grid, :snow)

"""
    $SIGNATURES

Return the spatial discretization of the canopy domain of `grid`, or `nothing` if the canopy is not
vertically resolved. Grids which are not land grids never resolve a canopy domain.
"""
@inline canopy_domain(::AbstractGrid) = nothing
@inline canopy_domain(grid::LandGrid) = getfield(grid, :canopy)

"""
    $SIGNATURES

Return the spatial discretization of `grid` for the model domain selected by `domain`, or `nothing`
if `grid` does not resolve that domain vertically.
"""
@inline get_domain(grid::AbstractGrid, ::Ground) = ground_domain(grid)
@inline get_domain(grid::AbstractGrid, ::Snow) = snow_domain(grid)
@inline get_domain(grid::AbstractGrid, ::Canopy) = canopy_domain(grid)
# The surface is an interface, not a vertical domain: it is never discretized in its own right, so
# its (necessarily two-dimensional) variables fall back to the shared horizontal discretization.
@inline get_domain(::AbstractGrid, ::Surface) = nothing

"""
    $SIGNATURES

Return the spatial discretization on which a variable at `loc` is allocated.

A domain which the model does not resolve vertically still has variables: a single-layer snowpack has
a snow water equivalent, a big-leaf canopy has a temperature. Those variables carry no vertical
dimension, so they are allocated on the shared horizontal discretization, which is the ground
domain's. A variable which *is* vertically resolved ([`XYZ`](@ref)) cannot be placed on a domain with
no vertical discretization, and asking for one is a configuration error.
"""
@inline function variable_grid(grid::AbstractGrid, loc::VarLocation)
    domain_grid = get_domain(grid, vardomain(loc))
    return isnothing(domain_grid) ? default_domain_grid(grid, vardims(loc), vardomain(loc)) : domain_grid
end

@inline default_domain_grid(grid::AbstractGrid, ::VarDims, ::VarDomain) = ground_domain(grid)

default_domain_grid(grid::AbstractGrid, dims::XYZ, domain::VarDomain) = throw(
    ArgumentError(
        "cannot allocate a vertically resolved ($(typeof(dims))) variable on the $(summary(domain)) " *
            "domain: this $(nameof(typeof(grid))) does not discretize it vertically. Either declare the " *
            "variable on the ground domain, or construct the grid with a $(summary(domain)) discretization."
    )
)

# Properties which are not the land grid's own resolve against the ground domain grid, so that a
# ground discretization which is itself a wrapper (e.g. `ColumnRingGrid`) can expose its own
# properties (`rings`, `mask`) through the land grid.
@inline Base.getproperty(grid::LandGrid, name::Symbol) = hasfield(typeof(grid), name) ? getfield(grid, name) : getproperty(ground_domain(grid), name)

Base.propertynames(grid::LandGrid) = (fieldnames(typeof(grid))..., propertynames(ground_domain(grid))...)

@adapt_structure LandGrid

function Architectures.on_architecture(arch::AbstractArchitecture, grid::LandGrid)
    return LandGrid(
        on_architecture(arch, ground_domain(grid)),
        on_architecture(arch, snow_domain(grid)),
        on_architecture(arch, canopy_domain(grid)),
    )
end

function Base.show(io::IO, ::MIME"text/plain", grid::LandGrid{NF}) where {NF}
    domain_summary(domain) = isnothing(domain) ? "not vertically resolved" : summary(domain)
    println(io, "LandGrid{$NF} on $(architecture(grid)) with")
    println(io, "├── ground: $(summary(ground_domain(grid)))")
    println(io, "├── snow:   $(domain_summary(snow_domain(grid)))")
    println(io, "└── canopy: $(domain_summary(canopy_domain(grid)))")
    return nothing
end

"""
    create_land_grid(grid::AbstractGrid, soil, snow, vegetation)::AbstractLandGrid

Construct the [`LandGrid`](@ref) for a model discretized on the spatial discretization `grid` whose
ground, snow, and canopy domains are determined by the `soil`, `snow`, and `vegetation` process
components. This is the standard interface through which model constructors turn the grid given by
the user into the land grid stored by the model.

The default implementation returns a ground-only land grid, i.e. one for which the snow and canopy
domains are not vertically resolved. This is correct for all currently implemented snow and
vegetation schemes, which are 0D; schemes which resolve a vertical snow or canopy profile should
add a method here which builds the corresponding domain grids.
"""
create_land_grid(grid::AbstractGrid, soil = nothing, snow = nothing, vegetation = nothing) = LandGrid(grid)

# Land grids are already land grids; `create_land_grid` is idempotent so that a model can be constructed
# either from a spatial discretization or from a pre-built land grid.
create_land_grid(grid::AbstractLandGrid, args...) = grid
