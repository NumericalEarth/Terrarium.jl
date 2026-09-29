"""
    $SIGNATURES

Construct the cell interfaces of a soil column of `N` layers of constant thickness `Δz`, spanning
a total depth of `N * Δz` with the surface at `z = 0`. The number format of the returned interfaces
follows that of `Δz`, so it also determines the default number format of a grid built from them.

This is a convenience for the uniformly spaced case, which Oceananigans expresses as a plain range
of cell interfaces; `UniformSpacing(Δz = 0.1, N = 10)` is equivalent to
`range(-10 * 0.1, 0, length = 11)`. See [`ExponentialSpacing`](@ref) for a column whose layers grow
with depth.

```jldoctest
using Terrarium

z = UniformSpacing(Δz = 0.1, N = 10)
(num_layers(z), first(z), last(z))

# output
(10, -1.0, 0.0)
```
"""
function UniformSpacing(; Δz::Real = 0.1, N::Int = 100)
    N > 0 || throw(ArgumentError("number of layers must be positive, got N = $N"))
    Δz > 0 || throw(ArgumentError("layer thickness must be positive, got Δz = $Δz"))
    return range(-N * Δz, zero(Δz), length = N + 1)
end

"""
    $SIGNATURES

Construct an Oceananigans `ExponentialDiscretization` for a soil column of `N` layers whose
thickness grows geometrically with depth, from `Δz_min` at the surface to `Δz_max` at the bottom.
The total depth of the column follows from the layer thicknesses,

    D = Δz_min (ρᴺ - 1) / (ρ - 1),    ρ = (Δz_max / Δz_min)^(1 / (N - 1))

where `ρ` is the ratio between the thicknesses of successive layers. Oceananigans parameterizes the
same geometric family by the e-folding `scale` of its exponential coordinate mapping, which for the
spacings above is `D / (N log ρ)`. Note that Oceananigans orders cell interfaces bottom-up, so the
first cell of the returned discretization is the thick bottom layer.

For a column of constant layer thickness, use [`UniformSpacing`](@ref) instead; the exponential
mapping is degenerate when `Δz_min == Δz_max`.

```jldoctest
using Terrarium

z = ExponentialSpacing(Δz_min = 0.05, Δz_max = 100.0, N = 50)
(num_layers(z), round(z.faces[1], digits = 3), z.faces[end])

# output
(50, -695.654, 0.0)
```
"""
function ExponentialSpacing(; Δz_min::Real = 0.05, Δz_max::Real = 100.0, N::Int = 50)
    N > 1 || throw(ArgumentError("number of layers must be > 1, got N = $N"))
    Δz_min > 0 || throw(ArgumentError("minimum layer thickness must be positive, got Δz_min = $Δz_min"))
    Δz_max > Δz_min || throw(ArgumentError("maximum layer thickness $Δz_max must exceed the minimum $Δz_min; pass a uniformly spaced range of cell interfaces for a column of constant thickness"))
    # Ratio between the thicknesses of successive layers, and the total depth that they span.
    ρ = (Δz_max / Δz_min)^(1 / (N - 1))
    depth = Δz_min * (ρ^N - 1) / (ρ - 1)
    # Oceananigans maps a uniform coordinate through `expm1`, which gives thicknesses that grow by a
    # constant factor `exp(Δξ / scale)` per cell with `Δξ = depth / N`; matching that factor to `ρ`
    # reproduces the layer thicknesses above.
    scale = depth / (N * log(ρ))
    return ExponentialDiscretization(N, -depth, zero(depth); scale, bias = :right)
end

"""
The vertical coordinates accepted by Terrarium's grid constructors: either a vector (or range) of
cell interfaces, such as the one returned by [`UniformSpacing`](@ref), or one of the Oceananigans
discretizations which determine their own number of cells, such as the one returned by
[`ExponentialSpacing`](@ref).

Note that a coordinate given as a `Tuple` of endpoints, which Oceananigans also accepts, does not
determine the number of layers and so cannot be used here.
"""
const VerticalCoordinate = Union{AbstractVector, CallableDiscretization}

"""
    $SIGNATURES

Return the number of layers spanned by the given vertical coordinate.
"""
num_layers(z::AbstractVector) = length(z) - 1
num_layers(z::CallableDiscretization) = length(z)
