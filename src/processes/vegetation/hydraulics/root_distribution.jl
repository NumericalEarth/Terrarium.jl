"""
    $TYPEDEF

Static vegetation root distribution implementation in [willeitPALADYNV10Comprehensive2016](@cite)
based on the scheme proposed by [zengGlobalVegetationRoot2001](@cite). The continuous density of the
root distribution is modeled as
```math
\\frac{\\partial R}{\\partial z} = \\frac{1}{2} \\left[ a \\exp(a z) + b \\exp(b z) \\right]
```
which is then integrated over the soil column and normalized to sum to unity. Note that
this is effectively the average of two exponential distributions with rates `a` and `b`, both
with units m⁻¹. The resulting CDF of this distribution determines the root distribution.

Properties:
$FIELDS

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
* [zengGlobalVegetationRoot2001](@cite) Zeng, Journal of Hydrometeorology (2001)
"""
@parameterized @kwdef struct StaticExponentialRootDistribution{NF} <: AbstractRootDistribution{NF}
    "First empirical rate parameter for root distribution"
    @param a::NF = 7.0 (units = u"m^-1", bounds = Positive) # TODO PFT-specific parameter (here needleleaf tree)

    "Second empirical rate parameter for root distribution"
    @param b::NF = 2.0 (units = u"m^-1", bounds = Positive) # TODO PFT-specific parameter (here needleleaf tree)
end

StaticExponentialRootDistribution(::Type{NF}; kwargs...) where {NF} = StaticExponentialRootDistribution{NF}(; kwargs...)

"""
    $TYPEDSIGNATURES

Compute the continuous density function of the root distirbution as a function of depth `z`.
"""
@inline function root_density(rd::StaticExponentialRootDistribution{NF}, z) where {NF}
    (; a, b) = rd
    ∂R∂z = NF(0.5) * (a * exp(a * z) + b * exp(b * z))
    return ∂R∂z
end

variables(rootdist::StaticExponentialRootDistribution) = (
    auxiliary(:root_fraction, Ground(XYZ()), root_fraction, rootdist), # Static root fraction defined as function
)

"""
    $TYPEDEF

Returns a `FunctionField` that lazily computes the static root distribution on a 1D column grid.

The `FunctionField` takes `(x, z)` arguments, so this is restricted to land grids with a `Flat`
lateral (`y`) dimension, i.e. those built on column discretizations.
"""
function root_fraction(grid::AbstractLandGrid{<:Any, <:Any, Flat}, clock, fields, rootdist::StaticExponentialRootDistribution{NF}) where {NF}
    ground_grid = ground_domain(grid)
    # define pdf of root distribution as a continuous function of depth
    ∂R∂z = FunctionField{Center, Center, Center}(ground_grid, parameters = rootdist) do x, z, params
        root_density(params, z)
    end
    # scale by the layer thicknesses
    Δz = zspacings(ground_grid, Center(), Center(), Center())
    R = ∂R∂z * Δz
    # and normalize
    R_norm = R / sum(R, dims = 3)
    return R_norm
end

compute_auxiliary!(state, grid, ::StaticExponentialRootDistribution, args...) = nothing
