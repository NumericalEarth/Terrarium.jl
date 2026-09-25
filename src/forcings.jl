"""
Type alias for a `UnionAll` over `DiscreteForcing` and `ContinuousForcing` types from Oceananigans.
"""
const AbstractForcing{P} = Union{DiscreteForcing{P}, ContinuousForcing{LX, LY, LZ, P}} where {LX, LY, LZ}

"""
    forcing(i, j, k, grid, clock, fields, forcing::AbstractForcing, target::AbstractProcess, args...)

Return the value computed by the given `Oceananigans` forcing type, which should be an instance of
either `DiscreteForcing` or `ContinuousForcing`. Note that `target` and additional `args` are only included
for interface consistency and are not passed through to `forcing`.
"""
@propagate_inbounds function forcing(i, j, k, grid, clock, fields, forcing::AbstractForcing, target::AbstractProcess, args...)
    return forcing(i, j, k, ground_domain(grid), clock, fields)
end

@inline function forcing(i, j, k, grid, clock, fields, ::Nothing, target::AbstractProcess, args...)
    return zero(eltype(grid))
end
