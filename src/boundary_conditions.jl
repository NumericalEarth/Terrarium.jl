"""
Alias for union type of `FieldBoundaryConditions` or a named tuple of `BoundaryCondition`s with
keys corresponding to boundary locations (i.e. top, bottom, etc.)
"""
const FieldBC = Union{FieldBoundaryConditions, NamedTuple{locs, <:Tuple{Vararg{BoundaryCondition}}}} where {locs}

"""
Alias for a `NamedTuple` of `FieldBC` types where the keys correspond to field/variable names.
"""
const FieldBCs{names, BCs} = NamedTuple{names, BCs} where {names, BCs <: Tuple{Vararg{FieldBC}}}

"""
    merge_boundary_conditions(bcs::FieldBCs...)

Recursively merge an arbitrary number of field/variable boundary conditions.
"""
merge_boundary_conditions(bcs::FieldBCs...) = merge_recursive(bcs...)

"""
    fill_halo_regions!(field::AbstractField, state::StateVariables)

Alias for `fill_halo_regions!(field, state.clock, state.inputs)`.
"""
@inline function BoundaryConditions.fill_halo_regions!(field::Field, state::StateVariables)
    fill_halo_regions!(field, state.clock, state.inputs)
    return nothing
end

"""
    getbc(::Variable{name}, i::Integer, j::Integer, grid::AbstractGrid, clock, fields) where {name}

Implementation of `Oceananigans.BoundaryConditions.getbc` for variable placeholders that retrieves the input `Field` from
`fields` and returns the value at the given index.
"""
@inline function BoundaryConditions.getbc(::Variable{name}, i::Integer, j::Integer, grid::AbstractGrid, clock, fields) where {name}
    field = getproperty(fields, name)
    return @inbounds field[i, j]
end

