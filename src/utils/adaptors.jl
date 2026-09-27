struct NumberFormatAdaptor{NF} end

"""
    $TYPEDSIGNATURES

Adaptor that reconstructs arbitrary data structures with all numeric values
converted to the specified number format `NF`.
"""
function Adapt.adapt(::NumberFormatAdaptor{NF}, obj) where {NF <: Number}
    vals = map(NF, flatten(obj, flattenable, Number))
    return reconstruct(obj, vals, flattenable, Number)
end

"""
    $SIGNATURES

Return the `NF` type of the given type(s) or object(s).
"""
@inline number_format(typ::Type) = eltype(typ)
@inline number_format(arg) = number_format(typeof(arg))
@inline number_format(args...) = promote_type(map(number_format, args)...)