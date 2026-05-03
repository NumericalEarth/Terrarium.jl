global DEBUG::Bool = haskey(ENV, "TERRARIUM_DEBUG") && ENV["TERRARIUM_DEBUG"] == "true"

"""
    debug!(debug::Bool)

Enable or disable global debug mode for Terrarium. Debug mode 
"""
function debug!(debug::Bool)
    global DEBUG = debug
    DEBUG && @warn "Debug mode enabled! Debugging hooks will now be active and performance may be degraded."
    return DEBUG
end

"""
    $SIGNATURES

Check whether the given `field` has any `NaN` or `Inf` values and raise an error if `NaN`s are detected.
"""
checkfinite!(field::AbstractField, name = nothing) = any(!isfinite, parent(field)) && error("Found NaN/Inf values in Field $name: $field")
function checkfinite!(nt::NamedTuple)
    for key in keys(nt)
        checkfinite!(nt[key], key)
    end
    return nothing
end

"""
    $SIGNATURES

Provides a "hook" for handling debug calls from relevant callsites. Default implementations for
`Field` and `NamedTuple` (assumed to be of `Field`s) simply forward to [`checkfinite!`](@ref).
"""
@inline debughook!(args...) = nothing
@inline debughook!(field::AbstractField) = checkfinite!(field)
@inline debughook!(nt::NamedTuple) = checkfinite!(nt)

"""
    $SIGNATURES

Utility method that forwards `args` to `debughook!` *if and only if debug mode is enabled*. Debug mode is set by
the global variable `DEBUG` which can be toggled by the user facing API [`debug!`](@ref).
"""
@inline function debugsite!(args...)
    if DEBUG
        debughook!(args...)
    end
    return nothing
end
