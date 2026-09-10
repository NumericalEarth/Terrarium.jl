"""
    $SIGNATURES

Return `true` when Reactant is loaded, `false` otherwise.

Note, that `ReactantCore.within_compile` is not used because it doesn't work properly when KA 
kernels aren't raised.
"""
@inline uses_reactant(_) = false
@inline uses_reactant() = uses_reactant(ReactantMarker())

"""
Marker type used to dispatch [`uses_reactant`](@ref).
"""
struct ReactantMarker end

"""
    @assert_kernel cond [text]

Drop-in replacement for `Base.@assert` that is disabled once Reactant is loaded.
"""
macro assert_kernel(cond, text...)
    assertion = esc(Expr(:macrocall, GlobalRef(Base, Symbol("@assert")), __source__, cond, text...))
    return quote
        if !$(uses_reactant)()
            $assertion
        end
        nothing
    end
end

"""
    $TYPEDSIGNATURES

Return a `KernelFunctionOperation` over `grid` with the given `state` and trailing `args`.
"""
function kernel_operation(func, state, grid, args...; location = (Center, Center, Nothing), with_clock = false)
    fields = get_fields(state, args...)
    fgrid = get_field_grid(grid)
    if with_clock
        return KernelFunctionOperation{location...}(func, fgrid, state.clock, fields, args...)
    else
        return KernelFunctionOperation{location...}(func, fgrid, fields, args...)
    end
end

kernel_operation2D(func, state, grid, args...; X = Center, Y = Center, with_clock = false) = kernel_operation(func, state, grid, args...; location = (X, Y, Nothing), with_clock)
kernel_operation3D(func, state, grid, args...; X = Center, Y = Center, Z = Center, with_clock = false) = kernel_operation(func, state, grid, args...; location = (X, Y, Z), with_clock)

struct KernelFunction{autonomous, Func, Args}
    func::Func
    args::Args
end

function (op::KernelFunction{false})(var, grid, clock, fields, args...)
    loc = map(typeof, location(vardims(var)))
    return KernelFunctionOperation{loc...}(op.func, get_field_grid(grid), clock, fields, args..., op.args...)
end

function (op::KernelFunction{true})(var, grid, clock, fields, args...)
    loc = map(typeof, location(vardims(var)))
    return KernelFunctionOperation{loc...}(op.func, get_field_grid(grid), fields, args..., op.args...)
end

"""
    kernel(func, args...; clock = false)

Return a `KernelFunction` that lazily constructs a `KernelFunctionOperation` from the given
`func` and tuple of `args` when invoked, i.e:

```juila
ctor = kerenl(my_function, arg1, arg2)
...
kfo = ctor(grid, clock, fields)
```

This is intended to be used as a constructor for `AuxiliaryVariable`s:

```julia
myvar(i, j, k, grid, fields) = clamp(fields.x[i, j, k], zero(eltype(grid)), one(eltype(grid)))

auxvar = auxiliary(:myvar, XYZ(), kernel(myvar))
```
"""
function kernel(func, args...; clock = false)
    return KernelFunction{!clock, typeof(func), typeof(args)}(func, args)
end

"""
    findfirst_z(i, j, condition_func, z_nodes, field)

2D kernel function that finds the first coordinate in `z_nodes` where `condition_func(field[i, j, k])`.
This implementation performs a linear scan over the z-axis and thus has time complexity O(N_z).
"""
@propagate_inbounds function findfirst_z(i, j, condition_func, z_nodes, field)
    n = length(z_nodes)
    idx = -1
    for k in 1:n
        found = (idx < 0) & condition_func(field[i, j, k])
        idx = ifelse(found, k, idx)
    end
    # Select the index (not the nodes): ifelse evaluates both branches, so indexing in the
    # unselected branch with idx = -1 would be out of bounds.
    return z_nodes[ifelse(idx > 0, idx, n)]
end

"""
    min_zᵃᵃᶠ(i, j, k, grid, x)
    min_zᵃᵃᶠ(i, j, k, grid, f, args...)

Computes the field or function at the vertical (z-axis) face by taking the `min` of the two adjacent vertical layers.
"""
@inline min_zᵃᵃᶠ(i, j, k, grid, c) = @inbounds min(c[i, j, k], c[i, j, k - 1])
@inline min_zᵃᵃᶠ(i, j, k, grid, f, args...) = @inbounds min(f(i, j, k, grid, args...), f(i, j, k - 1, grid, args...))
