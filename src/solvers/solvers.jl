"""
    $TYPEDEF

Represents an objective function for nonlinear solvers. The name `target`
refers to the output `Field` which should updated on each iteration. The
objective function should have the signature
```julia
func(out, indices..., grid, fields, func_args...; func_kwargs...)
```
where `indices` are the grid indices passed to [`solve!`](@ref) and directly
return the scalar residual. If an analytical derivative is provided via `dfunc`,
it should follow the same signature as `func` and return the derivative of
the residual with respect to the target.
"""
struct ObjectiveFunction{target, F, DF}
    "Objective function for the nonlinear solver"
    func::F

    "Optional analytical derivative of the objective function"
    dfunc::DF

    ObjectiveFunction(func, dfunc, target::Symbol) = new{target, typeof(func), typeof(dfunc)}(func, dfunc)
    ObjectiveFunction(func, target::Symbol) = ObjectiveFunction(func, nothing, target)
end

@inline (func::ObjectiveFunction)(args...) = func.func(args...)

"""
    $TYPEDEF

Simple relaxation scheme for iterative solvers.
"""
@kwdef struct RelaxationFactor{NF}
    "Relaxation factor"
    factor::NF = 0.5

    "Upper limit for relaxed updates"
    upper_limit::NF = oftype(factor, Inf)

    "Lower limit for relaxed updates"
    lower_limit::NF = oftype(factor, -Inf)
end

"""
    $TYPEDSIGNATURES

Apply the under-relaxed fixed-point update to the skin temperature. 
"""
@inline function relaxed_update(relax::RelaxationFactor{NF}, x_target::NF, x_old::NF) where {NF}
    ω = relax.factor
    x_new = (1 - ω) * x_old + ω * x_target
    return clamp(x_new, relax.lower_limit, relax.upper_limit)
end

relaxed_update(::Nothing, x_target, x_old) = x_target

# Interface

"""
    solve!(out, indices, grid, fields, objective_func!::ObjectiveFunction, solver, args...; kwargs...)

Solve the nonlinear problem defined by `objective_func!` for its `target` field at the given
`indices`, mutating `out` in place. The objective returns the residual `F(x)` whose root is
sought; on return, the target field holds the converged estimate and the method returns the
root (and, for some solvers, the number of iterations performed). Dispatches on the concrete
`solver` type, e.g. [`RootSolver`](@ref), [`NewtonSolver`](@ref) or [`FixedPointSolver`](@ref).
"""
function solve! end

# Solvers

include("fixed_point.jl")
include("newton.jl")
include("root_solvers.jl")

# Implicit solvers

include("linear_operators.jl")
include("implicit_solvers.jl")
include("picard.jl")
