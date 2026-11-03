"""
    $TYPEDEF

Wrapper for RootSolvers.jl root-finding methods.
"""
struct RootSolver{NF, M, S, Tolerance <: RootSolvers.AbstractTolerance{NF}}
    "Numerical tolerance of the root finding iteration"
    tolerance::Tolerance

    "Maximum number of iterations to run"
    max_iterations::Int

    "Solution type (e.g., VerboseSolution, CompactSolution)"
    solution_type::S

    function RootSolver(::Type{NF}, ::Type{M}, tolerance, max_iterations, solution_type) where {NF, M <: RootSolvers.RootSolvingMethod}
        return new{NF, M, typeof(solution_type), typeof(tolerance)}(tolerance, max_iterations, solution_type)
    end
end

function RootSolver(
        ::Type{NF};
        method = RootSolvers.NewtonsMethod{NF},
        tolerance = RootSolvers.ResidualTolerance(sqrt(eps(NF))),
        max_iterations::Int = 100,
        solution_type::S = RootSolvers.CompactSolution()
    ) where {NF, S}
    return RootSolver(NF, method, tolerance, max_iterations, solution_type)
end

@propagate_inbounds function solve!(
        out, indices::NTuple{N, Integer}, grid, fields,
        step_func!::ObjectiveFunction{target},
        solver::RootSolver{NF, M, S},
        func_args...;
        func_kwargs...
    ) where {NF, N, M, S, target}
    target_field = getproperty(out, target)
    # Build the residual function, with derivatives if necessary
    residual! = build_residual(out, indices, grid, fields, step_func!, target_field, solver, func_args...; func_kwargs...)

    # Solve for the root using RootSolvers.jl
    x₀ = target_field[field_indices(indices)...]
    method = M(x₀)
    tolerance = solver.tolerance
    result = RootSolvers.find_zero(residual!, method, solver.solution_type, tolerance, solver.max_iterations)

    # Extract the root and number of iterations from the result
    x_root = result.root

    # Ensure the target field holds the final estimate
    target_field[field_indices(indices)...] = x_root

    if S <: RootSolvers.CompactSolution
        return x_root
    else
        return x_root, result.iter_performed
    end
end

build_residual(
    out, indices, grid, fields,
    step_func!::ObjectiveFunction{<:Any, F, DF},
    target_field,
    solver::RootSolver{NF},
    func_args...;
    func_kwargs...
) where {NF, F, DF} = step_func!

@propagate_inbounds function build_residual(
        out, indices, grid, fields,
        step_func!::ObjectiveFunction{<:Any, F, DF},
        target_field,
        solver::RootSolver{NF, RootSolvers.NewtonsMethod{NF}},
        func_args...;
        func_kwargs...
    ) where {NF, F, DF}

    dstep_func! = step_func!.dfunc

    # RootSolvers.jl calls these closures, so the caller's `@inbounds` cannot propagate into them;
    # `indices` must be valid for `target_field`, as they are when `solve!` is called from a kernel
    function residual!(x)
        @inbounds target_field[field_indices(indices)...] = x
        return @inbounds step_func!(out, indices..., grid, fields, func_args...; func_kwargs...)
    end

    function dresidual!(x)
        @inbounds target_field[field_indices(indices)...] = x
        return @inbounds dstep_func!(out, indices..., grid, fields, func_args...; func_kwargs...)
    end

    return x -> (residual!(x), dresidual!(x))
end

@propagate_inbounds function build_residual(
        out, indices, grid, fields,
        step_func!::ObjectiveFunction{<:Any, F, Nothing},
        target_field,
        solver::RootSolver{NF, RootSolvers.NewtonsMethod{NF}},
        func_args...;
        func_kwargs...
    ) where {NF, F}

    # RootSolvers.jl calls these closures, so the caller's `@inbounds` cannot propagate into them;
    # `indices` must be valid for `target_field`, as they are when `solve!` is called from a kernel
    function residual!(x)
        @inbounds target_field[field_indices(indices)...] = x
        return @inbounds step_func!(out, indices..., grid, fields, func_args...; func_kwargs...)
    end

    function residual_with_derivative!(x)
        h = sqrt(eps(NF)) * (1 + abs(x))
        r = residual!(x)
        drdx = (residual!(x + h) - r) / h
        return r, drdx
    end

    return residual_with_derivative!
end
