using Terrarium
using Test

using Terrarium: FixedPointSolver, NewtonSolver, RootSolver, RootSolvers, ObjectiveFunction, RelaxationFactor, solve!

# Build an objective function from a scalar map g(x). The solvers now expect the
# objective to *return* the fixed-point residual F(x) = x - g(x) evaluated at the
# current value of the target field, rather than writing g(x) into the field; the
# solver itself owns the field update. Both solver families solve x = g(x).
# The `grid` and `fields` arguments are unused for these scalar tests.
function scalar_objective(g, ::Val{target}) where {target}
    function residual(out, i, _grid, _fields)
        x = out.x[i]
        return x - g(x)
    end
    return ObjectiveFunction(residual, :x)
end

# Run a solver on g starting from x0 and return (root, iterations).
function run_solver(solver, g, x0)
    out = (; x = [x0])
    objective = scalar_objective(g, Val{:x}())
    return solve!(out, (1,), nothing, nothing, objective, solver)
end

for NF in (Float32, Float64)
    tol = sqrt(eps(NF))

    @testset "FixedPointSolver ($NF)" begin
        # Use an undamped (factor = 1, or relax = nothing) update so the iteration matches the plain
        # fixed-point map for these contractive examples.
        solver = FixedPointSolver(NF; tolerance = tol, relax = RelaxationFactor(factor = NF(1.0)), max_iterations = 1000)

        @testset "Linear map" begin
            # g(x) = x/2 + 1 has the unique fixed point x = 2
            x, iters = run_solver(solver, x -> x / 2 + 1, NF(0.0))
            @test isapprox(x, NF(2.0); atol = tol)
            @test iters <= solver.max_iterations
        end

        @testset "Dottie number (cos)" begin
            # x = cos(x) ⇒ Dottie number ≈ 0.7390851332151607
            x, iters = run_solver(solver, cos, NF(1.0))
            @test isapprox(x, NF(0.7390851332151607); atol = tol)
            @test iters <= solver.max_iterations
        end

        @testset "Square root (Heron)" begin
            # g(x) = (x + 2/x)/2 has fixed point √2
            x, iters = run_solver(solver, x -> (x + 2 / x) / 2, NF(1.0))
            @test isapprox(x, sqrt(NF(2.0)); atol = tol)
            @test iters <= solver.max_iterations
        end

        @testset "Iteration limit" begin
            # g(x) = 2x is non-contractive; the iteration cannot converge and must
            # exhaust the iteration budget.
            limited = FixedPointSolver(NF; tolerance = tol, relax = RelaxationFactor(factor = NF(1.0)), max_iterations = 5)
            _, iters = run_solver(limited, x -> 2 * x, NF(1.0))
            @test iters > limited.max_iterations
        end

        @testset "Type stability" begin
            # The full solve path must infer to a concrete (NF, Int) tuple with no
            # boxing (see the header note on avoiding `NF` capture in objectives).
            result = @inferred run_solver(solver, x -> x / 2 + 1, NF(0.0))
            @test result isa Tuple{NF, Int}
        end
    end

    @testset "RootSolver ($NF)" begin
        tolerance = RootSolvers.ResidualTolerance(tol)
        solver = RootSolver(NF; tolerance, max_iterations = 100)

        @testset "Linear map" begin
            # Same linear fixed point x = 2 as above
            x = run_solver(solver, x -> x / 2 + 1, NF(0.0))
            @test isapprox(x, NF(2.0); atol = tol)
        end

        @testset "Dottie number (cos)" begin
            x = run_solver(solver, cos, NF(1.0))
            @test isapprox(x, NF(0.7390851332151607); atol = tol)
        end

        @testset "Square root via x^2 - 2 = 0" begin
            # The fixed-point form g(x) = x^2 + x - 2 corresponds to the residual
            # F(x) = x - g(x) = 2 - x^2, whose positive root is √2.
            x = run_solver(solver, x -> x^2 + x - 2, NF(1.0))
            @test isapprox(x, sqrt(NF(2.0)); atol = tol)
        end

        @testset "Iteration limit" begin
            # A stiff, slowly-converging residual paired with a tight iteration budget:
            # the solver must stop after exactly `max_iterations` steps.
            limited_solver = RootSolver(NF; tolerance, max_iterations = 2, solution_type = RootSolvers.VerboseSolution())
            _, iters = run_solver(limited_solver, x -> (x - 10)^2 + log(cbrt(x + 2) + exp(x - 1)), NF(0.0))
            @test iters == limited_solver.max_iterations
        end

        @testset "Type stability" begin
            # With the default CompactSolution the solve path must infer to a
            # concrete scalar root of type NF with no boxing.
            x = @inferred run_solver(solver, x -> x / 2 + 1, NF(0.0))
            @test x isa NF
            @test isapprox(x, NF(2.0); atol = tol)
        end

        @testset "Verbose solution type" begin
            # VerboseSolution returns the iteration count alongside the root,
            # whereas the default CompactSolution returns only the scalar root.
            verbose_solver = RootSolver(
                NF;
                tolerance,
                solution_type = RootSolvers.VerboseSolution(),
                max_iterations = 100
            )
            x, iters = run_solver(verbose_solver, x -> x / 2 + 1, NF(0.0))
            @test isapprox(x, NF(2.0); atol = tol)
            @test iters isa Integer
            @test 0 < iters <= verbose_solver.max_iterations
        end
    end

    @testset "NewtonSolver ($NF)" begin
        # NewtonSolver has no convergence test: it always runs exactly `iterations` steps, which is
        # what makes it unrollable (and therefore raisable under Reactant / cheap to differentiate).
        solver = NewtonSolver(NF; iterations = 20)

        @testset "Linear map" begin
            x, iters = run_solver(solver, x -> x / 2 + 1, NF(0.0))
            @test isapprox(x, NF(2.0); atol = tol)
            @test iters == Terrarium.iterations(solver)
        end

        @testset "Dottie number (cos)" begin
            x, _ = run_solver(solver, cos, NF(1.0))
            @test isapprox(x, NF(0.7390851332151607); atol = tol)
        end

        @testset "Square root via x^2 - 2 = 0" begin
            # Same residual form as the RootSolver case: F(x) = x - g(x) = 2 - x^2
            x, _ = run_solver(solver, x -> x^2 + x - 2, NF(1.0))
            @test isapprox(x, sqrt(NF(2.0)); atol = tol)
        end

        @testset "Fixed iteration count" begin
            # The iteration count is fixed by the type and does not depend on convergence,
            # so a non-contractive map exhausts exactly the same budget as a contractive one.
            few = NewtonSolver(NF; iterations = 3)
            _, iters = run_solver(few, x -> 2 * x, NF(1.0))
            @test iters == 3
        end

        @testset "Type stability" begin
            result = @inferred run_solver(solver, x -> x / 2 + 1, NF(0.0))
            @test result isa Tuple{NF, Int}
        end
    end
end

@testset "TridiagonalOperator with upwind advection" begin
    NF = Float64
    Nz = 8
    grid = ground_domain(ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.05, Δz_max = 0.3, N = Nz)))
    Δz = [Terrarium.Δzᵃᵃᶜ(1, 1, k, grid) for k in 1:Nz]
    Δt = 3600.0
    operator = TridiagonalOperator(grid; advection = true)
    C = 1.0 .+ (1:Nz) ./ Nz
    κ = 1.0e-5 .* (1.0 .+ sin.(1:(Nz + 1)))
    # velocities of both signs, with advective Courant numbers well above one
    u = 1.0e-4 .* cos.(1:(Nz + 1))
    # outflow through both boundary faces
    u[1] = -2.0e-4
    u[Nz + 1] = 2.0e-4
    interior(operator.diagonal_coefficient)[1, 1, :] .= C
    interior(operator.face_conductivity)[1, 1, :] .= κ
    interior(operator.face_velocity)[1, 1, :] .= u
    rhs = Field{Center, Center, Center}(grid)
    interior(rhs)[1, 1, :] .= 1.0 .+ (1:Nz) .^ 2 ./ Nz
    ϕ = Field{Center, Center, Center}(grid)
    Terrarium.solve!(ϕ, operator, rhs, Δt)
    # the solution satisfies the system in flux form, with the tendencies of the conservative update
    residual = map(1:Nz) do k
        diffusion = Terrarium.compute_implicit_diffusion_tendency(1, 1, k, grid, operator.face_conductivity, ϕ)
        advection = Terrarium.compute_implicit_advection_tendency(1, 1, k, grid, operator.face_velocity, ϕ)
        C[k] * interior(ϕ)[1, 1, k] - Δt * (diffusion + advection) - interior(rhs)[1, 1, k]
    end
    @test maximum(abs, residual) < 1.0e-12 * maximum(abs, interior(rhs))
    # dense matrix of the coefficients: an M-matrix, whose Δz-weighted column sums are Δz C plus the
    # outflow through the boundary faces (conservation)
    A = zeros(NF, Nz, Nz)
    for k in 1:Nz
        A[k, k] = Terrarium.compute_implicit_diffusion_diagonal(1, 1, k, grid, operator.face_conductivity, operator.face_velocity, operator.diagonal_coefficient, Δt)
        k < Nz && (A[k, k + 1] = Terrarium.compute_implicit_diffusion_upper_diagonal(1, 1, k, grid, operator.face_conductivity, Δt) + Terrarium.compute_implicit_advection_upper_diagonal(1, 1, k, grid, operator.face_velocity, Δt))
        k > 1 && (A[k, k - 1] = Terrarium.compute_implicit_diffusion_lower_diagonal(1, 1, k - 1, grid, operator.face_conductivity, Δt) + Terrarium.compute_implicit_advection_lower_diagonal(1, 1, k - 1, grid, operator.face_velocity, Δt))
    end
    @test all(A[i, j] <= 0 for i in 1:Nz, j in 1:Nz if i != j)
    @test all(>(0), [A[k, k] for k in 1:Nz])
    boundary_outflow = zeros(NF, Nz)
    boundary_outflow[1] = -Δt * min(u[1], 0)
    boundary_outflow[Nz] = Δt * max(u[Nz + 1], 0)
    @test vec(sum(Δz .* A; dims = 1)) ≈ Δz .* C .+ boundary_outflow rtol = 1.0e-12
    @test A \ interior(rhs)[1, 1, :] ≈ interior(ϕ)[1, 1, :] rtol = 1.0e-12
    # without a face velocity, the operator reduces to pure diffusion
    diffusion_only = TridiagonalOperator(grid)
    @test isnothing(diffusion_only.face_velocity)
    @test Terrarium.compute_implicit_advection_tendency(1, 1, 3, grid, nothing, ϕ) == 0
    @test Terrarium.compute_implicit_advection_diagonal(1, 1, 3, grid, nothing, Δt) == 0
end
