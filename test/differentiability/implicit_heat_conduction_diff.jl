using Terrarium
using Test

using Enzyme
using Statistics

function build_implicit_soil_energy_model(arch, ::Type{NF}; iterations = 1) where {NF}
    grid = ColumnGrid(arch, NF, ExponentialSpacing(N = 10))
    initializer = SoilInitializer(eltype(grid))
    solver = Picard(NF; iterations)
    timestepper = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver))
    model = SoilModel(grid; initializer, timestepper)
    return model
end

# Mean soil temperature after one implicit step taken from the internal energy held by `integrator`. The
# closure is applied first so that the temperature at tⁿ is a function of the (perturbed) internal energy.
function mean_soil_temperature_implicit_step!(integrator, Δt)
    Terrarium.closure!(integrator.state, integrator.model)
    timestep!(integrator, Δt)
    return mean(interior(integrator.state.temperature))
end

@testset "Implicit soil energy model: timestep! (iterations = $iterations)" for iterations in (1, 2)
    NF = Float64
    model = build_implicit_soil_energy_model(CPU(), NF; iterations)
    boundary_conditions = SoilHeatFlux(NF(10))
    integrator = initialize(model; boundary_conditions)
    dintegrator = make_zero(integrator)
    Δt = NF(3600)
    @time Enzyme.autodiff(set_runtime_activity(Reverse), mean_soil_temperature_implicit_step!, Active, Duplicated(integrator, dintegrator), Const(Δt))
    ∂U = interior(dintegrator.state.internal_energy)
    @test all(isfinite, ∂U)
    @test any(!iszero, ∂U)
    # central finite-difference check of the sensitivity to the initial internal energy of the uppermost cell
    Nz = size(∂U, 3)
    ε = NF(1.0e2)
    perturbed_step(δ) = begin
        integrator = initialize(model; boundary_conditions)
        interior(integrator.state.internal_energy)[1, 1, Nz] += δ
        mean_soil_temperature_implicit_step!(integrator, Δt)
    end
    ∂U_fd = (perturbed_step(ε) - perturbed_step(-ε)) / (2ε)
    @info "Implicit soil energy sensitivity (iterations = $iterations)" ∂U[1, 1, Nz] ∂U_fd
    @test ∂U[1, 1, Nz] ≈ ∂U_fd rtol = 1.0e-4
end
