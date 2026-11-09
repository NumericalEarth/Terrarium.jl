using Terrarium
using Test

using Enzyme
using Statistics

function build_implicit_richards_model(arch, ::Type{NF}; iterations = 2) where {NF}
    grid = ColumnGrid(arch, NF, UniformSpacing(Δz = NF(0.1), N = 10))
    swrc = VanGenuchten(α = NF(2), n = NF(2))
    hydraulic_properties = ConstantSoilHydraulics(NF; swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(NF))
    hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(NF; hydrology)
    solver = Picard(NF; iterations)
    timestepper = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver))
    return SoilModel(grid; soil, timestepper)
end

# Mean pressure head after one implicit step taken from the saturation held by `integrator`. The closure
# is applied first so that the pressure head at tⁿ is a function of the (perturbed) saturation.
function mean_pressure_head_implicit_step!(integrator, Δt)
    Terrarium.closure!(integrator.state, integrator.model)
    timestep!(integrator, Δt)
    return mean(interior(integrator.state.pressure_head))
end

@testset "Implicit Richards model: timestep! (iterations = 2)" begin
    NF = Float64
    model = build_implicit_richards_model(CPU(), NF)
    # constant infiltration (saturation flux, positive upward) at the top
    boundary_conditions = InfiltrationFlux(NF(-1.0e-6))
    initializers = (temperature = NF(5), saturation_water_ice = (x, z) -> NF(0.5) - NF(0.1) * z)
    integrator = initialize(model; boundary_conditions, initializers)
    @test timestepping(integrator.state, Val(:saturation_water_ice)) === Implicit()
    dintegrator = make_zero(integrator)
    Δt = NF(3600)
    @time Enzyme.autodiff(set_runtime_activity(Reverse), mean_pressure_head_implicit_step!, Active, Duplicated(integrator, dintegrator), Const(Δt))
    ∂s = interior(dintegrator.state.saturation_water_ice)
    @test all(isfinite, ∂s)
    @test any(!iszero, ∂s)
    # central finite-difference check of the sensitivity to the initial saturation of the uppermost cell
    Nz = size(∂s, 3)
    ε = NF(1.0e-5)
    perturbed_step(δ) = begin
        integrator = initialize(model; boundary_conditions, initializers)
        interior(integrator.state.saturation_water_ice)[1, 1, Nz] += δ
        mean_pressure_head_implicit_step!(integrator, Δt)
    end
    ∂s_fd = (perturbed_step(ε) - perturbed_step(-ε)) / (2ε)
    @info "Implicit Richards sensitivity" ∂s[1, 1, Nz] ∂s_fd
    @test ∂s[1, 1, Nz] ≈ ∂s_fd rtol = 1.0e-4
end
