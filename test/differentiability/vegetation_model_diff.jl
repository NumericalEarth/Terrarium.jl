using Terrarium
using Test

using Enzyme
using Statistics

using Terrarium: compute_λc, compute_respiration_assimilation, compute_GPP

# Differentiability of the prescribed vegetation setup, in which the leaf area index is an input that drives
# photosynthesis and stomatal conductance.

# Midday conditions
const VEG_DIFF_T_AIR = 20.0 # °C
const VEG_DIFF_SWDOWN = 400.0 # W/m²
const VEG_DIFF_PRESSURE = 1.0e5 # Pa
const VEG_DIFF_CO2 = 400.0 # ppm
const VEG_DIFF_VPD = 1000.0 # Pa

# GPP (kgC/m²/s) as a function of the photosynthesis and stomatal conductance parameters and the leaf area index
function midday_GPP(photo, stomcond, traits, material, LAI)
    λc = compute_λc(stomcond, VEG_DIFF_VPD)
    Rd, An = compute_respiration_assimilation(
        photo, traits, material, VEG_DIFF_T_AIR, VEG_DIFF_SWDOWN, VEG_DIFF_PRESSURE, VEG_DIFF_CO2, LAI, λc, 1.0
    )
    return compute_GPP(photo, An, Rd)
end

function central_difference(f, x; relative_step = 1.0e-5)
    h = relative_step * abs(x)
    return (f(x + h) - f(x - h)) / (2h)
end

@testset "GPP: gradients w.r.t. parameters and leaf area index" begin
    photo = LUEPhotosynthesis(Float64)
    stomcond = MedlynStomatalConductance(Float64)
    traits = PlantTraits(Float64)
    material = MaterialConstants(Float64)
    LAI = 3.0
    grads = Enzyme.autodiff(
        Reverse, midday_GPP, Active,
        Active(photo), Active(stomcond), Const(traits), Const(material), Active(LAI)
    )[1]
    ∂α_C3 = grads[1].α_C3
    ∂g₁ = grads[2].g₁
    ∂LAI = grads[5]
    # finite-difference references, reconstructing each process with a perturbed parameter
    @test ∂α_C3 ≈ central_difference(x -> midday_GPP(LUEPhotosynthesis(Float64; α_C3 = x), stomcond, traits, material, LAI), photo.α_C3) rtol = 1.0e-5
    @test ∂g₁ ≈ central_difference(x -> midday_GPP(photo, MedlynStomatalConductance(Float64; g₁ = x), traits, material, LAI), stomcond.g₁) rtol = 1.0e-5
    @test ∂LAI ≈ central_difference(x -> midday_GPP(photo, stomcond, traits, material, x), LAI) rtol = 1.0e-5
    # more quantum efficiency, wider stomata (higher cᵢ), and more leaves all increase assimilation
    @test ∂α_C3 > 0
    @test ∂g₁ > 0
    @test ∂LAI > 0
end

function build_prescribed_vegetation_integrator(LAI)
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 5))
    model = VegetationModel(grid; vegetation = PrescribedVegetation(Float64))
    integrator = initialize(model)
    set!(integrator.state.air_temperature, VEG_DIFF_T_AIR)
    set!(integrator.state.surface_shortwave_down, VEG_DIFF_SWDOWN)
    set!(integrator.state.leaf_area_index, LAI)
    return integrator
end

function mean_GPP_step!(integrator, Δt)
    timestep!(integrator, Δt)
    return mean(interior(integrator.state.gross_primary_production))
end

@testset "Prescribed vegetation model: timestep! adjoint w.r.t. leaf area index" begin
    LAI = 3.0
    Δt = 3600.0
    integrator = build_prescribed_vegetation_integrator(LAI)
    dintegrator = make_zero(integrator)
    Enzyme.autodiff(
        set_runtime_activity(Reverse),
        mean_GPP_step!,
        Active,
        Duplicated(integrator, dintegrator),
        Const(Δt)
    )
    dLAI = only(Array(interior(dintegrator.state.leaf_area_index)))
    @test isfinite(dLAI)
    @test dLAI > 0
    # finite-difference reference from two fresh integrators with a perturbed prescribed LAI
    GPP_after_step(x) = mean_GPP_step!(build_prescribed_vegetation_integrator(x), Δt)
    @test dLAI ≈ central_difference(GPP_after_step, LAI) rtol = 1.0e-4
end
