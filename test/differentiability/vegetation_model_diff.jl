using Terrarium
using Test

using Enzyme
using Statistics

using Terrarium: compute_λc, compute_respiration_assimilation, compute_GPP, compute_Ra, compute_NPP
using Terrarium: compute_balanced_leaf_area_index, compute_C_veg_tend

# Midday conditions for a closed needleleaf stand
const VEG_DIFF_T_AIR = 20.0 # °C
const VEG_DIFF_T_SOIL = 15.0 # °C
const VEG_DIFF_SWDOWN = 400.0 # W/m²
const VEG_DIFF_PRESSURE = 1.0e5 # Pa
const VEG_DIFF_CO2 = 400.0 # ppm
const VEG_DIFF_VPD = 1000.0 # Pa
const VEG_DIFF_C_VEG = 8.0 # kgC/m²

# GPP, NPP, and the vegetation carbon tendency (all kgC/m²/s) as functions of the process parameters
function midday_carbon_fluxes(photo, stomcond, autoresp, vegcarbon_dynamics, traits, material)
    LAI_b = compute_balanced_leaf_area_index(vegcarbon_dynamics, traits, VEG_DIFF_C_VEG)
    λc = compute_λc(stomcond, VEG_DIFF_VPD)
    Rd, An = compute_respiration_assimilation(
        photo, traits, material, VEG_DIFF_T_AIR, VEG_DIFF_SWDOWN, VEG_DIFF_PRESSURE, VEG_DIFF_CO2, LAI_b, λc, 1.0
    )
    GPP = compute_GPP(photo, An, Rd)
    Ra = compute_Ra(autoresp, vegcarbon_dynamics, traits, VEG_DIFF_T_AIR, VEG_DIFF_T_SOIL, Rd, 1.0, VEG_DIFF_C_VEG, GPP)
    NPP = compute_NPP(autoresp, GPP, Ra)
    C_veg_tendency = compute_C_veg_tend(vegcarbon_dynamics, traits, LAI_b, NPP)
    return GPP, NPP, C_veg_tendency
end

midday_GPP(args...) = midday_carbon_fluxes(args...)[1]
midday_NPP(args...) = midday_carbon_fluxes(args...)[2]
midday_C_veg_tendency(args...) = midday_carbon_fluxes(args...)[3]

function central_difference(f, x; relative_step = 1.0e-5)
    h = relative_step * abs(x)
    return (f(x + h) - f(x - h)) / (2h)
end

@testset "Vegetation carbon fluxes: parameter gradients" begin
    photo = LUEPhotosynthesis(Float64)
    stomcond = MedlynStomatalConductance(Float64)
    autoresp = PALADYNAutotrophicRespiration(Float64)
    vegcarbon_dynamics = PALADYNCarbonDynamics(Float64)
    traits = PlantTraits(Float64)
    material = MaterialConstants(Float64)

    # finite-difference references, reconstructing each process with a perturbed parameter
    flux_with(f; α_C3 = photo.α_C3, g₁ = stomcond.g₁, γL = vegcarbon_dynamics.γL) = f(
        LUEPhotosynthesis(Float64; α_C3), MedlynStomatalConductance(Float64; g₁), autoresp,
        PALADYNCarbonDynamics(Float64; γL), traits, material
    )

    for (name, f) in (("GPP", midday_GPP), ("NPP", midday_NPP), ("C_veg tendency", midday_C_veg_tendency))
        @testset "$name" begin
            grads = Enzyme.autodiff(
                Reverse, f, Active,
                Active(photo), Active(stomcond), Const(autoresp), Active(vegcarbon_dynamics), Const(traits), Const(material)
            )[1]
            ∂α_C3 = grads[1].α_C3
            ∂g₁ = grads[2].g₁
            ∂γL = grads[4].γL
            @test ∂α_C3 ≈ central_difference(x -> flux_with(f; α_C3 = x), photo.α_C3) rtol = 1.0e-5
            @test ∂g₁ ≈ central_difference(x -> flux_with(f; g₁ = x), stomcond.g₁) rtol = 1.0e-5
            @test ∂γL ≈ central_difference(x -> flux_with(f; γL = x), vegcarbon_dynamics.γL) rtol = 1.0e-5 atol = 1.0e-20
            # more quantum efficiency and wider stomata (higher cᵢ) both increase assimilation
            @test ∂α_C3 > 0
            @test ∂g₁ > 0
        end
    end
end

function mean_vegetation_carbon_step!(integrator, Δt)
    timestep!(integrator, Δt)
    return mean(interior(integrator.state.carbon_vegetation))
end

@testset "Vegetation carbon cycle model: timestep!" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 5))
    model = VegetationModel(grid; vegetation = VegetationCarbonCycle(Float64))
    initializers = (carbon_vegetation = 5.0, vegetation_area_fraction = 0.5, growing_degree_days = 300.0)
    integrator = initialize(model; initializers)
    set!(integrator.state.air_temperature, 15.0)
    set!(integrator.state.surface_shortwave_down, 300.0)
    dintegrator = make_zero(integrator)
    Δt = 86400.0
    Enzyme.autodiff(
        set_runtime_activity(Reverse),
        mean_vegetation_carbon_step!,
        Active,
        Duplicated(integrator, dintegrator),
        Const(Δt)
    )
    dC_veg = Array(interior(dintegrator.state.carbon_vegetation))
    @test all(isfinite.(dC_veg))
    # one daily step changes the carbon pool by a tiny fraction, so ∂C_veg(t + Δt)/∂C_veg(t) ≈ 1
    @test all(abs.(dC_veg .- 1) .< 1.0e-3)
    @test all(isfinite.(Array(interior(dintegrator.state.air_temperature))))
end
