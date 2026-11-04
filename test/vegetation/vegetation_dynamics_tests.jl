using Terrarium
using Terrarium: compute_γv, compute_ν_star, compute_ν_tendency, compute_λ_NPP, seconds_per_year
using Test

@testset "compute_γv test" begin
    veg_dynamics = PALADYNVegetationDynamics()
    # For now, γv = γv_min, converted from 1/yr to 1/s
    γv = compute_γv(veg_dynamics)
    @test γv == veg_dynamics.γv_min / seconds_per_year(Float64)
end

@testset "compute_ν_star test" begin
    veg_dynamics = PALADYNVegetationDynamics()
    # Test ν < ν_seed (ν_star should be equal to ν_seed)
    ν = veg_dynamics.ν_seed / 2
    ν_star = compute_ν_star(veg_dynamics, ν)
    @test ν_star == veg_dynamics.ν_seed

    # Test ν > ν_seed (ν_star should be equal to ν)
    ν = veg_dynamics.ν_seed * 2
    ν_star = compute_ν_star(veg_dynamics, ν)
    @test ν_star == ν

    # Test ν == ν_seed (ν_star should be equal to ν_seed)
    ν = veg_dynamics.ν_seed
    ν_star = compute_ν_star(veg_dynamics, ν)
    @test ν_star == veg_dynamics.ν_seed
end

@testset "compute_ν_tendency test" begin
    veg_dynamics = PALADYNVegetationDynamics()
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    LAI_b = (traits.minimum_leaf_area_index + traits.maximum_leaf_area_index) / 2.0
    C_veg = 10.0 # kgC/m²
    ν = 0.3
    NPP = 3.0e-8 # kgC/m²/s, about 1 kgC/m²/yr
    ν_tendency = compute_ν_tendency(veg_dynamics, vegcarbon_dynamics, traits, LAI_b, C_veg, NPP, ν)
    # Willeit & Ganopolski (2016), Eq. 73
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)
    @test ν_tendency ≈ λ_NPP * NPP / C_veg * ν * (1 - ν) - compute_γv(veg_dynamics) * ν
    # without productivity, the vegetated fraction decays at the disturbance rate
    @test compute_ν_tendency(veg_dynamics, vegcarbon_dynamics, traits, LAI_b, C_veg, 0.0, ν) ≈ -compute_γv(veg_dynamics) * ν
end
