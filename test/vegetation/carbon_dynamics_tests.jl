using Terrarium
using Terrarium: compute_λ_NPP, compute_balanced_leaf_area_index, compute_Λ_loc, compute_C_veg_tend
using Terrarium: compute_leaf_carbon, compute_root_carbon, compute_stem_carbon
using Test

@testset "λ_NPP test" begin
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    # Test LAI_b < minimum_leaf_area_index (λ_NPP should be 0)
    LAI_b = traits.minimum_leaf_area_index / 2.0
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)
    @test λ_NPP == 0.0

    # Test LAI_b == minimum_leaf_area_index (λ_NPP should be 0)
    LAI_b = traits.minimum_leaf_area_index
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)
    @test λ_NPP == 0.0

    # Test minimum_leaf_area_index < LAI_b < maximum_leaf_area_index (λ_NPP should be between 0 and 1)
    LAI_b = (traits.minimum_leaf_area_index + traits.maximum_leaf_area_index) / 2.0
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)
    @test 0 < λ_NPP < 1

    # Test LAI_b == maximum_leaf_area_index (λ_NPP should be 1)
    LAI_b = traits.maximum_leaf_area_index
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)
    @test λ_NPP == 1.0

    # Test LAI_b > maximum_leaf_area_index (λ_NPP should be 1)
    LAI_b = traits.maximum_leaf_area_index * 2.0
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)
    @test λ_NPP == 1.0
end

@testset "LAI_b test" begin
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    # Test LAI_b should be positive for a positive C_veg
    C_veg = 5.0 # kgC/m²
    LAI_b = compute_balanced_leaf_area_index(vegcarbon_dynamics, traits, C_veg)
    @test isfinite(LAI_b) && LAI_b > 0.0
    # and proportional to it
    @test compute_balanced_leaf_area_index(vegcarbon_dynamics, traits, 2C_veg) ≈ 2LAI_b
    # a closed needleleaf stand of 10-15 kgC/m² has a balanced LAI of roughly 3-7
    @test 3 < compute_balanced_leaf_area_index(vegcarbon_dynamics, traits, 10.0) < 7
end

@testset "Tissue carbon partitioning" begin
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    for C_veg in (0.1, 1.0, 5.0, 15.0) # kgC/m²
        LAI_b = compute_balanced_leaf_area_index(vegcarbon_dynamics, traits, C_veg)
        C_leaf = compute_leaf_carbon(vegcarbon_dynamics, traits, LAI_b)
        C_root = compute_root_carbon(vegcarbon_dynamics, traits, LAI_b)
        C_stem = compute_stem_carbon(vegcarbon_dynamics, traits, LAI_b)
        # the diagnosed tissue pools partition the single vegetation carbon pool
        @test C_leaf + C_root + C_stem ≈ C_veg
        @test C_root == C_leaf
        @test C_leaf * traits.specific_leaf_area ≈ LAI_b
    end
end

@testset "Λ_loc test" begin
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    # Test Λ_loc should be positive for a positive LAI_b
    LAI_b = (traits.minimum_leaf_area_index + traits.maximum_leaf_area_index) / 2.0
    Λ_loc = compute_Λ_loc(vegcarbon_dynamics, traits, LAI_b)
    @test isfinite(Λ_loc) && Λ_loc > 0.0
    # litterfall is proportional to the balanced leaf area index
    @test compute_Λ_loc(vegcarbon_dynamics, traits, 2LAI_b) ≈ 2Λ_loc
    # the implied turnover time of the vegetation carbon pool is decades for a needleleaf forest
    C_veg = LAI_b * (2 / traits.specific_leaf_area + traits.awl)
    turnover_time_years = C_veg / Λ_loc / (365.25 * 86400)
    @test 5 < turnover_time_years < 50
end

@testset "C_veg tendency test" begin
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    LAI_b = (traits.minimum_leaf_area_index + traits.maximum_leaf_area_index) / 2.0
    NPP = 3.0e-8 # kgC/m²/s, about 1 kgC/m²/yr
    C_veg_tendency = compute_C_veg_tend(vegcarbon_dynamics, traits, LAI_b, NPP)
    # Willeit & Ganopolski (2016), Eq. 72
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)
    @test C_veg_tendency ≈ (1 - λ_NPP) * NPP - compute_Λ_loc(vegcarbon_dynamics, traits, LAI_b)
end
