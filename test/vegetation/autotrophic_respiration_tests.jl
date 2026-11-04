using Terrarium
using Terrarium: compute_f_temp, compute_Rm, compute_Rg, compute_Ra, compute_NPP
using Test

@testset "f_temp test" begin
    autoresp = PALADYNAutotrophicRespiration()
    # Test that f_temp_soil > 0 and f_temp_air > 0
    T_air = 10.0 # °C
    T_soil = 5.0 # °C
    f_temp_air, f_temp_soil = compute_f_temp(autoresp, T_air, T_soil)
    @test isfinite(f_temp_air) && f_temp_air > 0.0
    @test isfinite(f_temp_soil) && f_temp_soil > 0.0
    # Test that f_temp_soil > 0 and f_temp_air > 0
    T_air = 15.0 # °C
    T_soil = 10.0 # °C
    f_temp_air, f_temp_soil = compute_f_temp(autoresp, T_air, T_soil)
    @test isfinite(f_temp_air) && f_temp_air > 0.0
    @test isfinite(f_temp_soil) && f_temp_soil > 0.0
    # Test that f_temp_soil == f_temp_air when soil and air temperatures match
    T_air = 10.0 # °C
    T_soil = 10.0 # °C
    f_temp_air, f_temp_soil = compute_f_temp(autoresp, T_air, T_soil)
    @test f_temp_air == f_temp_soil
end

@testset "resp10 test" begin
    autoresp = PALADYNAutotrophicRespiration()
    # PALADYN tissue respiration rate at 10 °C (kgC/kgN/day)
    @test autoresp.resp10 == 0.066
end

@testset "Rm test" begin
    autoresp = PALADYNAutotrophicRespiration()
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    # Test Rm should be positive for a positive C_veg and Rd
    T_air = 20.0 # °C
    T_soil = 15.0 # °C
    Rd = 0.2 # Mock value
    phen = 1.0
    C_veg = 0.5 # Mock value
    Rm = compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, Rd, phen, C_veg)
    @test isfinite(Rm) && Rm > 0.0
end

@testset "Rg test" begin
    autoresp = PALADYNAutotrophicRespiration()
    # Test Rg should be finite
    GPP = 0.5 # Mock value
    Rm = 0.2 # Mock value
    Rg = compute_Rg(autoresp, GPP, Rm)
    @test isfinite(Rg)

    # TODO Rg could be negative?
end

@testset "Ra test" begin
    autoresp = PALADYNAutotrophicRespiration()
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    # Test Ra should be finite
    T_air = 20.0 # °C
    T_soil = 15.0 # °C
    Rd = 0.2 # Mock value
    phen = 1.0
    C_veg = 0.5 # Mock value
    GPP = 0.5 # Mock value
    Ra = compute_Ra(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, Rd, phen, C_veg, GPP)
    @test isfinite(Ra)

    # TODO Ra could be negative?
end

@testset "NPP test" begin
    autoresp = PALADYNAutotrophicRespiration()
    # Test NPP should be the difference between GPP and Ra
    GPP = 0.5 # Mock value
    Ra = 0.3 # Mock value
    NPP = compute_NPP(autoresp, GPP, Ra)
    @test NPP == GPP - Ra

    # TODO is NPP always positive ? GPP always > Ra?
end
