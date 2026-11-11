using Terrarium
using Terrarium: compute_f_temp, compute_resp10, compute_Rm, compute_Rg, compute_Ra, compute_NPP
using Test

# Realistic inputs for the default (needleleaf) parameters; all carbon fluxes in kgC/m²/s.
# 1.0e-7 kgC/m²/s ≈ 8.3 μmol CO₂/m²/s, a typical midday GPP.
const GPP_MIDDAY = 1.0e-7 # kgC/m²/s
const RD_MIDDAY = 6.0e-9 # kgC/m²/s, about 6 % of GPP
const μMOL_CO2 = 0.012 * 1.0e-6 # kgC/m²/s per μmol CO₂/m²/s

@testset "f_temp test" begin
    autoresp = PALADYNAutotrophicRespiration()
    # Lloyd & Taylor (1994) factor is one at the reference temperature of 10 °C
    f_temp_air, f_temp_soil = compute_f_temp(autoresp, 10.0, 10.0)
    @test f_temp_air ≈ 1
    @test f_temp_air == f_temp_soil
    # and increases monotonically with temperature
    f_temps = [first(compute_f_temp(autoresp, T, T)) for T in -10.0:5.0:30.0]
    @test issorted(f_temps)
    @test all(f -> isfinite(f) && f > 0, f_temps)
end

@testset "resp10 test" begin
    autoresp = PALADYNAutotrophicRespiration()
    # For now, a placeholder returning the PALADYN tissue respiration rate at 10 °C (kgC/kgN/day)
    @test compute_resp10(autoresp) == 0.066
end

@testset "Rm test" begin
    autoresp = PALADYNAutotrophicRespiration()
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    T_air = 15.0 # °C
    T_soil = 10.0 # °C
    phen = 1.0

    # Stem and root respiration scale linearly with the vegetation carbon pool
    Rm_5 = compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, 0.0, phen, 5.0)
    Rm_10 = compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, 0.0, phen, 10.0)
    @test Rm_10 ≈ 2 * Rm_5
    @test compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, 0.0, phen, 0.0) == 0

    # Root respiration scales with the phenology factor, stem respiration does not
    Rm_dormant = compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, 0.0, 0.0, 5.0)
    @test 0 < Rm_dormant < Rm_5

    # Leaf respiration enters additively
    @test compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, RD_MIDDAY, phen, 5.0) ≈ Rm_5 + RD_MIDDAY

    # Plausibility: stem and root maintenance respiration of a forest stand with 1-15 kgC/m² is
    # a small fraction of midday GPP (0.1-2 μmol CO₂/m²/s, i.e. 0.05-0.75 kgC/m²/yr)
    for C_veg in (1.0, 5.0, 10.0, 15.0)
        Rm = compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, 0.0, phen, C_veg)
        @test 0.05 * μMOL_CO2 < Rm < 2.5 * μMOL_CO2
        @test Rm < 0.3 * GPP_MIDDAY
    end
end

@testset "Rg test" begin
    autoresp = PALADYNAutotrophicRespiration()
    # Growth respiration is a quarter of the carbon left after maintenance respiration
    Rm = 3.0e-8 # kgC/m²/s
    @test compute_Rg(autoresp, GPP_MIDDAY, Rm) ≈ 0.25 * (GPP_MIDDAY - Rm)
end

@testset "Ra test" begin
    autoresp = PALADYNAutotrophicRespiration()
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    T_air = 20.0 # °C
    T_soil = 15.0 # °C
    phen = 1.0
    C_veg = 5.0 # kgC/m²
    Rm = compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, RD_MIDDAY, phen, C_veg)
    Rg = compute_Rg(autoresp, GPP_MIDDAY, Rm)
    Ra = compute_Ra(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, RD_MIDDAY, phen, C_veg, GPP_MIDDAY)
    @test Ra ≈ Rm + Rg
    # At midday, autotrophic respiration is a fraction of GPP
    @test 0.2 * GPP_MIDDAY < Ra < 0.6 * GPP_MIDDAY
end

@testset "NPP test" begin
    autoresp = PALADYNAutotrophicRespiration()
    Ra = 4.0e-8 # kgC/m²/s
    @test compute_NPP(autoresp, GPP_MIDDAY, Ra) == GPP_MIDDAY - Ra
end
