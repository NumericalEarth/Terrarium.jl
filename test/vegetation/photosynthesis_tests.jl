using Terrarium
using Terrarium: compute_kinetic_parameters, compute_Γ_star, compute_PAR, compute_APAR, compute_pres_i
using Terrarium: compute_temperature_stress, compute_assimilation_factors, compute_Vc_max, compute_JE_JC
using Terrarium: compute_Rd, compute_Ag, compute_GPP, compute_respiration_assimilation, compute_photosynthesis
using Terrarium: compute_λc
using Test

@testset "Kinetic parameters" begin
    photo = LUEPhotosynthesis()
    # Test kinetic parameters should be positive
    T_air = 20.0 # °C
    τ, Kc, Ko = compute_kinetic_parameters(photo, T_air)
    @test isfinite(τ) && τ > 0
    @test isfinite(Kc) && Kc > 0
    @test isfinite(Ko) && Ko > 0

    # Test temperature dependence, kinetic parameters should increase/decrease with T_air
    T_air_warm = 30.0 # °C
    τ_warm, Kc_warm, Ko_warm = compute_kinetic_parameters(photo, T_air_warm)
    @test τ_warm < τ
    @test Kc_warm > Kc
    @test Ko_warm > Ko
end

@testset "Γ_star" begin
    photo = LUEPhotosynthesis()
    # Test Γ_star should be positive
    τ = 3000.0 # Value for T_air = 20.0 °C
    pres_O2 = 20.9e3 # Pa
    Γ_star = compute_Γ_star(photo, τ, pres_O2)
    @test isfinite(Γ_star) && Γ_star > 0

    # Test τ dependence, Γ_star inv. proportional to τ
    τ_warm = 2000.0 # Value for T_air = 30.0 °C
    Γ_star_warm = compute_Γ_star(photo, τ_warm, pres_O2)
    @test Γ_star_warm > Γ_star
end

@testset "PAR" begin
    photo = LUEPhotosynthesis()
    # Test PAR should be positive for a positive swdown
    swdown = 50.0 # W/m²
    PAR = compute_PAR(photo, swdown)
    @test isfinite(PAR) && PAR > 0

    # Test swdown = 0 (PAR should be 0)
    swdown = 0.0 # W/m²
    PAR = compute_PAR(photo, swdown)
    @test PAR == 0

    # Test linearity, PAR should be linear in swdown
    swdown = 50.0 # W/m²
    PAR_50 = compute_PAR(photo, swdown)
    swdown = 100.0 # W/m²
    PAR_100 = compute_PAR(photo, swdown)
    @test PAR_100 ≈ 2.0 * PAR_50
end

@testset "APAR" begin
    photo = LUEPhotosynthesis()
    traits = PlantTraits()
    swdown = 50.0 # W/m²
    # Test APAR should be positive for a positive LAI
    LAI = 5.0
    APAR = compute_APAR(photo, traits, swdown, LAI)
    @test isfinite(APAR) && APAR > 0.0

    # Test LAI = 0 (APAR should be 0)
    LAI = 0.0
    APAR = compute_APAR(photo, traits, swdown, LAI)
    @test APAR == 0

    # Test LAI = Inf (APAR should be = α_a*PAR)
    LAI = Inf
    APAR = compute_APAR(photo, traits, swdown, LAI)
    @test APAR == photo.α_a * compute_PAR(photo, swdown)
end

@testset "press_i" begin
    photo = LUEPhotosynthesis()
    # Test λc = 0 (pi should be 0)
    λc = 0.0
    pres_a = 40.0 # Pa
    pres_i = compute_pres_i(photo, λc, pres_a)
    @test  pres_i == 0.0

    # Test λc = 1 (pi should be pa)
    λc = 1.0
    pres_i = compute_pres_i(photo, λc, pres_a)
    @test pres_i == pres_a

    # Test pi should be positive and less than pres_a for positive λc
    λc = 0.5
    pres_a = 40.0 # Pa
    pres_i = compute_pres_i(photo, λc, pres_a)
    @test 0 < pres_i < pres_a
end

@testset "Temperature stress" begin
    photo = LUEPhotosynthesis()
    # Test T_air < T_CO2_low (T_stress should be 0)
    T_air = photo.T_CO2_low * 2 # since photo.T_CO2_low < 0
    T_stress = compute_temperature_stress(photo, T_air)
    @test T_stress == 0.0

    # Test T_air = T_CO2_low (T_stress should be 0)
    T_air = photo.T_CO2_low
    T_stress = compute_temperature_stress(photo, T_air)
    @test T_stress == 0.0

    # Test T_air > T_CO2_high (T_stress should be 0)
    T_air = photo.T_CO2_high * 2
    T_stress = compute_temperature_stress(photo, T_air)
    @test T_stress == 0.0

    # Test T_air = T_CO2_high (T_stress should be 0)
    T_air = photo.T_CO2_high
    T_stress = compute_temperature_stress(photo, T_air)
    @test T_stress == 0.0

    # Test T_CO2_low < T_air < T_CO2_high (T_stress should be between 0 and 1)
    T_air = (photo.T_CO2_low + photo.T_CO2_high) / 2
    T_stress = compute_temperature_stress(photo, T_air)
    @test 0.0 < T_stress < 1.0
end

@testset "Assimilation factors c1 and c2" begin
    photo = LUEPhotosynthesis()
    constants = MaterialConstants(Float64)
    T_air = 20.0 # °C
    Γ_star = 3.0 # Value for T_air = 20.0 °C and pres_O2 = 20.9e3 Pa
    Kc = 20.0 # Value for T_air = 20.0 °C
    Ko = 3.0e4 # Value for T_air = 20.0 °C
    pres_O2 = 20.9e3 # Pa

    # Test pi = Γ_star (c1 and c2 should be 0)
    pres_i = Γ_star
    T_stress = compute_temperature_stress(photo, T_air)
    c_1, c_2 = compute_assimilation_factors(photo, constants, Γ_star, T_stress, Kc, Ko, pres_i, pres_O2)
    @test c_1 == 0
    @test c_2 == 0

    # Test pi < Γ_star (c1 and c2 should be negative, c1 can be 0 if T_stress=0)
    # TODO can this happen?
    pres_i = Γ_star / 2.0
    c_1, c_2 = compute_assimilation_factors(photo, constants, Γ_star, T_stress, Kc, Ko, pres_i, pres_O2)
    @test isfinite(c_1) && c_1 <= 0
    @test isfinite(c_2) && c_2 < 0

    # Test pi > Γ_star (c1 and c2 should be positive, c1 can be 0 if T_stress=0)
    pres_i = Γ_star * 2.0
    c_1, c_2 = compute_assimilation_factors(photo, constants, Γ_star, T_stress, Kc, Ko, pres_i, pres_O2)
    @test isfinite(c_1) && c_1 >= 0
    @test isfinite(c_2) && c_2 > 0
end

@testset "Vc_max" begin
    photo = LUEPhotosynthesis()
    c_1 = 6.0e-4 # kgC/mol
    Kc = 20.0 # Value for T_air = 20.0 °C
    Ko = 3.0e4 # Value for T_air = 20.0 °C
    Γ_star = 3.0 # Value for T_air = 20.0 °C and pres_O2 = 20.9e3 Pa
    pres_i = 20.0 # Pa
    pres_O2 = 20.9e3 # Pa

    # Test APAR = 0 (Vc_max should be 0)
    APAR = 0.0
    Vc_max = compute_Vc_max(photo, c_1, APAR, Kc, Ko, Γ_star, pres_i, pres_O2)
    @test Vc_max == 0.0

    # Test Vc_max should be finite
    APAR = 5.0e-4 # mol/m²/s, for swdown ≈ 400 W/m² and LAI = 4
    Vc_max = compute_Vc_max(photo, c_1, APAR, Kc, Ko, Γ_star, pres_i, pres_O2)
    @test isfinite(Vc_max) && Vc_max > 0

    # TODO not clear for pres_i = Γ_star?
    # TODO Vc_max can be negative?
end

@testset "JE and JC" begin
    photo = LUEPhotosynthesis()
    # Test c1 = c2 = 0 (JE and JC should be 0)
    c_1 = 0.0
    c_2 = 0.0
    APAR = 5.0e-4 # mol/m²/s
    Vc_max = 0.0 # since c_1 = 0
    JE, JC = compute_JE_JC(photo, c_1, c_2, APAR, Vc_max)
    @test JE == 0.0
    @test JC == 0.0

    # Test APAR = Vcmax = 0 (JE and JC should be 0)
    c_1 = 6.0e-4 # kgC/mol
    c_2 = 0.3
    APAR = 0.0
    Vc_max = 0.0
    JE, JC = compute_JE_JC(photo, c_1, c_2, APAR, Vc_max)
    @test JE == 0.0
    @test JC == 0.0

    # Test JE, JC should be finite for the other cases
    c_1 = 6.0e-4 # kgC/mol
    c_2 = 0.3
    APAR = 5.0e-4 # mol/m²/s
    Vc_max = 1.0e-6 # kgC/m²/s
    JE, JC = compute_JE_JC(photo, c_1, c_2, APAR, Vc_max)
    @test isfinite(JE)
    @test isfinite(JC)

    # TODO JE, JC can be negative?
end

@testset "Rd" begin
    photo = LUEPhotosynthesis()
    # Test β=0 (Rd should be 0)
    β = 0.0
    Vc_max = 1.0e-6 # kgC/m²/s
    Rd = compute_Rd(photo, Vc_max, β)
    @test Rd == 0.0

    # Test β=1 (Rd should be = leaf_respiration_fraction*Vc_max)
    β = 1.0
    Vc_max = 1.0e-6 # kgC/m²/s
    Rd = compute_Rd(photo, Vc_max, β)
    @test Rd == photo.leaf_respiration_fraction * Vc_max

    # Test Rd should be between 0 and leaf_respiration_fraction*Vc_max for 0<β<1 and Vc_max positive
    β = 0.5
    Vc_max = 1.0e-6 # kgC/m²/s
    Rd = compute_Rd(photo, Vc_max, β)
    @test 0.0 < Rd < photo.leaf_respiration_fraction * Vc_max

    # TODO test for Vc_max<0, Rd can be negative?
end

@testset "Ag" begin
    photo = LUEPhotosynthesis()
    c_1 = 6.0e-4 # kgC/mol
    c_2 = 0.3
    APAR = 5.0e-4 # mol/m²/s
    Vc_max = 1.0e-6 # kgC/m²/s

    # Test with β = 0 (Ag should be 0)
    β = 0.0
    Ag = compute_Ag(photo, c_1, c_2, APAR, Vc_max, β)
    @test Ag == 0

    # Test Ag should be finite for all other cases
    β = 0.5
    Ag = compute_Ag(photo, c_1, c_2, APAR, Vc_max, β)
    @test isfinite(Ag)

    # TODO Ag can be negative?
end

@testset "Photosynthesis (An, Rd, and GPP)" begin
    photo = LUEPhotosynthesis()
    traits = PlantTraits()
    constants = MaterialConstants(Float64)
    swdown = 50.0 # W/m²
    pres = 1.0e5 # Pa
    co2 = 400.0 # ppm
    λc = 0.7 # typical C3 ratio of leaf-internal to ambient CO₂
    β = 1.0 # full soil moisture availability

    # Test T_air < -3 (An and Rd should be 0)
    T_air = -5.0 # °C
    LAI = 5.0 # m²/m²
    Rd, An = compute_respiration_assimilation(photo, traits, constants, T_air, swdown, pres, co2, LAI, λc, β)
    @test An == 0.0
    @test Rd == 0.0

    # Test T_air > -3 and LAI=0 (An and Rd should be 0)
    T_air = 20.0 # °C
    LAI = 0.0
    Rd, An = compute_respiration_assimilation(photo, traits, constants, T_air, swdown, pres, co2, LAI, λc, β)
    @test An == 0.0
    @test Rd == 0.0

    # Test T_air > -3 and LAI > 0 (An and Rd should be finite)
    T_air = 20.0 # °C
    LAI = 5.0 # m²/m²
    Rd, An = compute_respiration_assimilation(photo, traits, constants, T_air, swdown, pres, co2, LAI, λc, β)
    @test isfinite(An)
    @test isfinite(Rd)
    @test Rd > 0
    GPP = compute_GPP(photo, An, Rd)
    @test GPP ≈ An + Rd

    # TODO An, Rd can be negative?
end

@testset "Coordination hypothesis: JE == JC" begin
    photo = LUEPhotosynthesis()
    constants = MaterialConstants(Float64)
    T_air = 20.0 # °C
    τ, Kc, Ko = compute_kinetic_parameters(photo, T_air)
    pres_O2 = 0.209 * 1.0e5 # Pa
    Γ_star = compute_Γ_star(photo, τ, pres_O2)
    pres_i = 28.0 # Pa
    T_stress = compute_temperature_stress(photo, T_air)
    APAR = 5.0e-4 # mol/m²/s
    c_1, c_2 = compute_assimilation_factors(photo, constants, Γ_star, T_stress, Kc, Ko, pres_i, pres_O2)
    # Vc_max is acclimated so that light- and Rubisco-limited rates coincide
    Vc_max = compute_Vc_max(photo, c_1, APAR, Kc, Ko, Γ_star, pres_i, pres_O2)
    JE, JC = compute_JE_JC(photo, c_1, c_2, APAR, Vc_max)
    @test JE ≈ JC
    # Vc_max and leaf respiration scale linearly with APAR
    @test compute_Vc_max(photo, c_1, 2APAR, Kc, Ko, Γ_star, pres_i, pres_O2) ≈ 2Vc_max
    @test compute_Rd(photo, Vc_max, 1.0) / Vc_max ≈ photo.leaf_respiration_fraction
end

@testset "Photosynthesis: plausible midday rates" begin
    photo = LUEPhotosynthesis()
    traits = PlantTraits()
    constants = MaterialConstants(Float64)
    stomcond = MedlynStomatalConductance()
    μmol_CO2 = 0.012 * 1.0e-6 # kgC/m²/s per μmol CO₂/m²/s
    # (air temperature °C, shortwave W/m², LAI, VPD Pa) for a cool cloudy to a warm clear midday
    for (T_air, swdown, LAI, vpd) in ((15.0, 200.0, 3.0, 500.0), (20.0, 400.0, 4.0, 1000.0), (25.0, 800.0, 5.0, 1500.0))
        λc = compute_λc(stomcond, vpd)
        Rd, An = compute_respiration_assimilation(photo, traits, constants, T_air, swdown, 1.0e5, 400.0, LAI, λc, 1.0)
        GPP = compute_GPP(photo, An, Rd)
        # canopy GPP of a C3 forest at midday: roughly 5-30 μmol CO₂/m²/s
        @test 5 * μmol_CO2 < GPP < 30 * μmol_CO2
        # leaf (dark) respiration is a few percent of gross assimilation
        @test 0.03 < Rd / GPP < 0.15
    end
end

@testset "Photosynthesis: dependence on leaf area index" begin
    photo = LUEPhotosynthesis()
    traits = PlantTraits()
    constants = MaterialConstants(Float64)
    λc = compute_λc(MedlynStomatalConductance(), 1000.0)
    function GPP(LAI)
        # 20 °C, 400 W/m², 1000 hPa, 400 ppm, no soil moisture stress
        Rd, An = compute_respiration_assimilation(photo, traits, constants, 20.0, 400.0, 1.0e5, 400.0, LAI, λc, 1.0)
        return compute_GPP(photo, An, Rd)
    end
    LAIs = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    GPPs = GPP.(LAIs)
    # no leaves, no photosynthesis; GPP increases and saturates with LAI
    @test GPPs[1] == 0
    @test all(>(0), GPPs[2:end])
    @test issorted(GPPs)
    slopes = diff(collect(GPPs)) ./ diff(collect(LAIs))
    @test issorted(slopes; rev = true)
    # With the acclimated Vc_max, all rates scale linearly with APAR, so GPP follows the fraction of absorbed
    # light, 1 - exp(-k LAI) (Beer's law)
    k = traits.extinction_coefficient
    for LAI in LAIs[2:end]
        @test GPP(LAI) / GPP(4.0) ≈ (1 - exp(-k * LAI)) / (1 - exp(-k * 4.0))
    end
end
