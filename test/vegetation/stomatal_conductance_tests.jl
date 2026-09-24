using Terrarium
using Terrarium: compute_stomatal_conductance, compute_λc, celsius_to_kelvin, ppm_to_mole_fraction
using Test

# ── λc (leaf-internal / air CO₂ ratio) tests ────────────────────────────────

@testset "λc sanity checks" begin
    stomcond = MedlynStomatalConductance()

    # VPD = 0 → λc ≈ 1 (no drawdown)
    vpd = 0.0
    λc = compute_λc(stomcond, vpd)
    @test λc ≈ 1.0

    # Realistic VPD: λc ∈ (0, 1)
    vpd = 1000.0 # Pa
    λc = compute_λc(stomcond, vpd)
    @test 0.0 < λc < 1.0
end

@testset "λc monotonicity with VPD" begin
    stomcond = MedlynStomatalConductance()
    vpds = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    prev_λc = Inf
    for vpd in vpds
        λc = compute_λc(stomcond, vpd)
        @test λc ≤ prev_λc + eps(Float64) * 10 # non-increasing (allow tiny tolerance)
        prev_λc = λc
    end
end

@testset "λc sensitivity to g₁" begin
    vpd = 500.0 # Pa
    g₁_low = MedlynStomatalConductance(g₁ = 1.0)
    g₁_high = MedlynStomatalConductance(g₁ = 4.0)
    λc_low = compute_λc(g₁_low, vpd)
    λc_high = compute_λc(g₁_high, vpd)
    # Larger g₁ → larger denominator in 1/(1+g₁/√vpd) → smaller subtraction → larger λc
    @test λc_high > λc_low
end

# ── Stomatal conductance tests ─────────────────────────────────────────────

@testset "compute_stomatal_conductance sanity checks" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    # Typical midday conditions (leaf-level rates)
    vpd = 800.0      # Pa
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    An = 1.0e-4      # gC m⁻² s⁻¹ (≈ 25 μmol CO₂/m²/s, typical midday C3 leaf rate)
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    β = 1.0          # no soil moisture stress
    g_stm = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, LAI, β)
    @test isfinite(g_stm) && g_stm > 0
end

@testset "compute_stomatal_conductance: LAI dependence" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    vpd = 800.0      # Pa
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    An = 1.0e-4      # gC m⁻² s⁻¹
    co2 = 415.0      # ppm
    β = 1.0
    g_low = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, 0.5, β)
    g_high = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, 5.0, β)
    # Higher LAI → larger minimum-conductance contribution (light extinction term)
    @test g_high > g_low
end

@testset "compute_stomatal_conductance: soil moisture stress" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    vpd = 800.0      # Pa
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    An = 1.0e-4      # gC m⁻² s⁻¹
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    g_no_stress = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, LAI, 1.0)
    g_stressed = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, LAI, 0.3)
    @test g_stressed < g_no_stress
end

@testset "compute_stomatal_conductance: VPD response" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    An = 1.0e-4      # gC m⁻² s⁻¹
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    β = 1.0
    g_low_vpd = compute_stomatal_conductance(stomcond, traits, constants, 200.0, T_air, pres, co2, An, LAI, β)
    g_high_vpd = compute_stomatal_conductance(stomcond, traits, constants, 2000.0, T_air, pres, co2, An, LAI, β)
    # Higher VPD → smaller conductance (1/√VPD term)
    @test g_low_vpd > g_high_vpd
end

@testset "compute_stomatal_conductance: assimilation scaling" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    vpd = 800.0      # Pa
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    β = 1.0
    g_low_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, 5.0e-5, LAI, β)
    g_high_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, 4.0e-4, LAI, β)
    # Higher assimilation → higher conductance (quasi-linear relationship)
    @test g_high_A > g_low_A
end

@testset "compute_stomatal_conductance: CO₂ dependence" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    vpd = 800.0      # Pa
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    An = 1.0e-4      # gC m⁻² s⁻¹
    LAI = 3.0        # m²/m²
    β = 1.0
    g_low_co2 = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, 350.0, An, LAI, β)
    g_high_co2 = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, 500.0, An, LAI, β)
    # Higher CO₂ → lower conductance (An/CO₂ ratio decreases)
    @test g_low_co2 > g_high_co2
end

@testset "compute_stomatal_conductance: zero assimilation" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    vpd = 800.0      # Pa
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    β = 1.0
    g_zero = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, 0.0, LAI, β)
    # With An = 0 only the minimum-conductance term remains (positive)
    @test isfinite(g_zero) && g_zero > 0
end

@testset "compute_stomatal_conductance: minimum conductance contribution" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    vpd = 800.0      # Pa
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    An = 0.0         # net assimilation
    β = 1.0
    # g_min = 0.5 mm/s → 0.5e-3 m/s; with LAI = 3 the extinction term is (1 - exp(-k*LAI))
    g_zero_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, 0.0, LAI, β)
    # The minimum-conductance term should be positive and finite
    @test isfinite(g_zero_A) && g_zero_A > 0
    # It should scale with g_min
    stomcond_high_gmin = MedlynStomatalConductance(g_min = 2.0)
    g_high_gmin = compute_stomatal_conductance(stomcond_high_gmin, traits, constants, vpd, T_air, pres, co2, An, LAI, β)
    @test g_high_gmin > g_zero_A
end

@testset "compute_stomatal_conductance: physically realistic magnitude" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    vpd = 800.0      # Pa
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    An = 1.0e-4      # gC m⁻² s⁻¹ (typical midday C3 rate)
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    β = 1.0
    g_stm = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, LAI, β)
    # Canopy-level conductance can exceed leaf-level values; ~0.001–0.2 m/s is reasonable
    @test 1.0e-4 < g_stm < 0.2
end

# ── Unit convention for g₁ ─────────────────────────────────────────────────

@testset "g₁ and VPD share the √kPa convention" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    g₁ = stomcond.g₁
    D = stomcond.diffusivity_ratio_water_co2

    # VPD at which the humidity term g₁/√VPD equals one, i.e. VPD = g₁² kPa. Both
    # `compute_stomatal_conductance` and `compute_λc` must place it here; reading VPD
    # as Pa in one and kPa in the other displaces it by a factor of 1000.
    vpd = g₁^2 * 1.0e3 # Pa

    # λc = (g₁/√VPD)/(1 + g₁/√VPD) = 1/2 at this VPD
    @test compute_λc(stomcond, vpd) ≈ 1 // 2

    # and the Medlyn factor b = D(1 + g₁/√VPD) = 2D, isolated by differencing out the
    # minimum-conductance term g₀ (which does not depend on An)
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    An = 1.0e-4      # gC m⁻² s⁻¹
    β = 1.0
    g_zero_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, 0.0, LAI, β)
    g_with_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, LAI, β)

    M_C = constants.material.atomic_weight_carbon
    M_air = constants.material.molecular_weight_dry_air / 1.0e3
    R = constants.thermodynamics.gas_constant_dry_air * M_air
    F = R * celsius_to_kelvin(constants.thermodynamics, T_air) / pres
    cₐ = ppm_to_mole_fraction(co2)
    @test g_with_A - g_zero_A ≈ 2D * An / cₐ / M_C * F
end

@testset "midday conductance is physically plausible" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    # Typical midday conditions; VPD = 0.8 kPa, An ≈ 8 μmol CO₂ m⁻² s⁻¹.
    g_stm = compute_stomatal_conductance(stomcond, traits, constants, 800.0, 20.0, 101325.0, 415.0, 1.0e-4, 3.0, 1.0)
    # Bracketed tightly enough to catch a Pa/kPa mix-up in the g₁/√VPD term, which
    # would put this near 1.2 mm/s instead.
    @test 2.5e-3 < g_stm < 4.0e-3
end

@testset "gₛ and λc obey Fick's law" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    D = stomcond.diffusivity_ratio_water_co2

    # `gₛ` and `λc` are two faces of one closure and agree only if the water vapor
    # conductance they jointly imply satisfies Fick's law, gᴴ²ᴼ = D gᶜᴼ² = D Aₙ/(cₐ - cᵢ).
    # Writing gₛ - g₀ = b Aₙ/cₐ and cᵢ = λc cₐ, this reduces to b (1 - λc) = D when g₀ = 0.
    T_air = 20.0     # °C
    pres = 101325.0  # Pa
    co2 = 415.0      # ppm
    LAI = 3.0        # m²/m²
    An = 1.0e-4      # gC m⁻² s⁻¹
    β = 1.0

    M_C = constants.material.atomic_weight_carbon
    M_air = constants.material.molecular_weight_dry_air / 1.0e3
    R = constants.thermodynamics.gas_constant_dry_air * M_air
    F = R * celsius_to_kelvin(constants.thermodynamics, T_air) / pres
    cₐ = ppm_to_mole_fraction(co2)

    for vpd in (200.0, 800.0, 2000.0, 5000.0)
        g_zero_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, 0.0, LAI, β)
        g_with_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, LAI, β)
        λc = compute_λc(stomcond, vpd)
        # Fickian water vapor conductance (m/s) implied by the CO₂ flux and that cᵢ
        fickian = D * (An / M_C) / (cₐ * (1 - λc)) * F
        @test g_with_A - g_zero_A ≈ fickian
    end
end
