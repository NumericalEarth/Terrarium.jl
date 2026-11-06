using Terrarium
using Terrarium: Variables, varunits
using Terrarium: compute_kinetic_parameters, compute_Γ_star, compute_PAR, compute_APAR, compute_pres_i
using Terrarium: compute_temperature_stress, compute_assimilation_factors, compute_Vc_max, compute_JE_JC
using Terrarium: compute_Rd, compute_Ag, compute_GPP, compute_respiration_assimilation
using Terrarium: compute_stomatal_conductance, compute_λc
using Terrarium: compute_f_temp, compute_Rm, compute_Rg, compute_Ra, compute_NPP
using Terrarium: compute_λ_NPP, compute_balanced_leaf_area_index, compute_leaf_carbon, compute_root_carbon, compute_stem_carbon
using Terrarium: compute_Λ_loc, compute_C_veg_tend, compute_γv, compute_ν_star, compute_ν_tendency
using Terrarium: compute_phenology_factor, compute_leaf_area_index, compute_gdd_tendency
using Unitful: Unitful, dimension
using Test

# Each kernel function that contains a unit conversion is compared with a reference written in
# Unitful quantities. Where PALADYN has a counterpart, the reference is transcribed from the
# CLIMBER-X/PALADYN source (src/lnd/photosynthesis.f90 and dyn_veg.f90), which uses per-day rates;
# the conversion to the per-second rates of Terrarium is then done by Unitful, not by hand.

@testset "Unitful reference: PAR" begin
    photo = LUEPhotosynthesis()
    swdown = 400.0 # W/m²
    # half of the shortwave radiation is PAR; 4.6 μmol photons per J at 550 nm
    PAR_ref = 0.5 * swdown * u"W/m^2" * (1 - photo.α_leaf) * 4.6u"μmol/J"
    @test compute_PAR(photo, swdown) ≈ ustrip(u"mol/m^2/s", PAR_ref)
end

@testset "Unitful reference: light-limited assimilation in kgC" begin
    photo = LUEPhotosynthesis()
    constants = MaterialConstants(Float64)
    T_air = 20.0 # °C
    τ, Kc, Ko = compute_kinetic_parameters(photo, T_air)
    pres_O2 = 0.209 * 1.0e5 # Pa
    Γ_star = compute_Γ_star(photo, τ, pres_O2)
    pres_i = 28.0 # Pa
    T_stress = compute_temperature_stress(photo, T_air)
    APAR = 5.0e-4 # mol/m²/s
    c_1, _ = compute_assimilation_factors(photo, constants, Γ_star, T_stress, Kc, Ko, pres_i, pres_O2)
    # PALADYN: c_1 = alpha_c3 * tstress * cmass * (pi - gamma_star) / (pi + 2 gamma_star), cmass = 12 gC/mol
    c_1_ref = photo.α_C3 * T_stress * 12u"g/mol" * (pres_i - Γ_star) / (pres_i + 2Γ_star)
    @test c_1 ≈ ustrip(u"kg/mol", c_1_ref)
    # JE = c_1 * APAR (Haxeltine & Prentice, 1996, Eq. 3)
    JE, _ = compute_JE_JC(photo, c_1, 0.0, APAR, 0.0)
    @test JE ≈ ustrip(u"kg/m^2/s", c_1_ref * APAR * u"mol/m^2/s")
end

@testset "Unitful reference: Medlyn conductance from assimilation" begin
    stomcond = MedlynStomatalConductance()
    traits = PlantTraits()
    constants = PhysicalConstants()
    vpd = 800.0 # Pa
    T_air = 20.0 # °C
    pres = 101325.0 # Pa
    co2 = 415.0 # ppm
    LAI = 3.0
    An = 1.0e-7 # kgC/m²/s
    g_zero_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, 0.0, LAI, 1.0)
    g_with_A = compute_stomatal_conductance(stomcond, traits, constants, vpd, T_air, pres, co2, An, LAI, 1.0)
    # gₛ - g₀ = D (1 + g₁/√VPD) Aₙ/cₐ, converted from a molar to a volumetric conductance with the ideal gas law
    b = stomcond.diffusivity_ratio_water_co2 * (1 + stomcond.g₁ / sqrt(vpd / 1000))
    molar_flux = An * u"kg/m^2/s" / 12u"g/mol"
    molar_volume = 8.314462618u"J/(mol*K)" * (T_air + 273.15) * u"K" / (pres * u"Pa")
    g_ref = b * molar_flux / (co2 * 1.0e-6) * molar_volume
    @test g_with_A - g_zero_A ≈ ustrip(u"m/s", g_ref) rtol = 1.0e-4
end

@testset "Unitful reference: maintenance respiration (PALADYN)" begin
    autoresp = PALADYNAutotrophicRespiration()
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    traits = PlantTraits()
    T_air = 15.0 # °C
    T_soil = 10.0 # °C
    phen = 0.8
    C_veg = 8.0 # kgC/m²
    f_temp_air, f_temp_soil = compute_f_temp(autoresp, T_air, T_soil)
    # PALADYN allometry (dyn_veg.f90): leaf_c = lai_bal/sla, root_c = leaf_c, stem_c = awl*lai_bal
    LAI_b = C_veg / (2 / traits.specific_leaf_area + traits.awl)
    root_c = LAI_b / traits.specific_leaf_area * u"kg/m^2"
    stem_c = traits.awl * LAI_b * u"kg/m^2"
    # PALADYN (photosynthesis.f90), in kgC/m²/day:
    # resp_stem = resp10*stem_c/aws/cn_sapwood*ftemp_air, resp_root = resp10*root_c/cn_root*ftemp_soil*phen
    resp10 = 0.066u"d^-1"
    resp_stem = resp10 * stem_c / autoresp.aws / autoresp.cn_sapwood * f_temp_air
    resp_root = resp10 * root_c / autoresp.cn_root * f_temp_soil * phen
    Rm = compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, 0.0, phen, C_veg)
    @test Rm ≈ ustrip(u"kg/m^2/s", resp_stem + resp_root)
    # leaf respiration enters additively and is already in kgC/m²/s
    Rd = 3.0e-9 # kgC/m²/s
    @test compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, Rd, phen, C_veg) - Rm ≈ Rd
end

@testset "Unitful reference: turnover and disturbance rates" begin
    vegcarbon_dynamics = PALADYNCarbonDynamics()
    veg_dynamics = PALADYNVegetationDynamics()
    traits = PlantTraits()
    LAI_b = 4.0
    leaf_c = LAI_b / traits.specific_leaf_area * u"kg/m^2"
    stem_c = traits.awl * LAI_b * u"kg/m^2"
    # Willeit & Ganopolski (2016), Eq. 75, with turnover rates in 1/yr
    Λ_ref = vegcarbon_dynamics.γL * u"yr^-1" * leaf_c +
        vegcarbon_dynamics.γR * u"yr^-1" * leaf_c +
        vegcarbon_dynamics.γS * u"yr^-1" * stem_c
    @test compute_Λ_loc(vegcarbon_dynamics, traits, LAI_b) ≈ ustrip(u"kg/m^2/s", Λ_ref)
    @test compute_γv(veg_dynamics) ≈ ustrip(u"s^-1", veg_dynamics.γv_min * u"yr^-1")
end

@testset "Unitful reference: growing degree days" begin
    phenol = PALADYNPhenology{Float64}()
    # 20 °C is 15 K above the base temperature, accumulated per day; gdd is in K·d
    @test compute_gdd_tendency(phenol, 0.0, 20.0) ≈ ustrip(u"K*d/s", 15u"K" * 1u"d" / 1u"d")
    # below the base temperature, gdd relaxes with the relaxation timescale (in days)
    gdd = 300.0
    relaxation_ref = gdd * u"K*d" / (phenol.gdd_relaxation_time * u"d")
    @test compute_gdd_tendency(phenol, gdd, 2.0) ≈ -ustrip(u"K*d/s", relaxation_ref)
end

@testset "Declared units of carbon variables are kgC" begin
    areal_mass_flux = dimension(u"kg/m^2/s")
    areal_mass = dimension(u"kg/m^2")
    for vegetation in (VegetationCarbonCycle(Float64), PrescribedVegetation(Float64))
        vars = Variables(vegetation)
        for group in (vars.prognostic, vars.auxiliary, vars.inputs), var in values(group)
            units = varunits(var)
            if dimension(units) == areal_mass_flux
                @test units == u"kg/m^2/s"
            elseif dimension(units) == areal_mass
                @test units == u"kg/m^2"
            end
        end
    end
    @test MaterialConstants(Float64).atomic_weight_carbon == 0.012 # kgC/mol
end

@testset "Kernel functions preserve Float32" begin
    NF = Float32
    photo = LUEPhotosynthesis(NF)
    traits = PlantTraits(NF)
    material = MaterialConstants(NF)
    constants = PhysicalConstants(NF)
    stomcond = MedlynStomatalConductance(NF)
    autoresp = PALADYNAutotrophicRespiration(NF)
    vegcarbon_dynamics = PALADYNCarbonDynamics(NF)
    veg_dynamics = PALADYNVegetationDynamics(NF)
    phenol = PALADYNPhenology(NF)
    T = NF(20)
    calls = (
        compute_kinetic_parameters => (photo, T),
        compute_Γ_star => (photo, NF(2600), NF(20900)),
        compute_PAR => (photo, NF(400)),
        compute_APAR => (photo, traits, NF(400), NF(4)),
        compute_pres_i => (photo, NF(0.7), NF(40)),
        compute_temperature_stress => (photo, T),
        compute_assimilation_factors => (photo, material, NF(4), NF(0.9), NF(30), NF(30000), NF(28), NF(20900)),
        compute_Vc_max => (photo, NF(6.0e-4), NF(5.0e-4), NF(30), NF(30000), NF(4), NF(28), NF(20900)),
        compute_JE_JC => (photo, NF(6.0e-4), NF(0.3), NF(5.0e-4), NF(1.0e-6)),
        compute_Rd => (photo, NF(1.0e-6), NF(1)),
        compute_Ag => (photo, NF(6.0e-4), NF(0.3), NF(5.0e-4), NF(1.0e-6), NF(1)),
        compute_respiration_assimilation => (photo, traits, material, T, NF(400), NF(1.0e5), NF(400), NF(4), NF(0.7), NF(1)),
        compute_GPP => (photo, NF(1.0e-7), NF(1.0e-8)),
        compute_stomatal_conductance => (stomcond, traits, constants, NF(800), T, NF(101325), NF(415), NF(1.0e-7), NF(3), NF(1)),
        compute_λc => (stomcond, NF(800)),
        compute_f_temp => (autoresp, T, NF(10)),
        compute_Rm => (autoresp, vegcarbon_dynamics, traits, T, NF(10), NF(1.0e-8), NF(1), NF(5)),
        compute_Rg => (autoresp, NF(1.0e-7), NF(3.0e-8)),
        compute_Ra => (autoresp, vegcarbon_dynamics, traits, T, NF(10), NF(1.0e-8), NF(1), NF(5), NF(1.0e-7)),
        compute_NPP => (autoresp, NF(1.0e-7), NF(4.0e-8)),
        compute_λ_NPP => (vegcarbon_dynamics, traits, NF(3)),
        compute_balanced_leaf_area_index => (vegcarbon_dynamics, traits, NF(5)),
        compute_leaf_carbon => (vegcarbon_dynamics, traits, NF(3)),
        compute_root_carbon => (vegcarbon_dynamics, traits, NF(3)),
        compute_stem_carbon => (vegcarbon_dynamics, traits, NF(3)),
        compute_Λ_loc => (vegcarbon_dynamics, traits, NF(3)),
        compute_C_veg_tend => (vegcarbon_dynamics, traits, NF(3), NF(3.0e-8)),
        compute_γv => (veg_dynamics,),
        compute_ν_star => (veg_dynamics, NF(0.3)),
        compute_ν_tendency => (veg_dynamics, vegcarbon_dynamics, traits, NF(3), NF(5), NF(3.0e-8), NF(0.3)),
        compute_phenology_factor => (phenol, NF(100), T),
        compute_leaf_area_index => (phenol, NF(0.5), NF(4)),
        compute_gdd_tendency => (phenol, NF(100), T),
    )
    @testset "$(nameof(f))" for (f, args) in calls
        result = @inferred f(args...)
        @test all(x -> x isa NF, result)
    end
end
