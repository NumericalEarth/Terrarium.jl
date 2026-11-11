using Terrarium
using Terrarium: compute_kinetic_parameters, compute_Γ_star, compute_PAR, compute_APAR, compute_pres_i
using Terrarium: compute_temperature_stress, compute_assimilation_factors, compute_Vc_max, compute_JE_JC
using Terrarium: compute_Rd, compute_Ag, compute_GPP, compute_respiration_assimilation
using Terrarium: compute_stomatal_conductance, compute_λc
using Terrarium: compute_f_temp, compute_resp10, compute_Rm, compute_Rg, compute_Ra, compute_NPP
using Terrarium: compute_λ_NPP, compute_balanced_leaf_area_index, compute_leaf_carbon, compute_root_carbon, compute_stem_carbon
using Terrarium: compute_Λ_loc, compute_C_veg_tend, compute_γv, compute_ν_star, compute_ν_tendency
using Terrarium: compute_phenology_factor, compute_leaf_area_index, compute_gdd_tendency
using Test

# Kernel functions must not promote Float32 inputs to Float64 (for example through Float64 literals)
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
        compute_resp10 => (autoresp,),
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
