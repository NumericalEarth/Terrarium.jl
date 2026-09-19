"""
    $TYPEDEF

Coupled process type representing the major carbon cycle processes for natural vegetation.
"""
@kwdef struct VegetationCarbonCycle{
        NF,
        Photosynthesis <: AbstractPhotosynthesis{NF},
        StomatalConductance <: AbstractStomatalConductance{NF},
        AutotrophicRespiration <: AbstractAutotrophicRespiration{NF},
        Phenology <: AbstractPhenology{NF},
        CarbonDynamics <: AbstractVegetationCarbonDynamics{NF},
        VegetationDynamics <: Optional{AbstractVegetationDynamics},
        RootDistribution <: Optional{AbstractRootDistribution},
        PAW <: Optional{AbstractPlantAvailableWater},
    } <: AbstractVegetation{NF}
    "Photosynthesis scheme"
    photosynthesis::Photosynthesis # not prognostic

    "Stomatal conductance scheme"
    stomatal_conductance::StomatalConductance # not prognostic

    "Autotrophic respiration scheme"
    autotrophic_respiration::AutotrophicRespiration # not prognostic

    "Phenology scheme"
    phenology::Phenology # not prognostic

    "Vegetation carbon pool dynamics"
    carbon_dynamics::CarbonDynamics # prognostic

    "Vegetation population density or coverage fraction dynamics"
    vegetation_dynamics::VegetationDynamics # prognostic

    "Vegetation root distribution"
    root_distribution::RootDistribution

    "Plant available water"
    plant_available_water::PAW

    "Plant-specific trait parameters"
    traits::PlantTraits{NF}
end

function VegetationCarbonCycle(
        ::Type{NF};
        photosynthesis = LUEPhotosynthesis(NF),
        stomatal_conductance = MedlynStomatalConductance(NF),
        autotrophic_respiration = PALADYNAutotrophicRespiration(NF),
        phenology = PALADYNPhenology(NF),
        carbon_dynamics = PALADYNCarbonDynamics(NF),
        vegetation_dynamics = PALADYNVegetationDynamics(NF),
        root_distribution = StaticExponentialRootDistribution(NF),
        plant_available_water = FieldCapacityLimitedPAW(NF),
        traits = PlantTraits(NF)
    ) where {NF}
    return VegetationCarbonCycle(;
        photosynthesis,
        stomatal_conductance,
        autotrophic_respiration,
        phenology,
        carbon_dynamics,
        vegetation_dynamics,
        root_distribution,
        plant_available_water,
        traits
    )
end

function initialize!(state, grid, veg::VegetationCarbonCycle)
    initialize!(state, grid, veg.plant_available_water)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute auxiliary variables for all vegetation component processes based on the given
atmospheric inputs defined by `atmos` and (optionally) `soil` state. If `soil = nothing`,
stress factors due to soil temperature and moisture availability will be ignored.
"""
function compute_auxiliary!(
        state, grid,
        veg::VegetationCarbonCycle,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::Optional{AbstractSoil} = nothing,
        args...
    )
    # Compute auxiliary variables for each component
    # Roots: need soil state and computes root_fraction
    compute_auxiliary!(state, grid, veg.root_distribution, soil)

    # PAW: needs soil saturation profile and computes soil_moisture_limiting_factor
    compute_auxiliary!(state, grid, veg.plant_available_water, soil)

    # Veg. carbon dynamics: needs C_veg(t) and computes LAI_b(t)
    compute_auxiliary!(state, grid, veg.carbon_dynamics, veg.traits)

    # Phenology: needs LAI_b(t) and air temperature(t) and computes LAI(t) and phen(t)
    compute_auxiliary!(state, grid, veg.phenology, veg.carbon_dynamics, atmos)

    # Photosynthesis: needs atm. inputs(t), LAI(t), and computes Rd(t) and GPP(t)
    # N.B. We break the usual dependency pattern here to resolve the tight coupling between photosynthesis and
    # stomatal conductance. Photosynthesis does *not* depend on the auxiliary state of stomatal conductance but
    # requires its parameters to compute λc.
    compute_auxiliary!(state, grid, veg.photosynthesis, veg.stomatal_conductance, veg.traits, constants, atmos)

    # Stomatal conductance: needs atm. inputs(t) and computes g_can(t)
    compute_auxiliary!(state, grid, veg.stomatal_conductance, veg.traits, constants, atmos)

    # Autotrophic respiration: needs atm. inputs(t), GPP(t), Rd(t), C_veg(t), phen(t) and computes Ra(t) and NPP(t)
    compute_auxiliary!(state, grid, veg.autotrophic_respiration, veg.carbon_dynamics, veg.phenology, veg.traits, atmos)

    # Note: vegetation_dynamics compute_auxiliary! does nothing for now
    compute_auxiliary!(state, grid, veg.vegetation_dynamics)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute tendencies for carbon and vegetation dynamics.
"""
function compute_tendencies!(state, grid, veg::VegetationCarbonCycle, constants::PhysicalConstants, atmos::AbstractAtmosphere, args...)
    # Needs NPP(t), C_veg(t), LAI_b(t) and computes tendency for C_veg
    compute_tendencies!(state, grid, veg.carbon_dynamics, veg.traits)

    # Needs NPP(t), C_veg(t), LAI_b(t), ν(t) and computes tendency for ν
    compute_tendencies!(state, grid, veg.vegetation_dynamics, veg.carbon_dynamics, veg.traits)

    # Needs air temperature(t) and computes tendency for growing degree days (prognostic phenology)
    compute_tendencies!(state, grid, veg.phenology, atmos)

    return nothing
end

@propagate_inbounds function vegetation_area_fraction(i, j, grid, fields, veg::VegetationCarbonCycle)
    if isnothing(veg.vegetation_dynamics)
        LAI = fields.leaf_area_index[i, j, end]
        LAI_max = maximum_leaf_area_index(i, j, grid, fields, veg.traits)
        f_veg = LAI / LAI_max
        return f_veg
    else
        return vegetation_area_fraction(i, j, grid, fields, veg.vegetation_dynamics)
    end
end
