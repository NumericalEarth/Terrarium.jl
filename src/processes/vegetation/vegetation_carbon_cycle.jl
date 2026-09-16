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
    # Roots: need soil state and computes root_fraction (a lazy `FunctionField`, no launch)
    compute_auxiliary!(state, grid, veg.root_distribution, soil)

    # PAW: needs soil saturation profile and materializes soil_moisture_limiting_factor (a derived field).
    # It runs before the fused kernel below, whose photosynthesis and stomatal conductance stages read it.
    compute_auxiliary!(state, grid, veg.plant_available_water, soil)

    # The carbon-cycle chain — carbon dynamics → phenology → photosynthesis → stomatal conductance →
    # autotrophic respiration — is fused into a single launch. Each stage reads the previous stage's
    # output within a cell, so the dependency chain resolves without returning to the host.
    carbon_dynamics = veg.carbon_dynamics
    phenology = veg.phenology
    photosynthesis = veg.photosynthesis
    stomatal_conductance = veg.stomatal_conductance
    autotrophic_respiration = veg.autotrophic_respiration
    out = filter(v -> v isa Field, auxiliary_fields(state, carbon_dynamics, phenology, photosynthesis,
                                                     stomatal_conductance, autotrophic_respiration))
    # Full fields (no `except`): within a cell the kernel writes `out.foo` and a later stage reads
    # `fields.foo` — the same `Field` object, so the write is visible to the stages below.
    fields = get_fields(state, carbon_dynamics, phenology, photosynthesis, stomatal_conductance,
                        autotrophic_respiration, atmos)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, veg, constants, atmos)

    # Note: vegetation_dynamics compute_auxiliary! does nothing for now
    compute_auxiliary!(state, grid, veg.vegetation_dynamics)
    return nothing
end

"""
    $TYPEDSIGNATURES

Fused auxiliary kernel for the vegetation carbon cycle. Each component's per-cell mutating variant runs in
dependency order — carbon dynamics (a pure producer of `balanced_leaf_area_index` from the vegetation carbon
pool), then phenology (which reads it to set `leaf_area_index` and `phenology_factor`), then photosynthesis
(which reads `leaf_area_index` and the soil moisture limiting factor to set `net_assimilation` and
`gross_primary_production`), then stomatal conductance (which reads `net_assimilation` to set
`canopy_water_conductance`), then autotrophic respiration (which reads `gross_primary_production` and
`phenology_factor` to set `net_primary_production`) — so the chain resolves within a single launch.
"""
@kernel inbounds = true function compute_auxiliary_kernel!(
        out, grid, fields,
        veg::VegetationCarbonCycle,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    )
    i, j = @index(Global, NTuple)
    compute_veg_carbon_auxiliary!(out, i, j, grid, fields, veg.carbon_dynamics, veg.traits)
    compute_phenology!(out, i, j, grid, fields, veg.phenology, atmos)
    # Photosynthesis reads stomatal conductance's parameters (λc) but not its auxiliary state
    compute_photosynthesis!(out, i, j, grid, fields, veg.photosynthesis, veg.stomatal_conductance,
                            veg.traits, constants, atmos)
    compute_stomatal_conductance!(out, i, j, grid, fields, veg.stomatal_conductance, veg.traits, constants, atmos)
    compute_autotrophic_respiration!(out, i, j, grid, fields, veg.autotrophic_respiration,
                                     veg.carbon_dynamics, veg.phenology, veg.traits, atmos)
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
        LAI = fields.leaf_area_index[i, j, 1]
        LAI_max = maximum_leaf_area_index(i, j, grid, fields, veg.traits)
        f_veg = LAI / LAI_max
        return f_veg
    else
        return vegetation_area_fraction(i, j, grid, fields, veg.vegetation_dynamics)
    end
end
