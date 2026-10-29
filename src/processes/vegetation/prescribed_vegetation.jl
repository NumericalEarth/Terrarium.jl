"""
    $TYPEDEF

Coupled vegetation process representing natural vegetation with a *prescribed* leaf area index. Unlike
[`VegetationCarbonCycle`](@ref), the vegetation carbon pool and leaf area are not prognostic: leaf area
index is imposed externally (via the [`PrescribedPhenology`](@ref) scheme) and drives photosynthesis,
stomatal conductance, and the associated water/energy exchange. There is consequently no prognostic
carbon-pool or vegetation-dynamics component and no autotrophic respiration; `compute_tendencies!` is a
no-op. Plant functional-type parameters (including the maximum leaf area index used to derive the
phenology factor) are supplied through the `traits` component.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct PrescribedVegetation{NF, Phenology, Photosynthesis, StomatalConductance, RootDistribution, PAW} <: AbstractVegetation{NF}
    "Phenology scheme"
    @component phenology::Phenology

    "Photosynthesis scheme"
    @component photosynthesis::Photosynthesis

    "Stomatal conductance scheme"
    @component stomatal_conductance::StomatalConductance

    "Plant vertical root distribution"
    @component root_distribution::RootDistribution

    "Plant available water determining soil moisture stress"
    @component plant_available_water::PAW

    "Plant physical traits"
    @component traits::PlantTraits{NF}
end

function PrescribedVegetation(
        ::Type{NF};
        phenology = PrescribedPhenology(NF),
        photosynthesis = LUEPhotosynthesis(NF),
        stomatal_conductance = MedlynStomatalConductance(NF),
        root_distribution = StaticExponentialRootDistribution(NF),
        plant_available_water = FieldCapacityLimitedPAW(NF),
        traits = PlantTraits(NF)
    ) where {NF}
    return PrescribedVegetation(; phenology, photosynthesis, stomatal_conductance, root_distribution, plant_available_water, traits)
end

variables(vegetation::PrescribedVegetation) = tuplejoin(
    variables(vegetation.phenology, vegetation.traits), # include traits in variables
    variables(vegetation.photosynthesis),
    variables(vegetation.stomatal_conductance),
    variables(vegetation.root_distribution),
    variables(vegetation.plant_available_water)
)

function initialize!(state, grid, veg::PrescribedVegetation, args...)
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
        veg::PrescribedVegetation,
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

    # Phenology (prescribed): needs LAI(t) and air temperature(t) as inputs and computes phen(t)
    compute_auxiliary!(state, grid, veg.phenology, atmos)

    # Photosynthesis: needs atm. inputs(t), LAI(t), and computes Rd(t) and GPP(t)
    compute_auxiliary!(state, grid, veg.photosynthesis, veg.stomatal_conductance, veg.traits, constants, atmos)

    # Stomatal conductance: needs atm. inputs(t) and computes λc(t)
    compute_auxiliary!(state, grid, veg.stomatal_conductance, veg.traits, constants, atmos)
    return nothing
end

# The vegetated fraction is diagnosed from the relative leaf area
@propagate_inbounds vegetation_area_fraction(i, j, grid, fields, veg::PrescribedVegetation) =
    compute_relative_leaf_area(i, j, grid, fields, veg.traits)

# No tendencies for prescribed vegetation
@inline compute_tendencies!(state, grid, veg::PrescribedVegetation, args...) = nothing
