# Vegetation component types

"""
    $TYPEDEF

Base type for coupled vegetation (carbon) processes.
"""
abstract type AbstractVegetation{NF} <: AbstractCoupledProcesses{NF} end

# Plant traits

"""
    $TYPEDEF

Base type for plant-dependent trait parameterizations.
"""
abstract type AbstractPlantTraits{NF} end

"""
    plant_traits(i, j, grid, fields, ::AbstractPlantTraits)

Return the localized plant traits at the given grid cell indices.
"""
function plant_traits end

# Vegetation process types

"""
    $TYPEDEF

Base type for photosynthesis schemes.
"""
abstract type AbstractPhotosynthesis{NF} <: AbstractProcess{NF} end

"""
	compute_photosynthesis(i, j, grid, fields, photo::AbstractPhotosynthesis, atmos::AbstractAtmosphere)

Cell-level photosynthesis computation. Implementations compute leaf respiration
and net assimilation for a single horizontal cell and return the pair
`(Rd, An, GPP)` or similar outputs as required by the photosynthesis scheme.
"""
function compute_photosynthesis end

"""
    $TYPEDEF

Base type for stomatal conductance schemes.
"""
abstract type AbstractStomatalConductance{NF} <: AbstractProcess{NF} end

"""
	compute_stomatal_conductance(
        i, j, grid, fields,
        stomcond::AbstractStomatalConductance,
        photo::AbstractPhotosynthesis,
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants,
        args...
    )

Cell-level stomatal conductance computation. Returns stomatal/canopy conductance
and internal CO₂ ratio for the specified cell.
"""
function compute_stomatal_conductance end

"""
    $TYPEDEF

Base type for autotrophic respiration schemes.
"""
abstract type AbstractAutotrophicRespiration{NF} <: AbstractProcess{NF} end

"""
	compute_autotrophic_respiration(
        i, j, grid, fields,
        autoresp::AbstractAutotrophicRespiration,
        vegcarbon::AbstractVegetationCarbonDynamics,
        atmos::AbstractAtmosphere,
        args...
    )

Cell-level autotrophic respiration computation. Implementations should compute
autotrophic respiration and related diagnostics (e.g. `NPP`) for the given cell.
"""
function compute_autotrophic_respiration end

"""
    $TYPEDEF

Base type for vegetation dynamics schemes.
"""
abstract type AbstractVegetationDynamics{NF} <: AbstractProcess{NF} end

"""
	compute_ν_tendency(
        i, j, grid, fields,
        veg_dynamics::AbstractVegetationDynamics,
        vegcarbon::AbstractVegetationCarbonDynamics
    )

Cell-level vegetation-fraction tendency computation used by vegetation
dynamics. Implementations compute the local tendency value for `ν` at the given index `i, j`.
"""
function compute_ν_tendency end

"""
    vegetation_area_fraction(i, j, grid, fields, ::AbstractVegetationDynamics)

Return the fraction of the grid cell `i, j` covered by vegetation of any type.
"""
function vegetation_area_fraction end

"""
    $TYPEDEF

Base type for vegetation phenology schemes.
"""
abstract type AbstractPhenology{NF} <: AbstractProcess{NF} end

"""
	compute_phenology(i, j, grid, fields, phenol::AbstractPhenology, atmos::AbstractAtmosphere)

Cell-level phenology computation. Implementations return phenology factors
and derived LAI at the given index `i, j`, using atmospheric inputs (e.g. air temperature)
where required by the scheme.
"""
function compute_phenology end

"""
    $TYPEDEF

Base type for vegetation carbon dynamics schemes.
"""
abstract type AbstractVegetationCarbonDynamics{NF} <: AbstractProcess{NF} end

"""
	compute_veg_carbon_tendency(i, j, grid, fields, vegcarbon::AbstractVegetationCarbonDynamics)

Cell-level vegetation-carbon tendency computation. Implementations compute the
tendency for the total vegetation carbon pool at the given index `i, j`.
"""
function compute_veg_carbon_tendency end

"""
    $TYPEDEF

Base type for processes that comptue the plant available water fraction in each soil layer.
"""
abstract type AbstractPlantAvailableWater{NF} <: AbstractProcess{NF} end

"""
	compute_plant_available_water(
        i, j, k, grid, fields,
        paw::AbstractPlantAvailableWater,
        soil::AbstractSoil
    )

Comptue the plant-available water fraction for grid cell `i, j` and soil layer `k`.
"""
function compute_plant_available_water end

"""
    $TYPEDEF

Base type for vegetation root distribution schemes.
"""
abstract type AbstractRootDistribution{NF} <: AbstractProcess{NF} end

"""
    root_density(::AbstractRootDistribution, z, args...)

Compute the continuous density function of the given root distirbution as a function of depth `z`.
Note that this function must be integrated and normalized over the root zone in order to obtain the
cumulative root fraction in each soil layer.
"""
function root_density end
