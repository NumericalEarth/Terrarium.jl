# Component types

"""
    $TYPEDEF

Base type for coupled ground processes.
"""
abstract type AbstractGround{NF} <: AbstractCoupledProcesses{NF} end

"""
    get_stratigraphy(ground)

Return the ground stratigraphy parameterization associated with `ground`.
"""
@inline get_stratigraphy(ground::AbstractGround) = ground.strat

"""
    get_energy(ground)

Return the energy balance scheme associated with `ground`.
"""
@inline get_energy(ground::AbstractGround) = ground.energy

"""
    get_hydrology(ground)

Return the hydrology scheme associated with `ground`.
"""
@inline get_hydrology(ground::AbstractGround) = ground.hydrology

"""
    $TYPEDEF

Base type for coupled soil processes.
"""
abstract type AbstractSoil{NF} <: AbstractGround{NF} end

"""
    get_biogeochemistry(soil)

Return the biogeochemistry scheme associated with `soil`.
"""
@inline get_biogeochemistry(soil::AbstractSoil) = soil.biogeochem

# Soil process types

abstract type AbstractSoilEnergyBalance{NF} <: AbstractProcess{NF} end

"""
    compute_energy_tendency(i, j, k, grid, ::SoilEnergyBalance, args...)

Compute the internal energy tendency `∂U∂t` at index `i, j, k`.
"""
function compute_energy_tendency end

"""
    compute_thermal_conductivity(i, j, k, grid, ::SoilEnergyBalance, args...)

Compute the thermal conductivity at index `i, j, k`.
"""
function compute_thermal_conductivity end

abstract type AbstractSoilHydrology{NF} <: AbstractProcess{NF} end

abstract type AbstractSoilBiogeochemistry{NF} <: AbstractProcess{NF} end

"""
    density_pure_soc(bgc::AbstractSoilBiogeochemistry)

Return the assumed constant density of pure organic material. The default implementation assumes
there to be a property `ρ_org` defined on the type of `bgc`.
"""
density_pure_soc(bgc::AbstractSoilBiogeochemistry) = bgc.ρ_org

"""
    density_soc(i, j, k, grid, fields, bgc::AbstractSoilBiogeochemistry{NF}) where {NF}

Calculate the organic solid fraction based on the prescribed SOC and natural porosity/density of
the organic material.
"""
@inline density_soc(i, j, k, grid, fields, bgc::AbstractSoilBiogeochemistry{NF}) where {NF} = zero(NF)

# Parameterization types

"""
    $TYPEDEF

Base type for mineral soil texture parameterizations.
"""
abstract type AbstractSoilTexture{NF} end

"""
    $TYPEDEF

Base type for parameterizations of soil porosity.
"""
abstract type AbstractSoilPorosity{NF} end

"""
    mineral_porosity(::AbstractSoilPorosity, texture::SoilTexture)

Compute or retrieve the natural porosity of the mineral soil constitutents, i.e.
excluding organic material.
"""
function mineral_porosity end

"""
    organic_porosity(::AbstractSoilPorosity, texture::SoilTexture)

Compute or retrieve the natural porosity of the organic soil constitutents, i.e.
excluding mineral material.
"""
function organic_porosity end

"""
    $TYPEDEF

Base type for soil stratigraphy parameterizations.
"""
abstract type AbstractStratigraphy{NF} end

"""
    soil_texture(i, j, k, grid, fields, ::AbstractStratigraphy, args...)

Return the texture of the soil at index `i, j, k` for the given stratigraphy parameterization.
"""
function soil_texture end

"""
    soil_matrix(i, j, k, grid, fields, ::AbstractStratigraphy, args...)

Return the solid matrix of the soil at index `i, j, k` for the given stratigraphy parameterization.
"""
function soil_matrix end

"""
    soil_volume(i, j, k, grid, fields, ::AbstractStratigraphy, args...)

Return a description of the full material composition of the soil volume at index `i, j, k` for the
given stratigraphy parameterization.
"""
function soil_volume end

"""
    $TYPEDEF

Base type for formulations of the heat transfer operator.
"""
abstract type AbstractHeatOperator end

"""
    $TYPEDEF

Base type for closure relations between energy and temperature in soil volumes.
"""
abstract type AbstractSoilEnergyClosure <: AbstractClosureRelation end

"""
    $TYPEDEF

Base type for closure relations between water saturation and potential in soil volumes.
"""
abstract type AbstractSoilWaterClosure <: AbstractClosureRelation end
