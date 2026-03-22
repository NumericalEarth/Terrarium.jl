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

include("stratigraphy/abstract_types.jl")

include("biogeochem/abstract_types.jl")

include("hydrology/abstract_types.jl")

include("energy/abstract_types.jl")
