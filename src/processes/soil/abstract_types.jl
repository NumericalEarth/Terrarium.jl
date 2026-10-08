# Component types

"""
    $TYPEDEF

Base type for coupled soil processes.
"""
abstract type AbstractSoil{NF} <: AbstractCoupledProcesses{NF} end

"""
    get_stratigraphy(soil::AbstractSoil)

Return the stratigraphy parameterization associated with `soil`.
"""
@inline get_stratigraphy(soil::AbstractSoil) = soil.strat

"""
    get_energy_balance(soil::AbstractSoil)

Return the energy balance scheme associated with `soil`.
"""
@inline get_energy_balance(soil::AbstractSoil) = soil.energy

"""
    get_hydrology(soil::AbstractSoil)

Return the hydrology scheme associated with `soil`.
"""
@inline get_hydrology(soil::AbstractSoil) = soil.hydrology

"""
    get_biogeochemistry(soil::AbstractSoil)

Return the biogeochemistry scheme associated with `soil`.
"""
@inline get_biogeochemistry(soil::AbstractSoil) = soil.biogeochem

# Soil process types

include("stratigraphy/abstract_types.jl")

include("biogeochem/abstract_types.jl")

include("hydrology/abstract_types.jl")

include("energy/abstract_types.jl")
