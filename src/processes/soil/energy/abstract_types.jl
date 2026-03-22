"""
    $TYPEDEF

Base type for soil energy balance process implementations. Subtypes should define
state variables for soil `temperature`, `internal_energy`, and any other relevant
thermal properties or state variables.
"""
abstract type AbstractSoilEnergyBalance{NF} <: AbstractProcess{NF} end

"""
    get_thermal_properties(energy::AbstractSoilEnergyBalance)

Return the thermal properites associated with the given soil energy balance process.
"""
function get_thermal_properties end

# Parameterizations

"""
Base type for bulk weighting/mixing schemes that calculate weighted mixture of material properties
such as conductivities or densities.
"""
abstract type AbstractBulkWeighting end

"""
    $TYPEDEF

Base type for formulations of the heat transfer operator.
"""
abstract type AbstractHeatOperator end

# Closures

"""
    $TYPEDEF

Base type for closure relations between energy and temperature in soil volumes.
"""
abstract type AbstractSoilEnergyClosure <: AbstractClosureRelation end

# Kernel functions

"""
    compute_energy_tendency(i, j, k, grid, ::AbstractSoilEnergyBalance, args...)

Compute the internal energy tendency `∂U∂t` at index `i, j, k`.
"""
function compute_energy_tendency end

"""
    compute_thermal_conductivity(i, j, k, grid, ::AbstractSoilEnergyBalance, args...)

Compute the thermal conductivity at index `i, j, k`.
"""
function compute_thermal_conductivity end
