"""
    $TYPEDEF

Base type for energy balance process implementations that evolve the **internal energy** of a
solid or porous material medium.
"""
abstract type AbstractThermodynamics{NF} <: AbstractProcess{NF} end

"""
    get_thermal_properties(energy::AbstractThermodynamics)

Return the thermal properties associated with the given energy balance process.
"""
function get_thermal_properties end

# Closures

"""
    $TYPEDEF

Base type for closure relations between internal energy and temperature in a material volume.
"""
abstract type AbstractEnergyClosure <: AbstractClosureRelation end

# Kernel functions

"""
    compute_energy_tendency(i, j, k, grid, ::AbstractThermodynamics, args...)

Compute the internal energy tendency `∂U∂t` at index `i, j, k`.
"""
function compute_energy_tendency end

"""
    compute_thermal_conductivity(i, j, k, grid, ::AbstractThermodynamics, args...)

Compute the thermal conductivity at index `i, j, k`.
"""
function compute_thermal_conductivity end
