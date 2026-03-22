"""
    $TYPEDEF

Base type for soil hydrology implementations. Subtypes should define
state variables for `saturation_water_ice`, `hydraulic_conductivity`, 
`liquid_water_fraction`, and the current `water_table` level, along with
any other implementation-specific state variables.
"""
abstract type AbstractSoilHydrology{NF} <: AbstractProcess{NF} end

"""
    get_hydraulic_properties(hydrology::AbstractSoilHydrology)

Return the soil hydraulic properties defined by the given soil `hydrology` configuration.
"""
function get_hydraulic_properties end

"""
    compute_hydraulics!(state, grid, hydrology::SoilHydrology, soil::AbstractSoil, args...)

Compute all state-dependent hydraulic auxiliaries such as hydraulic conductivity and
field capacity, and wilting point.
"""
function compute_hydraulics! end

"""
    get_swrc(hydrology::AbstractSoilHydrology)

Return the soil water retention curve from the `hydraulic_properties` associated with
the given soil hydrology configuration.
"""
get_swrc(hydrology::AbstractSoilHydrology) = get_swrc(get_hydraulic_properties(hydrology))

# Closures

"""
    $TYPEDEF

Base type for closure relations between water saturation and potential in soil volumes.
"""
abstract type AbstractSoilWaterClosure <: AbstractClosureRelation end

# Kernel functions

"""
    saturation_water_ice(i, j, k, grid, fields, ::AbstractSoilHydrology)

Compute or retrieve the current saturation level of water + ice in the pore space.
"""
function saturation_water_ice end

"""
    liquid_water_fraction(i, j, k, grid, fields, ::AbstractSoilHydrology)

Compute or retrieve the current fraction of unfrozen water in the pore space.
"""
function liquid_water_fraction end

"""
    water_table(i, j, k, grid, fields, ::AbstractSoilHydrology)

Compute or retrieve the current water table level relative to the surface.
"""
function water_table end

"""
    surface_excess_water(i, j, k, grid, fields, ::AbstractSoilHydrology)

Retrieve the current saturation level of water + ice in the pore space.
"""
function surface_excess_water end
