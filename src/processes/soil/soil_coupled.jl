"""
    $TYPEDEF

Coupled process type that encapsulates the coupling of soil energy, water, and carbon dynamics.
The stratigraphy parameterization determines how the vertical layering of the soil is parameterized.
"""
struct SoilEnergyWaterCarbon{
        NF,
        Stratigraphy <: AbstractStratigraphy{NF},
        Energy <: AbstractSoilThermodynamics{NF},
        Hydrology <: AbstractSoilHydrology{NF},
        Biogeochemistry <: AbstractSoilBiogeochemistry{NF},
    } <: AbstractSoil{NF}
    "Soil stratigraphy parameterization"
    strat::Stratigraphy

    "Soil energy balance process"
    energy::Energy

    "Soil hydrology (water balance) process"
    hydrology::Hydrology

    "Soil biogeochemistry process"
    biogeochem::Biogeochemistry
end

function SoilEnergyWaterCarbon(
        ::Type{NF};
        strat = HomogeneousSoilStratigraphy(NF),
        energy = SoilThermodynamics(NF),
        hydrology = SoilHydrology(NF),
        biogeochem = ConstantSoilCarbonDensity(NF)
    ) where {NF}
    return SoilEnergyWaterCarbon(strat, energy, hydrology, biogeochem)
end

# Process interface methods

"""
    $TYPEDSIGNATURES

Initialize the soil energy, water, and carbon state variables on `grid` given
the parameter values in `constants`.
"""
function initialize!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants
    )
    initialize!(state, grid, soil.hydrology, soil, constants)
    initialize!(state, grid, soil.biogeochem, soil, constants)
    initialize!(state, grid, soil.energy, soil, constants)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute auxiliary variables for soil energy, water, and carbon state variables
on `grid` based on the given values in `constants`.
"""
function compute_auxiliary!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants
    )
    # TODO: consider implementing fused kernel here?
    compute_auxiliary!(state, grid, soil.hydrology, soil, constants)
    compute_auxiliary!(state, grid, soil.biogeochem, soil, constants)
    compute_auxiliary!(state, grid, soil.energy, soil, constants)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute boundary conditions (and halo regions) for soil energy and hydrology.
"""
function compute_boundary_conditions!(state, grid, soil::SoilEnergyWaterCarbon)
    compute_boundary_conditions!(state, grid, soil.hydrology, soil.strat, soil.biogeochem)
    compute_boundary_conditions!(state, grid, soil.energy)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute tendencies for soil energy, water, and carbon state variables on `grid`
based on the given values in `constants`.
"""
function compute_tendencies!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants
    )
    # TODO: consider implementing fused kernel here?
    compute_tendencies!(state, grid, soil.hydrology, soil, constants)
    compute_tendencies!(state, grid, soil.biogeochem, soil, constants)
    compute_tendencies!(state, grid, soil.energy, soil, constants)
    return nothing
end

# Closures

"""
    $TYPEDSIGNATURES

Compute the forward closure mapping for soil hydrology and energy, in that order.

An optional `surface_hydrology` process may be supplied (by the coupled `LandModel`) so that excess
water removed from an oversaturated soil surface is routed into the `surface_excess_water` pool
owned by its runoff scheme ([`get_surface_runoff`](@ref)). Without it (standalone soil), the excess
is discarded.
"""
function closure!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        surface_hydrology::Optional{AbstractSurfaceHydrology} = nothing
    )
    runoff = isnothing(surface_hydrology) ? nothing : get_surface_runoff(surface_hydrology)
    closure!(state, grid, get_closure(soil.hydrology), soil.hydrology, soil, runoff)
    closure!(state, grid, get_closure(soil.energy), soil.energy, soil, constants)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the inverse closure mapping for soil hydrology and energy, in that order.

See [`closure!`](@ref) for the role of the optional `surface_hydrology` process.
"""
function invclosure!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        surface_hydrology::Optional{AbstractSurfaceHydrology} = nothing
    )
    runoff = isnothing(surface_hydrology) ? nothing : get_surface_runoff(surface_hydrology)
    invclosure!(state, grid, get_closure(soil.hydrology), soil.hydrology, soil, runoff)
    invclosure!(state, grid, get_closure(soil.energy), soil.energy, soil, constants)
    return nothing
end
