"""
    $TYPEDEF

Vegetation dynamics implementation following PALADYN (Willeit 2016) for a single PFT
based on the Lotka–Volterra approach.

Authors: Maha Badri 

Properties:
$TYPEDFIELDS
"""
@kwdef struct PALADYNVegetationDynamics{NF} <: AbstractVegetationDynamics{NF}
    "Vegetation seed fraction [-]"
    ν_seed::NF = 0.001

    "Minimum vegetation disturbance rate [1/year]"
    # TODO this parameter is yearly, should be changed to daily for now
    γv_min::NF = 0.002
end

PALADYNVegetationDynamics(::Type{NF}; kwargs...) where {NF} = PALADYNVegetationDynamics(; kwargs...)

variables(::PALADYNVegetationDynamics) = (
    prognostic(:vegetation_area_fraction, XY()), # PFT fractional area coverage [-]
    input(:balanced_leaf_area_index, XY()),
    input(:carbon_vegetation, XY(), units = u"kg/m^2"),
)

"""
    $SIGNATURES

Computes the disturbance rate`γv`,
Eq. 80, PALADYN (Willeit 2016).
"""
@inline function compute_γv(veg_dynamics::PALADYNVegetationDynamics)
    # TODO add PALADYN implemetation for the disturbance rate (depends on soil moisture)
    # Placeholder for now γv = min. disturbance rate
    return veg_dynamics.γv_min
end

"""
    $SIGNATURES

Computes `ν_star` which is the maximum between the current vegetation fraction `ν` and the seed fraction `ν_seed` [-],
to ensure that a PFT is always seeded.
"""
@inline function compute_ν_star(veg_dynamics::PALADYNVegetationDynamics, ν)
    return max(ν, veg_dynamics.ν_seed)
end

"""
    $SIGNATURES

Computes the vegetation fraction tendency for a single PFT,
Eq. 73, PALADYN (Willeit 2016).
"""
@inline function compute_ν_tendency(
        veg_dynamics::PALADYNVegetationDynamics,
        vegcarbon_dynamics::PALADYNCarbonDynamics{NF},
        LAI_b::NF,
        C_veg::NF,
        ν::NF
    ) where {NF}

    # Compute λ_NPP
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, LAI_b)

    # Compute the disturbance rate
    γv = compute_γv(veg_dynamics)

    # Compute ν_star
    ν_star = compute_ν_star(veg_dynamics, ν)

    # Compute the vegetation fraction tendency
    ν_tendency = (λ_NPP * C_veg / ν_star) * (NF(1.0) - ν) - γv * ν_star
    return ν_tendency
end

# Process methods

function compute_auxiliary!(state, grid, veg_dynamics::PALADYNVegetationDynamics, args...)
    # Nothing needed here for now
    return nothing
end

function compute_tendencies!(
        state, grid,
        veg_dynamics::PALADYNVegetationDynamics,
        vegcarbon_dynamics::PALADYNCarbonDynamics,
        args...
    )
    tend = tendency_fields(state, veg_dynamics)
    fields = get_fields(state, veg_dynamics, vegcarbon_dynamics)
    launch!(grid, XY, compute_tendencies_kernel!, tend, fields, veg_dynamics, vegcarbon_dynamics)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute vegetation area fraction tendency at a single grid point from NPP-productivity and disturbance rates.
"""
@propagate_inbounds function compute_ν_tendency(
        i, j, grid, fields,
        veg_dynamics::PALADYNVegetationDynamics,
        vegcarbon_dynamics::PALADYNCarbonDynamics
    )
    # Get inputs
    LAI_b = fields.balanced_leaf_area_index[i, j]
    C_veg = fields.carbon_vegetation[i, j]

    # Current state
    ν = fields.vegetation_area_fraction[i, j]

    # Compute the vegetation fraction tendency
    ν_tendency = compute_ν_tendency(veg_dynamics, vegcarbon_dynamics, LAI_b, C_veg, ν)
    return ν_tendency
end


"""
    $TYPEDSIGNATURES

Mutating wrapper for [`compute_ν_tendency`](@ref) that stores the result in `tend`.
"""
@propagate_inbounds function compute_ν_tendencies!(
        tend, i, j, grid, fields,
        veg_dynamics::PALADYNVegetationDynamics,
        vegcarbon_dynamics::PALADYNCarbonDynamics
    )
    tend.vegetation_area_fraction[i, j, 1] = compute_ν_tendency(i, j, grid, fields, veg_dynamics, vegcarbon_dynamics)
    return tend
end

# Kernels

@kernel function compute_tendencies_kernel!(tend, grid, fields, veg_dynamics::AbstractVegetationDynamics, args...)
    i, j = @index(Global, NTuple)
    compute_ν_tendencies!(tend, i, j, grid, fields, veg_dynamics, args...)
end
