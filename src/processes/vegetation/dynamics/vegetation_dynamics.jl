"""
    $TYPEDEF

Vegetation dynamics implementation following [willeitPALADYNV10Comprehensive2016](@cite) for a single PFT
based on the Lotka–Volterra approach.

Authors: Maha Badri 

Properties:
$TYPEDFIELDS

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@parameterized @kwdef struct PALADYNVegetationDynamics{NF} <: AbstractVegetationDynamics{NF}
    "Vegetation seed fraction"
    @param ν_seed::NF = 0.001 (bounds = UnitInterval,)

    "Minimum vegetation disturbance rate"
    # TODO this parameter is yearly, should be changed to daily for now
    @param γv_min::NF = 0.002 (units = u"yr^-1", bounds = Positive)
end

PALADYNVegetationDynamics(::Type{NF}; kwargs...) where {NF} = PALADYNVegetationDynamics{NF}(; kwargs...)

variables(::PALADYNVegetationDynamics) = (
    prognostic(:vegetation_area_fraction, Canopy(XY())), # PFT fractional area coverage [-]
    input(:net_primary_production, Canopy(XY()), units = u"kg/m^2/s"),
)

@propagate_inbounds vegetation_area_fraction(i, j, grid, fields, ::PALADYNVegetationDynamics) = fields.vegetation_area_fraction[i, j]

"""
    $SIGNATURES

Computes the disturbance rate`γv`,
[willeitPALADYNV10Comprehensive2016; Eq. (80)](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_γv(veg_dynamics::PALADYNVegetationDynamics)
    # TODO add PALADYN implemetation for the disturbance rate (depends on soil moisture)
    # Placeholder for now γv = min. disturbance rate
    return veg_dynamics.γv_min / (365.25 * 24 * 3600) # convert to seconds
end

"""
    $SIGNATURES

Computes `ν_star` which is the maximum of the current vegetation fraction `ν`
and the seed fraction `ν_seed` [-], to ensure that a PFT is always seeded.
"""
@inline function compute_ν_star(veg_dynamics::PALADYNVegetationDynamics, ν)
    return max(ν, veg_dynamics.ν_seed)
end

"""
    $SIGNATURES

Computes the vegetation fraction tendency for a single PFT,
[willeitPALADYNV10Comprehensive2016; Eq. (73)](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_ν_tendency(
        veg_dynamics::PALADYNVegetationDynamics,
        vegcarbon_dynamics::PALADYNCarbonDynamics{NF},
        traits::PlantTraits{NF},
        LAI_b::NF,
        C_veg::NF,
        NPP::NF,
        ν::NF
    ) where {NF}

    # Compute λ_NPP
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)

    # Compute the disturbance rate
    γv = compute_γv(veg_dynamics)

    # Compute ν_star
    ν_star = compute_ν_star(veg_dynamics, ν)

    # Compute the vegetation fraction tendency
    ν_tendency = (λ_NPP * NPP / C_veg) * ν_star * (NF(1.0) - ν) - γv * ν_star
    return ν_tendency
end

# Top-level interface methods

""" $TYPEDSIGNATURES """
function compute_auxiliary!(state, grid, veg_dynamics::PALADYNVegetationDynamics, args...)
    # Nothing needed here for now
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_tendencies!(
        state, grid,
        veg_dynamics::PALADYNVegetationDynamics,
        vegcarbon_dynamics::PALADYNCarbonDynamics,
        traits::PlantTraits,
        args...
    )
    tend = tendency_fields(state, veg_dynamics)
    fields = get_fields(state, veg_dynamics, vegcarbon_dynamics)
    launch!(grid, XY, compute_tendencies_kernel!, tend, fields, veg_dynamics, vegcarbon_dynamics, traits)
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
        vegcarbon_dynamics::PALADYNCarbonDynamics,
        traits::PlantTraits
    )
    LAI_b = fields.balanced_leaf_area_index[i, j]
    C_veg = fields.carbon_vegetation[i, j]
    NPP = fields.net_primary_production[i, j]
    ν = fields.vegetation_area_fraction[i, j]
    ν_tendency = compute_ν_tendency(veg_dynamics, vegcarbon_dynamics, traits, LAI_b, C_veg, NPP, ν)
    return ν_tendency
end


"""
    $TYPEDSIGNATURES

Mutating wrapper for [`compute_ν_tendency`](@ref) that stores the result in `tend`.
"""
@propagate_inbounds function compute_ν_tendencies!(
        tend, i, j, grid, fields,
        veg_dynamics::PALADYNVegetationDynamics,
        vegcarbon_dynamics::PALADYNCarbonDynamics,
        traits::PlantTraits
    )
    tend.vegetation_area_fraction[i, j, end] = compute_ν_tendency(i, j, grid, fields, veg_dynamics, vegcarbon_dynamics, traits)
    return tend
end

# Kernels

@kernel function compute_tendencies_kernel!(tend, grid, fields, veg_dynamics::AbstractVegetationDynamics, args...)
    i, j = @index(Global, NTuple)
    compute_ν_tendencies!(tend, i, j, grid, fields, veg_dynamics, args...)
end
