"""
    $TYPEDEF

Vegetation carbon dynamics implementation following PALADYN (Willeit 2016) but considering only the sum of the vegetation
carbon pools. The subsequent splitting into C_leaf, C_stem, C_root is not implemented for now.

Authors: Maha Badri

Properties:
$TYPEDFIELDS
"""
@kwdef struct PALADYNCarbonDynamics{NF} <: AbstractVegetationCarbonDynamics{NF}
    "Specific leaf area (Kattge et al. 2011) [m²/kgC], PFT specific"
    SLA::NF = 10.0 # Value for Needleleaf tree PFT

    "Allometric coefficient, modified from Cox 2001 to account for bwl=1 [kgC/m²], PFT specific"
    awl::NF = 2.0 # Value for Needleleaf tree PFT

    "Minimum Leaf Area Index modified from Clark et al. 2011 [m²/m²], PFT specific"
    LAI_min::NF = 1.0 # Value for Needleleaf tree PFT

    "Maximum Leaf Area Index modified from Clark et al. 2011 [m²/m²], PFT specific"
    LAI_max::NF = 6.0 # Value for Needleleaf tree PFT

    "Leaf turnover rate (Kattge et al. 2011) [1/year], PFT specific"
    # TODO this parameter is yearly, should be changed to daily for now
    γL::NF = 0.3 # Value for Needleleaf tree PFT

    "Root turnover rate [1/year], PFT specific"
    # TODO this parameter is yearly, should be changed to daily for now
    γR::NF = 0.3 # Value for Needleleaf tree PFT

    "Stem turnover rate modified from Clark et al. 2011 [1/year], PFT specific"
    # TODO this parameter is yearly, should be changed to daily for now
    γS::NF = 0.05 # Value for Needleleaf tree PFT
end

PALADYNCarbonDynamics(::Type{NF}; kwargs...) where {NF} = PALADYNCarbonDynamics{NF}(; kwargs...)

variables(::PALADYNCarbonDynamics) = (
    prognostic(:carbon_vegetation, XY(), units = u"kg/m^2"), # Vegetation carbon pool [kgC/m²]
    auxiliary(:balanced_leaf_area_index, XY()), # Balanced Leaf Area Index [m²/m²]
    input(:net_primary_production, XY(), units = u"kg/m^2/s"), # Net Primary Production [kgC/m²/s]
)

"""
    $SIGNATURES

Computes `λ_NPP`,a factor determining the partitioning of NPP between increase of vegetation carbon of the existing 
vegetated area and spreading of the given PFT based on the balanced Leaf Area Index `LAI_b`,
Eq. 74, PALADYN (Willeit 2016).
"""
@inline function compute_λ_NPP(vegcarbon_dynamics::PALADYNCarbonDynamics{NF}, LAI_b) where {NF}
    if LAI_b < vegcarbon_dynamics.LAI_min
        λ_NPP = zero(NF)
    elseif LAI_b <= vegcarbon_dynamics.LAI_max
        λ_NPP = (LAI_b - vegcarbon_dynamics.LAI_min) /
            (vegcarbon_dynamics.LAI_max - vegcarbon_dynamics.LAI_min)
    else
        λ_NPP = NF(1.0)
    end
    return λ_NPP
end

"""
    $SIGNATURES

Computes `LAI_b`, the balanced Leaf Area Index based on the vegetation carbon pool `C_veg` (assuming with bwl = 1),
Eqs. 76-79, PALADYN (Willeit 2016).
"""
@inline function compute_LAI_b(vegcarbon_dynamics::PALADYNCarbonDynamics{NF}, C_veg) where {NF}
    LAI_b = ((NF(2.0) / vegcarbon_dynamics.SLA) + vegcarbon_dynamics.awl) / (C_veg + eps(NF)) # division by zero risk
    return LAI_b
end

"""
    $SIGNATURES
Computes the local litterfall rate `Λ_loc` based on the balanced Leaf Area Index `LAI_b` (assuming evergreen PFTs),
Eq. 75, PALADYN (Willeit 2016).
"""
@inline function compute_Λ_loc(vegcarbon_dynamics::PALADYNCarbonDynamics{NF}, LAI_b) where {NF}
    Λ_loc = (
        vegcarbon_dynamics.γL / vegcarbon_dynamics.SLA +
            vegcarbon_dynamics.γR / vegcarbon_dynamics.SLA +
            vegcarbon_dynamics.γS * vegcarbon_dynamics.awl
    ) * LAI_b
    return Λ_loc
end

"""
    $SIGNATURES
Computes the `C_veg` tendency based on `NPP` and the balanced Leaf Area Index `LAI_b`,
Eq. 72, PALADYN (Willeit 2016) 
"""
@inline function compute_C_veg_tend(vegcarbon_dynamics::PALADYNCarbonDynamics{NF}, LAI_b::NF, NPP::NF) where {NF}
    # Compute λ_NPP
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, LAI_b)

    # Compute local litterfall rate
    Λ_loc = compute_Λ_loc(vegcarbon_dynamics, LAI_b)

    # Compute C_veg tendency
    C_veg_tendency = (NF(1.0) - λ_NPP) * NPP - Λ_loc

    return C_veg_tendency
end

# Process methods

function compute_auxiliary!(state, grid, vegcarbon_dynamics::PALADYNCarbonDynamics, args...)
    out = auxiliary_fields(state, vegcarbon_dynamics)
    fields = get_fields(state, vegcarbon_dynamics; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, vegcarbon_dynamics)
    return nothing
end

function compute_tendencies!(state, grid, vegcarbon_dynamics::PALADYNCarbonDynamics, args...)
    out = tendency_fields(state, vegcarbon_dynamics)
    fields = get_fields(state, vegcarbon_dynamics)
    launch!(grid, XY, compute_tendencies_kernel!, out, fields, vegcarbon_dynamics)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the tendency for the carbon vegetation pool given fields `LAI_b` and `NPP`.
"""
@propagate_inbounds function compute_veg_carbon_tendency(i, j, grid, fields, vegcarbon_dynamics::PALADYNCarbonDynamics)
    # Get inputs
    LAI_b = fields.balanced_leaf_area_index[i, j]
    NPP = fields.net_primary_production[i, j]

    # Compute the vegetation carbon pool tendency
    C_veg_tendency = compute_C_veg_tend(vegcarbon_dynamics, LAI_b, NPP)
    return C_veg_tendency
end

"""
    $TYPEDSIGNATURES

Mutating wrapper for [`compute_LAI_b`](@ref) that stores the result in `out.balanced_leaf_area_index`.
"""
@propagate_inbounds function compute_veg_carbon_auxiliary!(out, i, j, grid, fields, vegcarbon_dynamics::PALADYNCarbonDynamics)
    # Compute balanced Leaf Area Index
    out.balanced_leaf_area_index[i, j, 1] = compute_LAI_b(vegcarbon_dynamics, fields.carbon_vegetation[i, j])
    return nothing
end

"""
    $TYPEDSIGNATURES

Calls [`compute_veg_carbon_tendency`](@ref) and stores the result in `out`.
"""
@propagate_inbounds function compute_veg_carbon_tendencies!(tend, i, j, grid, fields, vegcarbon_dynamics::PALADYNCarbonDynamics)
    # Compute and store C_veg tendency
    tend.carbon_vegetation[i, j, 1] = compute_veg_carbon_tendency(i, j, grid, fields, vegcarbon_dynamics)
    return nothing
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, vegcarbon_dynamics::AbstractVegetationCarbonDynamics, args...)
    i, j = @index(Global, NTuple)
    compute_veg_carbon_auxiliary!(out, i, j, grid, fields, vegcarbon_dynamics, args...)
end

@kernel inbounds = true function compute_tendencies_kernel!(tend, grid, fields, vegcarbon_dynamics::AbstractVegetationCarbonDynamics, args...)
    i, j = @index(Global, NTuple)
    compute_veg_carbon_tendencies!(tend, i, j, grid, fields, vegcarbon_dynamics, args...)
end
