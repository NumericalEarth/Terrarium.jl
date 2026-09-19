"""
    $TYPEDEF

Vegetation carbon dynamics implementation following [willeitPALADYNV10Comprehensive2016](@cite) but considering only the sum of the vegetation
carbon pools. The subsequent splitting into C_leaf, C_stem, C_root is not implemented for now.

Authors: Maha Badri

Properties:
$TYPEDFIELDS

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
* [coxDescriptionTRIFFIDDynamic2001](@cite) Cox, Hadley Centre Technical Note (2001)
* [clarkJointUKLand2011](@cite) Clark et al., Geoscientific Model Development (2011)
* [kattgeTRYGlobalDatabase2011](@cite) Kattge et al., Global Change Biology (2011)
"""
@parameterized @kwdef struct PALADYNCarbonDynamics{NF} <: AbstractVegetationCarbonDynamics{NF}
    "Leaf turnover rate ([kattgeTRYGlobalDatabase2011](@cite)). PFT specific."
    @param γL::NF = 0.3 (units = u"yr^-1", bounds = Positive) # Value for Needleleaf tree PFT

    "Root turnover rate. PFT specific."
    @param γR::NF = 0.3 (units = u"yr^-1", bounds = Positive) # Value for Needleleaf tree PFT

    "Stem src/processes/vegetation/hydraulicsturnover rate modified from [clarkJointUKLand2011](@cite). PFT specific."
    @param γS::NF = 0.05 (units = u"yr^-1", bounds = Positive) # Value for Needleleaf tree PFT
end

PALADYNCarbonDynamics(::Type{NF}; kwargs...) where {NF} = PALADYNCarbonDynamics{NF}(; kwargs...)

variables(::PALADYNCarbonDynamics) = (
    prognostic(:carbon_vegetation, Canopy(XY()), units = u"kg/m^2"), # Vegetation carbon pool [kgC/m²]
    auxiliary(:balanced_leaf_area_index, Canopy(XY())), # Balanced Leaf Area Index [m²/m²]
    input(:net_primary_production, Canopy(XY()), units = u"kg/m^2/s"), # Net Primary Production [kgC/m²/s]
)

"""
    $SIGNATURES

Computes `λ_NPP`,a factor determining the partitioning of NPP between increase of vegetation carbon of the existing 
vegetated area and spreading of the given PFT based on the balanced Leaf Area Index `LAI_b`,
[willeitPALADYNV10Comprehensive2016; Eq. (74)](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_λ_NPP(vegcarbon_dynamics::PALADYNCarbonDynamics{NF}, traits::PlantTraits{NF}, LAI_b) where {NF}
    LAI_min = traits.minimum_leaf_area_index
    LAI_max = traits.maximum_leaf_area_index
    λ_NPP = ifelse(
        LAI_b < LAI_min,
        zero(NF),
        ifelse(
            LAI_b <= LAI_max,
            (LAI_b - LAI_min) / (LAI_max - LAI_min),
            one(NF)
        )
    )
    return λ_NPP
end

"""
    $SIGNATURES

Computes `LAI_b`, the balanced Leaf Area Index based on the vegetation carbon pool `C_veg` (assuming with bwl = 1),
[willeitPALADYNV10Comprehensive2016; Eqs. (76-79)](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_balanced_leaf_area_index(vegcarbon_dynamics::PALADYNCarbonDynamics{NF}, traits::PlantTraits{NF}, C_veg) where {NF}
    SLA = traits.specific_leaf_area
    awl = traits.awl
    LAI_b = C_veg / ((NF(2.0) / SLA) + awl)
    return LAI_b
end

"""
    $SIGNATURES
Computes the local litterfall rate `Λ_loc` based on the balanced Leaf Area Index `LAI_b` (assuming evergreen PFTs),
[willeitPALADYNV10Comprehensive2016; Eq. (75)](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_Λ_loc(vegcarbon_dynamics::PALADYNCarbonDynamics{NF}, traits::PlantTraits{NF}, LAI_b) where {NF}
    # TODO: Change tendencies to units of per day and add automatic unit normalization
    γL = vegcarbon_dynamics.γL / (365.25 * 24 * 3600) # convert to seconds
    γR = vegcarbon_dynamics.γR / (365.25 * 24 * 3600)
    γS = vegcarbon_dynamics.γS / (365.25 * 24 * 3600)
    Λ_loc = LAI_b * (
        γL / traits.specific_leaf_area +
            γR / traits.specific_leaf_area +
            γS * traits.awl
    )
    return Λ_loc
end

"""
    $SIGNATURES
Computes the `C_veg` tendency based on `NPP` and the balanced Leaf Area Index `LAI_b`,
[willeitPALADYNV10Comprehensive2016; Eq. (72)](@cite)

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016) 
"""
@inline function compute_C_veg_tend(vegcarbon_dynamics::PALADYNCarbonDynamics{NF}, traits::PlantTraits{NF}, LAI_b::NF, NPP::NF) where {NF}
    λ_NPP = compute_λ_NPP(vegcarbon_dynamics, traits, LAI_b)
    Λ_loc = compute_Λ_loc(vegcarbon_dynamics, traits, LAI_b)
    C_veg_tendency = (NF(1.0) - λ_NPP) * NPP - Λ_loc
    return C_veg_tendency
end

# Top-level interface methods

""" $TYPEDSIGNATURES """
function compute_auxiliary!(state, grid, vegcarbon_dynamics::PALADYNCarbonDynamics, traits::PlantTraits, args...)
    out = auxiliary_fields(state, vegcarbon_dynamics)
    fields = get_fields(state, vegcarbon_dynamics; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, vegcarbon_dynamics, traits)
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_tendencies!(state, grid, vegcarbon_dynamics::PALADYNCarbonDynamics, traits::PlantTraits, args...)
    out = tendency_fields(state, vegcarbon_dynamics)
    fields = get_fields(state, vegcarbon_dynamics)
    launch!(grid, XY, compute_tendencies_kernel!, out, fields, vegcarbon_dynamics, traits)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the tendency for the carbon vegetation pool given fields `LAI_b` and `NPP`.
"""
@propagate_inbounds function compute_veg_carbon_tendency(i, j, grid, fields, vegcarbon_dynamics::PALADYNCarbonDynamics, traits::PlantTraits)
    # Get inputs
    LAI_b = fields.balanced_leaf_area_index[i, j]
    NPP = fields.net_primary_production[i, j]

    # Compute the vegetation carbon pool tendency
    C_veg_tendency = compute_C_veg_tend(vegcarbon_dynamics, traits, LAI_b, NPP)
    return C_veg_tendency
end

"""
    $TYPEDSIGNATURES

Mutating wrapper for [`compute_balanced_leaf_area_index`](@ref) that stores the result in `out.balanced_leaf_area_index`.
"""
@propagate_inbounds function compute_veg_carbon_auxiliary!(
        out, i, j, grid, fields,
        vegcarbon_dynamics::PALADYNCarbonDynamics,
        traits::PlantTraits
    )
    # Compute balanced Leaf Area Index
    out.balanced_leaf_area_index[i, j, end] = compute_balanced_leaf_area_index(vegcarbon_dynamics, traits, fields.carbon_vegetation[i, j])
    return nothing
end

"""
    $TYPEDSIGNATURES

Calls [`compute_veg_carbon_tendency`](@ref) and stores the result in `out`.
"""
@propagate_inbounds function compute_veg_carbon_tendencies!(
        tend, i, j, grid, fields,
        vegcarbon_dynamics::PALADYNCarbonDynamics,
        traits::PlantTraits
    )
    # Compute and store C_veg tendency
    tend.carbon_vegetation[i, j, end] = compute_veg_carbon_tendency(i, j, grid, fields, vegcarbon_dynamics, traits)
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
