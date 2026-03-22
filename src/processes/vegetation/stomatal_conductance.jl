"""
    $TYPEDEF
Stomatal conductance implementation from PALADYN (Willeit 2016) following the optimal stomatal conductance model
(Medlyn et al. 2011).

Authors: Maha Badri and Matteo Willeit

Properties:
$TYPEDFIELDS
"""
@kwdef struct MedlynStomatalConductance{NF} <: AbstractStomatalConductance{NF}
    "Parameter in optimal stomatal conductance formulation representing the quasi-linear
    relationship between conductance and net assimilation, Lin et al. 2015 [-], PFT specific"
    g₁::NF = 2.3 # TODO: value for Needleleaf tree PFT

    "Minimum stomatal condutance parameter [mm s⁻¹]"
    g_min::NF = 0.5
end

MedlynStomatalConductance(::Type{NF}; kwargs...) where {NF} = MedlynStomatalConductance(; kwargs...)

variables(::MedlynStomatalConductance) = (
    auxiliary(:canopy_water_conductance, XY(), units = u"m/s"), # Canopy conducatance for water vapor
    auxiliary(:leaf_to_air_co2_ratio, XY()), # Ratio of leaf-internal and air CO2 concentration [-]
)

@inline @propagate_inbounds stomatal_conductance(i, j, grid, fields, ::MedlynStomatalConductance) = fields.canopy_water_conductance[i, j]

"""
    $TYPEDSIGNATURES

Compute canopy-level water conductance [m/s] from the Medlyn et al. (2011) optimal stomatal conductance model.
Includes minimum conductance and light extinction effects based on LAI, scaled by soil moisture factor β.
"""
@inline function compute_gw_can(
        stomcond::MedlynStomatalConductance{NF},
        photo::LUEPhotosynthesis{NF},
        vpd, An, co2, LAI, β
    ) where {NF}
    # Preconditions
    @assert isfinite(vpd) && abs(vpd) > zero(NF) "vapor pressure deficit must be greater than zero"
    @assert isfinite(An) "An must be finite"
    @assert isfinite(co2) && co2 > zero(NF) "CO2 must be positive and finite"
    @assert isfinite(LAI) "LAI must be finite"
    @assert isfinite(β) && 0 <= β <= 1 "β must be finite and between 0 and 1"
    # Compute stomatal conductance gw_can
    let g_min = stomcond.g_min / 1000, # convert mm/s to m/s
            g₁ = stomcond.g₁,
            k_ext = photo.k_ext
        g₀ = g_min * (1 - exp(-k_ext * LAI)) * β
        gw_can = g₀ + (1 + g₁ / sqrt(vpd)) * An / co2 * NF(1.0e6)
        return gw_can
    end
end

"""
    $SIGNATURES

Computes the ratio of leaf-internal and air CO2 concentration `λc`, 
derived from the optimal stomatal conductance model (Medlyn et al. 2011),
Eq. 71, PALADYN (Willeit 2016).
"""
@inline function compute_λc(stomcond::MedlynStomatalConductance{NF}, vpd) where {NF}
    λc = NF(1.0) - NF(1.6) / (NF(1.0) + stomcond.g₁ / sqrt(vpd * NF(1.0e-3)))
    return λc
end

# Process methods

function compute_auxiliary!(
        state, grid,
        stomcond::MedlynStomatalConductance,
        photo::LUEPhotosynthesis,
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants,
        args...
    )
    out = auxiliary_fields(state, stomcond)
    fields = get_fields(state, stomcond, photo, atmos, constants; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, stomcond, photo, atmos, constants)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute stomatal conductance (gw_can) and leaf-to-air CO₂ ratio (λc) at a grid point.
Returns tuple (gw_can, λc) for use in photosynthesis and transpiration calculations.
"""
@propagate_inbounds function compute_stomatal_conductance(i, j, grid, fields, stomcond::MedlynStomatalConductance{NF}, photo::LUEPhotosynthesis{NF}, atmos::AbstractAtmosphere, constants::PhysicalConstants, args...) where {NF}
    # Get inputs
    An = fields.net_assimilation[i, j]
    CO2 = fields.CO2[i, j]
    LAI = fields.leaf_area_index[i, j]
    β = fields.soil_moisture_limiting_factor[i, j]

    # Compute vpd [Pa]
    vpd = compute_vpd(i, j, grid, fields, atmos, constants)

    # Compute conductance gw_can and internal CO2 ratio λc
    gw_can = compute_gw_can(stomcond, photo, vpd, An, CO2, LAI, β)
    λc = compute_λc(stomcond, vpd)

    return gw_can, λc
end

"""
    $TYPEDSIGNATURES

Calls [`compute_stomatal_conductance`](@ref) and stores the result in `out`.
"""
@propagate_inbounds function compute_stomatal_conductance!(out, i, j, grid, fields, stomcond::MedlynStomatalConductance{NF}, photo::LUEPhotosynthesis{NF}, atmos::AbstractAtmosphere, constants::PhysicalConstants, args...) where {NF}
    gw_can, λc = compute_stomatal_conductance(i, j, grid, fields, stomcond, photo, atmos, constants, args...)
    out.canopy_water_conductance[i, j, 1] = gw_can
    out.leaf_to_air_co2_ratio[i, j, 1] = λc
    return out
end

# Kernels

@kernel function compute_auxiliary_kernel!(out, grid, fields, stomcond::AbstractStomatalConductance, args...)
    i, j = @index(Global, NTuple)
    compute_stomatal_conductance!(out, i, j, grid, fields, stomcond, args...)
end
