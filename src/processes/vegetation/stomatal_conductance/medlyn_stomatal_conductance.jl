"""
    $TYPEDEF
Stomatal conductance implementation from [willeitPALADYNV10Comprehensive2016](@cite) following the optimal stomatal conductance model
of [medlynReconcilingOptimalEmpirical2011](@cite).

Authors: Maha Badri and Matteo Willeit

Properties:
$TYPEDFIELDS

# References

* [linOptimalStomatalBehaviour2015](@cite) Lin et al., Nature Climate Change (2015)
* [medlynReconcilingOptimalEmpirical2011](@cite) Medlyn et al., Global Change Biology (2011)
* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@parameterized @kwdef struct MedlynStomatalConductance{NF} <: AbstractStomatalConductance{NF}
    "Parameter in optimal stomatal conductance formulation representing the quasi-linear
    relationship between conductance and net assimilation, [linOptimalStomatalBehaviour2015](@cite). PFT specific."
    @param g₁::NF = 2.3 (bounds = Positive,) # TODO: value for Needleleaf tree PFT

    "Minimum stomatal conductance parameter"
    @param g_min::NF = 0.5 (units = u"mm/s", bounds = Positive)

    "Diffusivity ratio of water to CO₂"
    diffusivity_ratio_water_co2::NF = 1.6
end

MedlynStomatalConductance(::Type{NF}; kwargs...) where {NF} = MedlynStomatalConductance{NF}(; kwargs...)

variables(::MedlynStomatalConductance) = (
    auxiliary(:canopy_water_conductance, Canopy(XY()), units = u"m/s"), # Canopy conducatance for water vapor
    input(:leaf_area_index, Canopy(XY()), units = u"m^2/m^2"),
    input(:net_assimilation, Canopy(XY()), units = u"g/m^2/s"), # Net photosynthesis rate [gC/m²/s]
    input(:soil_moisture_limiting_factor, Ground(XY())),
)

@inline @propagate_inbounds stomatal_conductance(i, j, grid, fields, ::MedlynStomatalConductance) = fields.canopy_water_conductance[i, j]

"""
    $TYPEDSIGNATURES

Compute canopy-level water conductance [m/s] from the [medlynReconcilingOptimalEmpirical2011](@cite) optimal stomatal conductance model.
Includes minimum conductance and light extinction effects based on LAI, scaled by soil moisture factor β.

# References

* [medlynReconcilingOptimalEmpirical2011](@cite) Medlyn et al., Global Change Biology (2011)
"""
@inline function compute_stomatal_conductance(
        stomcond::MedlynStomatalConductance{NF},
        traits::PlantTraits{NF},
        constants::PhysicalConstants{NF},
        vpd, T_air, pres, co2_ppm, An, LAI, β
    ) where {NF}
    # Compute stomatal conductance g_stm
    let g_min = stomcond.g_min / NF(1.0e3) # convert mm/s to m/s
        g₁ = stomcond.g₁
        D = stomcond.diffusivity_ratio_water_co2
        k_ext = traits.extinction_coefficient
        # We clamp VPD from below at 10 Pa (0.01 kPa) for numerical stability (division by zero risk)
        vpd = max(vpd, NF(10.0))
        # We similarly clamp net assimilation from below at zero;
        # this assumes that the stomata are closed during respiration (An < 0) and ensures that
        # stomatal conductance remains nonnegative
        An = max(An, NF(0))
        cₐ = ppm_to_mole_fraction(co2_ppm) # convert CO₂ concentration to mole fraction
        # Compute minimum conductance g₀ from g_min and LAI
        g₀ = g_min * (1 - exp(-k_ext * LAI)) * β # m/s
        # Collect constants for An conversion factor
        M_C = constants.material.atomic_weight_carbon # atomic weight of carbon in gC/mol
        M_air = constants.material.molecular_weight_dry_air / NF(1.0e3)
        R = constants.thermodynamics.gas_constant_dry_air * M_air # Universal gas constant in J/(mol·K)
        T_K = celsius_to_kelvin(constants.thermodynamics, T_air)
        # Compute ideal gas conversion factor in m³/mol
        F = R * T_K / pres
        # Dimensionless scaling coefficient (from Medlyn model)
        b = D * (1 + g₁ / sqrt(vpd))
        # Stomatal conductance [m/s] as g₀ [m/s] + b * An / cₐ [gC/m²/s] / M_C [gC/mol] × F [m³/mol]
        # An is converted from gC/m²/s → mol C/m²/s, then to m/s via ideal gas law
        g_stm = g₀ + b * An / cₐ / M_C * F
        return g_stm
    end
end

"""
    $SIGNATURES

Computes the ratio of leaf-internal and air CO2 concentration `λc`, 
derived from the optimal stomatal conductance model ([medlynReconcilingOptimalEmpirical2011](@cite)),
[willeitPALADYNV10Comprehensive2016; Eq. (71)](@cite).

# References

* [medlynReconcilingOptimalEmpirical2011](@cite) Medlyn et al., Global Change Biology (2011)
* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_λc(stomcond::MedlynStomatalConductance{NF}, vpd) where {NF}
    # here we allow zero VPD since lim x⁻¹ as x → ∞ ≈ 0
    g₁ = stomcond.g₁
    D = stomcond.diffusivity_ratio_water_co2
    λc = NF(1.0) - D / (NF(1.0) + g₁ / sqrt(vpd * NF(1.0e-3)))
    return λc
end

# Top-level interface methods

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        stomcond::MedlynStomatalConductance,
        traits::PlantTraits,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    )
    out = auxiliary_fields(state, stomcond)
    fields = get_fields(state, stomcond, traits, constants, atmos; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, stomcond, traits, constants, atmos)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute stomatal conductance (g_stm) and leaf-to-air CO₂ ratio (λc) at a grid point.
Returns tuple (g_stm, λc) for use in photosynthesis and transpiration calculations.
"""
@propagate_inbounds function compute_stomatal_conductance(
        i, j, grid, fields,
        stomcond::MedlynStomatalConductance{NF},
        traits::PlantTraits{NF},
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    ) where {NF}
    # Get inputs
    An = fields.net_assimilation[i, j]
    CO2 = fields.CO2[i, j]
    LAI = fields.leaf_area_index[i, j]
    β = fields.soil_moisture_limiting_factor[i, j]
    Tair = air_temperature(i, j, grid, fields, atmos)
    pres = air_pressure(i, j, grid, fields, atmos)

    # Compute vpd [Pa]
    vpd = compute_vapor_pressure_deficit(i, j, grid, fields, atmos, constants)

    # Compute conductance g_stm and internal CO2 ratio λc
    g_stm = compute_stomatal_conductance(stomcond, traits, constants, vpd, Tair, pres, CO2, An, LAI, β)

    return g_stm
end

"""
    $TYPEDSIGNATURES

Calls [`compute_stomatal_conductance`](@ref) and stores the result in `out`.
"""
@propagate_inbounds function compute_stomatal_conductance!(
        out, i, j, grid, fields,
        stomcond::MedlynStomatalConductance{NF},
        traits::PlantTraits{NF},
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    ) where {NF}
    g_stm = compute_stomatal_conductance(i, j, grid, fields, stomcond, traits, constants, atmos, args...)
    out.canopy_water_conductance[i, j, end] = g_stm
    return out
end

# Kernels

@kernel function compute_auxiliary_kernel!(out, grid, fields, stomcond::AbstractStomatalConductance, args...)
    i, j = @index(Global, NTuple)
    compute_stomatal_conductance!(out, i, j, grid, fields, stomcond, args...)
end
