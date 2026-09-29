"""
    $TYPEDEF
Photosynthesis implementation from PALADYN [willeitPALADYNV10Comprehensive2016](@cite) for C3 PFTs following the mechanistic
approach of [haxeltineBIOME3EquilibriumTerrestrial1996](@cite). Computes instantaneous photosynthetic rates as differential
equations that are integrated over arbitrary timesteps by the timestepper.

Authors: Maha Badri and Matteo Willeit

Properties:
$TYPEDFIELDS

# References

* [haxeltineBIOME3EquilibriumTerrestrial1996](@cite) Haxeltine & Prentice, Global Biogeochemical Cycles (1996)
* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)

"""
@parameterized @kwdef struct LUEPhotosynthesis{NF} <: AbstractPhotosynthesis{NF}
    "Rubisco specificity factor at 25°C. Ratio of carboxylation to oxygenation rates."
    @param τ25::NF = 2600.0 (bounds = Positive,)

    "Michaelis-Menten constant for CO₂ at 25°C. PALADYN value for needleleaf trees."
    @param Kc25::NF = 30.0 (units = u"Pa", bounds = Positive)

    "Michaelis-Menten constant for O₂ at 25°C. PALADYN value for needleleaf trees."
    @param Ko25::NF = 3.0e4 (units = u"Pa", bounds = Positive)

    "Q10 temperature sensitivity for τ. Controls temperature dependence of specificity."
    @param q10_τ::NF = 0.57 (bounds = Positive,)

    "Q10 temperature sensitivity for Kc. Controls temperature dependence of CO₂ affinity."
    @param q10_Kc::NF = 2.1 (bounds = Positive,)

    "Q10 temperature sensitivity for Ko. Controls temperature dependence of O₂ affinity."
    @param q10_Ko::NF = 1.2 (bounds = Positive,)

    "Leaf albedo in PAR range"
    @param α_leaf::NF = 0.17 (bounds = UnitInterval,)

    "Fraction of PAR assimilated at ecosystem level, relative to leaf level"
    @param α_a::NF = 0.5 (bounds = UnitInterval,)

    "Intrinsic quantum efficiency of CO2 uptake in C3 plants"
    @param α_C3::NF = 0.08 (bounds = UnitInterval,)

    "Conversion factor for solar radiation at 550 nm from J/m² to mol/m²"
    @param cq::NF = 4.6e-6 (units = u"mol/J", bounds = Positive, scale = 1.0e-6)

    "Upper temperature threshold for CO₂/O₂ specificity factor. Above this, photosynthesis rapidly declines. PFT-specific, needleleaf tree value."
    @param T_CO2_high::NF = 42.0 (units = u"°C",)

    "Lower temperature threshold for CO₂/O₂ specificity factor. Below this, photosynthesis rapidly declines. PFT-specific, needleleaf tree value."
    @param T_CO2_low::NF = -4.0 (units = u"°C",)

    "Upper temperature threshold for light-limited photosynthesis rate. Peak photosynthesis capacity. PFT-specific, needleleaf tree value."
    @param T_photos_high::NF = 30.0 (units = u"°C",)

    "Lower temperature threshold for light-limited photosynthesis rate. Minimum for photosynthesis. PFT-specific, needleleaf tree value."
    @param T_photos_low::NF = 15.0 (units = u"°C",)

    "Root of quadratic mean shape parameter. Controls smoothness of interpolation between light and RuBisCO limitations (0.7 for smooth, 0.5 for arithmetic mean)."
    @param θ_r::NF = 0.7 (bounds = UnitInterval,)
end

LUEPhotosynthesis(::Type{NF}; kwargs...) where {NF} = LUEPhotosynthesis{NF}(; kwargs...)

variables(::LUEPhotosynthesis{NF}) where {NF} = (
    auxiliary(:net_assimilation, Canopy(XY()), units = u"g/m^2/s"), # Net photosynthesis rate [gC/m²/s]
    auxiliary(:leaf_respiration, Canopy(XY()), units = u"g/m^2/s"), # Leaf respiration rate [gC/m²/s]
    auxiliary(:gross_primary_production, Canopy(XY()), units = u"kg/m^2/s"), # Gross primary production rate [kgC/m²/s]
    input(:soil_moisture_limiting_factor, Ground(XY()), default = NF(1)), # soil moisture limiting factor with default value of 1
    input(:leaf_area_index, Canopy(XY())), # Leaf Area Index [m²/m²]
)

"""
    $TYPEDSIGNATURES
Computes kinetic parameters `τ`, `Kc`, `Ko` based on temperature using Q10 temperature response.
Follows enzyme kinetics from [haxeltineBIOME3EquilibriumTerrestrial1996](@cite), Appendix C.

- τ: Rubisco specificity factor (CO₂ to O₂ carboxylation ratio), dimensionless
- Kc: Michaelis-Menten constant for CO₂ [Pa]
- Ko: Michaelis-Menten constant for O₂ [Pa]

# References

* [haxeltineBIOME3EquilibriumTerrestrial1996](@cite) Haxeltine & Prentice, Global Biogeochemical Cycles (1996)
"""
@inline function compute_kinetic_parameters(photo::LUEPhotosynthesis{NF}, T_air::NF) where {NF}
    τ = photo.τ25 * photo.q10_τ^((T_air - NF(25.0)) * NF(0.1))
    Kc = photo.Kc25 * photo.q10_Kc^((T_air - NF(25.0)) * NF(0.1))
    Ko = photo.Ko25 * photo.q10_Ko^((T_air - NF(25.0)) * NF(0.1))
    return τ, Kc, Ko
end


"""
    $TYPEDSIGNATURES
Computes the CO₂ compensation point `Γ_star` [Pa].
The intercellular CO₂ partial pressure at which gross photosynthesis equals respiration.
Follows [willeitPALADYNV10Comprehensive2016; Eq. (C6)](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_Γ_star(photo::LUEPhotosynthesis{NF}, τ::NF, pres_O2::NF) where {NF}
    Γ_star = pres_O2 / (NF(2.0) * τ)
    return Γ_star
end

"""
    $TYPEDSIGNATURES
Computes NET Photosynthetically Active Radiation `PAR` [mol/m²/s].
"""
@inline function compute_PAR(photo::LUEPhotosynthesis{NF}, swdown::NF) where {NF}
    PAR = NF(0.5) * swdown * (NF(1.0) - photo.α_leaf) * photo.cq
    return PAR
end

"""
    $TYPEDSIGNATURES
Computes absorbed PAR limited by the fraction of PAR assimilated at ecosystem level `APAR` [mol/m²/s],
[willeitPALADYNV10Comprehensive2016; Eq. (62)](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_APAR(photo::LUEPhotosynthesis{NF}, traits::PlantTraits{NF}, swdown::NF, LAI::NF) where {NF}
    PAR = compute_PAR(photo, swdown)
    k_ext = traits.extinction_coefficient
    APAR = photo.α_a * PAR * (NF(1.0) - exp(-k_ext * LAI))
    return APAR
end

"""
    $TYPEDSIGNATURES
Computes intercellular CO2 partial pressure [Pa],
[willeitPALADYNV10Comprehensive2016; Eq. (67)](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_pres_i(photo::LUEPhotosynthesis, λc, pres_a)
    pres_i = λc * pres_a
    return pres_i
end

"""
    $TYPEDSIGNATURES

Computes the temperature stress factor `T_stress` based on the air temperature.
"""
@inline function compute_temperature_stress(photo::LUEPhotosynthesis{NF}, T_air::NF) where {NF}
    # Temperature stress function implements a double-sigmoid response to temperature.
    # T_CO2_low/high: thresholds for Rubisco specificity (τ) optimal activity range
    # T_photos_low/high: thresholds for light-limited photosynthesis (JE) optimal activity range
    # k1: slope parameter for lower sigmoid (temperature rise from T_CO2_low to T_photos_low)
    # Derived from logistic function to transition from 0.01 to 0.99 over the temperature interval.
    k1 = NF(2.0) * log(NF(1.0) / NF(0.99) - NF(1.0)) / (photo.T_CO2_low - photo.T_photos_low)

    # k2: inflection point (midpoint) of the lower sigmoid
    # Set to the average of the two lower thresholds for symmetric transition.
    k2 = NF(0.5) * (photo.T_CO2_low + photo.T_photos_low)

    # k3: slope parameter for upper sigmoid (temperature rise from T_photos_high to T_CO2_high)
    # Controls how rapidly photosynthesis declines above the optimal temperature range.
    k3 = log(NF(0.99) / NF(0.01)) / (photo.T_CO2_high - photo.T_photos_high)

    # Compute T_stress as product of lower and upper sigmoid curves
    # T_stress varies from 0 (outside optimal range) to ~1 (within optimal range [T_photos_low, T_photos_high])
    if photo.T_CO2_low < T_air < photo.T_CO2_high
        low = NF(1.0) / (NF(1.0) + exp(k1 * (k2 - T_air)))
        high = NF(1.0) - NF(0.01) * exp(k3 * (T_air - photo.T_photos_high))
        T_stress = low * high
    else
        T_stress = zero(NF)
    end

    return T_stress
end

"""
    $TYPEDSIGNATURES

Computes factors for light-limited `c_1` [gC/mol] and RuBisCO-limited `c_2` [dimensionless] assimilation.
Follows [willeitPALADYNV10Comprehensive2016; Eqs. (C4-C5)](@cite) and [haxeltineBIOME3EquilibriumTerrestrial1996](@cite).

- c_1: quantum efficiency × temperature factor × carbon mass / denominator, used in JE = c_1 × APAR
- c_2: dimensionless coefficient relating enzyme capacity to assimilation, used in JC = c_2 × Vc_max

# References

* [haxeltineBIOME3EquilibriumTerrestrial1996](@cite) Haxeltine & Prentice, Global Biogeochemical Cycles (1996)
* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)H
"""
@inline function compute_assimilation_factors(
        photo::LUEPhotosynthesis{NF},
        constants::MaterialConstants{NF},
        Γ_star::NF, T_stress::NF, Kc::NF, Ko::NF, pres_i::NF, pres_O2::NF
    ) where {NF}
    # The factor of 2 in the c_1 denominator relates to the partial pressure terms in the compensation point.
    c_1 = photo.α_C3 * T_stress * constants.atomic_weight_carbon * (pres_i - Γ_star) / (pres_i + NF(2.0) * Γ_star)
    c_2 = (pres_i - Γ_star) / (pres_i + Kc * (NF(1.0) + pres_O2 / Ko))
    return c_1, c_2
end

"""
    $TYPEDSIGNATURES
    
Computes the maximum rate of net photosynthesis `Vc_max` [gC/m²/s],
following the coordination hypothesis (acclimation), see [harrisonEcoEvolutionaryOptimalityMeans2021](@cite) Box 2.
Note: this is not the same formula as in the [willeitPALADYNV10Comprehensive2016](@cite) paper, this implementaion is taken from the code

# References

* [harrisonEcoEvolutionaryOptimalityMeans2021](@cite) Harrison et al., New Phytologist (2021)
* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_Vc_max(photo::LUEPhotosynthesis{NF}, c_1::NF, PAR::NF, Kc::NF, Ko::NF, Γ_star::NF, pres_i::NF, pres_O2::NF) where {NF}
    Vc_max = c_1 * PAR * (pres_i + Kc * (NF(1.0) + pres_O2 / Ko)) / (pres_i - Γ_star)
    return Vc_max
end

"""
    $TYPEDSIGNATURES

Computes the PAR-limited and the rubisco-activity-limited photosynthesis rates `JE` and `JC` [gC/m²/s],
[haxeltineBIOME3EquilibriumTerrestrial1996; Eqs. (3, 5)](@cite).

# References

* [haxeltineBIOME3EquilibriumTerrestrial1996](@cite) Haxeltine & Prentice, Global Biogeochemical Cycles (1996)
"""
@inline function compute_JE_JC(photo::LUEPhotosynthesis{NF}, c_1::NF, c_2::NF, APAR::NF, Vc_max::NF) where {NF}
    JE = c_1 * APAR
    JC = c_2 * Vc_max
    return JE, JC
end

"""
    $TYPEDSIGNATURES

Computes the leaf respiration rate `Rd` [gC/m²/s],
[haxeltineBIOME3EquilibriumTerrestrial1996; Eq. (10)](@cite) and [willeitPALADYNV10Comprehensive2016; Eq. (10)](@cite).

# References

* [haxeltineBIOME3EquilibriumTerrestrial1996](@cite) Haxeltine & Prentice, Global Biogeochemical Cycles (1996)
* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@inline function compute_Rd(photo::LUEPhotosynthesis, Vc_max::NF, β::NF) where {NF}
    Rd = photo.α_C3 * Vc_max * β
    return Rd
end

"""
    $TYPEDSIGNATURES

Computes the gross photosynthesis rate `Ag` [gC/m²/s],
[haxeltineBIOME3EquilibriumTerrestrial1996; Eq. (2)](@cite).

# References

* [haxeltineBIOME3EquilibriumTerrestrial1996](@cite) Haxeltine & Prentice, Global Biogeochemical Cycles (1996)
"""
@inline function compute_Ag(photo::LUEPhotosynthesis{NF}, c_1::NF, c_2::NF, APAR::NF, Vc_max::NF, β::NF) where {NF}
    # Compute JE and Jc, PAR-limited and rubisco-activity-limited photosynthesis rates
    JE, JC = compute_JE_JC(photo, c_1, c_2, APAR, Vc_max)

    # TODO photosyntheis downregulation ignored for now
    Ag = (JE + JC - sqrt((JE + JC)^2 - NF(4) * photo.θ_r * JE * JC)) / (NF(2) * photo.θ_r) * β
    return Ag
end

"""
    $TYPEDSIGNATURES

Compute and return leaf respiration [gC/m²/s] and net assimilation [gC/m²/s] rates.
"""
function compute_respiration_assimilation(
        photo::LUEPhotosynthesis{NF},
        traits::PlantTraits{NF},
        constants::MaterialConstants{NF},
        T_air::NF, swdown::NF, pres::NF, co2::NF, LAI::NF, λc::NF, β::NF,
    ) where {NF}
    # Compute partial CO2 and O2 pressures in [Pa]
    pres_O2 = partial_pressure_O2(pres)
    pres_a = partial_pressure_CO2(pres, co2)

    # Minimum light and temperature thresholds for photosynthesis
    # No photosynthesis occurs below -3°C or without incident shortwave radiation
    # TODO: Move this threshold parameter to the struct
    if swdown > zero(NF) && T_air > NF(-3.0)
        # Compute kinetic parameters: Rubisco specificity τ, and Michaelis-Menten constants Kc, Ko
        τ, Kc, Ko = compute_kinetic_parameters(photo, T_air)

        # Compute CO₂ compensation point in Pa
        Γ_star = compute_Γ_star(photo, τ, pres_O2)

        # Only compute photosynthesis if there is leaf area (LAI)
        if LAI > zero(NF)

            # Compute absorbed PAR [mol/m²/s]
            APAR = compute_APAR(photo, traits, swdown, LAI)

            # Compute pres_i, intercellular CO2 partial pressure [Pa]
            pres_i = compute_pres_i(photo, λc, pres_a)

            # Compute temperature stress
            T_stress = compute_temperature_stress(photo, T_air)

            # Compute c1 and c2 parameters for C3 photosynthesis enzyme kinetics
            c_1, c_2 = compute_assimilation_factors(photo, constants, Γ_star, T_stress, Kc, Ko, pres_i, pres_O2)

            # Compute Vc_max, maximum rate of carboxylation [gC/m²/s]
            Vc_max = compute_Vc_max(photo, c_1, APAR, Kc, Ko, Γ_star, pres_i, pres_O2)

            # Compute leaf respiration rate [gC/m²/s]
            Rd = compute_Rd(photo, Vc_max, β)

            # Compute Ag, the gross photosynthesis rate [gC/m²/s]
            Ag = compute_Ag(photo, c_1, c_2, APAR, Vc_max, β)

            # Compute An, the net photosynthesis rate [gC/m²/s]
            An = Ag - Rd
        else
            # No leaves, no photosynthesis
            An = zero(NF)
            # No leaves, no leaf respiration
            Rd = zero(NF)
        end
    else
        # No net photosynthesis without sufficient light or warmth
        An = zero(NF)
        # Note: Leaf respiration (Rd) could continue in darkness, but without photosynthesis
        # computation we cannot compute Vc_max and thus Rd. This represents a simplified
        # assumption that respiration is negligible relative to photosynthesis in lit conditions.
        Rd = zero(NF)
    end

    return Rd, An
end

"""
    $TYPEDSIGNATURES

Compute the Gross Primary Production rate [kgC/m²/s].
"""
@inline function compute_GPP(::LUEPhotosynthesis{NF}, An::NF) where {NF}
    # Convert from gC/m²/s to kgC/m²/s
    GPP = An * NF(1.0e-3)
    return GPP
end

# Top-level interface methods

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        photo::LUEPhotosynthesis,
        stomcond::AbstractStomatalConductance,
        traits::PlantTraits,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    )
    out = auxiliary_fields(state, photo)
    # N.B. in this specific case, we do not need state fields from the stomatal condctuance dependency
    fields = get_fields(state, photo, atmos; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, photo, stomcond, traits, constants, atmos, args...)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute photosynthesis, leaf respiration, and gross primary production at a single grid point.
Returns instantaneous rates in [gC/m²/s] and [kgC/m²/s] for integration by the timestepper.
"""
@propagate_inbounds function compute_photosynthesis(
        i, j, grid, fields,
        photo::LUEPhotosynthesis,
        stomcond::AbstractStomatalConductance,
        traits::PlantTraits,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    # Get inputs from atmosphere
    T_air = air_temperature(i, j, grid, fields, atmos)
    pres = air_pressure(i, j, grid, fields, atmos)
    swdown = shortwave_down(i, j, grid, fields, atmos)
    co2 = ambient_co2(i, j, grid, fields, atmos)

    # Get vegetation inputs
    β = fields.soil_moisture_limiting_factor[i, j]
    LAI = fields.leaf_area_index[i, j]

    # Compute VPD + leaf-to-air CO2 ratio λc
    vpd = compute_vapor_pressure_deficit(i, j, grid, fields, atmos, constants)
    λc = compute_λc(stomcond, vpd)

    # Compute Rd, leaf respiration rate in [gC/m²/s],
    # An, net photosynthesis rate in [gC/m²/s]
    Rd, An = compute_respiration_assimilation(photo, traits, constants.material, T_air, swdown, pres, co2, LAI, λc, β)

    # Compute GPP, Gross Primary Production in [kgC/m²/s]
    GPP = compute_GPP(photo, An)

    return Rd, An, GPP
end

"""
    $TYPEDSIGNATURES

Calls [`compute_photosynthesis`](@ref) and stores the results in `out`.
"""
@propagate_inbounds function compute_photosynthesis!(
        out, i, j, grid, fields,
        photo::LUEPhotosynthesis,
        stomcond::AbstractStomatalConductance,
        traits::PlantTraits,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    Rd, An, GPP = compute_photosynthesis(i, j, grid, fields, photo, stomcond, traits, constants, atmos)
    out.leaf_respiration[i, j, end] = Rd
    out.net_assimilation[i, j, end] = An
    out.gross_primary_production[i, j, end] = GPP
    return out
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, photo::AbstractPhotosynthesis, args...)
    i, j = @index(Global, NTuple)
    compute_photosynthesis!(out, i, j, grid, fields, photo, args...)
end
