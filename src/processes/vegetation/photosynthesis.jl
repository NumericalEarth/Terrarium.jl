"""
    $TYPEDEF
Photosynthesis implementation from PALADYN (Willeit 2016) for C3 PFTs following the mechanistic
approach of Haxeltine and Prentice (1996). Computes instantaneous photosynthetic rates as differential
equations that are integrated over arbitrary timesteps by the timestepper.

Authors: Maha Badri and Matteo Willeit

Properties:
$TYPEDFIELDS
"""
@kwdef struct LUEPhotosynthesis{NF} <: AbstractPhotosynthesis{NF}
    "Rubisco specificity factor at 25°C [dimensionless]. Ratio of carboxylation to oxygenation rates."
    τ25::NF = 2600.0

    "Michaelis-Menten constant for CO₂ at 25°C [Pa]. PALADYN value for needleleaf trees."
    Kc25::NF = 30.0

    "Michaelis-Menten constant for O₂ at 25°C [Pa]. PALADYN value for needleleaf trees."
    Ko25::NF = 3.0e4

    "Q10 temperature sensitivity for τ [dimensionless]. Controls temperature dependence of specificity."
    q10_τ::NF = 0.57

    "Q10 temperature sensitivity for Kc [dimensionless]. Controls temperature dependence of CO₂ affinity."
    q10_Kc::NF = 2.1

    "Q10 temperature sensitivity for Ko [dimensionless]. Controls temperature dependence of O₂ affinity."
    q10_Ko::NF = 1.2

    "Leaf albedo in PAR range [-]"
    α_leaf::NF = 0.17

    "Fraction of PAR assimilated at ecosystem level, relative to leaf level [-]"
    α_a::NF = 0.5

    "Intrinsic quantum efficiency of CO2 uptake in C3 plants [mol/mol]"
    α_C3::NF = 0.08

    "Conversion factor for solar radiation at 550 nm from J/m² to mol/m² [mol/J]"
    cq::NF = 4.6e-6

    "Extinction coefficient for radiation through vegetation [-]"
    k_ext::NF = 0.5

    "Upper temperature threshold for CO₂/O₂ specificity factor [°C]. Above this, photosynthesis rapidly declines. PFT-specific, needleleaf tree value."
    T_CO2_high::NF = 42.0

    "Lower temperature threshold for CO₂/O₂ specificity factor [°C]. Below this, photosynthesis rapidly declines. PFT-specific, needleleaf tree value."
    T_CO2_low::NF = -4.0

    "Upper temperature threshold for light-limited photosynthesis rate [°C]. Peak photosynthesis capacity. PFT-specific, needleleaf tree value."
    T_photos_high::NF = 30.0

    "Lower temperature threshold for light-limited photosynthesis rate [°C]. Minimum for photosynthesis. PFT-specific, needleleaf tree value."
    T_photos_low::NF = 15.0

    "Root of quadratic mean shape parameter [-]. Controls smoothness of interpolation between light and RuBisCO limitations (0.7 for smooth, 0.5 for arithmetic mean)."
    θ_r::NF = 0.7
end

LUEPhotosynthesis(::Type{NF}; kwargs...) where {NF} = LUEPhotosynthesis{NF}(; kwargs...)

variables(::LUEPhotosynthesis{NF}) where {NF} = (
    auxiliary(:net_assimilation, XY(), units = u"g/m^2/s"), # Net photosynthesis rate [gC/m²/s]
    auxiliary(:leaf_respiration, XY(), units = u"g/m^2/s"), # Leaf respiration rate [gC/m²/s]
    auxiliary(:gross_primary_production, XY(), units = u"kg/m^2/s"), # Gross primary production rate [kgC/m²/s]
    input(:soil_moisture_limiting_factor, XY(), default = NF(1)), # soil moisture limiting factor with default value of 1
    input(:leaf_area_index, XY()), # Leaf Area Index [m²/m²]
)

"""
    $TYPEDSIGNATURES
Computes kinetic parameters `τ`, `Kc`, `Ko` based on temperature using Q10 temperature response.
Follows enzyme kinetics from Haxeltine & Prentice (1996), Appendix C.
- τ: Rubisco specificity factor (CO₂ to O₂ carboxylation ratio), dimensionless
- Kc: Michaelis-Menten constant for CO₂ [Pa]
- Ko: Michaelis-Menten constant for O₂ [Pa]
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
Follows Eq. C6, PALADYN (Willeit 2016).
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
Eq. 62, PALADYN (Willeit 2016).
"""
@inline function compute_APAR(photo::LUEPhotosynthesis{NF}, swdown::NF, LAI::NF) where {NF}
    PAR = compute_PAR(photo, swdown)
    APAR = photo.α_a * PAR * (NF(1.0) - exp(-photo.k_ext * LAI))
    return APAR
end

"""
    $TYPEDSIGNATURES
Computes intercellular CO2 partial pressure [Pa],
Eq. 67, PALADYN (Willeit 2016).
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
Follows Eq. C4-C5 from PALADYN (Willeit 2016) and Haxeltine & Prentice (1996).
- c_1: quantum efficiency × temperature factor × carbon mass / denominator, used in JE = c_1 × APAR
- c_2: dimensionless coefficient relating enzyme capacity to assimilation, used in JC = c_2 × Vc_max
"""
@inline function compute_assimilation_factors(
        photo::LUEPhotosynthesis{NF},
        constants::PhysicalConstants{NF},
        Γ_star::NF, T_stress::NF, Kc::NF, Ko::NF, pres_i::NF, pres_O2::NF
    ) where {NF}
    # The factor of 2 in the c_1 denominator relates to the partial pressure terms in the compensation point.
    c_1 = photo.α_C3 * T_stress * constants.C_mass * (pres_i - Γ_star) / (pres_i + NF(2.0) * Γ_star)
    c_2 = (pres_i - Γ_star) / (pres_i + Kc * (NF(1.0) + pres_O2 / Ko))
    return c_1, c_2
end

"""
    $TYPEDSIGNATURES
    
Computes the maximum rate of net photosynthesis `Vc_max` [gC/m²/s],
following the coordination hypothesis (acclimation), see Harrison 2021 Box 2.
Note: this is not the same formula in PALADYN paper, this implementaion is taken from the code
"""
@inline function compute_Vc_max(photo::LUEPhotosynthesis{NF}, c_1::NF, PAR::NF, Kc::NF, Ko::NF, Γ_star::NF, pres_i::NF, pres_O2::NF) where {NF}
    Vc_max = c_1 * PAR * (pres_i + Kc * (NF(1.0) + pres_O2 / Ko)) / (pres_i - Γ_star)
    return Vc_max
end

"""
    $TYPEDSIGNATURES

Computes the PAR-limited and the rubisco-activity-limited photosynthesis rates `JE` and `JC` [gC/m²/s],
Eqn 3+5, Haxeltine & Prentice 1996.
"""
@inline function compute_JE_JC(photo::LUEPhotosynthesis{NF}, c_1::NF, c_2::NF, APAR::NF, Vc_max::NF) where {NF}
    JE = c_1 * APAR
    JC = c_2 * Vc_max
    return JE, JC
end

"""
    $TYPEDSIGNATURES

Computes the leaf respiration rate `Rd` [gC/m²/s],
Eqn 10, Haxeltine & Prentice 1996 and Eq. 10 PALADYN (Willeit 2016).
"""
@inline function compute_Rd(photo::LUEPhotosynthesis, Vc_max::NF, β::NF) where {NF}
    Rd = photo.α_C3 * Vc_max * β
    return Rd
end

"""
    $TYPEDSIGNATURES

Computes the gross photosynthesis rate `Ag` [gC/m²/s],
Eqn 2, Haxeltine & Prentice 1996
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
        constants::PhysicalConstants{NF},
        T_air::NF, swdown::NF, pres::NF, co2::NF, LAI::NF, λc::NF, β::NF,
    ) where {NF}
    # Compute partial CO2 and O2 pressures in [Pa]
    pres_O2 = partial_pressure_O2(pres)
    pres_a = partial_pressure_CO2(pres, co2)

    # Minimum light and temperature thresholds for photosynthesis
    # No photosynthesis occurs below -3°C or without incident shortwave radiation
    if swdown > zero(NF) && T_air > NF(-3.0)
        # Compute kinetic parameters: Rubisco specificity τ, and Michaelis-Menten constants Kc, Ko
        τ, Kc, Ko = compute_kinetic_parameters(photo, T_air)

        # Compute CO₂ compensation point in Pa
        Γ_star = compute_Γ_star(photo, τ, pres_O2)

        # Only compute photosynthesis if there is leaf area (LAI)
        if LAI > zero(NF)

            # Compute absorbed PAR [mol/m²/s]
            APAR = compute_APAR(photo, swdown, LAI)

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

# Process methods

function compute_auxiliary!(
        state, grid,
        photo::LUEPhotosynthesis,
        stomcond::AbstractStomatalConductance,
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants,
        args...
    )
    out = auxiliary_fields(state, photo)
    fields = get_fields(state, photo, stomcond, atmos; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, photo, atmos, constants)
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
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants
    )
    # Get inputs from fields/atmosphere
    T_air = air_temperature(i, j, grid, fields, atmos)
    pres = air_pressure(i, j, grid, fields, atmos)
    swdown = shortwave_down(i, j, grid, fields, atmos)
    co2 = fields.CO2[i, j]
    β = fields.soil_moisture_limiting_factor[i, j]
    LAI = fields.leaf_area_index[i, j]
    λc = fields.leaf_to_air_co2_ratio[i, j]

    # Compute Rd, leaf respiration rate in [gC/m²/s],
    # An, net photosynthesis rate in [gC/m²/s]
    Rd, An = compute_respiration_assimilation(photo, constants, T_air, swdown, pres, co2, LAI, λc, β)

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
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants
    )
    Rd, An, GPP = compute_photosynthesis(i, j, grid, fields, photo, atmos, constants)
    out.leaf_respiration[i, j, 1] = Rd
    out.net_assimilation[i, j, 1] = An
    out.gross_primary_production[i, j, 1] = GPP
    return out
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, photo::AbstractPhotosynthesis, args...)
    i, j = @index(Global, NTuple)
    compute_photosynthesis!(out, i, j, grid, fields, photo, args...)
end
