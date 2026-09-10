# Material-agnostic enthalpy-temperature relations
#
# These scalar maps relate volumetric internal energy `U` (J/m³) to the unfrozen (liquid) water
# fraction θ_w/θ, , with θ the sum of the volumetric fractions of water and ice, and to temperature,
# given the volumetric latent-heat content `ρLθ` (J/m³) and the volumetric heat capacity `C` (J/m³/K).

"""
    $TYPEDSIGNATURES

Calculate the unfrozen water content from the given internal energy `U` (J/m³) and 
volumetric latent heat content `ρLθ` (J/m³).
"""
@inline function liquid_water_fraction(::FreeWater, U::NF, ρLθ::NF) where {NF}
    # Case 1: U ≥ 0          → thawed (liq = 1)
    # Case 2a: -ρLθ ≤ U < 0  → phase change (liq = 1 - U/(-ρLθ))
    # Case 2b: U < -ρLθ      → frozen (liq = 0), enforced by the (U ≥ -ρLθ) factor.
    return ifelse(
        U >= zero(U),
        NF(1),
        (U >= -ρLθ) * (NF(1) - safediv(U, -ρLθ)),
    )
end

"""
    $TYPEDSIGNATURES
    
Calculate the inverse enthalpy function given the internal energy `U` (J/m³), 
volumetric latent heat content `ρLθ` (J/m³), and volumetric heat capacity `C` (J/m³/K)
under the free water freezing characteristic.
"""
@inline function energy_to_temperature(::FreeWater, U::NF, ρLθ::NF, C::NF) where {NF}
    # Case 1:  U < -ρLθ      → frozen       (T = (U + ρLθ)/C)
    # Case 2a: U ≥ 0         → thawed       (T = U/C)
    # Case 2b: -ρLθ ≤ U < 0  → phase change (T = 0)
    return ifelse(
        U < -ρLθ,
        (U + ρLθ) / C,
        ifelse(U >= zero(U), U / C, zero(NF)),
    )
end
