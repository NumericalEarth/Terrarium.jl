# Note: maybe change the name later, if the PALADYN autotrophic respiration approach has a more specific name
"""
    $TYPEDEF

Autotrophic respiration implementation from PALADYN (Willeit 2016).

Authors: Maha Badri and Matteo Willeit

Properties:
$TYPEDFIELDS
"""
@kwdef struct PALADYNAutotrophicRespiration{NF} <: AbstractAutotrophicRespiration{NF}
    # TODO check physical meaning of this parameter + add unit
    "Sapwood parameter"
    cn_sapwood::NF = 330.0

    # TODO check physical meaning of this parameter + add unit
    "Root parameter"
    cn_root::NF = 29.0

    "Ratio of total to respiring stem carbon, Cox 2001, PFT specific [-]"
    aws::NF = 10.0 # Value for Needleleaf tree PFT
end

PALADYNAutotrophicRespiration(::Type{NF}; kwargs...) where {NF} = PALADYNAutotrophicRespiration{NF}(; kwargs...)

variables(::PALADYNAutotrophicRespiration) = (
    auxiliary(:autotrophic_respiration, XY(), units = u"kg/m^2/s"), # Autotrophic respiration [kgC/m²/s]
    auxiliary(:net_primary_production, XY(), units = u"kg/m^2/s"), # Net Primary Production [kgC/m²/s]
    input(:gross_primary_production, XY(), units = u"kg/m^2/s"), # Gross Primary Production [kgC/m²/s]
    input(:daily_leaf_respiration, XY(), units = u"g/m^2/s"), # Daily leaf respiration [gC/m²/s]
    input(:phenology_factor, XY()), # Phenology factor [-]
    input(:ground_temperature, XY(), default = 10.0, units = u"°C"), # Ground surface temperature [°C]
)

"""
    $SIGNATURES

Computes temperature factors `f_temp_air` and `f_temp_soil` for autotrophic respiration.
"""
@inline function compute_f_temp(
        autoresp::PALADYNAutotrophicRespiration{NF},
        T_air::NF,
        T_soil::NF
    ) where {NF}
    # TODO: These hardcoded constants need to be moved either into the model struct as
    # parameters or into the PhysicalConstants struct
    f_temp(T) = exp(NF(308.56) * (NF(1.0) / NF(56.02) - NF(1.0) / (NF(46.02) + T)))

    # Compute f_temp_soil
    # TODO: This hard bound at 7°C comes from CLIMBER-X/PALADYN but is there not further justified.
    # Maybe these functions can be considered candidates for further improvement or data-driven replacement.
    f_temp_soil = (T_soil > 7) * f_temp(T_soil)

    # Compute f_temp_air
    f_temp_air = f_temp(T_air)

    return f_temp_air, f_temp_soil
end

"""
$SIGNATURES

Computes `resp10` 
"""
@inline function compute_resp10(autoresp::PALADYNAutotrophicRespiration{NF}) where {NF}
    # TODO check physical meaning of this variable + add unit
    # TODO add resp10 implementation
    # For now, placeholder as a constant value
    resp10 = NF(0.066)

    return resp10
end

"""
$SIGNATURES

Computes maintenance respiration `Rm` in [kgC/m²/day].
"""
@inline function compute_Rm(
        autoresp::PALADYNAutotrophicRespiration{NF},
        vegcarbon_dynamics::PALADYNCarbonDynamics{NF},
        T_air,
        T_soil,
        Rd,
        phen,
        C_veg
    ) where {NF}

    # Compute f_temp for autotrophic respiration
    f_temp_air, f_temp_soil = compute_f_temp(autoresp, T_air, T_soil)

    # Compute resp10
    resp10 = compute_resp10(autoresp)

    # Compute leaf respiration
    R_leaf = Rd / NF(1000.0) # convert from gC/m²/day to kgC/m²/day

    # Compute stem respiration
    R_stem = resp10 * f_temp_air * (vegcarbon_dynamics.awl * ((NF(2.0) / vegcarbon_dynamics.SLA) + vegcarbon_dynamics.awl)) /
        (C_veg * autoresp.aws * autoresp.cn_sapwood)

    # Compute root respiration
    R_root = resp10 * f_temp_soil * phen * (NF(2.0) / vegcarbon_dynamics.SLA) /
        (vegcarbon_dynamics.SLA * C_veg * autoresp.cn_root)

    # Compute maintenance respiration Rm
    Rm = R_leaf + R_stem + R_root

    return Rm
end

"""
$SIGNATURES

Computes growth respiration `Rg` in [kgC/m²/s].
"""
@inline function compute_Rg(autoresp::PALADYNAutotrophicRespiration{NF}, GPP, Rm) where {NF}
    Rg = NF(0.25) * (GPP - Rm)
    return Rg
end

"""
$SIGNATURES

Computes autotrophic respiration `Ra` as the sum of maintenance respiration `Rm` and growth respiration `Rg` in [kgC/m²/s].
"""
@inline function compute_Ra(autoresp::PALADYNAutotrophicRespiration, vegcarbon_dynamics::PALADYNCarbonDynamics, T_air, T_soil, Rd, phen, C_veg, GPP)
    # Compute Rm, maintenance respiration
    Rm = compute_Rm(autoresp, vegcarbon_dynamics, T_air, T_soil, Rd, phen, C_veg)

    # Compute Rg, growth respiration
    Rg = compute_Rg(autoresp, GPP, Rm)

    # Compute Ra, autotrophic respiration
    Ra = Rm + Rg
    return Ra
end

"""
$SIGNATURES

Computes Net Primary Productivity `NPP` as the difference between Gross Primary Production `GPP` and autotrophic respiration `Ra`
in [kgC/m²/s].
"""
@inline function compute_NPP(autoresp::PALADYNAutotrophicRespiration, GPP, Ra)
    NPP = GPP - Ra
    return NPP
end

# Process methods

function compute_auxiliary!(
        state, grid,
        autoresp::PALADYNAutotrophicRespiration,
        vegcarbon::AbstractVegetationCarbonDynamics,
        atmos::AbstractAtmosphere
    )
    out = auxiliary_fields(state, autoresp)
    fields = get_fields(state, autoresp, vegcarbon, atmos; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, autoresp, vegcarbon, atmos)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute autotrophic respiration following the scheme of PALADYN (Willeit 2016).
"""
@propagate_inbounds function compute_autotrophic_respiration(
        i, j, grid, fields,
        autoresp::PALADYNAutotrophicRespiration,
        vegcarbon_dynamics::PALADYNCarbonDynamics,
        atmos::AbstractAtmosphere
    )
    # Get inputs
    T_air = air_temperature(i, j, grid, fields, atmos)
    T_soil = fields.ground_temperature[i, j]
    Rd = fields.daily_leaf_respiration[i, j]
    phen = fields.phenology_factor[i, j]
    C_veg = fields.carbon_vegetation[i, j]
    GPP = fields.gross_primary_production[i, j]

    # Compute autotrophic respiration Ra
    Ra = compute_Ra(autoresp, vegcarbon_dynamics, T_air, T_soil, Rd, phen, C_veg, GPP)

    # Compute net primary product (NPP)
    NPP = compute_NPP(autoresp, GPP, Ra)

    # Return both Ra and NPP
    return Ra, NPP
end

"""
    $TYPEDSIGNATURES

Mutating wrapper for [`compute_autotrophic_respiration`](@ref) that stores the results in `out`.
"""
@propagate_inbounds function compute_autotrophic_respiration!(out, i, j, grid, fields, autoresp::AbstractAutotrophicRespiration, args...)
    # Compute and store results
    Ra, NPP = compute_autotrophic_respiration(i, j, grid, fields, autoresp, args...)
    out.autotrophic_respiration[i, j, 1] = Ra
    out.net_primary_production[i, j, 1] = NPP
    return out
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, autoresp::AbstractAutotrophicRespiration, args...)
    i, j = @index(Global, NTuple)
    compute_autotrophic_respiration!(out, i, j, grid, fields, autoresp, args...)
end
