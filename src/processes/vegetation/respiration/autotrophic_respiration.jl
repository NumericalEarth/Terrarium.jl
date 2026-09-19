# Note: maybe change the name later, if the PALADYN autotrophic respiration approach has a more specific name
"""
    $TYPEDEF

Autotrophic respiration implementation from [willeitPALADYNV10Comprehensive2016](@cite).

Authors: Maha Badri and Matteo Willeit

Properties:
$TYPEDFIELDS

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
* [coxDescriptionTRIFFIDDynamic2001](@cite) Cox, Hadley Centre Technical Note (2001)
"""
@parameterized @kwdef struct PALADYNAutotrophicRespiration{NF} <: AbstractAutotrophicRespiration{NF}
    # TODO check physical meaning of this parameter + add unit
    "Sapwood parameter"
    @param cn_sapwood::NF = 330.0 (bounds = Positive,)

    # TODO check physical meaning of this parameter + add unit
    "Root parameter"
    @param cn_root::NF = 29.0 (bounds = Positive,)

    "Ratio of total to respiring stem carbon, [coxDescriptionTRIFFIDDynamic2001](@cite). PFT specific."
    @param aws::NF = 10.0 (bounds = Positive,) # Value for Needleleaf tree PFT
end

PALADYNAutotrophicRespiration(::Type{NF}; kwargs...) where {NF} = PALADYNAutotrophicRespiration{NF}(; kwargs...)

variables(::PALADYNAutotrophicRespiration) = (
    auxiliary(:autotrophic_respiration, Canopy(XY()), units = u"kg/m^2/s"), # Autotrophic respiration [kgC/m²/s]
    auxiliary(:net_primary_production, Canopy(XY()), units = u"kg/m^2/s"), # Net Primary Production [kgC/m²/s]
    input(:gross_primary_production, Canopy(XY()), units = u"kg/m^2/s"), # Gross Primary Production [kgC/m²/s]
    input(:daily_leaf_respiration, Canopy(XY()), units = u"g/m^2/s"), # Daily leaf respiration [gC/m²/s]
    input(:ground_temperature, Ground(Top(z = Center())), default = 10.0, units = u"°C"), # Ground surface temperature [°C]
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

    # Compute soil temperature factor
    f_temp_soil = f_temp(T_soil)

    # Compute air temperature factor
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
        traits::PlantTraits{NF},
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
    R_stem = resp10 * f_temp_air * (traits.awl * ((NF(2.0) / traits.specific_leaf_area) + traits.awl)) /
        (C_veg * autoresp.aws * autoresp.cn_sapwood)

    # Compute root respiration
    R_root = resp10 * f_temp_soil * phen * (NF(2.0) / traits.specific_leaf_area) /
        (traits.specific_leaf_area * C_veg * autoresp.cn_root)

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
@inline function compute_Ra(autoresp::PALADYNAutotrophicRespiration, vegcarbon_dynamics::PALADYNCarbonDynamics, traits::PlantTraits, T_air, T_soil, Rd, phen, C_veg, GPP)
    Rm = compute_Rm(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, Rd, phen, C_veg)
    Rg = compute_Rg(autoresp, GPP, Rm)
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

# Top-level interface methods

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        autoresp::PALADYNAutotrophicRespiration,
        vegcarbon::AbstractVegetationCarbonDynamics,
        phenology::AbstractPhenology,
        traits::PlantTraits,
        atmos::AbstractAtmosphere
    )
    out = auxiliary_fields(state, autoresp)
    fields = get_fields(state, autoresp, vegcarbon, phenology, atmos; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, autoresp, vegcarbon, phenology, traits, atmos)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute autotrophic respiration following the scheme of [willeitPALADYNV10Comprehensive2016](@cite).

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
"""
@propagate_inbounds function compute_autotrophic_respiration(
        i, j, grid, fields,
        autoresp::PALADYNAutotrophicRespiration,
        vegcarbon_dynamics::PALADYNCarbonDynamics,
        phenology::AbstractPhenology,
        traits::PlantTraits,
        atmos::AbstractAtmosphere
    )
    T_air = air_temperature(i, j, grid, fields, atmos)
    T_soil = fields.ground_temperature[i, j]
    Rd = fields.daily_leaf_respiration[i, j]
    phen = fields.phenology_factor[i, j]
    C_veg = fields.carbon_vegetation[i, j]
    GPP = fields.gross_primary_production[i, j]
    Ra = compute_Ra(autoresp, vegcarbon_dynamics, traits, T_air, T_soil, Rd, phen, C_veg, GPP)
    NPP = compute_NPP(autoresp, GPP, Ra)
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
