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
    "Carbon-to-nitrogen mass ratio of sapwood (kgC/kgN)"
    @param cn_sapwood::NF = 330.0 (bounds = Positive,)

    "Carbon-to-nitrogen mass ratio of fine roots (kgC/kgN)"
    @param cn_root::NF = 29.0 (bounds = Positive,)

    "Ratio of total to respiring stem carbon, [coxDescriptionTRIFFIDDynamic2001](@cite). PFT specific."
    @param aws::NF = 10.0 (bounds = Positive,) # Value for Needleleaf tree PFT

    "Tissue respiration rate at 10 °C per unit tissue nitrogen (kgC/kgN/day), as in [willeitPALADYNV10Comprehensive2016](@cite)"
    @param resp10::NF = 0.066 (units = u"d^-1", bounds = Positive)
end

PALADYNAutotrophicRespiration(::Type{NF}; kwargs...) where {NF} = PALADYNAutotrophicRespiration{NF}(; kwargs...)

variables(::PALADYNAutotrophicRespiration) = (
    auxiliary(:autotrophic_respiration, Canopy(XY()), units = u"kg/m^2/s"), # Autotrophic respiration [kgC/m²/s]
    auxiliary(:net_primary_production, Canopy(XY()), units = u"kg/m^2/s"), # Net Primary Production [kgC/m²/s]
    input(:gross_primary_production, Canopy(XY()), units = u"kg/m^2/s"), # Gross Primary Production [kgC/m²/s]
    input(:daily_leaf_respiration, Canopy(XY()), units = u"g/m^2/s"), # Daily leaf respiration [gC/m²/s]
    input(:ground_temperature, Ground(Top(z = Center())), default = 10, units = u"°C"), # Ground surface temperature [°C]
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

Computes maintenance respiration `Rm` in [kgC/m²/s].
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

    # Tissue respiration rate at 10 °C, converted from per day to per second
    resp10 = autoresp.resp10 / seconds_per_day(NF)

    # Compute leaf respiration
    R_leaf = Rd / NF(1000.0) # convert from gC/m²/day to kgC/m²/day

    # Diagnose the stem and root carbon pools from the single vegetation carbon pool via the
    # allometry (Willeit & Ganopolski, 2016, Eqs. 76-79)
    LAI_b = compute_balanced_leaf_area_index(vegcarbon_dynamics, traits, C_veg)
    C_stem = compute_stem_carbon(vegcarbon_dynamics, traits, LAI_b)
    C_root = compute_root_carbon(vegcarbon_dynamics, traits, LAI_b)

    # Compute stem respiration of the respiring (sapwood) fraction 1/aws of the stem carbon
    R_stem = resp10 * f_temp_air * C_stem / (autoresp.aws * autoresp.cn_sapwood)

    # Compute root respiration
    R_root = resp10 * f_temp_soil * phen * C_root / autoresp.cn_root

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
    out.autotrophic_respiration[i, j, end] = Ra
    out.net_primary_production[i, j, end] = NPP
    return out
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, autoresp::AbstractAutotrophicRespiration, args::Vararg{Any, N}) where {N}
    i, j = @index(Global, NTuple)
    compute_autotrophic_respiration!(out, i, j, grid, fields, autoresp, args...)
end
