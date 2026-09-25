"""
    $TYPEDEF

Prognostic, growing-degree-day (GDD) based vegetation phenology following the cold-deciduous scheme of
[willeitPALADYNV10Comprehensive2016](@cite) and [sitchEvaluationEcosystemDynamics2003](@cite).

The instantaneous leaf area index is `LAI = ϕ·LAI_b`, where `LAI_b` is the balanced (annual-maximum)
leaf area index and `ϕ` is the phenology factor. For a cold-deciduous PFT, `ϕ` ramps up linearly with
accumulated growing degree days above a base temperature `T_gdd_base` at a rate set by `gdd_crit`
[willeitPALADYNV10Comprehensive2016; Eq. (83)](@cite), holds at 1 once fully leafed out, and declines
linearly during senescence as air temperature falls from `T_gdd_base` to `T_gdd_base − T_senescence_range`.

Unlike the discrete daily accumulation of the original scheme in PALADYN, the growing degree days are integrated as
a prognostic state variable, so no history of past inputs needs to be stored. To recover a periodic seasonal
cycle without a discrete annual reset (which would violate the continuous-time design), a cold-season
relaxation term drains the accumulator with timescale `gdd_relaxation_time` whenever air temperature is
below `T_gdd_base`.

The deciduous/evergreen distinction, set in PALADYN by the coldest-month temperature, is represented here
by the prescribed `f_deciduous` fraction: `ϕ = f_deciduous·ϕ_deciduous + (1 − f_deciduous)`, so an evergreen
PFT (`f_deciduous = 0`) has `ϕ = 1` (and `LAI = LAI_b`) independent of temperature.

Authors: Maha Badri and Matteo Willeit

Properties:
$TYPEDFIELDS

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
* [sitchEvaluationEcosystemDynamics2003](@cite) Sitch et al., Global Change Biology (2003)
"""
@parameterized @kwdef struct PALADYNPhenology{NF} <: AbstractPhenology{NF}
    # TODO: PFT-specific values from PALADYN Table 5 (Willeit & Ganopolski, 2016); defaults below are
    # placeholders representative of a temperate cold-deciduous PFT.
    "Base temperature for growing-degree-day accumulation and senescence onset"
    @param T_gdd_base::NF = 5.0 (units = u"°C",)

    "Critical growing-degree-day sum for full leaf-out"
    @param gdd_crit::NF = 200.0 (units = u"K*d", bounds = Positive)

    "Air-temperature range below `T_gdd_base` over which leaves fully senesce"
    @param T_senescence_range::NF = 5.0 (units = u"K", bounds = Positive)

    # Note: this parameter has no counterpart in the original scheme; it replaces the discrete annual
    # GDD reset with a continuous cold-season relaxation. See the type docstring.
    "Cold-season relaxation timescale for the continuous growing-degree-day reset"
    @param gdd_relaxation_time::NF = 30.0 (units = u"d", bounds = Positive)

    "Deciduous fraction (0 = evergreen, 1 = fully deciduous); prescribed per PFT via the coldest-month-temperature criterion"
    @param f_deciduous::NF = 1.0 (bounds = UnitInterval,)
end

PALADYNPhenology(::Type{NF}; kwargs...) where {NF} = PALADYNPhenology{NF}(; kwargs...)

variables(::PALADYNPhenology) = (
    prognostic(:growing_degree_days, Canopy(XY()), units = u"K*d"), # Growing degree days [K⋅day]
    auxiliary(:phenology_factor, Canopy(XY())), # Phenology factor [-]
    auxiliary(:leaf_area_index, Canopy(XY())), # Leaf Area Index [m²/m²]
)

"""
    $SIGNATURES

Compute the phenology factor `ϕ` [-] from accumulated growing degree days `gdd` and air temperature `T_air`.

The cold-deciduous factor combines the growth-phase ramp `gdd / gdd_crit`
[willeitPALADYNV10Comprehensive2016; Eq. (83)](@cite) with a senescence ramp that declines linearly from 1
at `T_gdd_base` to 0 at `T_gdd_base − T_senescence_range`. Taking the minimum reproduces all three PALADYN
regimes (linear green-up, held at 1 when mature and warm, temperature-driven senescence). The result is then
blended with the evergreen value of 1 via the prescribed deciduous fraction.
"""
@inline function compute_phenology_factor(phenol::PALADYNPhenology{NF}, gdd, T_air) where {NF}
    T_base = phenol.T_gdd_base
    # growth-phase ramp: linear in accumulated growing degree days
    growth = gdd / phenol.gdd_crit
    # senescence ramp: linear in air temperature below the base temperature
    senescence = clamp((T_air - (T_base - phenol.T_senescence_range)) / phenol.T_senescence_range, zero(NF), one(NF))
    # deciduous factor: green-up capped by leaf-out (1) and drawn down by senescence
    ϕ_deciduous = min(growth, senescence)
    # blend deciduous and evergreen (ϕ = 1) behavior
    f_deciduous = phenol.f_deciduous
    ϕ = f_deciduous * ϕ_deciduous + (one(NF) - f_deciduous)
    return ϕ
end

"""
    $SIGNATURES

Compute the instantaneous leaf area index `LAI = ϕ·LAI_b` from the phenology factor `ϕ` and the balanced
leaf area index `LAI_b`.
"""
@inline compute_leaf_area_index(phenol::PALADYNPhenology, ϕ, LAI_b) = ϕ * LAI_b

"""
    $SIGNATURES

Compute the growing-degree-day tendency `d(gdd)/dt` [K⋅day/s] at a single grid point.

Heat above the base temperature is accumulated (in degree-days per second) and, below the base temperature,
the accumulator relaxes toward zero with timescale `gdd_relaxation_time`. The relaxation provides a
continuous surrogate for the discrete annual reset of the original scheme.
"""
@inline function compute_gdd_tendency(phenol::PALADYNPhenology{NF}, gdd, T_air) where {NF}
    T_base = phenol.T_gdd_base
    seconds_per_day_NF = seconds_per_day(NF)
    # heat accumulation above base temperature, expressed in degree-days per second
    accumulation = max(T_air - T_base, zero(NF)) / seconds_per_day_NF
    # cold-season relaxation toward zero (continuous reset) when below base temperature
    is_cold = T_air < T_base
    relaxation = ifelse(is_cold, gdd / (phenol.gdd_relaxation_time * seconds_per_day_NF), zero(NF))
    return accumulation - relaxation
end

# Top-level interface methods

""" $TYPEDSIGNATURES """
function compute_auxiliary!(state, grid, phenol::PALADYNPhenology, vegcarbon::PALADYNCarbonDynamics, atmos::AbstractAtmosphere)
    out = auxiliary_fields(state, phenol)
    fields = get_fields(state, phenol, vegcarbon, atmos; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, phenol, atmos)
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_tendencies!(state, grid, phenol::PALADYNPhenology, atmos::AbstractAtmosphere)
    tend = tendency_fields(state, phenol)
    fields = get_fields(state, phenol, atmos)
    launch!(grid, XY, compute_tendencies_kernel!, tend, fields, phenol, atmos)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the phenology factor and instantaneous leaf area index (LAI) at a single grid point.
"""
@propagate_inbounds function compute_phenology(i, j, grid, fields, phenol::PALADYNPhenology, atmos::AbstractAtmosphere)
    # Get inputs and prognostic state
    gdd = fields.growing_degree_days[i, j]
    LAI_b = fields.balanced_leaf_area_index[i, j]
    T_air = air_temperature(i, j, grid, fields, atmos)

    # Compute phenology factor and LAI
    ϕ = compute_phenology_factor(phenol, gdd, T_air)
    LAI = compute_leaf_area_index(phenol, ϕ, LAI_b)

    return ϕ, LAI
end

"""
    $TYPEDSIGNATURES

Mutating wrapper for [`compute_phenology`](@ref) that stores the result in `out`.
"""
@propagate_inbounds function compute_phenology!(out, i, j, grid, fields, phenol::PALADYNPhenology, atmos::AbstractAtmosphere)
    ϕ, LAI = compute_phenology(i, j, grid, fields, phenol, atmos)
    out.phenology_factor[i, j, end] = ϕ
    out.leaf_area_index[i, j, end] = LAI
    return out
end

"""
    $TYPEDSIGNATURES

Mutating wrapper for [`compute_gdd_tendency`](@ref) that stores the growing-degree-day tendency in `tend`.
"""
@propagate_inbounds function compute_gdd_tendency!(tend, i, j, grid, fields, phenol::PALADYNPhenology, atmos::AbstractAtmosphere)
    gdd = fields.growing_degree_days[i, j]
    T_air = air_temperature(i, j, grid, fields, atmos)
    tend.growing_degree_days[i, j, end] = compute_gdd_tendency(phenol, gdd, T_air)
    return tend
end

# Kernels

@kernel function compute_auxiliary_kernel!(out, grid, fields, phenol::AbstractPhenology, args...)
    i, j = @index(Global, NTuple)
    compute_phenology!(out, i, j, grid, fields, phenol, args...)
end

@kernel function compute_tendencies_kernel!(tend, grid, fields, phenol::PALADYNPhenology, args...)
    i, j = @index(Global, NTuple)
    compute_gdd_tendency!(tend, i, j, grid, fields, phenol, args...)
end
