"""
    $TYPEDEF

Plant-dependent trait parameters for a single plant functional type (PFT), constant in space.
Default values are those of the needleleaf tree PFT unless noted otherwise.

Properties:
$TYPEDFIELDS

# References

* [willeitPALADYNV10Comprehensive2016](@cite) Willeit & Ganopolski, Geoscientific Model Development (2016)
* [coxDescriptionTRIFFIDDynamic2001](@cite) Cox, Hadley Centre Technical Note (2001)
* [clarkJointUKLand2011](@cite) Clark et al., Geoscientific Model Development (2011)
* [kattgeTRYGlobalDatabase2011](@cite) Kattge et al., Global Change Biology (2011)
* [linOptimalStomatalBehaviour2015](@cite) Lin et al., Nature Climate Change (2015)
"""
@parameterized @kwdef struct PlantTraits{NF} <: AbstractPlantTraits{NF}
    "Specific leaf area (m²/kgC), i.e. leaf area per unit leaf carbon ([kattgeTRYGlobalDatabase2011](@cite))"
    @param specific_leaf_area::NF = 10.0 (units = u"m^2/kg", bounds = Positive)

    "Allometric coefficient (kgC/m²) relating stem carbon to balanced leaf area index, modified from [coxDescriptionTRIFFIDDynamic2001](@cite) to account for bwl=1"
    @param awl::NF = 2.0 (units = u"kg/m^2", bounds = Positive)

    "Minimum Leaf Area Index modified from [clarkJointUKLand2011](@cite)"
    @param minimum_leaf_area_index::NF = 1.0 (bounds = Positive,)

    "Maximum Leaf Area Index modified from [clarkJointUKLand2011](@cite)"
    @param maximum_leaf_area_index::NF = 6.0 (bounds = Positive,)

    "Extinction coefficient for radiation through vegetation"
    @param extinction_coefficient::NF = 0.5 (bounds = Positive,)

    "Snow-free canopy albedo for broadband shortwave radiation"
    @param albedo::NF = 0.15 (bounds = UnitInterval,)

    "Snow-free canopy emissivity for broadband longwave radiation (default value from PALADYN: 0.96)"
    @param emissivity::NF = 0.96 (bounds = UnitInterval,)

    "Leaf turnover rate `γL` ([kattgeTRYGlobalDatabase2011](@cite))"
    @param leaf_turnover_rate::NF = 0.3 (units = u"yr^-1", bounds = Positive)

    "Root turnover rate `γR`"
    @param root_turnover_rate::NF = 0.3 (units = u"yr^-1", bounds = Positive)

    "Stem turnover rate `γS` modified from [clarkJointUKLand2011](@cite)"
    @param stem_turnover_rate::NF = 0.05 (units = u"yr^-1", bounds = Positive)

    "Ratio `aws` of total to respiring stem carbon, [coxDescriptionTRIFFIDDynamic2001](@cite)"
    @param total_to_respiring_stem_carbon_ratio::NF = 10.0 (bounds = Positive,)

    "Slope parameter `g₁` of the optimal stomatal conductance model relating conductance to net assimilation, [linOptimalStomatalBehaviour2015](@cite)"
    @param stomatal_slope::NF = 2.3 (units = u"kPa^(1/2)", bounds = Positive)

    "Lower temperature limit `T_CO2_low` of photosynthesis, below which it rapidly declines to zero"
    @param photosynthesis_minimum_temperature::NF = -4.0 (units = u"°C",)

    "Upper temperature limit `T_CO2_high` of photosynthesis, above which it rapidly declines to zero"
    @param photosynthesis_maximum_temperature::NF = 42.0 (units = u"°C",)

    "Lower bound `T_photos_low` of the optimal temperature range for photosynthesis"
    @param photosynthesis_optimum_temperature_low::NF = 15.0 (units = u"°C",)

    "Upper bound `T_photos_high` of the optimal temperature range for photosynthesis"
    @param photosynthesis_optimum_temperature_high::NF = 30.0 (units = u"°C",)

    "Fraction `f_deciduous` of deciduous behavior (0 = evergreen, 1 = fully deciduous); placeholder for a temperate cold-deciduous PFT"
    @param deciduous_fraction::NF = 1.0 (bounds = UnitInterval,)

    "Base temperature `T_gdd_base` for growing-degree-day accumulation and senescence onset; placeholder for a temperate cold-deciduous PFT"
    @param growing_degree_day_base_temperature::NF = 5.0 (units = u"°C",)

    "Critical growing-degree-day sum `gdd_crit` for full leaf-out; placeholder for a temperate cold-deciduous PFT"
    @param critical_growing_degree_days::NF = 200.0 (units = u"K*d", bounds = Positive)

    "Air-temperature range `T_senescence_range` below the base temperature over which leaves fully senesce; placeholder for a temperate cold-deciduous PFT"
    @param senescence_temperature_range::NF = 5.0 (units = u"K", bounds = Positive)
end

PlantTraits(::Type{NF}; kwargs...) where {NF} = PlantTraits{NF}(; kwargs...)

"""
    $TYPEDSIGNATURES

Return the plant traits at grid cell `i, j`. Since `PlantTraits` is constant in space, this
returns `traits` itself.
"""
@inline plant_traits(i, j, grid, fields, traits::PlantTraits) = traits

"""
    $TYPEDSIGNATURES

Return the leaf area index at grid point `i, j` relative to the maximum leaf area index of the given
plant `traits`, `LAI / LAI_max`, clamped to the unit interval.
"""
@propagate_inbounds function compute_relative_leaf_area(i, j, grid, fields, traits::AbstractPlantTraits{NF}) where {NF}
    local_traits = plant_traits(i, j, grid, fields, traits)
    LAI = fields.leaf_area_index[i, j, end]
    LAI_max = local_traits.maximum_leaf_area_index
    return clamp(LAI / LAI_max, zero(NF), one(NF))
end
