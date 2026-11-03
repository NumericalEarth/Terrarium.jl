abstract type AbstractSnowThermalConductivity{NF} end

number_format(::Type{<:AbstractSnowThermalConductivity{NF}}) where {NF} = NF

"""
    $TYPEDEF

Power law parameterization for snow thermal conductivity as a function of density following [yenReviewThermalProperties1981; Eq. (34)](@cite).

Properties:
$TYPEDFIELDS

# References
* [yenReviewThermalProperties1981](@cite) Yen et al. 1981
"""
@parameterized @kwdef struct PowerLawSnowThermalConductivity{NF} <: AbstractSnowThermalConductivity{NF}
    "Coefficient `a` in the thermal conductivity power law `κ = a·(ρ_snow/ρ_w)^b`"
    @param conductivity_coefficient::NF = 2.22362 (units = u"W/m/K", bounds = Positive)

    "Exponent `b` in the thermal conductivity power law `κ = a·(ρ_snow/ρ_w)^b`"
    @param conductivity_exponent::NF = 1.885 (bounds = Positive,)
end

PowerLawSnowThermalConductivity(::Type{NF}; kwargs...) where {NF} = PowerLawSnowThermalConductivity{NF}(; kwargs...)


"""
    $TYPEDSIGNATURES

Bulk snow thermal conductivity via the density power law `κ_snow = a·(ρ_snow/ρ_w)^b` ([yenReviewThermalProperties1981](@cite)).
"""
@inline function compute_thermal_conductivity(
        cond::PowerLawSnowThermalConductivity{NF},
        constants::MaterialConstants{NF},
        ρ_snow::NF
    ) where {NF}
    a = cond.conductivity_coefficient
    b = cond.conductivity_exponent
    ρ_w = constants.density_water
    κ_s = a * (ρ_snow / ρ_w)^b
    return κ_s
end

"""
    $TYPEDEF

Logarithmic snow thermal conductivity parameterization of [sturmThermalConductivitySeasonal1997](@cite).

Properties:
$TYPEDFIELDS

# References
* [sturmThermalConductivitySeasonal1997](@cite)
"""
@parameterized @kwdef struct LogarithmicSnowThermalConductivity{NF} <: AbstractSnowThermalConductivity{NF}
    @param scale::NF = 2.65
    @param shift::NF = -1.652
end

LogarithmicSnowThermalConductivity(::Type{NF}; kwargs...) where {NF} = LogarithmicSnowThermalConductivity{NF}(; kwargs...)

function compute_thermal_conductivity(
        cond::LogarithmicSnowThermalConductivity,
        constants::MaterialConstants{NF},
        ρ_snow::NF
    ) where {NF}
    a, b = cond.scale, cond.shift
    # convert density to g/cm^3
    ρ_snow = ustrip(u"g/cm^3", ρ_snow * u"kg/m^3")
    κ_s = 10^(a * ρ_snow + b)
    return κ_s
end

"""
    $TYPEDEF

Piecewise quadratic snow thermal conductivity parameterization of [sturmThermalConductivitySeasonal1997](@cite).
Default conductivity of ice from [wellerNewDataThermal1971](@cite). 

Properties:
$TYPEDFIELDS

# References
* [sturmThermalConductivitySeasonal1997](@cite) Sturm et al., Journal of Glaciology (1997)
* [wellerNewDataThermal1971](@cite) Weller & Schwerdtfeger, Journal of Glaciology (1971)
"""
@parameterized @kwdef struct QuadraticSnowThermalConductivity{NF} <: AbstractSnowThermalConductivity{NF}
    @component func_hi::QuadraticFunction{NF} = QuadraticFunction(a = 3.233, b = -1.01, c = 0.138)
    @component func_lo::QuadraticFunction{NF} = QuadraticFunction(a = 0.0, b = 0.234, c = 0.023)
    @param threshold::NF = 0.156 (units = u"g/cm^3", bounds = UnitInterval)
    @param κ_max::NF = 2.2 (units = u"W/m/K", bounds = Positive) # assumed conductivity of ice
end

QuadraticSnowThermalConductivity(::Type{NF}; kwargs...) where {NF} = QuadraticSnowThermalConductivity{NF}(; kwargs...)

function compute_thermal_conductivity(
        cond::QuadraticSnowThermalConductivity,
        constants::MaterialConstants{NF},
        ρ_snow::NF
    ) where {NF}
    # convert density to g/cm^3
    ρ_snow = ustrip(u"g/cm^3", ρ_snow * u"kg/m^3")
    func = ifelse(ρ_snow >= cond.threshold, cond.func_hi, cond.func_lo)
    # calculate conductivity
    κ_s = func(ρ_snow)
    return min(κ_s, cond.κ_max)
end
