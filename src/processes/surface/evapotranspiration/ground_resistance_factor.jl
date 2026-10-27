# Ground resistance to evaporation

"""
    $TYPEDEF

Represents a spatiotemporally constant ground evaporation resistance factor.
"""
@parameterized @kwdef struct ConstantEvaporationResistanceFactor{NF} <: AbstractGroundEvaporationResistanceFactor
    "Unit interval factor that determines resistance to evaporation; zero corresponds to no evaporation"
    @param factor::NF = 1.0 (bounds = UnitInterval,)
end

ConstantEvaporationResistanceFactor(::Type{NF}; kwargs...) where {NF} = ConstantEvaporationResistanceFactor{NF}(; kwargs...)

@inline ground_evaporation_resistance_factor(i, j, grid, fields, res::ConstantEvaporationResistanceFactor, args...) = res.factor

"""
    $TYPEDEF

Implements the soil moisture limiting resistance factor of [leeEstimatingSoilSurface1992](@cite),

```math
\\beta =
\\frac{1}{4} \\left[1 - \\cos\\left(π \\theta_1/\\theta_{\\text{fc}} \\right)\\right] \\quad \\text{for } \\theta_1 < \\theta_{\\text{fc}}
```
otherwise ``\\beta=1``.

# References

* [leeEstimatingSoilSurface1992](@cite) Lee and Pielke, Journal of Applied Meteorology (1992)
"""
struct SoilMoistureResistanceFactor{NF} <: AbstractGroundEvaporationResistanceFactor end

SoilMoistureResistanceFactor(::Type{NF}) where {NF} = SoilMoistureResistanceFactor{NF}()

# Fallback implementation for interface consistency
ground_evaporation_resistance_factor(i, j, grid, fields, ::SoilMoistureResistanceFactor{NF}, args...) where {NF} = one(NF)

@inline function ground_evaporation_resistance_factor(
        i, j, grid, fields,
        res::SoilMoistureResistanceFactor{NF},
        soil::AbstractSoil
    ) where {NF}
    ground_grid = ground_domain(grid)
    strat = get_stratigraphy(soil)
    hydrology = get_hydrology(soil)
    bgc = get_biogeochemistry(soil)
    props = get_hydraulic_properties(hydrology)
    comp = soil_composition(i, j, ground_grid.Nz, grid, fields, strat, hydrology, bgc)
    texture = mineral_texture(comp)
    fracs = volumetric_fractions(comp)
    # Get field capacity, water content, and residual water content
    θfc = field_capacity(get_hydraulic_properties(hydrology), texture)
    θw = fracs.water
    por = porosity(comp)
    θres = props.residual_saturation * por
    return ground_evaporation_resistance_factor(res, θw, θfc, θres)
end

@inline function ground_evaporation_resistance_factor(::SoilMoistureResistanceFactor{NF}, θw, θfc, θres) where {NF}
    if θw < θfc
        fc_sat = ifelse(θfc > θres, (θw - θres) / (θfc - θres), zero(NF)) # guard against edge case where θfc == θres
        β = (1 - cos(π * fc_sat))^2 / 4
    else
        β = NF(1)
    end
    return β
end
