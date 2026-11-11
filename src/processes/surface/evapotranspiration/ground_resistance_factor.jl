# Ground resistance to evaporation

"""
    $TYPEDEF

Represents a spatiotemporally constant ground evaporation resistance factor.

Note that this factor does not limit ground evaporation by the soil water supply: with a factor greater
than zero, the ground evaporates at the same fraction of its potential rate regardless of how dry the soil
is. It is therefore only suitable where soil water is not limiting; use [`SoilMoistureResistanceFactor`](@ref)
otherwise.
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

    β = ¼ (1 − cos(π x))²,  x = clamp((θ₁ − θ_res) / (θ_fc − θ_res), 0, 1)

where `θ₁` is the liquid water content of the uppermost soil layer, `θ_fc` the field capacity, and
`θ_res` the residual water content. The factor vanishes at and below the residual water content and
equals one at and above field capacity.

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
    # Water content normalized between residual water content and field capacity, clamped to the unit
    # interval so that β = 0 at and below θres (where the unclamped cosine would grow again) and β = 1 at
    # and above θfc; the lower bound on the denominator guards against the edge case θfc ≤ θres.
    x = clamp((θw - θres) / max(θfc - θres, eps(NF)), zero(NF), one(NF))
    β = (1 - cos(π * x))^2 / 4
    return β
end
