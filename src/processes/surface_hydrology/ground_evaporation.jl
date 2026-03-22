"""
    GroundEvaporation{NF, GR} <: AbstractEvapotranspiration

Evaporation scheme for bare ground that calculates the humidity flux as

```math
E = \\beta \\frac{\\Delta q}{r_a}
```
where `Δq` is the vapor pressure deficit in terms of specific humidity, `rₐ` is aerodynamic resistance,
and `β` is an evaporation limiting factor.
"""
struct GroundEvaporation{NF, GR <: AbstractGroundEvaporationResistanceFactor} <: AbstractEvapotranspiration{NF}
    "Parameterization for ground resistance to evaporation/sublimation"
    ground_resistance::GR
end

GroundEvaporation(
    ::Type{NF};
    ground_resistance::GR = ConstantEvaporationResistanceFactor(one(NF))
) where {NF, GR} = GroundEvaporation{NF, GR}(; ground_resistance)

@propagate_inbounds surface_humidity_flux(i, j, grid, fields, evtr::GroundEvaporation) = fields.evaporation_ground[i, j]

# Process methods

variables(::GroundEvaporation) = (
    auxiliary(:evaporation_ground, XY(), units = u"m/s", desc = "Ground evaporation contribution to surface humidity flux"),
    input(:skin_temperature, XY(), units = u"°C", desc = "Skin temperature of the surface"),
)

function compute_auxiliary!(
        state, grid,
        evap::GroundEvaporation,
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants,
        soil::Optional{AbstractSoil} = nothing
    )
    out = auxiliary_fields(state, evap)
    fields = get_fields(state, evap, atmos, soil; except = out)
    launch!(grid, XY, compute_evaporation_kernel!, out, fields, evap, atmos, constants, soil)
    return nothing
end

# Kernel functions

@propagate_inbounds function compute_evaporation_kernel!(
        out, i, j, grid, fields,
        evap::GroundEvaporation,
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants,
        soil::Optional{AbstractSoil} = nothing
    )
    i, j = @index(Global, NTuple)
    Ts = fields.skin_temperature[i, j]
    rₐ = aerodynamic_resistance(i, j, grid, fields, aerodynamic_resistance) # aerodynamic resistance
    β = ground_evaporation_resistance_factor(i, j, grid, fields, evap.ground_resistance, soil)
    Δq = compute_humidity_vpd(i, j, grid, fields, atmos, constants, Ts)
    # Calculate water evaporation flux in m/s (positive upwards)
    return out.evaporation_ground[i, j, 1] = β * Δq / rₐ
end

# Ground resistance to evaporation

"""
    $TYPEDEF

Represents a spatiotemporally constant, prescribed ground evaporation resistance factor.
"""
@kwdef struct ConstantEvaporationResistanceFactor{NF} <: AbstractGroundEvaporationResistanceFactor
    "Unit interval factor that determines resistance to evaporation; zero corresponds to no evaporation"
    factor::NF = 1.0
end

ConstantEvaporationResistanceFactor(::Type{NF}; kwargs...) where {NF} = ConstantEvaporationResistanceFactor{NF}(; kwargs...)

@inline ground_evaporation_resistance_factor(i, j, grid, fields, res::ConstantEvaporationResistanceFactor, args...) = res.factor

"""
    $TYPEDEF

Implements the soil moisture limiting resistance factor of Lee and Pielke (1992),

```math
\\beta =
\\frac{1}{4} \\left[1 - \\cos\\left(π \\theta_1/\\theta_{\\text{fc}} \\right)\\right] \\quad \\text{for } \\theta_1 < \\theta_{\\text{fc}}
```
otherwise ``\\beta=1``.
"""
struct SoilMoistureResistanceFactor{NF} <: AbstractGroundEvaporationResistanceFactor end

SoilMoistureResistanceFactor(::Type{NF}) where {NF} = SoilMoistureResistanceFactor{NF}()

# Fallback implementation for interface consistency
ground_evaporation_resistance_factor(i, j, grid, fields, ::SoilMoistureResistanceFactor{NF}, args...) where {NF} = one(NF)

@inline function ground_evaporation_resistance_factor(
        i, j, grid, fields,
        ::SoilMoistureResistanceFactor{NF},
        soil::AbstractSoil
    ) where {NF}
    fgrid = get_field_grid(grid)
    strat = get_stratigraphy(soil)
    hydrology = get_hydrology(soil)
    bgc = get_biogeochemistry(soil)
    soil = soil_volume(i, j, fgrid.Nz, grid, fields, strat, hydrology, bgc)
    fc = field_capacity(get_hydraulic_properties(hydrology), soil)
    fracs = volumetric_fractions(soil)
    if fracs.water < fc
        β = (1 - cos(π * fracs.water / fc))^2 / 4
    else
        β = NF(1)
    end
    return β
end

# Forcing interface for soil hydrology

"""
    forcing(i, j, k, grid, clock, fields, evtr::AbstractEvapotranspiration, ::AbstractSoilHydrology)

Compute and return the evapotranspiration forcing for soil moisture at the given indices `i, j, k`.
The ET forcing is just the `surface_humidity_flux` rescaled by the thickness of layer `k`.
"""
@inline function forcing(i, j, k, grid, clock, fields, evtr::AbstractEvapotranspiration, ::AbstractSoilHydrology)
    let Δz = Δzᵃᵃᶜ(i, j, k, grid),
            Qh = surface_humidity_flux(i, j, grid, fields, evtr)
        ∂θ∂t = -Qh / Δz # rescale by layer thickness to get water content flux
        return ∂θ∂t * (k == grid.Nz)
    end
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, evtr::AbstractEvapotranspiration, args...)
    i, j = @index(Global, NTuple)
    compute_evapotranspiration!(out, i, j, grid, fields, evtr, args...)
end
