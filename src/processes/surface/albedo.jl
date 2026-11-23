"""
    $TYPEDEF

Properties:
$TYPEDFIELDS
"""
@kwdef struct PrescribedAlbedo{NF} <: AbstractAlbedo{NF} end

PrescribedAlbedo(::Type{NF}) where {NF} = PrescribedAlbedo{NF}()

variables(::PrescribedAlbedo) = (
    input(:albedo, Surface(XY()), bounds = UnitInterval, desc = "Surface albedo, i.e. ratio of outgoing to incoming shortwave radiation [-]"),
    input(:emissivity, Surface(XY()), bounds = UnitInterval, desc = "Surface emissivity, i.e. efficiency of longwave emission [-]"),
)

"""
    $TYPEDEF

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct ConstantAlbedo{NF} <: AbstractAlbedo{NF}
    "Surface albedo, i.e. ratio of outgoing to incoming shortwave radiation"
    @param albedo::NF = 0.3 (bounds = UnitInterval,)

    "Surface emissivity, i.e. fraction of thermal radiation emitted from the surface"
    @param emissivity::NF = 0.97 (bounds = UnitInterval,)
end

ConstantAlbedo(::Type{NF}; kwargs...) where {NF} = ConstantAlbedo{NF}(; kwargs...)

"""
    $TYPEDSIGNATURES

Return the surface albedo at grid point `i, j`.
"""
@inline albedo(i, j, grid, fields, albedo::ConstantAlbedo) = albedo.albedo

"""
    $TYPEDSIGNATURES

Return the surface emissivity at grid point `i, j`.
"""
@inline emissivity(i, j, grid, fields, albedo::ConstantAlbedo) = albedo.emissivity

"""
    $TYPEDEF

Diagnosed surface albedo and emissivity. The values are computed in [`compute_auxiliary!`](@ref) as an
area-weighted blend of bare ground, vegetation, and snow,
`α = (1 − f_snow)·(1 − f_veg)·α_bg + (1 − f_snow)·f_veg·α_veg + f_snow·α_snow` (and likewise for
emissivity), where `f_veg` is the vegetated area fraction of the optional vegetation component and
`f_snow` the snow-covered area fraction of the optional snow component passed to `compute_auxiliary!`.
Snow-covered bare and vegetated areas are assumed to share the same snow albedo and emissivity. Without
a vegetation or snow component (`nothing`), the corresponding fraction is zero.

The background albedo of bare ground varies linearly with the moisture of the uppermost soil layer
between a dry and a wet value, `α_bg = α_dry + (α_wet − α_dry)·x`, where
`x = clamp((θ₁ − θ_res) / (θ_fc − θ_res), 0, 1)` is the liquid water content `θ₁` of the uppermost soil
layer relative to the residual water content `θ_res` and the field capacity `θ_fc`. Without a soil
component (`nothing`), the dry value is used. The dry and wet albedos and the background emissivity are
spatially constant parameters.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct DiagnosticAlbedo{NF} <: AbstractAlbedo{NF}
    "Albedo of dry bare ground"
    @param dry_ground_albedo::NF = 0.3 (bounds = UnitInterval,)

    "Albedo of wet bare ground"
    @param wet_ground_albedo::NF = 0.15 (bounds = UnitInterval,)

    "Snow- and vegetation-free (background) emissivity of bare ground"
    @param background_emissivity::NF = 0.97 (bounds = UnitInterval,)
end

DiagnosticAlbedo(::Type{NF}; kwargs...) where {NF} = DiagnosticAlbedo{NF}(; kwargs...)

variables(::DiagnosticAlbedo) = (
    auxiliary(:albedo, Surface(XY()), bounds = UnitInterval, desc = "Diagnosed surface albedo [-]"),
    auxiliary(:emissivity, Surface(XY()), bounds = UnitInterval, desc = "Diagnosed surface emissivity [-]"),
)

"""
    $TYPEDSIGNATURES

Diagnose the blended surface albedo and emissivity, weighting the bare-ground, vegetation, and snow
values by the vegetated area fraction of the optional `vegetation` component and the snow-covered area
fraction of the optional `snow` component. The bare-ground albedo depends on the moisture of the
uppermost layer of the optional `soil` component.
"""
function compute_auxiliary!(
        state, grid,
        albedo::DiagnosticAlbedo,
        vegetation::Optional{AbstractVegetation} = nothing,
        snow::Optional{AbstractSnow} = nothing,
        soil::Optional{AbstractSoil} = nothing,
        args...
    )
    out = auxiliary_fields(state, albedo)
    fields = get_fields(state, albedo, vegetation, snow, soil; except = out)
    launch!(grid, XY, compute_albedo_kernel!, out, fields, albedo, vegetation, snow, soil)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the liquid water content of the uppermost soil layer relative to the residual water content and
the field capacity, `x = clamp((θ₁ − θ_res) / (θ_fc − θ_res), 0, 1)`, which is zero for dry and one for
wet ground. Returns zero, i.e. dry ground, when there is no soil component.
"""
@propagate_inbounds function compute_relative_ground_wetness(i, j, grid, fields, soil::AbstractSoil)
    NF = eltype(grid)
    ground_grid = ground_domain(grid)
    strat = get_stratigraphy(soil)
    hydrology = get_hydrology(soil)
    bgc = get_biogeochemistry(soil)
    props = get_hydraulic_properties(hydrology)
    comp = soil_composition(i, j, ground_grid.Nz, grid, fields, strat, hydrology, bgc)
    texture = mineral_texture(comp)
    fracs = volumetric_fractions(comp)
    θ₁ = fracs.water
    θfc = field_capacity(props, texture)
    θres = props.residual_saturation * porosity(comp)
    # the lower bound on the denominator guards against the edge case θfc ≤ θres
    x = clamp((θ₁ - θres) / max(θfc - θres, eps(NF)), zero(NF), one(NF))
    return x
end

@inline compute_relative_ground_wetness(i, j, grid, fields, ::Nothing) = zero(eltype(grid))

"""
    $TYPEDSIGNATURES

Compute the albedo of bare ground, `α_bg = α_dry + (α_wet − α_dry)·x`, from the dry and wet ground albedos
and the relative wetness `x` of the uppermost layer of the optional `soil` component.
"""
@propagate_inbounds function compute_ground_albedo(i, j, grid, fields, albedo::DiagnosticAlbedo, soil::Optional{AbstractSoil})
    α_dry = albedo.dry_ground_albedo
    α_wet = albedo.wet_ground_albedo
    x = compute_relative_ground_wetness(i, j, grid, fields, soil)
    α_bg = α_dry + (α_wet - α_dry) * x
    return α_bg
end

@propagate_inbounds function compute_albedo(
        i, j, grid, fields,
        albedo::DiagnosticAlbedo{NF},
        vegetation::Optional{AbstractVegetation},
        snow::Optional{AbstractSnow},
        soil::Optional{AbstractSoil} = nothing,
        args...
    ) where {NF}
    α₀ = compute_ground_albedo(i, j, grid, fields, albedo, soil)
    ϵ₀ = albedo.background_emissivity
    f_snow = snow_cover_fraction(i, j, grid, fields, snow)
    α_snow = compute_albedo(i, j, grid, fields, snow)
    ϵ_snow = compute_emissivity(i, j, grid, fields, snow)
    f_veg = vegetation_area_fraction(i, j, grid, fields, vegetation)
    α_veg = compute_albedo(i, j, grid, fields, vegetation)
    ϵ_veg = compute_emissivity(i, j, grid, fields, vegetation)
    α_eff =
        (1 - f_snow) * (1 - f_veg) * α₀ +
        (1 - f_snow) * f_veg * α_veg +
        f_snow * α_snow # assume for now that bare and vegetated areas have same snow-covered albedo
    ϵ_eff =
        (1 - f_snow) * (1 - f_veg) * ϵ₀ +
        (1 - f_snow) * f_veg * ϵ_veg +
        f_snow * ϵ_snow # assume for now that bare and vegetated areas have same snow-covered emissivity
    return α_eff, ϵ_eff
end

""" $TYPEDSIGNATURES """
@propagate_inbounds function compute_albedo!(
        out, i, j, grid, fields,
        albedo::DiagnosticAlbedo,
        vegetation::Optional{AbstractVegetation},
        snow::Optional{AbstractSnow},
        soil::Optional{AbstractSoil} = nothing,
        args...
    )
    α, ϵ = compute_albedo(i, j, grid, fields, albedo, vegetation, snow, soil)
    out.albedo[i, j, end] = α
    out.emissivity[i, j, end] = ϵ
    return nothing
end

# Kernels

@kernel inbounds = true function compute_albedo_kernel!(out, grid, fields, albedo::DiagnosticAlbedo, args::Vararg{Any, N}) where {N}
    i, j = @index(Global, NTuple)
    compute_albedo!(out, i, j, grid, fields, albedo, args...)
end
