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

Diagnosed surface albedo and emissivity. The snow-free surface blends bare ground and vegetation,
`αˢᶠ = (1 − f_veg)·α_bg + f_veg·α_veg`, where `f_veg` is the vegetated area fraction of the optional
vegetation component (zero without vegetation). The snow-covered surface takes the albedo of the snow
component, `αˢⁿ = α_snow`, since snow is assumed to bury the vegetation. The emissivity is treated
likewise. See the methods of [`compute_albedo`](@ref) for [`SnowFree`](@ref) and [`SnowCovered`](@ref).

The grid-cell values computed in [`compute_auxiliary!`](@ref) and stored in `albedo` and `emissivity`
are the area-weighted blend `α = (1 − f_snow)·αˢᶠ + f_snow·αˢⁿ`, where `f_snow` is the snow-covered area
fraction of the optional snow component passed to `compute_auxiliary!` (zero without snow).

The background albedo of bare ground varies linearly with the moisture of the uppermost soil layer
between a dry and a wet value, `α_bg = α_dry + (α_wet − α_dry)·w`, where the relative ground wetness
`w = clamp((θ₁ − θ_res) / (θ_fc − θ_res), 0, 1)` is the liquid water content `θ₁` of the uppermost soil
layer relative to the residual water content `θ_res` and the field capacity `θ_fc`. Without a soil
component (`nothing`), the dry value is used. The dry and wet albedos and the ground emissivity are
spatially constant parameters; the ground emissivity does not (yet) depend on soil moisture.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct DiagnosticAlbedo{NF} <: AbstractAlbedo{NF}
    "Albedo of dry bare ground"
    @param dry_ground_albedo::NF = 0.3 (bounds = UnitInterval,)

    "Albedo of wet bare ground"
    @param wet_ground_albedo::NF = 0.15 (bounds = UnitInterval,)

    "Emissivity of bare ground"
    @param ground_emissivity::NF = 0.97 (bounds = UnitInterval,)
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

Compute the albedo of bare ground, `α_bg = α_dry + (α_wet − α_dry)·w`, from the dry and wet ground albedos and
the relative wetness `w = clamp((θ₁ − θ_res) / (θ_fc − θ_res), 0, 1)` of the uppermost layer of the `soil`
component (see [`relative_saturation`](@ref)). `w` is zero for dry ground and one for ground at or above field
capacity (not full saturation). Without a soil component, the ground is treated as dry.
"""
@propagate_inbounds function compute_ground_albedo(i, j, grid, fields, albedo::DiagnosticAlbedo, soil::AbstractSoil)
    ground_grid = ground_domain(grid)
    strat = get_stratigraphy(soil)
    hydrology = get_hydrology(soil)
    bgc = get_biogeochemistry(soil)
    (; θw, θres, θfc) = soil_water_content(i, j, ground_grid.Nz, grid, fields, strat, hydrology, bgc)
    w = relative_saturation(θw, θres, θfc)
    α_dry = albedo.dry_ground_albedo
    α_wet = albedo.wet_ground_albedo
    α_bg = α_dry + (α_wet - α_dry) * w
    return α_bg
end

# without soil, the ground is treated as dry
@inline compute_ground_albedo(i, j, grid, fields, albedo::DiagnosticAlbedo, ::Nothing) = albedo.dry_ground_albedo

"""
    $TYPEDSIGNATURES

Compute the albedo and emissivity `(α, ϵ)` of the snow-free tile at `i, j` as the blend of bare ground
and vegetation weighted by the vegetated area fraction `f_veg` of the optional `vegetation` component,
`α = (1 − f_veg)·α_bg + f_veg·α_veg` (and likewise for the emissivity), where the bare-ground albedo `α_bg`
depends on the moisture of the uppermost layer of the optional `soil` component (see
[`compute_ground_albedo`](@ref)).

This is the method for the snow-free tile ([`SnowFree`](@ref)), which is also the whole grid cell in
configurations without snow; it is not weighted by a snow-covered area fraction, and `snow` is not used.
"""
@propagate_inbounds function compute_albedo(
        i, j, grid, fields,
        ::SnowFree,
        albedo::DiagnosticAlbedo{NF},
        vegetation::Optional{AbstractVegetation},
        snow::Optional{AbstractSnow},
        soil::Optional{AbstractSoil} = nothing,
        args...
    ) where {NF}
    α₀ = compute_ground_albedo(i, j, grid, fields, albedo, soil)
    ϵ₀ = albedo.ground_emissivity
    f_veg = vegetation_area_fraction(i, j, grid, fields, vegetation)
    α_veg = compute_albedo(i, j, grid, fields, vegetation)
    ϵ_veg = compute_emissivity(i, j, grid, fields, vegetation)
    α = (1 - f_veg) * α₀ + f_veg * α_veg
    ϵ = (1 - f_veg) * ϵ₀ + f_veg * ϵ_veg
    return α, ϵ
end

"""
    $TYPEDSIGNATURES

Compute the albedo and emissivity `(α, ϵ)` of the snow-covered tile at `i, j`, i.e. those of the `snow`
component. Snow is assumed to bury the vegetation, so the snow-covered bare and vegetated areas share the
same snow albedo and emissivity.

This is the method for the snow-covered tile ([`SnowCovered`](@ref)); it is not weighted by the snow-covered
area fraction. The
`vegetation` and `soil` arguments are accepted for a uniform call signature and are not used.
"""
@propagate_inbounds function compute_albedo(
        i, j, grid, fields,
        ::SnowCovered,
        albedo::DiagnosticAlbedo{NF},
        vegetation::Optional{AbstractVegetation},
        snow::AbstractSnow,
        soil::Optional{AbstractSoil} = nothing,
        args...
    ) where {NF}
    α_snow = compute_albedo(i, j, grid, fields, snow)
    ϵ_snow = compute_emissivity(i, j, grid, fields, snow)
    return α_snow, ϵ_snow
end

"""
    $TYPEDSIGNATURES

Compute the grid-cell albedo and emissivity `(α, ϵ)` at `i, j` with a `snow` component: the snow-free and
snow-covered tile values (the methods of this function for [`SnowFree`](@ref) and [`SnowCovered`](@ref))
are weighted by the snow-covered area fraction
`f_snow`, `α = (1 − f_snow)·αˢᶠ + f_snow·αˢⁿ` (and likewise for the emissivity).
"""
@propagate_inbounds function compute_albedo(
        i, j, grid, fields,
        albedo::DiagnosticAlbedo{NF},
        vegetation::Optional{AbstractVegetation},
        snow::AbstractSnow,
        soil::Optional{AbstractSoil} = nothing,
        args...
    ) where {NF}
    # Transitional (Phase 3 of the snow tiling plan stores the tile values in the tile namespaces and
    # forms the grid-cell values in the aggregation): blend the tile values by the snow-covered fraction
    f_snow = snow_cover_fraction(i, j, grid, fields, snow)
    α_sf, ϵ_sf = compute_albedo(i, j, grid, fields, SnowFree(), albedo, vegetation, snow, soil)
    α_sn, ϵ_sn = compute_albedo(i, j, grid, fields, SnowCovered(), albedo, vegetation, snow, soil)
    α_eff = (1 - f_snow) * α_sf + f_snow * α_sn
    ϵ_eff = (1 - f_snow) * ϵ_sf + f_snow * ϵ_sn
    return α_eff, ϵ_eff
end

"""
    $TYPEDSIGNATURES

Compute the grid-cell albedo and emissivity `(α, ϵ)` at `i, j` without snow, i.e. the snow-free tile
values ([`SnowFree`](@ref)).
"""
@propagate_inbounds compute_albedo(
    i, j, grid, fields,
    albedo::DiagnosticAlbedo,
    vegetation::Optional{AbstractVegetation},
    snow::Nothing,
    soil::Optional{AbstractSoil} = nothing,
    args...
) = compute_albedo(i, j, grid, fields, SnowFree(), albedo, vegetation, snow, soil)

"""
    $TYPEDSIGNATURES

Compute the grid-cell albedo and emissivity (see [`compute_albedo`](@ref)) and store them into the
auxiliary output fields `out`.
"""
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
