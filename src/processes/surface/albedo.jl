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

Diagnosed surface albedo and emissivity. The values are computed in [`compute_auxiliary!`](@ref) as a
snow-cover-weighted blend of a snow-free background and snow, `α = (1 − f_snow)·α_bg + f_snow·α_snow`
(and likewise for emissivity), where `f_snow` is the snow-covered area fraction of the optional snow
component passed to `compute_auxiliary!`. Without a snow component (`snow === nothing`), `f_snow = 0` and
the background values are used.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct DiagnosticAlbedo{NF} <: AbstractAlbedo{NF}
    "Snow- and vegetation-free (background) albedo of bare ground"
    @param background_albedo::NF = 0.25 (bounds = UnitInterval,)

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

Diagnose the blended surface albedo and emissivity, weighting the snow-free background and snow values
by the snow-covered area fraction of the optional `snow` component.
"""
function compute_auxiliary!(
        state, grid,
        albedo::DiagnosticAlbedo,
        vegetation::Optional{AbstractVegetation} = nothing,
        snow::Optional{AbstractSnow} = nothing,
        args...
    )
    out = auxiliary_fields(state, albedo)
    fields = get_fields(state, albedo, vegetation, snow; except = out)
    launch!(grid, XY, compute_albedo_kernel!, out, fields, albedo, vegetation, snow)
    return nothing
end

# Kernel functions

@propagate_inbounds function compute_albedo(
        i, j, grid, fields,
        albedo::DiagnosticAlbedo{NF},
        vegetation::Optional{AbstractVegetation},
        snow::Optional{AbstractSnow},
        args...
    ) where {NF}
    α₀ = albedo.background_albedo
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
        args...
    )
    α, ϵ = compute_albedo(i, j, grid, fields, albedo, vegetation, snow)
    out.albedo[i, j, end] = α
    out.emissivity[i, j, end] = ϵ
    return nothing
end

# Kernels

@kernel inbounds = true function compute_albedo_kernel!(out, grid, fields, albedo::DiagnosticAlbedo, args...)
    i, j = @index(Global, NTuple)
    compute_albedo!(out, i, j, grid, fields, albedo, args...)
end
