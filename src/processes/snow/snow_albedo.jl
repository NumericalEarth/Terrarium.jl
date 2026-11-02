"""$TYPEDSIGNATURES"""
@propagate_inbounds compute_albedo(i, j, grid, fields, snow::AbstractSnow) = compute_albedo(i, j, grid, fields, snow.albedo)
@propagate_inbounds compute_albedo(i, j, grid, fields, snow::Nothing) = zero(eltype(grid))

"""$TYPEDSIGNATURES"""
@propagate_inbounds compute_emissivity(i, j, grid, fields, snow::AbstractSnow) = compute_emissivity(i, j, grid, fields, snow.albedo)
@propagate_inbounds compute_emissivity(i, j, grid, fields, snow::Nothing) = zero(eltype(grid))

"""
    $TYPEDEF

Basic constant albedo scheme for snow that treats both albedo and emissivity as both spatially and temporally constants.
The default values are for freshly fallen snow, taken from [westermannCryoGrid3Simulating2016](@cite).
"""
@parameterized @kwdef struct ConstantSnowAlbedo{NF} <: AbstractSnowAlbedo{NF}
    "Albedo of (fresh) snow"
    @param snow_albedo::NF = 0.8 (bounds = UnitInterval,)

    "Emissivity of (fresh) snow"
    @param snow_emissivity::NF = 0.99 (bounds = UnitInterval,)
end

ConstantSnowAlbedo(::Type{NF}; kwargs...) where {NF} = ConstantSnowAlbedo{NF}(; kwargs...)

"""$TYPEDSIGNATURES"""
@inline compute_albedo(i, j, grid, fields, albedo::ConstantSnowAlbedo) = albedo.snow_albedo

"""$TYPEDSIGNATURES"""
@inline compute_emissivity(i, j, grid, fields, albedo::ConstantSnowAlbedo) = albedo.snow_emissivity

# TODO: Add time-varying snow albedo
