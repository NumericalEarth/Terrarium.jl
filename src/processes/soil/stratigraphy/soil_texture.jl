"""
    $TYPEDEF

Represents soil texture as a fractional mixture of sand, silt, and clay.
"""
@parameterized @kwdef struct SoilTexture{NF}
    "Fraction of sand"
    @param sand::NF = 1.0 (bounds = UnitInterval,)

    "Fraction of clay"
    @param clay::NF = 0.0 (bounds = UnitInterval,)

    "Fraction of silt"
    @param silt::NF = 1 - sand - clay (bounds = UnitInterval,)

    SoilTexture(sand, clay, silt) = SoilTexture{promote_type(typeof(sand), typeof(silt), typeof(clay))}(sand, clay, silt)
    function SoilTexture{NF}(sand, clay, silt) where {NF <: Number}
        @assert zero(sand) <= sand <= one(sand)
        @assert zero(clay) <= clay <= one(clay)
        @assert zero(silt) <= silt <= one(silt)
        @assert sand + silt + clay ≈ 1.0 "sand, silt, and clay fractions must sum to unity"
        return new{NF}(sand, clay, silt)
    end
end

SoilTexture(::Type{NF}; kwargs...) where {NF} = SoilTexture{NF}(; kwargs...)

# Convenience Field setter
function Oceananigans.Fields.set!(fields, texture::SoilTexture)
    set!(fields.sand_fraction, texture.sand)
    set!(fields.clay_fraction, texture.clay)
    set!(fields.silt_fraction, texture.silt)
    return nothing
end

function normalize_texture(sand::NF, silt::NF, clay::NF) where {NF}
    total = sand + silt + clay
    return sand / total, silt / total, clay / total
end

"""
    $TYPEDSIGNATURES

Normalize the given `sand`, `silt`, and `clay` fraction `Field`s in place such that
the fractions sum to unity in each element, as required by the `SoilTexture` constructor.
"""
function normalize_texture!(
        sand::AbstractField{LX, LY, LZ, G, NF},
        silt::AbstractField{LX, LY, LZ, G, NF},
        clay::AbstractField{LX, LY, LZ, G, NF};
        default = SoilTexture(NF; sand = 1.0, silt = 0.0, clay = 0.0)
    ) where {LX, LY, LZ, G, NF}
    grid = sand.grid
    launch!(architecture(grid), grid, :xy, normalize_texture_kernel!, sand, silt, clay, default)
    return nothing
end


Base.eltype(::SoilTexture{NF}) where {NF} = NF

@inline @propagate_inbounds function Base.getindex(texture::SoilTexture{<:AbstractArray}, idx...)
    return SoilTexture(
        texture.sand[idx...],
        texture.clay[idx...],
        texture.silt[idx...]
    )
end

function Base.convert(::SoilTexture{NewType}, texture::SoilTexture) where {NewType}
    return SoilTexture(
        convert(NewType, texture.sand),
        convert(NewType, texture.clay),
        convert(NewType, texture.silt)
    )
end

# Presets for common soil textures
# Borrowed from CryoGrid.jl:
# https://github.com/CryoGrid/CryoGrid.jl/blob/master/src/Physics/Soils/soil_texture.jl
SoilTexture(name::Symbol) = SoilTexture(Float64, Val{name}())
SoilTexture(::Type{NF}, name::Symbol) where {NF} = SoilTexture(NF, Val{name}())
SoilTexture(::Type{NF}, ::Val{:sand}) where {NF} = SoilTexture(NF, sand = 1.0, silt = 0.0, clay = 0.0)
SoilTexture(::Type{NF}, ::Val{:silt}) where {NF} = SoilTexture(NF, sand = 0.0, silt = 1.0, clay = 0.0)
SoilTexture(::Type{NF}, ::Val{:clay}) where {NF} = SoilTexture(NF, sand = 0.0, silt = 0.0, clay = 1.0)
SoilTexture(::Type{NF}, ::Val{:sandyclay}) where {NF} = SoilTexture(NF, sand = 0.5, silt = 0.0, clay = 0.5)
SoilTexture(::Type{NF}, ::Val{:siltyclay}) where {NF} = SoilTexture(NF, sand = 0.0, silt = 0.5, clay = 0.5)
SoilTexture(::Type{NF}, ::Val{:loam}) where {NF} = SoilTexture(NF, sand = 0.4, silt = 0.4, clay = 0.2)
SoilTexture(::Type{NF}, ::Val{:sandyloam}) where {NF} = SoilTexture(NF, sand = 0.8, silt = 0.1, clay = 0.1)
SoilTexture(::Type{NF}, ::Val{:siltyloam}) where {NF} = SoilTexture(NF, sand = 0.1, silt = 0.8, clay = 0.1)
SoilTexture(::Type{NF}, ::Val{:clayloam}) where {NF} = SoilTexture(NF, sand = 0.3, silt = 0.3, clay = 0.4)

@kernel function normalize_texture_kernel!(sand, silt, clay, default::SoilTexture)
    i, j = @index(Global, NTuple)
    total = sand[i, j] + silt[i, j] + clay[i, j]
    sand[i, j, end] = ifelse(isnan(total), default.sand, sand[i, j] / total)
    silt[i, j, end] = ifelse(isnan(total), default.silt, silt[i, j] / total)
    clay[i, j, end] = ifelse(isnan(total), default.clay, clay[i, j] / total)
end
