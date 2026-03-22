"""
    $TYPEDEF

Represents soil texture as a fractional mixture of sand, silt, and clay. Accepts values 
"""
@kwdef struct SoilTexture{NF}
    sand::NF = 1.0
    clay::NF = 0.0
    silt::NF = 1 - sand - clay

    SoilTexture(sand, silt, clay) = SoilTexture{promote_type(typeof(sand), typeof(silt), typeof(clay))}(sand, silt, clay)
    function SoilTexture{NF}(sand, silt, clay) where {NF <: Number}
        @assert zero(sand) <= sand <= one(sand)
        @assert zero(silt) <= silt <= one(silt)
        @assert zero(clay) <= clay <= one(clay)
        @assert sand + silt + clay ≈ 1.0 "sand, silt, and clay fractions must sum to unity"
        return new{NF}(sand, silt, clay)
    end
end

SoilTexture(::Type{NF}; kwargs...) where {NF} = SoilTexture{NF}(; kwargs...)

Base.eltype(::SoilTexture{NF}) where {NF} = NF

@inline @propagate_inbounds function Base.getindex(texture::SoilTexture{<:AbstractArray}, idx...)
    return SoilTexture(
        texture.sand[idx...],
        texture.silt[idx...],
        texture.clay[idx...]
    )
end

function Base.convert(::SoilTexture{NewType}, texture::SoilTexture) where {NewType}
    return SoilTexture(
        convert(NewType, texture.sand),
        convert(NewType, texture.silt),
        convert(NewType, texture.clay)
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
