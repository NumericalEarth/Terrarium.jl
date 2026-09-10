"""
    $TYPEDEF

Represents a soil stratigraphy as a stack of named [soil horizons](https://en.wikipedia.org/wiki/Soil_horizon).
Each soil horizon is assumed to have internally homogeneous soil properties. The number of horizons and their
respective names are defined by the user.

Properties:
$TYPEDFIELDS
"""
struct SoilStratigraphy{NF, N, Horizons <: Tuple{Vararg{AbstractSoilHorizon{NF}, N}}} <: AbstractStratigraphy{NF}
    "Named tuple of soil horizons ordered from top to bottom"
    horizons::Horizons
end

function SoilStratigraphy(::Type{NF}, horizons::AbstractSoilHorizon...) where {NF}
    @assert length(horizons) > 0 "At least one soil horizon must be specified; for simple configurations, consider using HomogeneousSoilStratigraphy."
    return SoilStratigraphy(horizons)
end

# Convenience constructors

"""
    $TYPEDSIGNATURES

Convenience constructor that creates a `SoilStratigraphy` with a single `ConstantSoilHorizon`.
"""
function HomogeneousSoilStratigraphy(
        ::Type{NF};
        texture::SoilTexture = SoilTexture(NF),
        porosity::AbstractSoilPorosity = ConstantSoilPorosity(NF)
    ) where {NF}
    return SoilStratigraphy(NF, ConstantSoilHorizon(NF, :soil; texture, porosity))
end

"""
    $TYPEDSIGNATURES

Convenience constructor that creates a `SoilStratigraphy` with six horizons corresponding to
the six depth intervals defined by the SoilGridsV2 dataset. The `porosity` parameterization is
applied to all six default horizons; pass explicit `horizon*` arguments to override individual
horizons (in which case `porosity` does not apply to those).
"""
function SoilGridsStratigraphy(
        ::Type{NF};
        porosity::AbstractSoilPorosity = ConstantSoilPorosity(NF),
        horizon1::AbstractSoilHorizon = PrescribedSoilHorizon(NF, :horizon1; porosity, default_thickness = NF(0.05)),
        horizon2::AbstractSoilHorizon = PrescribedSoilHorizon(NF, :horizon2; porosity, default_thickness = NF(0.1)),
        horizon3::AbstractSoilHorizon = PrescribedSoilHorizon(NF, :horizon3; porosity, default_thickness = NF(0.2)),
        horizon4::AbstractSoilHorizon = PrescribedSoilHorizon(NF, :horizon4; porosity, default_thickness = NF(0.3)),
        horizon5::AbstractSoilHorizon = PrescribedSoilHorizon(NF, :horizon5; porosity, default_thickness = NF(0.4)),
        horizon6::AbstractSoilHorizon = PrescribedSoilHorizon(NF, :horizon6; porosity, default_thickness = NF(1.0))
    ) where {NF}
    return SoilStratigraphy((horizon1, horizon2, horizon3, horizon4, horizon5, horizon6))
end

# Base methods

Base.length(strat::SoilStratigraphy) = length(strat.horizons)
Base.iterate(strat::SoilStratigraphy) = Base.iterate(strat.horizons)
Base.iterate(strat::SoilStratigraphy, iter) = Base.iterate(strat.horizons, iter)

# Variables

function variables(strat::SoilStratigraphy)
    return map(strat.horizons) do horizon
        namespace(nameof(horizon), variables(horizon))
    end
end

# Process methods

compute_auxiliary!(state, grid, ::SoilStratigraphy, args...) = nothing

compute_tendencies!(state, grid, ::SoilStratigraphy, args...) = nothing

# Kernel functions

"""
    $TYPEDSIGNATURES

Retrieve the soil horizon for the soil volume at index `i, j, k`. The last
soil horizon in `strat` is assumed to extend to the bottom of the vertical
column regardless of its associated `thickness`. Note that, since `SoilStratigraphy`
uses namespaces for the state variables of each horizon, any methods defined on
`AbstractSoilHorizon` types should be passed the namespace, e.g:
```julia
horizon = soil_horizon(i, j, k, grid, fields, strat)
texture = soil_texture(i, j, grid, getproperty(fields, nameof(horizon)), horizon)
```
"""
@inline function soil_horizon(i, j, k, grid, fields, strat::SoilStratigraphy{NF}) where {NF}
    fgrid = get_field_grid(grid)
    # get midpoint of current node at k
    zₖ = znode(i, j, k, fgrid, Center(), Center(), Center())
    # initially set z to uppermost layer boundary
    z = znode(i, j, fgrid.Nz + 1, fgrid, Center(), Center(), Face())
    # initialize result to first horizon (will be updated in loop)
    horizon = first(strat.horizons)
    for next in strat.horizons
        # update result when zₖ is within this horizon's depth range
        horizon = ifelse(zₖ <= z, next, horizon)
        Δzₗ = soil_thickness(i, j, grid, getproperty(fields, nameof(next)), next)
        z -= Δzₗ
    end
    return horizon
end

"""
    $TYPEDSIGNATURES

Convenience method that invokes `func(i, j, grid, horizon_fields, horizon, args...; kwargs...)`
where `horizon` is the soil horizon returned by [`soil_horizon`](@ref) and `horizon_fields`
is `getproperty(fields, nameof(horizon))`.
"""
@inline function with_soil_horizon(
        func, i, j, k, grid, fields,
        strat::SoilStratigraphy,
        args...;
        kwargs...
    )
    fgrid = get_field_grid(grid)
    # midpoint of the current node at k
    zₖ = znode(i, j, k, fgrid, Center(), Center(), Center())
    # uppermost layer boundary
    z = znode(i, j, fgrid.Nz + 1, fgrid, Center(), Center(), Face())
    # Evaluate `func` and the thickness for every horizon up front via `fastmap`, which unrolls
    # the (heterogeneously typed) horizon tuple in a type-stable way. The resulting `(value,
    # thickness)` tuples are homogeneously typed, so the depth-based selection below is a plain,
    # GPU-compatible loop. We select on the `func` result rather than returning the matching
    # horizon, as `soil_horizon` does: the horizon name is a type parameter, so a runtime-
    # selected horizon would be type-unstable and trigger dynamic dispatch inside GPU kernels.
    results = fastmap(strat.horizons) do horizon
        horizon_fields = getproperty(fields, nameof(horizon))
        value = func(i, j, grid, horizon_fields, horizon, args...; kwargs...)
        return (value, soil_thickness(i, j, grid, horizon_fields, horizon))
    end
    # Walk from the top boundary downward and keep the value of the horizon containing zₖ.
    selected = first(results)[1]
    for (value, Δzₗ) in results
        selected = ifelse(zₖ <= z, value, selected)
        z -= Δzₗ
    end
    return selected
end

"""
    $TYPEDSIGNATURES

Retrieve the soil texture of the soil volume at index `i, j, k` in the given stratigraphy `strat`.
"""
@inline function soil_texture(i, j, k, grid, fields, strat::SoilStratigraphy{NF}) where {NF}
    texture = with_soil_horizon(soil_texture, i, j, k, grid, fields, strat)
    return texture
end

"""
    $TYPEDSIGNATURES

Compute the organic fraction of solid material in the soil volume at index `i, j, k`.
"""
@inline function organic_fraction(
        i, j, k, grid, fields,
        strat::SoilStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    ρ_soc = density_soc(i, j, k, grid, fields, bgc)
    ρ_org = density_pure_soc(bgc)
    por = with_soil_horizon(organic_porosity, i, j, k, grid, fields, strat)
    organic = ρ_soc / ((1 - por) * ρ_org)
    return organic
end

"""
    $TYPEDSIGNATURES

Compute the porosity of the soil volume at the given indices.
"""
@propagate_inbounds function porosity(
        i, j, k, grid, fields,
        strat::SoilStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    organic = organic_fraction(i, j, k, grid, fields, strat, bgc)
    por_m = with_soil_horizon(mineral_porosity, i, j, k, grid, fields, strat)
    por_o = with_soil_horizon(organic_porosity, i, j, k, grid, fields, strat)
    return (1 - organic) * por_m + organic * por_o
end

"""
    $TYPEDSIGNATURES

Porosity of the topmost soil layer at column `i, j`, i.e. `porosity(i, j, Nz, grid, fields, strat, bgc)`
where `Nz` is the number of grid cells in the z-dimension. Intended for use as a 2D (XY) diagnostic or
boundary condition term, e.g. via `KernelFunctionOperation{Center, Center, Nothing}`.
"""
@propagate_inbounds porosity_top(i, j, grid, fields, strat::AbstractStratigraphy, bgc::AbstractSoilBiogeochemistry) =
    porosity(i, j, grid.Nz, grid, fields, strat, bgc)

"""
    $SIGNATURES

Construct a `SoilComposition` object summarizing the material composition of the soil volume
at the given indices `i, j, k` on `grid`.
"""
@propagate_inbounds function soil_composition(
        i, j, k, grid, fields,
        strat::AbstractStratigraphy,
        hydrology::AbstractSoilHydrology,
        bgc::AbstractSoilBiogeochemistry
    )
    # get current saturation and liquid fraction
    sat = saturation_water_ice(i, j, k, grid, fields, hydrology)
    liq = liquid_water_fraction(i, j, k, grid, fields, hydrology)
    # compute porosity and solid fractions
    por = porosity(i, j, k, grid, fields, strat, bgc)
    solid = soil_matrix(i, j, k, grid, fields, strat, bgc)
    return SoilComposition(por, sat, liq, solid)
end

"""
    $TYPEDSIGNATURES

Compute and return the soil solid matrix at index `i, j, k` on `grid`. The default
implementation assumes a simple `MineralOrganic` parameterization of the solid material.
"""
@propagate_inbounds function soil_matrix(
        i, j, k, grid, fields,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    organic = organic_fraction(i, j, k, grid, fields, strat, bgc)
    texture = soil_texture(i, j, k, grid, fields, strat)
    return MineralOrganic(texture, organic)
end
