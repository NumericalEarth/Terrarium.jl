"""
    $TYPEDEF

Implementation of vegetation water availability (a.k.a "plant available water")
that computes the wilting fraction

```math
W_i = \\min\\left(\\frac{\\theta_{\\text{w},i} - \\theta_{\\text{wp},i}}{\\theta_{\\text{fc},i} - \\theta_{\\text{wp},i}}
, 1\\right)
```

where ``\\theta_{\\text{w},i}`` is the volumetric water content of the ``i``'th soil layer, ``\\theta_{\\text{fc},i}``
is the "field capacity", and ``\\theta_{\\text{wp},i}`` is the "wilting point". The water availability

Properties:
$TYPEDFIELDS
"""
@kwdef struct FieldCapacityLimitedPAW{NF} <: AbstractPlantAvailableWater{NF} end

FieldCapacityLimitedPAW(::Type{NF} = Float32) where {NF} = FieldCapacityLimitedPAW{NF}()

variables(paw::FieldCapacityLimitedPAW{NF}) where {NF} = (
    auxiliary(:plant_available_water, XYZ(), desc = "Fraction of soil water available for plant root water uptake"),
    auxiliary(:soil_moisture_limiting_factor, XY(), soil_moisture_limiting_factor, paw), # soil moisture limiting factor
    input(:root_fraction, XYZ(), desc = "Fraction of roots in each soil layer"),
)

"""
    $TYPEDSIGNATURES

Field constructor for the soil moisture limiting factor. Returns a derived `Field` that calculates
the integral of `W(z) * r(z)` where `W` is the water availability coefficient and `r` is the root fraction.
"""
function soil_moisture_limiting_factor(::FieldCapacityLimitedPAW, grid, clock, fields)
    Δz = zspacings(get_field_grid(grid), Center(), Center(), Center())
    PAW = Integral(fields.plant_available_water * fields.root_fraction / Δz, dims = 3)
    return Field(PAW)
end

# Process methods

function compute_auxiliary!(
        state, grid,
        paw::FieldCapacityLimitedPAW,
        soil::AbstractSoil,
        args...
    )
    # Unpack soil processes
    strat = get_stratigraphy(soil)
    hydrology = get_hydrology(soil)
    bgc = get_biogeochemistry(soil)
    # Unpack necessary Fields
    out = auxiliary_fields(state, paw)
    fields = get_fields(state, paw, soil; except = out)
    launch!(grid, XYZ, compute_auxiliary_kernel!, out, fields, paw, strat, hydrology, bgc)
    # compute the derived soil moisture limiting factor field
    return compute!(state.soil_moisture_limiting_factor)
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the plant avaialble water given the current soil stratigraphy, hydrology, and biogeochemistry
state in `fields`.
"""
@propagate_inbounds function compute_plant_available_water(
        i, j, k, grid, fields,
        paw::FieldCapacityLimitedPAW{NF},
        strat::AbstractStratigraphy,
        hydrology::AbstractSoilHydrology,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    # Compute soil composition and hydraulic properties
    vol = soil_volume(i, j, k, grid, fields, strat, hydrology, bgc)
    θfc = field_capacity(hydrology.hydraulic_properties, vol.solid.texture)
    θwp = wilting_point(hydrology.hydraulic_properties, vol.solid.texture)
    # Compute liquid water content
    fracs = volumetric_fractions(vol)
    θw = fracs.water
    # Compute PAW
    PAW = max(min(NF(1), (θw - θwp) / (θfc - θwp)), NF(0))
    return PAW
end

"""
    $TYPEDSIGNATURES

Mutating wrapper for [`compute_plant_available_water`](@ref) that stores the result in `out`.
"""
@propagate_inbounds function compute_plant_available_water!(
        out, i, j, k, grid, fields,
        paw::FieldCapacityLimitedPAW{NF},
        strat::AbstractStratigraphy,
        hydrology::AbstractSoilHydrology,
        bgc::AbstractSoilBiogeochemistry,
        args...
    ) where {NF}
    PAW = compute_plant_available_water(i, j, k, grid, fields, paw, strat, hydrology, bgc)
    out.plant_available_water[i, j, k] = PAW
    return out
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, paw::AbstractPlantAvailableWater, args...)
    i, j, k = @index(Global, NTuple)
    compute_plant_available_water!(out, i, j, k, grid, fields, paw, args...)
end
