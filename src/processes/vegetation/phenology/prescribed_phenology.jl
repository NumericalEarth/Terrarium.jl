"""
    $TYPEDEF

Prescribed vegetation phenology where `leaf_area_index` is treated as a (possibly time-varying) input variable.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct PrescribedPhenology{NF} <: AbstractPhenology{NF} end

PrescribedPhenology(::Type{NF}) where {NF} = PrescribedPhenology{NF}()

variables(::PrescribedPhenology) = (
    input(:leaf_area_index, Canopy(XY())), # Leaf Area Index [m²/m²]
)

# if PlantTraits are given, also compute phenology factor
variables(phenol::PrescribedPhenology, traits::PlantTraits) = (
    auxiliary(:phenology_factor, Canopy(XY()), kernel(phenology_factor, phenol, traits)),
    variables(phenol)...,
)

@propagate_inbounds phenology_factor(i, j, grid, fields, ::PrescribedPhenology{NF}, traits::PlantTraits{NF}) where {NF} =
    compute_relative_leaf_area(i, j, grid, fields, traits)

@inline compute_auxiliary!(state, grid, phenology::PrescribedPhenology, args...) = nothing

@inline compute_tendencies!(state, grid, phenology::PrescribedPhenology, args...) = nothing
