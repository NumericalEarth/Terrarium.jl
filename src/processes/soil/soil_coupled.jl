"""
    $TYPEDEF

Coupled process type that encapsulates the coupling of soil energy, water, and carbon dynamics.
The stratigraphy parameterization determines how the vertical layering of the soil is parameterized.
"""
struct SoilEnergyWaterCarbon{
        NF,
        Stratigraphy <: AbstractStratigraphy{NF},
        Energy <: AbstractSoilThermodynamics{NF},
        Hydrology <: AbstractSoilHydrology{NF},
        Biogeochemistry <: AbstractSoilBiogeochemistry{NF},
    } <: AbstractSoil{NF}
    "Soil stratigraphy parameterization"
    strat::Stratigraphy

    "Soil energy balance process"
    energy::Energy

    "Soil hydrology (water balance) process"
    hydrology::Hydrology

    "Soil biogeochemistry process"
    biogeochem::Biogeochemistry
end

function SoilEnergyWaterCarbon(
        ::Type{NF};
        strat = HomogeneousSoilStratigraphy(NF),
        energy = SoilThermodynamics(NF),
        hydrology = SoilHydrology(NF),
        biogeochem = ConstantSoilCarbonDensity(NF)
    ) where {NF}
    return SoilEnergyWaterCarbon(strat, energy, hydrology, biogeochem)
end

# Process interface methods

"""
    $TYPEDSIGNATURES

Initialize the soil energy, water, and carbon state variables on `grid` given
the parameter values in `constants`.
"""
function initialize!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants
    )
    initialize!(state, grid, soil.hydrology, soil, constants)
    initialize!(state, grid, soil.biogeochem, soil, constants)
    initialize!(state, grid, soil.energy, soil, constants)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute auxiliary variables for soil energy, water, and carbon state variables
on `grid` based on the given values in `constants`.
"""
function compute_auxiliary!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants
    )
    # TODO: consider implementing fused kernel here?
    compute_auxiliary!(state, grid, soil.hydrology, soil, constants)
    compute_auxiliary!(state, grid, soil.biogeochem, soil, constants)
    compute_auxiliary!(state, grid, soil.energy, soil, constants)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute boundary conditions (and halo regions) for soil energy and hydrology. The sensible heat of
water entering or leaving the column through flux boundary conditions on `saturation_water_ice` (e.g.
[`InfiltrationFlux`](@ref)) is added to the energy tendency by [`compute_boundary_advected_heat!`](@ref),
after the hydrology boundary conditions (the saturation tendency then holds only their contribution).
"""
function compute_boundary_conditions!(state, grid, soil::SoilEnergyWaterCarbon)
    compute_boundary_conditions!(state, grid, soil.hydrology, soil.strat, soil.biogeochem)
    compute_boundary_advected_heat!(state, grid, soil.energy, soil.hydrology, soil.strat, soil.biogeochem)
    compute_boundary_conditions!(state, grid, soil.energy)
    compute_boundary_conditions!(state, grid, soil.biogeochem)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute tendencies for soil energy, water, and carbon state variables on `grid`
based on the given values in `constants`.

An optional `surface_hydrology` process may be supplied (by the coupled `LandModel`) so that its
evapotranspiration scheme (`get_evapotranspiration`) is applied as a sink term in the soil
water tendency. Without it (standalone soil), no evapotranspiration is removed from the soil.

The parent model's `timestepper` resolves the [`Timestepping`](@ref) classes of `internal_energy` and
`saturation_water_ice`, which are passed on to the energy and hydrology `compute_tendencies!`. Without a
timestepper (`nothing`) both are `Explicit()`.
"""
function compute_tendencies!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        surface_hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        timestepper::Optional{AbstractTimeStepper} = nothing
    )
    energy_timestepping = timestepping(timestepper, state, Val(:internal_energy))
    hydrology_timestepping = timestepping(timestepper, state, Val(:saturation_water_ice))
    evapotranspiration = isnothing(surface_hydrology) ? nothing : get_evapotranspiration(surface_hydrology)
    # TODO: consider implementing fused kernel here?
    compute_tendencies!(state, grid, soil.hydrology, soil, constants, evapotranspiration, hydrology_timestepping)
    compute_tendencies!(state, grid, soil.biogeochem, soil, constants)
    compute_tendencies!(state, grid, soil.energy, soil, constants, energy_timestepping)
    return nothing
end

# Closures

"""
    $TYPEDSIGNATURES

Compute the forward closure mapping for soil hydrology and energy, in that order.

An optional `surface_hydrology` process may be supplied (by the coupled `LandModel`) so that excess
water removed from an oversaturated soil surface is routed into the `surface_excess_water` pool
owned by its runoff scheme ([`get_surface_runoff`](@ref)). Without it (standalone soil), the excess
is discarded.
"""
function closure!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        surface_hydrology::Optional{AbstractSurfaceHydrology} = nothing
    )
    runoff = isnothing(surface_hydrology) ? nothing : get_surface_runoff(surface_hydrology)
    closure!(state, grid, get_closure(soil.hydrology), soil.hydrology, soil, runoff)
    closure!(state, grid, get_closure(soil.energy), soil.energy, soil, constants)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the inverse closure mapping for soil hydrology and energy, in that order.

See [`closure!`](@ref) for the role of the optional `surface_hydrology` process.
"""
function invclosure!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        surface_hydrology::Optional{AbstractSurfaceHydrology} = nothing
    )
    runoff = isnothing(surface_hydrology) ? nothing : get_surface_runoff(surface_hydrology)
    invclosure!(state, grid, get_closure(soil.hydrology), soil.hydrology, soil, runoff)
    invclosure!(state, grid, get_closure(soil.energy), soil.energy, soil, constants)
    return nothing
end

# Implicit operators: `SoilEnergyWaterCarbon` methods mapping `internal_energy` to the energy process and
# `saturation_water_ice` to the hydrology process. The process methods are in `energy/soil_energy_implicit.jl`
# and `hydrology/soil_hydrology_implicit.jl`.

"""
    $TYPEDSIGNATURES

Return the implicit operator of the soil prognostic variables `names` on `grid`: a
[`SplitOperator`](@ref) of per-variable operators that solves `saturation_water_ice` before
`internal_energy`, whatever the order of `names`. The heat step then advects heat with the water flux of
the current step (`water_flux`, see [`advective_heat_flux`](@ref)) and sees the end-of-step saturation,
while the Richards step sees the beginning-of-step temperature and liquid water fraction (the effect of
frozen ground on the hydraulic conductivity is lagged by one step). Other names follow in their given
order and are rejected by the per-variable fallback.
"""
function implicit_operator(soil::AbstractSoil, state, grid, names::Tuple{Vararg{Symbol}})
    soil_order = (:saturation_water_ice, :internal_energy)
    ordered = (filter(in(names), soil_order)..., filter(!in(soil_order), names)...)
    operators = NamedTuple{ordered}(map(name -> implicit_operator(soil, state, grid, Val(name)), ordered))
    return SplitOperator(operators)
end

"""
    implicit_operator(soil::SoilEnergyWaterCarbon, state, grid, ::Val{name})

Return the implicit operator of the soil prognostic variable `name` on `grid`: that of the energy
process for `internal_energy` and of the hydrology process for `saturation_water_ice`. The fallback
throws an `ArgumentError`.
"""
function implicit_operator(soil::AbstractSoil, state, grid, ::Val{name}) where {name}
    throw(
        ArgumentError(
            "no implicit operator is defined for variable `$name` of $(nameof(typeof(soil))); route the variable explicitly"
        )
    )
end

# Coupled soil level: `internal_energy` is owned by the soil energy process

implicit_operator(soil::SoilEnergyWaterCarbon, state, grid, ::Val{:internal_energy}) = implicit_operator(soil.energy, soil.hydrology, state, grid)

function linearize!(
        operator::AbstractLinearOperator, rhs, state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        explicit_stage, Δt,
        ::Iteration{:internal_energy}
    )
    linearize!(operator, rhs, state, grid, soil.energy, soil, constants, explicit_stage, Δt)
    return nothing
end

function implicit_update!(state, grid, soil::SoilEnergyWaterCarbon, operator::AbstractLinearOperator, explicit_stage, Δt, ::Iteration{:internal_energy})
    implicit_update!(state, grid, soil.energy, soil, operator, explicit_stage, Δt)
    return nothing
end

# The optional `surface_hydrology` is that of a coupled `LandModel`; its runoff (if any) receives excess
# surface water, as in the explicit `closure!`.
function implicit_closure!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        surface_hydrology::Optional{AbstractSurfaceHydrology},
        iteration::Iteration{:internal_energy}
    )
    runoff = isnothing(surface_hydrology) ? nothing : get_surface_runoff(surface_hydrology)
    implicit_closure!(state, grid, soil.energy, soil, constants, runoff, iteration)
    return nothing
end

# Coupled soil level: `saturation_water_ice` is owned by the soil hydrology process

implicit_operator(soil::SoilEnergyWaterCarbon, state, grid, ::Val{:saturation_water_ice}) = implicit_operator(soil.hydrology, state, grid)

function linearize!(
        operator::AbstractLinearOperator, rhs, state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        explicit_stage, Δt,
        ::Iteration{:saturation_water_ice}
    )
    linearize!(operator, rhs, state, grid, soil.hydrology, soil, constants, explicit_stage, Δt)
    return nothing
end

function implicit_update!(state, grid, soil::SoilEnergyWaterCarbon, operator::AbstractLinearOperator, explicit_stage, Δt, ::Iteration{:saturation_water_ice})
    implicit_update!(state, grid, soil.hydrology, soil, operator, explicit_stage, Δt)
    return nothing
end

# The optional `surface_hydrology` is that of a coupled `LandModel`; its runoff (if any) receives excess
# surface water, as in the explicit `closure!`.
function implicit_closure!(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        surface_hydrology::Optional{AbstractSurfaceHydrology},
        iteration::Iteration{:saturation_water_ice}
    )
    runoff = isnothing(surface_hydrology) ? nothing : get_surface_runoff(surface_hydrology)
    implicit_closure!(state, grid, soil.hydrology, soil, constants, runoff, iteration)
    return nothing
end
