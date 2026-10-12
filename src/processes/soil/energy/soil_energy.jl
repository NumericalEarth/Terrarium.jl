"""
    $TYPEDEF

Soil thermal dynamics with freezing and thawing of pore water. The internal energy `U` evolves by
vertical heat conduction and advection of sensible heat by the water flux `q`:

    ∂U/∂t = ∂z(κ ∂z T) - ∂z(C_w T q),

with temperature `T` (°C), thermal conductivity `κ` (W/m/K), and volumetric heat capacity of liquid
water `C_w` (J/m³/K). The `closure` relates temperature and internal energy, \$U(T,\\phi)\$, with
additional parameters \$\\phi\$ set by the model configuration.

Properties:
$TYPEDFIELDS
"""
struct SoilThermodynamics{
        NF,
        EnergyClosure <: AbstractEnergyClosure,
        ThermalProps <: SoilThermalProperties{NF},
    } <: AbstractSoilThermodynamics{NF}
    "Closure relating energy and temperature"
    closure::EnergyClosure

    "Soil thermal properties"
    thermal_properties::ThermalProps
end

SoilThermodynamics(
    ::Type{NF};
    closure::AbstractEnergyClosure = SoilEnergyTemperatureClosure(),
    thermal_properties::SoilThermalProperties{NF} = SoilThermalProperties(NF),
) where {NF} = SoilThermodynamics(closure, thermal_properties)

# TODO: Add base `AbstractParameterization` type and then (hopefully) remove
Adapt.@adapt_structure SoilThermodynamics

variables(energy::SoilThermodynamics) = (
    prognostic(:internal_energy, Ground(XYZ()); closure = energy.closure, units = u"J/m^3", desc = "Internal energy of the soil volume, including both latent and sensible components"),
    auxiliary(:ground_temperature, Ground(Top(z = Center())), ground_temperature, energy, units = u"°C", desc = "Temperature of the uppermost ground or soil grid cell in °C"),
)

# Field constructor for ground_temperature that returns a view of the uppermost soil layer
function ground_temperature(grid, clock, fields, energy::SoilThermodynamics)
    ground_grid = ground_domain(grid)
    # Use uppermost soil layer as ground temperature
    # TODO: Revisit this if/when we extend the vertical layers to include snow and canopy
    return @view fields.temperature[:, :, ground_grid.Nz]
end

get_thermal_properties(energy::SoilThermodynamics) = energy.thermal_properties

get_closure(energy::SoilThermodynamics) = energy.closure

""" $TYPEDSIGNATURES """
function initialize!(
        state, grid,
        energy::SoilThermodynamics,
        soil::AbstractSoil,
        constants::PhysicalConstants,
        args...
    )
    # inverse closure (temperature -> energy); assumes the temperature is already initialized
    # TODO: generalize for the rare case where energy is the initial condition
    invclosure!(state, grid, energy.closure, energy, soil, constants)
    return nothing
end

""" $TYPEDSIGNATURES """
compute_auxiliary!(state, grid, energy::SoilThermodynamics, soil::AbstractSoil, args...) = nothing

""" $TYPEDSIGNATURES """
function compute_boundary_conditions!(state, grid, ::SoilThermodynamics)
    fill_halo_regions!(state.temperature, state)
    compute_z_bcs!(state.tendencies.internal_energy, state.internal_energy, architecture(grid), state.clock, state.inputs)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the internal energy tendency of `energy` for the [`Timestepping`](@ref) class `timestepping`
of `internal_energy` (see [`timestepping(state, ::Val)`](@ref timestepping)). [`Explicit`](@ref): the
full divergence of the diffusive and advective heat fluxes. [`Implicit`](@ref): only the diffusive
fluxes and the advective inflow through the boundary faces; the rest is integrated by the implicit
sub-stepper (see [`Picard`](@ref)). Flux boundary conditions are included in both cases.
"""
function compute_tendencies!(
        state, grid,
        energy::SoilThermodynamics,
        soil::AbstractSoil,
        constants::Optional{PhysicalConstants} = nothing,
        timestepping::Timestepping = Explicit()
    )
    procs = (get_hydrology(soil), get_stratigraphy(soil), get_biogeochemistry(soil))
    tendencies = tendency_fields(state, energy)
    fields = get_fields(state, energy, procs...)
    launch!(grid, XYZ, compute_tendencies_kernel!, tendencies, fields, energy, timestepping, procs...)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Add the full internal energy tendency ([`compute_energy_tendency`](@ref)) at `i, j, k` under
[`Explicit`](@ref) time stepping.
"""
@propagate_inbounds function compute_energy_tendencies!(
        tendencies, i, j, k, grid, fields,
        energy::SoilThermodynamics,
        ::Explicit,
        args...
    )
    tendencies.internal_energy[i, j, k] += compute_energy_tendency(i, j, k, grid, fields, energy, args...)
    return nothing
end

"""
    $TYPEDSIGNATURES

Add the explicit part of the internal energy tendency at `i, j, k` under [`Implicit`](@ref) time
stepping: the diffusive fluxes through the boundary faces ([`compute_boundary_energy_tendency`](@ref))
and the heat of water entering through them ([`compute_boundary_advective_energy_tendency`](@ref)).
"""
@propagate_inbounds function compute_energy_tendencies!(
        tendencies, i, j, k, grid, fields,
        energy::SoilThermodynamics,
        ::Implicit,
        args...
    )
    tendencies.internal_energy[i, j, k] +=
        compute_boundary_energy_tendency(i, j, k, grid, fields, energy, args...) +
        compute_boundary_advective_energy_tendency(i, j, k, grid, fields, energy, args...)
    return nothing
end

"""
    $TYPEDSIGNATURES

Upwind advective heat flux `C_w T_up q` (W/m², positive upward) at face `i, j, k`, with `q` (m/s) the
`water_flux` of the step ([`compute_advective_water_flux`](@ref)) and `C_w` the `water` heat capacity of
the [`SoilThermalProperties`](@ref), consistent with the energy closure. `T_up` is the temperature of cell
`k` for `q < 0` and of cell `k - 1` otherwise, clamped to the column at the boundary faces. Only the
sensible heat of liquid water is advected; ice is immobile.

Under [`Implicit`](@ref) time stepping, only the boundary inflow is explicit
([`compute_boundary_advective_energy_tendency`](@ref)); the rest is part of the implicit heat operator
([`compute_implicit_advective_velocity!`](@ref)). The heat of water entering through flux boundary
conditions on `saturation_water_ice` (e.g. [`InfiltrationFlux`](@ref)) is added by
[`compute_boundary_advected_heat!`](@ref).
"""
@propagate_inbounds function advective_heat_flux(
        i, j, k, grid, fields,
        energy::SoilThermodynamics,
        hydrology::SoilHydrology{NF, RichardsEq},
        args...
    ) where {NF}
    ground_grid = ground_domain(grid)
    Nz = size(ground_grid, 3)
    q = compute_advective_water_flux(i, j, k, grid, fields)
    # upwind cell, clamped to the column so that boundary faces use the adjacent interior cell
    k_up = clamp(ifelse(q < zero(q), k, k - 1), 1, Nz)
    T_up = fields.temperature[i, j, k_up]
    C_w = energy.thermal_properties.heat_capacities.water
    return C_w * T_up * q
end

"""
    $TYPEDSIGNATURES

Water flux `q` (m/s, positive upward) at face `i, j, k` that advects heat: the `water_flux` of the
step, i.e. the Darcy flux at `tⁿ` (boundary faces, and interior faces under explicit hydrology) or of the
implicit update (see [`compute_water_flux`](@ref) and [`compute_implicit_water_flux`](@ref)).
"""
@propagate_inbounds compute_advective_water_flux(i, j, k, grid, fields) = fields.water_flux[i, j, k]

"""
    $TYPEDSIGNATURES

Explicit advective energy tendency at `i, j, k` under [`Implicit`](@ref) time stepping: the heat of
water entering the column through the boundary faces with the Darcy flux at `tⁿ`, at the temperature of
the boundary cell. Outflow is implicit ([`compute_implicit_advection_tendency`](@ref)), so that it cannot
remove more heat than a cell holds.
"""
@propagate_inbounds function compute_boundary_advective_energy_tendency(
        i, j, k, grid, fields,
        energy::SoilThermodynamics,
        hydrology::SoilHydrology{NF, RichardsEq},
        args...
    ) where {NF}
    ground_grid = ground_domain(grid)
    Nz = size(ground_grid, 3)
    Cw = energy.thermal_properties.heat_capacities.water
    T = fields.temperature[i, j, k]
    # inflow through the bottom (upward) and top (downward) faces of the column
    q_bottom = max(compute_advective_water_flux(i, j, 1, grid, fields), zero(NF))
    q_top = min(compute_advective_water_flux(i, j, Nz + 1, grid, fields), zero(NF))
    F_bottom = ifelse(k == 1, Cw * T * q_bottom, zero(T))
    F_top = ifelse(k == Nz, Cw * T * q_top, zero(T))
    Q_adv = -(F_top - F_bottom) / Δzᵃᵃᶜ(i, j, k, ground_grid)
    return Q_adv
end

@inline advective_heat_flux(i, j, k, grid, fields, energy::SoilThermodynamics, hydrology::SoilHydrology{NF, NoFlow}, args...) where {NF} = zero(grid)

"""
    $TYPEDSIGNATURES

Add the sensible heat of water crossing the column boundaries through flux boundary conditions on
`saturation_water_ice` (e.g. [`InfiltrationFlux`](@ref)) to the internal energy tendency
([`compute_boundary_advected_energy_tendency`](@ref)). Must be called right after the hydrology boundary
conditions, while the saturation tendency holds only their contributions. No-op without water flow.
"""
function compute_boundary_advected_heat!(
        state, grid,
        energy::SoilThermodynamics,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    fields = get_fields(state, energy, hydrology, strat, bgc)
    launch!(
        grid, XYZ, compute_boundary_advected_heat_kernel!,
        state.tendencies.internal_energy, fields, state.tendencies.saturation_water_ice, energy, hydrology, strat, bgc
    )
    return nothing
end

compute_boundary_advected_heat!(state, grid, energy::AbstractSoilThermodynamics, hydrology::AbstractSoilHydrology, args...) = nothing

"""
    $TYPEDSIGNATURES

Internal energy tendency `C_w T φ G_s` (W/m³) at `i, j, k` of water entering (`G_s > 0`) or leaving
(`G_s < 0`) the cell through flux boundary conditions on `saturation_water_ice`, with
`G_s = saturation_tendency[i, j, k]`, porosity `φ`, and cell temperature `T`. Infiltration thus enters
at the temperature of the top cell.
"""
@propagate_inbounds function compute_boundary_advected_energy_tendency(
        i, j, k, grid, fields, saturation_tendency,
        energy::SoilThermodynamics,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    por = porosity(i, j, k, grid, fields, strat, bgc)
    T = fields.temperature[i, j, k]
    C_w = energy.thermal_properties.heat_capacities.water
    ∂s∂t = saturation_tendency[i, j, k]
    return C_w * T * por * ∂s∂t
end

"""
    compute_redistributed_heat!(out, i, j, k, grid, fields, energy, strat, bgc, k_source, Δs)

Change `out.internal_energy` of cell `i, j, k` by the sensible heat `C_w T φ Δs` of a saturation change
`Δs` by redistributed water, with `C_w` the `water` heat capacity of the `energy` process, `φ` the
porosity of cell `k`, and `T` the temperature of the cell `k_source` that the water leaves. A no-op
without a [`SoilThermodynamics`](@ref) `energy` process.
"""
@propagate_inbounds function compute_redistributed_heat!(
        out, i, j, k, grid, fields,
        energy::SoilThermodynamics,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        k_source,
        Δs
    )
    φ = porosity(i, j, k, grid, fields, strat, bgc)
    T = fields.temperature[i, j, k_source]
    Cw = energy.thermal_properties.heat_capacities.water
    out.internal_energy[i, j, k] += Cw * T * φ * Δs
    return nothing
end

""" $TYPEDSIGNATURES """
@propagate_inbounds function compute_thermal_conductivity(
        i, j, k, grid, fields,
        energy::SoilThermodynamics,
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    soil = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    return compute_thermal_conductivity(energy.thermal_properties, soil)
end

# Kernels

@kernel inbounds = true function compute_boundary_advected_heat_kernel!(energy_tendency, grid, fields, saturation_tendency, args...)
    i, j, k = @index(Global, NTuple)
    energy_tendency[i, j, k] += compute_boundary_advected_energy_tendency(i, j, k, grid, fields, saturation_tendency, args...)
end

@kernel inbounds = true function compute_tendencies_kernel!(tendencies, grid, fields, energy::SoilThermodynamics, timestepping::Timestepping, args...)
    i, j, k = @index(Global, NTuple)
    compute_energy_tendencies!(tendencies, i, j, k, grid, fields, energy, timestepping, args...)
end
