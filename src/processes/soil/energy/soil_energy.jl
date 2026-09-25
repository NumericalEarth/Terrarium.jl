"""
    $TYPEDEF

Standard implementation of the soil thermal dynamics accounting for freezing and thawing of pore water/ice.
The `closure` field represents the temperature-energy closure \$U(T,\\phi)\$ which relates temperature to internal
energy via an arbitrary set of additional parameters \$\\phi\$ which are determined by the model configuration.

Properties:
$TYPEDFIELDS
"""
struct SoilThermodynamics{
        NF,
        HeatOperator <: AbstractHeatOperator,
        EnergyClosure <: AbstractEnergyClosure,
        ThermalProps <: SoilThermalProperties{NF},
    } <: AbstractSoilThermodynamics{NF}
    "Heat transport operator"
    operator::HeatOperator

    "Closure relating energy and temperature"
    closure::EnergyClosure

    "Soil thermal properties"
    thermal_properties::ThermalProps
end

SoilThermodynamics(
    ::Type{NF};
    operator::AbstractHeatOperator = ExplicitTwoPhaseHeatConduction(),
    closure::AbstractEnergyClosure = SoilEnergyTemperatureClosure(),
    thermal_properties::SoilThermalProperties{NF} = SoilThermalProperties(NF),
) where {NF} = SoilThermodynamics(operator, closure, thermal_properties)

# TODO: Add base `AbstractParameterization` type and then (hopefully) remove
Adapt.@adapt_structure SoilThermodynamics

variables(energy::SoilThermodynamics) = (
    prognostic(:internal_energy, XYZ(); closure = energy.closure, units = u"J/m^3", desc = "Internal energy of the soil volume, including both latent and sensible components"),
    auxiliary(:ground_temperature, XY(), ground_temperature, energy, units = u"°C", desc = "Temperature of the uppermost ground or soil grid cell in °C"),
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
    # Initialize by evaluating inverse closure (temperature -> energy)
    # Note that this assumes the temperature state to have already been initialized!
    # TODO: We may need to generalize this for rare cases where energy is specified as
    # the initial condition.
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

""" $TYPEDSIGNATURES """
function compute_tendencies!(
        state, grid,
        energy::SoilThermodynamics,
        soil::AbstractSoil,
        args...
    )
    # Get dependencies
    procs = (get_hydrology(soil), get_stratigraphy(soil), get_biogeochemistry(soil))
    # Get output (tendency) fields
    tendencies = tendency_fields(state, energy)
    # Get other fields (does not include tendencies)
    fields = get_fields(state, energy, procs...)
    launch!(grid, XYZ, compute_tendencies_kernel!, tendencies, fields, energy, procs...)
    return nothing
end

# Kernel functions

""" $TYPEDSIGNATURES """
@propagate_inbounds function compute_energy_tendencies!(
        tendencies, i, j, k, grid, fields,
        energy::SoilThermodynamics,
        args...
    )
    tendencies.internal_energy[i, j, k] += compute_energy_tendency(i, j, k, grid, fields, energy, args...)
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

@kernel inbounds = true function compute_tendencies_kernel!(tendencies, grid, fields, energy::SoilThermodynamics, args...)
    i, j, k = @index(Global, NTuple)
    compute_energy_tendencies!(tendencies, i, j, k, grid, fields, energy, args...)
end
