"""
    $TYPEDEF

Represents an explicit formulation of the two-phase heat conduction operator in 1D:

```math
\\frac{\\partial U(T,\\phi)}{\\partial t} = \\boldsymbol{\\nabla} \\cdot \\left[ \\kappa(T) \\boldsymbol{\\nabla}_x T(x,t) \\right]
```
where \$T\$ is temperature [K], \$U\$ is internal energy [J m⁻³], and \$\\kappa\$ is the thermal conductivity [W m K⁻¹].
"""
@kwdef struct ExplicitTwoPhaseHeatConduction <: AbstractHeatOperator end

"""
    $TYPEDEF

Standard implementation of the soil energy balance accounting for freezing and thawing of pore water/ice.
The `closure` field represents the temperature-energy closure \$U(T,\\phi)\$ which relates temperature to internal
energy via an arbitrary set of additional parameters \$\\phi\$ which are determined by the model configuration.

Properties:
$TYPEDFIELDS
"""
struct SoilEnergyBalance{
        NF,
        HeatOperator <: AbstractHeatOperator,
        EnergyClosure <: AbstractSoilEnergyClosure,
        ThermalProps <: SoilThermalProperties{NF},
    } <: AbstractSoilEnergyBalance{NF}
    "Heat transport operator"
    operator::HeatOperator

    "Closure relating energy and temperature"
    closure::EnergyClosure

    "Soil thermal properties"
    thermal_properties::ThermalProps
end

SoilEnergyBalance(
    ::Type{NF};
    operator::AbstractHeatOperator = ExplicitTwoPhaseHeatConduction(),
    closure::AbstractSoilEnergyClosure = SoilEnergyTemperatureClosure(),
    thermal_properties::SoilThermalProperties{NF} = SoilThermalProperties(NF),
) where {NF} = SoilEnergyBalance(operator, closure, thermal_properties)

variables(energy::SoilEnergyBalance) = (
    prognostic(:internal_energy, XYZ(); closure = energy.closure, units = u"J/m^3", desc = "Internal energy of the soil volume, including both latent and sensible components"),
    auxiliary(:ground_temperature, XY(), ground_temperature, energy, units = u"°C", desc = "Temperature of the uppermost ground or soil grid cell in °C"),
)

# Field constructor for ground_temperature that returns a view of the uppermost soil layer
function ground_temperature(energy::SoilEnergyBalance, grid, clock, fields)
    fgrid = get_field_grid(grid)
    # Use uppermost soil layer as ground temperature
    # TODO: Revisit this if/when we extend the vertical layers to include snow and canopy
    return @view fields.temperature[:, :, fgrid.Nz]
end

get_thermal_properties(energy::SoilEnergyBalance) = energy.thermal_properties

get_closure(energy::SoilEnergyBalance) = energy.closure

""" $TYPEDSIGNATURES """
function initialize!(
        state, grid,
        energy::SoilEnergyBalance,
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
compute_auxiliary!(state, grid, energy::SoilEnergyBalance, soil::AbstractSoil, args...) = nothing

""" $TYPEDSIGNATURES """
function compute_tendencies!(
        state, grid,
        energy::SoilEnergyBalance,
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
        energy::SoilEnergyBalance,
        args...
    )
    tendencies.internal_energy[i, j, k] += compute_energy_tendency(i, j, k, grid, fields, energy, args...)
    return nothing
end

""" $TYPEDSIGNATURES """
@propagate_inbounds function compute_energy_tendency(
        i, j, k, grid, fields,
        energy::SoilEnergyBalance{NF, <:ExplicitTwoPhaseHeatConduction},
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    # Operators require the underlying Oceananigans grid
    field_grid = get_field_grid(grid)

    # Divergence of heat fluxes
    ∂U∂t = -∂zᵃᵃᶜ(i, j, k, field_grid, diffusive_heat_flux, fields, energy, hydrology, strat, bgc)
    return ∂U∂t
end

""" $TYPEDSIGNATURES """
@propagate_inbounds function compute_thermal_conductivity(
        i, j, k, grid, fields,
        energy::SoilEnergyBalance,
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    soil = soil_volume(i, j, k, grid, fields, strat, hydrology, bgc)
    return compute_thermal_conductivity(energy.thermal_properties, soil)
end

# Diffusive heat flux term passed to ∂z operator
""" $TYPEDSIGNATURES """
@propagate_inbounds function diffusive_heat_flux(i, j, k, grid, fields, args...)
    # Get temperature field
    T = fields.temperature
    # Compute and ∂U∂tinterpolate conductivity to grid cell faces
    κ = ℑzᵃᵃᶠ(i, j, k, grid, compute_thermal_conductivity, fields, args...)
    # Fourier's law: q = -κ ∂T/∂z
    q = -κ * ∂zᵃᵃᶠ(i, j, k, grid, T)
    return q
end

# Kernels

@kernel inbounds = true function compute_tendencies_kernel!(tendencies, grid, fields, energy::SoilEnergyBalance, args...)
    i, j, k = @index(Global, NTuple)
    compute_energy_tendencies!(tendencies, i, j, k, grid, fields, energy, args...)
end
