"""
    $TYPEDEF

Simple model wrapper for the `SurfaceEnergyBalance` that couples it with
an `AbstractAtmosphere` to provide meteorological inputs. This model type
is mostly intended for testing but could also be used for simple energy
balance calculations from prescribed meteorological and ground temperature
conditions.
"""
struct SurfaceEnergyModel{
        NF,
        GridType <: AbstractLandGrid{NF},
        SEB <: AbstractSurfaceEnergyBalance,
        Atmosphere <: AbstractAtmosphere,
        Initializer <: AbstractInitializer,
    } <: AbstractSurfaceEnergyModel{NF, GridType}
    "Spatial grid"
    grid::GridType

    "Atmospheric inputs"
    atmosphere::Atmosphere

    "Surface energy balance scheme"
    surface_energy_balance::SEB

    "Physical constants"
    constants::PhysicalConstants

    "State variable initializer"
    initializer::Initializer
end

function SurfaceEnergyModel(
        grid::AbstractLandGrid{NF},
        surface_energy_balance::AbstractSurfaceEnergyBalance = SurfaceEnergyBalance(NF);
        atmosphere::AbstractAtmosphere = PrescribedAtmosphere(NF),
        constants::PhysicalConstants = PhysicalConstants(NF),
        initializer::AbstractInitializer = DefaultInitializer(eltype(grid))
    ) where {NF}
    return SurfaceEnergyModel(grid, atmosphere, surface_energy_balance, constants, initializer)
end

function compute_auxiliary!(state, model::SurfaceEnergyModel)
    compute_auxiliary!(state, model.grid, model.atmosphere)
    compute_auxiliary!(state, model.grid, model.surface_energy_balance, model.constants, model.atmosphere)
    return nothing
end

function compute_tendencies!(state, ::SurfaceEnergyModel)
    return nothing
end
