"""
    $TYPEDEF

Simple model wrapper for the `SurfaceEnergyBalance` that couples it with
an `AbstractAtmosphere` to provide meteorological inputs. This model type
is mostly intended for testing but could also be used for simple energy
balance calculations from prescribed meteorological and ground temperature
conditions.
"""
@parameterized @kwdef struct SurfaceEnergyModel{
        NF,
        GridType <: AbstractGrid{NF},
        SEB <: AbstractSurfaceEnergyBalance,
        Atmosphere <: AbstractAtmosphere,
        Initializer <: AbstractInitializer,
        Timestepper <: AbstractTimeStepper{NF},
    } <: AbstractSurfaceEnergyModel{NF, GridType}
    "Spatial grid"
    grid::GridType

    "Atmospheric inputs"
    @component atmosphere::Atmosphere = PrescribedAtmosphere(eltype(grid))

    "Surface energy balance scheme"
    @component surface_energy_balance::SEB = SurfaceEnergyBalance(eltype(grid))

    "Physical constants"
    @component constants::PhysicalConstants{NF} = PhysicalConstants(eltype(grid))

    "State variable initializer"
    @component initializer::Initializer = DefaultInitializer(eltype(grid))

    "Time stepper: a single `AbstractTimeStepper` (e.g. `ForwardEuler`, `Heun`) or an `IMEX`"
    @component timestepper::Timestepper = default_timestepper(eltype(grid))
end

function compute_auxiliary!(state, model::SurfaceEnergyModel)
    compute_auxiliary!(state, model.grid, model.atmosphere)
    compute_auxiliary!(state, model.grid, model.surface_energy_balance)
    return nothing
end

function compute_boundary_conditions!(state, model::SurfaceEnergyModel)
    solve_surface_energy_balance!(state, model.grid, model.surface_energy_balance, model.constants, model.atmosphere)
    return nothing
end

function compute_tendencies!(state, ::SurfaceEnergyModel)
    return nothing
end
