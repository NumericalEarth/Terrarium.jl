"""
    $TYPEDEF

General implementation of a 1D column model of soil energy, water, and carbon transport.

Properties:
$(TYPEDFIELDS)
"""
@parameterized @kwdef struct SoilModel{
        NF,
        GridType <: AbstractGrid,
        Soil <: AbstractSoil,
        Initializer <: AbstractInitializer,
        Timestepper <: AbstractTimeStepper,
    } <: AbstractSoilModel{NF, GridType}
    "Spatial grid type"
    grid::GridType

    "Soil processes"
    @component soil::Soil = SoilEnergyWaterCarbon(eltype(grid))

    "Physical constants"
    @component constants::PhysicalConstants{NF} = PhysicalConstants(eltype(grid))

    "State variable initializer"
    @component initializer::Initializer = DefaultInitializer(eltype(grid))

    "Time stepper: a single `AbstractTimeStepper` (e.g. `ForwardEuler`, `Heun`) or an `IMEX`"
    @component timestepper::Timestepper = default_timestepper(eltype(grid))
end

# Model interface methods

function initialize!(state, model::SoilModel)
    # run model/field initializers
    initialize!(state, model, model.initializer)
    # run process initializers
    initialize!(state, model.grid, model.soil, model.constants)
    return nothing
end

function compute_auxiliary!(state, model::SoilModel)
    compute_auxiliary!(state, model.grid, model.soil, model.constants)
    return nothing
end

function compute_boundary_conditions!(state, model::SoilModel)
    compute_boundary_conditions!(state, model.grid, model.soil)
    return nothing
end

function compute_tendencies!(state, model::SoilModel)
    compute_tendencies!(state, model.grid, model.soil, model.constants)
    return nothing
end

# Closures

function closure!(state, model::SoilModel)
    closure!(state, model.grid, model.soil, model.constants)
    return nothing
end

function invclosure!(state, model::SoilModel)
    invclosure!(state, model.grid, model.soil, model.constants)
    return nothing
end
