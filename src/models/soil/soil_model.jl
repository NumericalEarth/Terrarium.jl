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

    "Time stepper: a single `AbstractTimeStepper` (e.g. `ForwardEuler`, `Heun`) or an `IMEX`; defaults to `IMEX` for Richards flow and `ForwardEuler` otherwise"
    @component timestepper::Timestepper = default_timestepper(SoilModel, soil)
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
    compute_tendencies!(state, model.grid, model.soil, model.constants, nothing, get_timestepper(model))
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

# Time stepping

# The default timestepper follows the default class of the soil water flow: IMEX (implicit heat conduction and
# Richards flow, Δt = 1 h) if it is implicit, and ForwardEuler otherwise (e.g. `NoFlow`), which also supports
# value boundary conditions.
default_timestepper(::Type{<:SoilModel}, soil::AbstractSoil{NF}) where {NF} =
    default_timestepper(SoilModel, NF, default_timestepping(get_hydrology(soil)))
default_timestepper(::Type{<:SoilModel}, ::Type{NF}, ::Explicit) where {NF} = ForwardEuler(NF)
default_timestepper(::Type{<:SoilModel}, ::Type{NF}, ::Implicit) where {NF} = IMEX(ForwardEuler(NF; Δt = 3600), BackwardEuler(NF))

# Under an IMEX timestepper, soil heat conduction is implicit by default, and so is soil water flow if it
# follows the Richards equation (see `default_timestepping`).
timestepping(::AbstractVariable{:internal_energy}, ::SoilModel, ::AbstractIMEX) = Implicit()
timestepping(::AbstractVariable{:saturation_water_ice}, model::SoilModel, ::AbstractIMEX) = default_timestepping(get_hydrology(model.soil))

# Forward the implicit operator query and the solver hooks to the coupled soil, which maps each variable to
# the process that owns it.

implicit_operator(model::SoilModel, state, names::Tuple{Vararg{Symbol}}) = implicit_operator(model.soil, state, get_grid(model), names)

function linearize!(operator::AbstractLinearOperator, rhs, state, model::SoilModel, explicit_stage, Δt, iteration::Iteration)
    linearize!(operator, rhs, state, get_grid(model), model.soil, model.constants, explicit_stage, Δt, iteration)
    return nothing
end

function implicit_update!(state, model::SoilModel, operator::AbstractLinearOperator, explicit_stage, Δt, iteration::Iteration)
    implicit_update!(state, get_grid(model), model.soil, operator, explicit_stage, Δt, iteration)
    return nothing
end

function implicit_closure!(state, model::SoilModel, iteration::Iteration)
    # no surface hydrology: excess surface water is discarded
    implicit_closure!(state, get_grid(model), model.soil, model.constants, nothing, iteration)
    return nothing
end
