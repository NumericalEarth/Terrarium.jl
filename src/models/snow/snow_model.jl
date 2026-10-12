"""
    $TYPEDEF

Minimal standalone model of a single-layer snowpack, intended for unit and differentiability testing.
Couples an [`AbstractSnow`](@ref) process with a prescribed atmosphere providing precipitation and air
temperature. The surface and basal heat fluxes (`surface_heat_flux`, `basal_heat_flux`) and the
`sublimation` rate are prescribed input fields; in a coupled land model these are supplied by the
surface energy balance and the snow→soil conduction.

Properties:
$(TYPEDFIELDS)
"""
@parameterized @kwdef struct SnowModel{
        NF,
        GridType <: AbstractGrid,
        Snow <: AbstractSnow,
        Atmosphere <: AbstractAtmosphere,
        Initializer <: AbstractInitializer,
        Timestepper <: AbstractTimeStepper,
    } <: AbstractSnowModel{NF, GridType}
    "Spatial grid type"
    grid::GridType

    "Snow processes"
    @component snow::Snow = SingleLayerSnow(eltype(grid))

    "Near-surface atmospheric conditions"
    @component atmosphere::Atmosphere = PrescribedAtmosphere(eltype(grid))

    "Physical constants"
    @component constants::PhysicalConstants{NF} = PhysicalConstants(eltype(grid))

    "State variable initializer"
    @component initializer::Initializer = DefaultInitializer(eltype(grid))

    "Time stepper: a single `AbstractTimeStepper` (e.g. `ForwardEuler`, `Heun`) or an `IMEX`"
    @component timestepper::Timestepper = default_timestepper(eltype(grid))
end

# Model interface methods

function initialize!(state, model::SnowModel)
    # run model/field initializers, then the snow process initializer (invclosure: T, W -> E)
    initialize!(state, model, model.initializer)
    initialize!(state, model.grid, model.snow, model.constants)
    return nothing
end

function compute_auxiliary!(state, model::SnowModel)
    compute_auxiliary!(state, model.grid, model.snow, model.constants, model.timestepper)
    return nothing
end

function compute_tendencies!(state, model::SnowModel)
    compute_tendencies!(state, model.grid, model.snow, model.constants, model.atmosphere, model.timestepper)
    return nothing
end

# Closures

function closure!(state, model::SnowModel)
    closure!(state, model.grid, get_closure(model.snow), model.snow, model.constants)
    return nothing
end

function invclosure!(state, model::SnowModel)
    invclosure!(state, model.grid, get_closure(model.snow), model.snow, model.constants)
    return nothing
end

# Time stepping

# The snow meltwater drainage is implicit under an IMEX timestepper.
timestepping(::AbstractVariable{:snow_water_equivalent}, ::SnowModel, ::AbstractIMEX) = Implicit()

# Forward the implicit operator query and the solver hooks to the snow.

function implicit_operator(model::SnowModel, state, names::Tuple{Vararg{Symbol}})
    operators = NamedTuple{names}(map(name -> implicit_operator(model.snow, state, model.grid, Val(name)), names))
    return SplitOperator(operators)
end

function linearize!(operator::AbstractLinearOperator, rhs, state, model::SnowModel, explicit_stage, Δt, iteration::Iteration)
    linearize!(operator, rhs, state, model.grid, model.snow, model.constants, explicit_stage, Δt, iteration)
    return nothing
end

function implicit_update!(state, model::SnowModel, operator::AbstractLinearOperator, explicit_stage, Δt, iteration::Iteration)
    implicit_update!(state, model.grid, model.snow, model.constants, operator, explicit_stage, Δt, iteration)
    return nothing
end

function implicit_closure!(state, model::SnowModel, iteration::Iteration)
    implicit_closure!(state, model.grid, model.snow, model.constants, iteration)
    return nothing
end

function Terrarium.timestep!(state::StateVariables, model::SnowModel, timestepper::Terrarium.AbstractTimeStepper, Δt)
    return timestep!(state, model, model.snow, timestepper, Δt)
end

function Terrarium.timestep!(
        state::StateVariables,
        model::AbstractModel{NF},
        snow::AbstractSnow,
        args...
    ) where {NF}
    # Enforce SWE >= 0
    # TODO: Rewrite to ensure mass conservation
    out = (; snow_water_equivalent, snow_energy) = state
    launch!(model.grid, XY, enforce_snow_constraints!, out, snow)
    return nothing
end

@kernel inbounds = true function enforce_snow_constraints!(out, grid, snow::AbstractSnow{NF}) where {NF}
    i, j = @index(Global, NTuple)
    out.snow_water_equivalent[i, j, end] = max(out.snow_water_equivalent[i, j], zero(NF))
    out.snow_energy[i, j, end] = out.snow_energy[i, j] * (out.snow_water_equivalent[i, j] > zero(NF))
end
