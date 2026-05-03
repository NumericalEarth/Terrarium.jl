"""
Base type for time-stepper state caches.
"""
abstract type AbstractTimeStepperCache{NF} end

# AbstractTimeStepper

"""
Base type for time steppers.
"""
abstract type AbstractTimeStepper{NF} end

"""
    is_initialized(timestepper::AbstractTimeStepper)

Return `true` if the timestepper is initialized, `false` otherwise.
"""
function is_initialized end

"""
    default_dt(timestepper::AbstractTimeStepper)

Get the current timestep size for the time stepper.
"""
function default_dt end

"""
    is_adaptive(timestepper::AbstractTimeStepper)

Return `true` if the given time stepper is adaptive, false otherwise.
"""
function is_adaptive end

"""
    timestep!(integrator::ModelIntegrator, timestepper::AbstractTimeStepper, Δt)

Advance prognostic variables of the `integrator` model by one time step based on the current state, or by `Δt` units of time.
"""
function timestep! end

"""
    timestep!(state, model::AbstractModel, timestepper::AbstractTimeStepper, Δt)

Apply any necessary corrections or model-specific time stepping logic after applying `timestepper` to the prognostic state
variables defined by `model`.
"""
timestep!(state, model::AbstractModel, timestepper::AbstractTimeStepper, Δt) = nothing

"""
    initialize(::AbstractTimeStepper, model, state) where {NF}

Initialize and return the time stepping state cache for the given time stepper.
"""
initialize(timestepper::AbstractTimeStepper, model, state) = timestepper

"""
    $SIGNATURES

Evaluate an explicit update `u += ∂u∂t*Δt` for all prognostic fields and their corresponding
tendencies. By default, this is implemented as a simple Euler update `u += dudt*Δt` which can
serve as a building block for more complex, multi-stage timesteppers. Where necessary,
additional dispatches of `explicit_step_kernel!(field, tendency, ::AbstractLandGrid, ::TimeStepper, Δt)`
can be defined to implement more specialized time-stepping schemes.
"""
function explicit_step!(state, grid::AbstractLandGrid, timestepper::AbstractTimeStepper, Δt)
    @assert is_initialized(timestepper)
    fastiterate(keys(state.prognostic)) do name
        # apply flux BCs, if present
        compute_z_bcs!(state.tendencies[name], state.prognostic[name], grid, state)
        # debug site post-BC
        debugsite!(explicit_step!, state.tendencies[name], name)
        # update prognostic state variable
        explicit_step!(state.prognostic[name], state.tendencies[name], grid, timestepper, Δt)
        # debug site post-step
        debugsite!(explicit_step!, state.prognostic[name], name)
    end
    fastiterate(state.namespaces) do ns
        explicit_step!(ns, grid, timestepper, Δt)
    end
    return nothing
end

"""
Accumulate `tendency*Δt` in the given prognostic `field`. This method can be overridden by specialized
timestepping schemes as needed.
"""
function explicit_step!(
        field::AbstractField{LX, LY, LZ},
        tendency::AbstractField{LX, LY, LZ},
        grid::AbstractLandGrid,
        timestepper::AbstractTimeStepper,
        Δt,
        args...
    ) where {LX, LY, LZ}
    launch!(
        grid, XYZ, explicit_step_xyz_kernel!,
        field, tendency, timestepper, Δt, args...
    )
    return nothing
end

function explicit_step!(
        field::AbstractField{LX, LY, Nothing},
        tendency::AbstractField{LX, LY, Nothing},
        grid::AbstractLandGrid,
        timestepper::AbstractTimeStepper,
        Δt,
        args...
    ) where {LX, LY}
    launch!(
        grid, XY, explicit_step_xy_kernel!,
        field, tendency, timestepper, Δt, args...
    )
    return nothing
end

@kernel function explicit_step_xyz_kernel!(
        field,
        grid,
        tendency,
        ::AbstractTimeStepper,
        Δt
    )
    i, j, k = @index(Global, NTuple)
    u = field
    ∂u∂t = tendency
    @inbounds let Δt = convert(eltype(tendency), Δt)
        u[i, j, k] += ∂u∂t[i, j, k] * Δt
    end
end

@kernel function explicit_step_xy_kernel!(
        field,
        grid,
        tendency,
        ::AbstractTimeStepper,
        Δt
    )
    i, j = @index(Global, NTuple)
    u = field
    ∂u∂t = tendency
    @inbounds let Δt = convert(eltype(tendency), Δt)
        u[i, j, 1] += ∂u∂t[i, j] * Δt
    end
end

# Default debug hooks
@inline debughook!(::typeof(explicit_step!), field, name) = checkfinite!(field, name)
