"""
    $TYPEDEF

Simple forward Euler time stepping scheme.
"""
@kwdef struct ForwardEuler{NF} <: AbstractTimeStepper{NF}
    "Initial timestep size in seconds"
    Δt::NF = 300.0
end

ForwardEuler(::Type{NF}; kwargs...) where {NF} = ForwardEuler{NF}(; kwargs...)

timestepping(::ForwardEuler) = Explicit()

default_dt(euler::ForwardEuler) = euler.Δt

"""
    $TYPEDSIGNATURES

Forward Euler step `u ← u + Δt ∂u∂t` for the prognostic variables in `names`: compute the auxiliary
variables and tendencies at the current state, then apply a single [`explicit_stage!`](@ref). The
closure is left to the caller (see [`explicit_step!`](@ref)).
"""
function explicit_step!(integrator::ModelIntegrator, timestepper::ForwardEuler, Δt, names::Tuple)
    # Compute auxiliaries and tendencies
    update_state!(integrator, compute_tendencies = true)
    # Euler update of the assigned prognostic variables
    explicit_stage!(integrator.state, get_grid(integrator.model), timestepper, Δt, names)
    return nothing
end
