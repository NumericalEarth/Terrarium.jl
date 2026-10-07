"""
    $TYPEDEF

Simple forward 2nd order Heun / improved Euler time stepping scheme.
"""
@kwdef struct Heun{NF} <: AbstractTimeStepper{NF}
    "Initial timestep size in seconds"
    Δt::NF = 300.0
end

Heun(::Type{NF}; kwargs...) where {NF} = Heun{NF}(; kwargs...)

timestepping(::Heun) = Explicit()

default_dt(heun::Heun) = heun.Δt

is_adaptive(heun::Heun) = false

"""
    $TYPEDEF

Cache for the [`Heun`](@ref) scheme, holding copies of the prognostic state `u₀` and the predictor
tendencies `∂u∂t₀` (Heun steps in-place on `state`, so only these two are needed). The cache mirrors the
namespace tree of the `state`: `namespaces` holds a sub-`HeunCache` per namespace so that namespaced
prognostic variables are staged (saved/restored/averaged) consistently with how [`explicit_step!`](@ref)
recurses into namespaces.
"""
struct HeunCache{NF, P, T, NS} <: AbstractTimeStepperCache{NF}
    prognostic::P
    tendencies::T
    namespaces::NS
end

# Allocate the Heun cache with a `similar` field for each prognostic and tendency field, recursing into
# each namespace so the cache tree mirrors the state tree.
function initialize(heun::Heun, state::AbstractStateVariables)
    prognostic = map(similar, getfield(state, :prognostic))
    tendencies = map(similar, getfield(state, :tendencies))
    namespaces = map(ns -> initialize(heun, ns), getfield(state, :namespaces))
    return HeunCache{eltype(state), typeof(prognostic), typeof(tendencies), typeof(namespaces)}(prognostic, tendencies, namespaces)
end

function Adapt.adapt_structure(to, cache::HeunCache{NF}) where {NF}
    prognostic = Adapt.adapt_structure(to, cache.prognostic)
    tendencies = Adapt.adapt_structure(to, cache.tendencies)
    namespaces = Adapt.adapt_structure(to, cache.namespaces)
    return HeunCache{NF, typeof(prognostic), typeof(tendencies), typeof(namespaces)}(prognostic, tendencies, namespaces)
end

# Save the prognostic/tendency fields named in `names` into the Heun cache, recursing into namespaces.
# As in `explicit_step!`, iterating the (static) prognostic names with an `∈ names` guard keeps this type
# stable; the same `names` selection is threaded into each namespace.
function save_cache!(cache::HeunCache, state::StateVariables, names::Tuple)
    fastiterate(prognostic_names(state)) do name
        if name ∈ names
            copyto!(cache.prognostic[name], getfield(state, :prognostic)[name])
            copyto!(cache.tendencies[name], getfield(state, :tendencies)[name])
        end
    end
    fastiterate((c, ns) -> save_cache!(c, ns, names), cache.namespaces, getfield(state, :namespaces))
    return nothing
end

# Restore the prognostic fields named in `names` from the Heun cache, recursing into namespaces.
function restore_prognostic!(cache::HeunCache, state::StateVariables, names::Tuple)
    fastiterate(prognostic_names(state)) do name
        if name ∈ names
            copyto!(getfield(state, :prognostic)[name], cache.prognostic[name])
        end
    end
    fastiterate((c, ns) -> restore_prognostic!(c, ns, names), cache.namespaces, getfield(state, :namespaces))
    return nothing
end

# Average the tendencies named in `names` with the saved predictor tendencies in-place, recursing into
# namespaces: state.tendencies ← (state.tendencies + cache.tendencies) / 2.
function average_tendencies!(cache::HeunCache, state::StateVariables, names::Tuple)
    fastiterate(prognostic_names(state)) do name
        if name ∈ names
            tendency = getfield(state, :tendencies)[name]
            tendency .= (tendency .+ cache.tendencies[name]) ./ 2
        end
    end
    fastiterate((c, ns) -> average_tendencies!(c, ns, names), cache.namespaces, getfield(state, :namespaces))
    return nothing
end

# Step only the prognostic variables named in `names` with the Heun scheme.
function timestep!(integrator::ModelIntegrator, timestepper::Heun, Δt, names::Tuple)
    (; model, state, inputs) = integrator
    grid = get_grid(model)
    # fetch Heun's cache (the whole timestepper_cache for a single Heun, or the explicit slot under IMEX)
    cache = get_cache(state.timestepper_cache, timestepper)

    # Predictor: compute tendencies ∂u∂t₀ at the current state u₀
    update_state!(state, model, inputs, compute_tendencies = true)
    # Cache u₀ and ∂u∂t₀ for the assigned variables
    save_cache!(cache, state, names)

    # Step state forward in place using ∂u∂t₀
    explicit_step!(state, grid, timestepper, Δt, names)
    # Call timestep! on model
    timestep!(state, model, timestepper, Δt)
    # Apply closure relations
    closure!(state, model)

    # Recompute tendencies ∂u∂t₁ at the predictor state (u_pred, t + Δt)
    update_state!(state, model, inputs, compute_tendencies = true)
    # Average tendencies in place: state.tendencies ← (∂u∂t₀ + ∂u∂t₁) / 2
    average_tendencies!(cache, state, names)
    # Restore the prognostic state to u₀ before the corrector explicit step
    restore_prognostic!(cache, state, names)

    # Corrector: u ← u₀ + Δt · averaged tendencies
    explicit_step!(state, grid, timestepper, Δt, names)
    # Call timestep! on model
    timestep!(state, model, timestepper, Δt)
    # Apply closure relations
    closure!(state, model)
    return nothing
end
