using Terrarium
using Test

# mock a simple model with exponential dynamics (and a constant offset) to test time steppers

@kwdef struct ExpModel{NF, Grid <: Terrarium.AbstractLandGrid{NF}, I, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
    grid::Grid
    initializer::I = DefaultInitializer(eltype(grid))
    timestepper::TS = ForwardEuler(eltype(grid))
end

Terrarium.variables(::ExpModel) = (
    Terrarium.prognostic(:u, Terrarium.Ground(Terrarium.XY())),
    Terrarium.auxiliary(:v, Terrarium.Ground(Terrarium.XY())),
)

# just a constant offset (we could do it differently but this is for testing auxilitary as wel)
function Terrarium.compute_auxiliary!(state, model::ExpModel)
    return state.auxiliary.v .= 0.1
end

# du/dt = u + c = u + 0.1
function Terrarium.compute_tendencies!(state, model::ExpModel)
    return set!(state.tendencies.u, state.prognostic.u + state.auxiliary.v)
end

@testset "ExpModel: Heun and Euler time steppers" begin

    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    model_euler = ExpModel(grid)
    model_heun = ExpModel(grid; timestepper = Heun())

    initializers = (u = 0.0, v = 0.1)
    integrator_heun = initialize(model_heun; initializers)
    integrator_euler = initialize(model_euler; initializers)

    # test that Heun estimate is more accurate (larger value than Euler here)
    # test that both are what we expect
    timestep!(integrator_heun)
    timestep!(integrator_euler)

    @test integrator_heun.state.u[2] > integrator_euler.state.u[2]

    # Euler: expected value: u = 0.1 * Δt
    dt_euler = default_dt(integrator_euler)
    @test integrator_euler.state.u[2] == 0.1 * dt_euler

    # Heun: expected value: u = (0.1Δt + (0.1 * Δt + 0.1) * Δt) / 2
    dt_heun = default_dt(integrator_heun)
    @test integrator_heun.state.u[2] == (0.1 * dt_heun + (0.1 * dt_heun + 0.1) * dt_heun) / 2
end

# Use timestep!(state, model, timestepper, Δt) to clip negative values in an super simple example sim
@testset "ExpModel: clip negative values" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    model = ExpModel(grid)

    Terrarium.timestep!(state, model::ExpModel, timestepper::ForwardEuler, Δt) = begin
        state.u[2] = max(state.u[2], 0.0)
    end

    initializers = (u = -20, v = -5.0)
    integrator = initialize(model; initializers)

    # Test that timestep! clips negative values
    timestep!(integrator)

    @test integrator.state.u[2] >= 0.0
end

# mock a model with the same exponential dynamics as `ExpModel`, but with the prognostic
# variable (and its auxiliary offset and an input) living inside a namespace `:inner`.
# This exercises time stepping of prognostic and input variables defined in namespaces and
# also tests that actually the correct timestepper is used in the namespace as well.
@kwdef struct NamespacedExpModel{NF, Grid <: Terrarium.AbstractLandGrid{NF}, I, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
    grid::Grid
    initializer::I = DefaultInitializer(eltype(grid))
    timestepper::TS = ForwardEuler(eltype(grid))
end

Terrarium.variables(::NamespacedExpModel) = (
    Terrarium.prognostic(:u, Terrarium.Ground(Terrarium.XY())),
    Terrarium.auxiliary(:v, Terrarium.Ground(Terrarium.XY())),
    Terrarium.namespace(
        :inner, (
            Terrarium.prognostic(:u, Terrarium.Ground(Terrarium.XY())),
            Terrarium.auxiliary(:v, Terrarium.Ground(Terrarium.XY())),
            Terrarium.input(:c, Terrarium.Ground(Terrarium.XY())),
        )
    ),
)

# constant offset at the root (v = 0.1); inside the namespace the offset is read from the
# namespaced input `c`, so that we also exercise input variables defined in a namespace.
function Terrarium.compute_auxiliary!(state, model::NamespacedExpModel)
    state.auxiliary.v .= 0.1
    state.namespaces.inner.auxiliary.v .= state.namespaces.inner.inputs.c
    return nothing
end

# du/dt = u + v at both the root and inside the namespace
function Terrarium.compute_tendencies!(state, model::NamespacedExpModel)
    set!(state.tendencies.u, state.prognostic.u + state.auxiliary.v)
    set!(state.namespaces.inner.tendencies.u, state.namespaces.inner.prognostic.u + state.namespaces.inner.auxiliary.v)
    return nothing
end

@testset "NamespacedExpModel: time stepping prognostic/input in a namespace" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    model_heun = NamespacedExpModel(grid; timestepper = Heun())
    model_euler = NamespacedExpModel(grid)

    # root initializers only; the namespaced prognostic `u` starts at its default (zero) and
    # the namespaced input `c` (which has no input source) is set explicitly below.
    initializers = (u = 0.0, v = 0.1)
    integrator_heun = initialize(model_heun; initializers)
    integrator_euler = initialize(model_euler; initializers)

    # the model must define namespaced variables (prognostic, auxiliary, and input)
    @test haskey(integrator_euler.state.namespaces, :inner)
    @test :u in keys(integrator_euler.state.namespaces.inner.prognostic)
    @test :c in keys(integrator_euler.state.namespaces.inner.inputs)

    # set the namespaced input; the namespaced offset `v` reads from it, giving the same
    # dynamics (du/dt = u + 0.1) inside the namespace as the root variable.
    set!(integrator_heun.state.namespaces.inner.inputs.c, 0.1)
    set!(integrator_euler.state.namespaces.inner.inputs.c, 0.1)

    timestep!(integrator_heun)
    timestep!(integrator_euler)

    u_inner_euler = integrator_euler.state.namespaces.inner.prognostic.u
    u_inner_heun = integrator_heun.state.namespaces.inner.prognostic.u

    # the namespaced prognostic actually integrated, and Heun is more accurate than Euler
    @test u_inner_heun[2] > u_inner_euler[2]

    # the namespaced prognostic integrates identically to the root variable (same dynamics),
    # confirming both the root and the namespace are stepped consistently
    @test integrator_euler.state.u[2] == u_inner_euler[2]
    @test integrator_heun.state.u[2] == u_inner_heun[2]

    # exact integrated values for the namespaced prognostic
    dt_euler = default_dt(integrator_euler)
    @test u_inner_euler[2] == 0.1 * dt_euler
    dt_heun = default_dt(integrator_heun)
    @test u_inner_heun[2] == (0.1 * dt_heun + (0.1 * dt_heun + 0.1) * dt_heun) / 2
end
