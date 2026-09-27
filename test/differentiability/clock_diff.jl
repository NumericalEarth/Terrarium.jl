using Terrarium
using Test

using Checkpointing
using Enzyme
using Oceananigans.TimeSteppers: Clock, tick!

using Enzyme: Reverse, set_runtime_activity

# `StateVariables` is a mutable struct, so the clock is reachable by reference from the
# differentiated integrator state and Enzyme descends into `tick!`. `TerrariumEnzymeExt` supplies
# custom reverse rules that keep the integer `iteration`/`stage` counters out of the activity
# analysis while still propagating the derivative of `clock.time`.

@testset "Clock: tick! reverse rule" begin
    @test Base.get_extension(Terrarium, :TerrariumEnzymeExt) !== nothing

    # `time += Δt`, so d(time²)/dΔt = 2(time + Δt). A blanket `EnzymeRules.inactive` on `tick!`
    # would return zero here; the custom rule must not.
    squared_time(clock, Δt) = (tick!(clock, Δt); clock.time^2)
    clock = Clock{Float64}(time = 3.0)
    dclock = Enzyme.make_zero(clock)
    (_, dΔt), = Enzyme.autodiff(Reverse, squared_time, Active, Duplicated(clock, dclock), Active(2.0))
    @test dΔt ≈ 2 * (3.0 + 2.0)
    @test clock.iteration == 1

    # Checkpointed reverse mode through `run!` must compile and give a nonzero, finite adjoint.
    NF = Float32
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing())
    model = SoilModel(grid; timestepper = ForwardEuler(NF), initializer = SoilInitializer(NF))
    boundary_conditions = PrescribedSurfaceTemperature(:T_ub, NF(1))
    integrator = initialize(model; boundary_conditions)
    dintegrator = Enzyme.make_zero(integrator)

    function layer_temperature(integrator)
        run!(integrator, Revolve(1), 10)
        return interior(integrator.state.temperature)[1, 1, 2]
    end

    Enzyme.autodiff(set_runtime_activity(Reverse), layer_temperature, Active, Duplicated(integrator, dintegrator))
    dT = interior(dintegrator.state.temperature)[1, 1, 2]
    @test isfinite(dT)
    @test dT != 0
    @test integrator.state.clock.iteration == 10
end
