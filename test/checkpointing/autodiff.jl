using Terrarium
using Test

using Checkpointing
using Enzyme

@testset "Soil energy model: timestepping with checkpointing" begin
    # Model setup
    arch = CPU()
    NF = Float32
    grid = ColumnGrid(arch, NF, UniformSpacing(Δz = 0.1, N = 10))
    initializer = SoilInitializer(NF)
    model = SoilModel(grid; timestepper = ForwardEuler(NF), initializer = initializer)
    # constant surface temperature of 1°C
    bcs = PrescribedSurfaceTemperature(:T_ub, NF(1.0))
    integrator = initialize(model, boundary_conditions = bcs)

    # AD setup
    scheme = Revolve(1)
    dintegrator = make_zero(integrator)

    # Take derivative of the mean soil temperature after 5 time steps
    # with respect to the initial conditions

    function mean_temperature(integrator)
        run!(integrator, scheme, 5)
        return mean(interior(integrator.state.temperature))
    end

    @time autodiff(
        set_runtime_activity(Reverse),
        mean_temperature,
        Active,
        Duplicated(integrator, dintegrator)
    )

    @test all(isfinite.(dintegrator.state.temperature))
end
