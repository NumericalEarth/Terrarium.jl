using Terrarium
using Test

import RingGrids
import Dates: Hour

@testset "run! SoilModel w/ ForwardEuler" begin
    grid = ColumnRingGrid(CPU(), Float64, ExponentialSpacing(N = 50), RingGrids.FullHEALPixGrid(16))
    model = SoilModel(grid)
    integrator = initialize(model)

    run!(integrator; steps = 2)
    @test all(isfinite.(integrator.state.temperature))

    run!(integrator; period = Hour(1))
    @test all(isfinite.(integrator.state.temperature))

    @test_throws ArgumentError run!(integrator; steps = 2, period = Hour(1))
    @test_throws ArgumentError run!(integrator)

    # test Oceananigans Simulation
    integrator = initialize(model)
    sim = Simulation(integrator; Δt = 900.0, stop_time = 3600.0)
    timestep!(sim)
    run!(sim)
    @test integrator.clock.time == 3600.0
end

@testset "run! LandModel on ColumnRingGrid" begin
    # the soil water and energy tendencies apply flux boundary conditions, which need the horizontal
    # metrics of the ColumnRingGrid
    grid = ColumnRingGrid(CPU(), Float64, ExponentialSpacing(N = 10), RingGrids.FullHEALPixGrid(4))
    model = LandModel(grid)
    # variably saturated soil and nonzero vegetation carbon (the default of zero carbon is not a valid
    # initial state for the vegetation dynamics)
    initializers = (
        temperature = (x, z) -> 5.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> min(1, 0.8 - 0.05 * z),
        carbon_vegetation = 0.1,
    )
    integrator = initialize(model; initializers)

    run!(integrator; steps = 2, Δt = 600.0)
    @test all(isfinite.(integrator.state.temperature))
    @test all(isfinite.(integrator.state.saturation_water_ice))
end

@testset "run! SoilModel w/ Heun" begin
    grid = ColumnRingGrid(CPU(), Float64, ExponentialSpacing(N = 50), RingGrids.FullHEALPixGrid(16))
    model = SoilModel(grid; timestepper = Heun())
    integrator = initialize(model)

    run!(integrator; steps = 2)
    @test all(isfinite.(integrator.state.temperature))

    run!(integrator; period = Hour(1))
    @test all(isfinite.(integrator.state.temperature))

    @test_throws ArgumentError run!(integrator; steps = 2, period = Hour(1))
    @test_throws ArgumentError run!(integrator)
end
