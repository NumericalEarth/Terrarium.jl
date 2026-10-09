using Terrarium
using Terrarium:
    compute_surface_drainage,
    compute_infiltration,
    compute_surface_runoff
using Test

@testset "compute_surface_drainage" begin
    runoff = DirectSurfaceRunoff(Float64)
    # Test drainage is zero when there is no excess water
    ∂S∂t = compute_surface_drainage(runoff, 0.0)
    @test iszero(∂S∂t)
    # Test that drainage is still zero when excess water is negative (mass balance violation)
    ∂S∂t = compute_surface_drainage(runoff, -0.1)
    @test iszero(∂S∂t)
    # Test drainage is equal to surface_water / τ_r
    # the calculation is simple enough to just test directly here
    ∂S∂t = compute_surface_drainage(runoff, 0.1)
    @test ∂S∂t ≈ 0.1 / runoff.τ_r
    # Test with alternative value of τ_r
    runoff = DirectSurfaceRunoff(τ_r = 24 * 3600)
    ∂S∂t = compute_surface_drainage(runoff, 0.1)
    @test ∂S∂t ≈ 0.1 / runoff.τ_r
end

@testset "compute_infiltration" begin
    runoff = DirectSurfaceRunoff(Float64)
    # Test that infiltration is zero when there is no flux
    sat_top = 0.5
    max_infil = 1.0e-5
    influx = 0.0
    infil = compute_infiltration(runoff, influx, sat_top, max_infil)
    @test iszero(infil)
    # Test that infiltration is positive when flux is positive
    influx = max_infil
    infil = compute_infiltration(runoff, influx, sat_top, max_infil)
    @test infil ≈ influx
    # Test that infiltration is capped at given max value
    influx = 2 * max_infil
    infil = compute_infiltration(runoff, influx, sat_top, max_infil)
    @test infil ≈ max_infil
    # Test that infiltration is zero when soil is saturated
    infil = compute_infiltration(runoff, influx, 1.0, max_infil)
    @test iszero(infil)
end

@testset "compute_surface_runoff" begin
    runoff = DirectSurfaceRunoff(Float64)
    # Check that surface runoff is zero when all terms are zero
    R = compute_surface_runoff(runoff, 0, 0, 0)
    @test iszero(R)
    # Check that surface runoff is equal to the defined sum
    precip = 1.0e-6
    surface_drainage = 1.0e-7
    infil = 1.0e-5
    R = compute_surface_runoff(runoff, precip, surface_drainage, infil)
    @test R ≈ precip + surface_drainage - infil
end

@testset "fresh rainfall infiltrates after surface water decays" begin
    grid = ColumnGrid(CPU(), Float32, Float32[-3, -2, -1, -0.5, -0.2, 0])
    runoff = DirectSurfaceRunoff(Float32)
    rainfall = 1.0f-7
    excess_water = eps(Float32) / 2
    drainage = excess_water / runoff.τ_r
    fields = (
        rainfall_ground = fill(rainfall, 1, 1, 1),
        surface_excess_water = fill(excess_water, 1, 1, 1),
        saturation_water_ice = fill(0.43f0, 1, 1, 5),
        hydraulic_conductivity = fill(4.0f-6, 1, 1, 6),
    )
    out = (infiltration = zeros(Float32, 1, 1, 1),
           surface_runoff = zeros(Float32, 1, 1, 1))

    Terrarium.compute_surface_runoff!(out, 1, 1, grid, fields, runoff,
                                      Terrarium.NoCanopyInterception(Float32),
                                      SoilHydrology(Float32, NoFlow()))

    @test out.infiltration[1, 1, 1] ≈ rainfall
    @test out.surface_runoff[1, 1, 1] ≈ drainage rtol = 1.0f-3

    # The seasonal replay reached this subnormal pool value; its drainage rounds to zero.
    fields.surface_excess_water[1, 1, 1] = 2.388f-42
    Terrarium.compute_surface_runoff!(out, 1, 1, grid, fields, runoff,
                                      Terrarium.NoCanopyInterception(Float32),
                                      SoilHydrology(Float32, NoFlow()))
    @test out.infiltration[1, 1, 1] ≈ rainfall
    @test iszero(out.surface_runoff[1, 1, 1])

    fields.surface_excess_water[1, 1, 1] = 1.0f-4
    Terrarium.compute_surface_runoff!(out, 1, 1, grid, fields, runoff,
                                      Terrarium.NoCanopyInterception(Float32),
                                      SoilHydrology(Float32, NoFlow()))
    @test out.infiltration[1, 1, 1] ≈ 1.0f-4 / runoff.τ_r
    @test out.surface_runoff[1, 1, 1] ≈ rainfall
end

@testset "surface_excess_water tendency" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 10))

    # The runoff scheme owns the `surface_excess_water` pool and drains it at the surface
    # drainage rate, capped so a single step cannot remove more than is present.
    runoff = DirectSurfaceRunoff(Float64)
    state = StateVariables(runoff, grid)
    S = 0.1
    set!(state.surface_excess_water, S)
    Terrarium.compute_tendencies!(state, grid, runoff)
    ∂S∂t = Array(interior(state.tendencies.surface_excess_water))[1, 1, 1]
    # The tendency is a negative removal rate, -min(D, S), so the pool is drawn down
    @test ∂S∂t ≈ -min(S / runoff.τ_r, S)
    @test ∂S∂t < 0

    # When the drainage rate exceeds the pool (short timescale), the cap min(D, S) = S binds
    runoff = DirectSurfaceRunoff(Float64; τ_r = 0.5)
    state = StateVariables(runoff, grid)
    set!(state.surface_excess_water, S)
    Terrarium.compute_tendencies!(state, grid, runoff)
    ∂S∂t = Array(interior(state.tendencies.surface_excess_water))[1, 1, 1]
    @test ∂S∂t ≈ -S

    # An empty pool has no tendency
    state = StateVariables(DirectSurfaceRunoff(Float64), grid)
    Terrarium.compute_tendencies!(state, grid, runoff)
    @test all(iszero.(state.tendencies.surface_excess_water))
end
