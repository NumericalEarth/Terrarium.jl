using Terrarium
using Terrarium: Clock, Variables, InputSource, initialize!, update_inputs!, variables, interior, InputSources, varname
using Test

using Dates
using NCDatasets
using Oceananigans.OutputReaders: Cyclical
using Rasters
using Rasters: X, Y, Ti

import RingGrids

# Get RasterInputSource from the loaded extension
import Base: get_extension
const TerrariumRastersExt = get_extension(Terrarium, :TerrariumRastersExt)
const RasterInputSource = TerrariumRastersExt.RasterInputSource

@testset "RasterInputSource" begin
    # Create a temporary directory for test files
    testdir = mktempdir()

    # Set up a ColumnRingGrid for testing
    ring_grid = RingGrids.FullHEALPixGrid(8)
    lat, lon = RingGrids.get_lat(ring_grid), RingGrids.get_lon(ring_grid)
    npoints = RingGrids.get_npoints(ring_grid)

    grid = ColumnRingGrid(UniformSpacing(Δz = 0.5, N = 5), ring_grid)

    # Generate random data matching the ring grid size
    nx, ny = length(lon), length(lat)
    static_data = rand(Float32, nx, ny)

    @testset "Static raster input" begin
        # Create a dummy NetCDF file with static (no time dimension) data
        static_file = joinpath(testdir, "static_data.nc")

        NCDataset(static_file, "c") do ds
            defDim(ds, "x", nx)
            defDim(ds, "y", ny)
            defVar(ds, "x", collect(1:nx), ("x",))
            defVar(ds, "y", collect(1:ny), ("y",))
            defVar(ds, "temperature", static_data, ("x", "y"); attrib = Dict("units" => "K"))
        end

        # Load with Rasters
        raster = Raster(static_file; name = :temperature)

        # A static (time-invariant) raster reduces to a plain FieldInputSource
        source = InputSource(grid, raster)
        @test isa(source, Terrarium.FieldInputSource)
        @test Terrarium.vardims(source) == XY()
        @test varname(source) == :temperature

        # Check variables are correctly inferred
        vars = variables(source)
        @test length(vars) == 1
        @test first(vars) == Terrarium.input(:temperature, XY())

        # Test initialization
        state = StateVariables(Variables(source), grid)
        @test hasproperty(state.inputs, :temperature)

        # Test that initialize! copies data correctly
        clock = Clock(time = 0.0)
        initialize!(state.inputs, grid, clock, state, source)

        # Verify data was copied (only masked points)
        idxmap = findall(vec(Array(grid.mask)))
        flat_data = vec(static_data)
        expected = flat_data[idxmap]
        @test all(interior(state.inputs.temperature)[:, 1, 1] .≈ expected)

        # Test that update_inputs! does nothing for static rasters
        original_data = copy(interior(state.inputs.temperature))
        update_inputs!(state.inputs, grid, clock, state, source)
        @test all(interior(state.inputs.temperature) .== original_data)
    end

    @testset "Time-varying raster input" begin
        # Create a dummy NetCDF file with time-varying data
        timeseries_file = joinpath(testdir, "timeseries_data.nc")

        ntimes = 5
        times = DateTime(2020, 1, 1):Hour(1):DateTime(2020, 1, 1, 4)
        timeseries_data = rand(Float32, nx, ny, ntimes)

        NCDataset(timeseries_file, "c") do ds
            defDim(ds, "x", nx)
            defDim(ds, "y", ny)
            defDim(ds, "time", ntimes)
            defVar(ds, "x", collect(1:nx), ("x",))
            defVar(ds, "y", collect(1:ny), ("y",))
            defVar(ds, "time", collect(times), ("time",))
            defVar(ds, "forcing", timeseries_data, ("x", "y", "time"); attrib = Dict("units" => "W/m^2"))
        end

        # Load with Rasters
        raster = Raster(timeseries_file; name = :forcing)

        # Create RasterInputSource with reference time
        reftime = DateTime(2020, 1, 1)
        source = InputSource(grid, raster; reftime)
        @test isa(source, RasterInputSource)
        @test varname(source) == :forcing
        @test source.reftime == reftime

        # Check variables
        vars = variables(source)
        @test length(vars) == 1
        @test first(vars) == Terrarium.input(:forcing, XY())

        # Test initialization and update
        state = StateVariables(Variables(source), grid)
        @test hasproperty(state.inputs, :forcing)

        # Initialize (should be no-op for time-varying)
        clock = Clock(time = 0.0)
        initialize!(state.inputs, grid, clock, state, source)

        # Update at t=0 (should get first timestep)
        update_inputs!(state.inputs, grid, clock, state, source)
        flat_t0 = vec(timeseries_data[:, :, 1])
        expected_t0 = flat_t0[source.idxmap]
        @test all(interior(state.inputs.forcing)[:, 1, 1] .≈ expected_t0)

        # Advance clock by 1 hour (3600 seconds) and update
        Terrarium.tick!(clock, 3600.0)
        update_inputs!(state.inputs, grid, clock, state, source)
        flat_t1 = vec(timeseries_data[:, :, 2])
        expected_t1 = flat_t1[source.idxmap]
        @test all(interior(state.inputs.forcing)[:, 1, 1] .≈ expected_t1)

        # Test interpolation at t=0.5 hours (1800 seconds from start)
        clock = Clock(time = 1800.0)
        update_inputs!(state.inputs, grid, clock, state, source)
        # Should be interpolated between t0 and t1
        expected_interp = (flat_t0[source.idxmap] .+ flat_t1[source.idxmap]) ./ 2
        @test all(isapprox.(interior(state.inputs.forcing)[:, 1, 1], expected_interp, atol = 1.0e-5))
    end

    @testset "Cyclic time-varying raster input" begin
        # Reuse the 5-record hourly series but treat the time axis as periodic
        timeseries_file = joinpath(testdir, "cyclic_data.nc")
        ntimes = 5
        times = DateTime(2020, 1, 1):Hour(1):DateTime(2020, 1, 1, 4)
        timeseries_data = rand(Float32, nx, ny, ntimes)
        NCDataset(timeseries_file, "c") do ds
            defDim(ds, "x", nx)
            defDim(ds, "y", ny)
            defDim(ds, "time", ntimes)
            defVar(ds, "x", collect(1:nx), ("x",))
            defVar(ds, "y", collect(1:ny), ("y",))
            defVar(ds, "time", collect(times), ("time",))
            defVar(ds, "forcing", timeseries_data, ("x", "y", "time"))
        end
        raster = Raster(timeseries_file; name = :forcing)
        source = InputSource(grid, raster; reftime = DateTime(2020, 1, 1), cycle = true)
        @test isa(source, RasterInputSource)
        @test isa(source.extrapolation, Cyclical)

        state = StateVariables(Variables(source), grid)
        idxmap = source.idxmap
        flat(n) = vec(timeseries_data[:, :, n])[idxmap]

        # span = 4 h, spacing Δ = 1 h, so the period is 5 h; t = 5 h wraps back to record 1
        clock = Clock(time = 5 * 3600.0)
        update_inputs!(state.inputs, grid, clock, state, source)
        @test all(isapprox.(interior(state.inputs.forcing)[:, 1, 1], flat(1), atol = 1.0e-5))

        # t = 6 h wraps to 1 h -> record 2
        clock = Clock(time = 6 * 3600.0)
        update_inputs!(state.inputs, grid, clock, state, source)
        @test all(isapprox.(interior(state.inputs.forcing)[:, 1, 1], flat(2), atol = 1.0e-5))

        # t = 4.5 h lies in the wrap gap: halfway between record 5 and record 1
        clock = Clock(time = 4.5 * 3600.0)
        update_inputs!(state.inputs, grid, clock, state, source)
        expected_wrap = (flat(5) .+ flat(1)) ./ 2
        @test all(isapprox.(interior(state.inputs.forcing)[:, 1, 1], expected_wrap, atol = 1.0e-5))
    end

    @testset "Multiple rasters" begin
        # Create two NetCDF files
        file1 = joinpath(testdir, "var1.nc")
        file2 = joinpath(testdir, "var2.nc")

        data1 = rand(Float32, nx, ny)
        data2 = rand(Float32, nx, ny) .* 100

        NCDataset(file1, "c") do ds
            defDim(ds, "x", nx)
            defDim(ds, "y", ny)
            defVar(ds, "x", collect(1:nx), ("x",))
            defVar(ds, "y", collect(1:ny), ("y",))
            defVar(ds, "var1", data1, ("x", "y"))
        end

        NCDataset(file2, "c") do ds
            defDim(ds, "x", nx)
            defDim(ds, "y", ny)
            defVar(ds, "x", collect(1:nx), ("x",))
            defVar(ds, "y", collect(1:ny), ("y",))
            defVar(ds, "var2", data2, ("x", "y"))
        end

        # Load both rasters
        raster1 = Raster(file1; name = :var1)
        raster2 = Raster(file2; name = :var2)

        # Create InputSources with two separate RasterInputSources
        sources = InputSources(
            InputSource(grid, raster1),
            InputSource(grid, raster2)
        )
        @test variables(sources) == (Terrarium.input(:var1, XY()), Terrarium.input(:var2, XY()))

        # Test initialization
        state = StateVariables(Variables(sources), grid)
        @test hasproperty(state.inputs, :var1)
        @test hasproperty(state.inputs, :var2)

        # Initialize and verify both fields
        clock = Clock(time = 0.0)
        initialize!(state.inputs, grid, clock, state, sources)
        idxmap = findall(vec(Array(grid.mask)))
        expected1 = vec(data1)[idxmap]
        expected2 = vec(data2)[idxmap]
        @test all(interior(state.inputs.var1)[:, 1, 1] .≈ expected1)
        @test all(interior(state.inputs.var2)[:, 1, 1] .≈ expected2)
    end

    # Clean up
    rm(testdir; recursive = true)
end
