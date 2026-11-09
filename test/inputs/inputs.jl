using Terrarium
using Terrarium: InputSources, FieldInputSource, FieldTimeSeriesInputSource, Variables
using Terrarium: initialize!, interior, varname
using Test
using Unitful

@testset "Input sources" begin
    # Field input source
    grid = ColumnGrid(ExponentialSpacing())
    X1 = Field(grid, XY())
    field_input = InputSource(grid, X1; name = :X1)
    @test isa(field_input, FieldInputSource)
    ## check that dimensions and name were inferred correctly
    @test Terrarium.vardims(field_input) == XY()
    @test varname(field_input) == :X1
    @test variables(field_input) == (Terrarium.input(:X1, XY()),)
    ## check state variable is allocated and initialize! copies data
    X1 .= 1.0f0
    state = StateVariables(Variables(field_input), grid)
    @test hasproperty(state.inputs, :X1)
    initialize!(state.inputs, grid, Clock(time = 0), state, field_input)
    @test all(state.inputs.X1 .≈ 1.0f0)

    # Multiple FieldInputSources via InputSources
    X2 = Field(grid, XY())
    field_sources = InputSources(InputSource(grid, X1; name = :X1), InputSource(grid, X2; name = :X2))
    @test variables(field_sources) == (Terrarium.input(:X1, XY()), Terrarium.input(:X2, XY()))
    state = StateVariables(Variables(field_sources), grid)
    @test hasproperty(state.inputs, :X1)
    @test hasproperty(state.inputs, :X2)

    # FieldTimeSeries input source
    grid = ColumnGrid(ExponentialSpacing())
    ts = 0.0:1.0:10.0
    S1 = FieldTimeSeries(grid, XY(), ts)
    fts_input = InputSource(S1; name = :S1)
    @test isa(fts_input, FieldTimeSeriesInputSource)
    ## check that dimensions and name were inferred correctly
    @test Terrarium.vardims(fts_input) == XY()
    @test varname(fts_input) == :S1
    @test fts_input.fts === S1
    # populate S1 with random data and check update_inputs!
    S1.data .= randn(size(S1))
    fields = (S1 = Field(grid, XY()),)
    clock = Clock(time = 0)
    update_inputs!(fields, grid, clock, fields, fts_input)
    @test all(fields.S1 .== S1[1])
    # advance clock and check that inputs are updated
    Terrarium.tick!(clock, 1.0)
    update_inputs!(fields, grid, clock, fields, fts_input)
    @test all(fields.S1 .== S1[2])

    # Multiple FTS via InputSources
    S2 = FieldTimeSeries(grid, XY(), ts)
    fts_sources = InputSources(InputSource(S1; name = :S1), InputSource(S2; name = :S2))
    @test variables(fts_sources) == (Terrarium.input(:S1, XY()), Terrarium.input(:S2, XY()))
    fields2 = (S1 = Field(grid, XY()), S2 = Field(grid, XY()))
    clock = Clock(time = 0)
    update_inputs!(fields2, grid, clock, fields2, fts_sources)
    @test all(fields2.S1 .== S1[1])
    @test all(fields2.S2 .== S2[1])
end

@testset "RingGrids.Field InputSource convenience" begin
    import RingGrids

    # Create a ColumnRingGrid
    ring_grid = RingGrids.FullHEALPixGrid(8)
    grid = ColumnRingGrid(UniformSpacing(Δz = 0.5, N = 5), ring_grid)

    # Create RingGrids fields
    ring_field1 = rand(ring_grid)
    ring_field2 = rand(ring_grid)

    # Test single field
    source = InputSource(grid, ring_field1; name = :temperature)
    @test isa(source, FieldInputSource)
    @test Terrarium.vardims(source) == XY()
    @test varname(source) == :temperature
    @test variables(source) == (Terrarium.input(:temperature, XY()),)

    # Test multiple fields via InputSources
    sources = InputSources(InputSource(grid, ring_field1; name = :temp), InputSource(grid, ring_field2; name = :pressure))
    @test length(variables(sources)) == 2

    # Test that data is correctly converted and stored
    state = StateVariables(Variables(sources), grid)
    @test hasproperty(state.inputs, :temp)
    @test hasproperty(state.inputs, :pressure)

    # Initialize and verify data was copied correctly
    initialize!(state.inputs, grid, Clock(time = 0.0), state, sources)
    expected1 = ring_field1.data[Array(grid.mask)]
    expected2 = ring_field2.data[Array(grid.mask)]
    @test all(interior(state.inputs.temp)[:, 1, 1] .≈ expected1)
    @test all(interior(state.inputs.pressure)[:, 1, 1] .≈ expected2)

    # Test 3D RingGrids field (with vertical dimension)
    ring_field_3d = rand(ring_grid, 5)  # 5 vertical levels
    source_3d = InputSource(grid, ring_field_3d; name = :field3d)
    @test isa(source_3d, FieldInputSource)
    @test Terrarium.vardims(source_3d) == XYZ()
end
