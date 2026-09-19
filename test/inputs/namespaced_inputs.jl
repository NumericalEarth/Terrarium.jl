using Terrarium
using Terrarium: Variables, Namespace, XY, XYZ, Ground, initialize!, interior, varname, matches_scope, with_scope
using Test

@testset "Namespaced input sources" begin
    grid = ColumnGrid(ExponentialSpacing())

    # source with a namespaced name
    X1 = Field(grid, Ground(XY()))
    src = InputSource(grid, X1; name = :ns1 => :x, domain = Ground())
    @test varname(src) == :x
    @test varpath(src) == (:ns1, :x)
    @test matches_scope(src, (:ns1,))
    @test !matches_scope(src, ())
    @test !matches_scope(src, (:ns2,))

    # variables(src) should declare the input inside the namespace
    vars = variables(src)
    @test length(vars) == 1 && isa(vars[1], Namespace)
    @test varname(vars[1]) == :ns1
    @test first(variables(vars[1])) == Terrarium.input(:x, Ground(XY()))

    # initialize state with a root input :x and two namespaces both declaring :x
    all_vars = Variables(
        Terrarium.input(:x, Ground(XY())),
        Terrarium.namespace(:ns1, (Terrarium.input(:x, Ground(XY())),)),
        Terrarium.namespace(:ns2, (Terrarium.input(:x, Ground(XY())),)),
    )
    state = StateVariables(all_vars, grid)
    X1 .= 3.0f0
    initialize!(state, grid, InputSources(src))
    ## only ns1.x should be set by the source
    @test all(interior(state.namespaces.ns1.x) .== 3.0f0)
    @test all(interior(state.namespaces.ns2.x) .== 0.0f0)
    @test all(interior(state.inputs.x) .== 0.0f0)

    # a root-level source should not leak into namespaces
    X2 = Field(grid, Ground(XY()))
    X2 .= 5.0f0
    root_src = InputSource(grid, X2; name = :x, domain = Ground())
    initialize!(state, grid, InputSources(root_src))
    @test all(interior(state.inputs.x) .== 5.0f0)
    @test all(interior(state.namespaces.ns1.x) .== 3.0f0)
    @test all(interior(state.namespaces.ns2.x) .== 0.0f0)

    # static field sources do not change values in update_inputs!
    X1 .= 7.0f0
    update_inputs!(state, grid, InputSources(src))
    @test all(interior(state.namespaces.ns1.x) .== 3.0f0)

    # time-varying sources follow the same scoping rules in update_inputs!
    ts = 0.0:1.0:10.0
    S = FieldTimeSeries(grid, Ground(XY()), ts)
    for (i, t) in enumerate(ts)
        S[i] .= Float32(t)
    end
    fts_src = InputSource(S; name = :ns2 => :x, domain = Ground())
    @test varpath(fts_src) == (:ns2, :x)
    update_inputs!(state, grid, InputSources(fts_src))
    @test all(interior(state.namespaces.ns2.x) .== 0.0f0)
    Terrarium.tick!(state.clock, 1.0)
    update_inputs!(state, grid, InputSources(fts_src))
    @test all(interior(state.namespaces.ns2.x) .== 1.0f0)
    ## other fields named x are untouched
    @test all(interior(state.namespaces.ns1.x) .== 3.0f0)
    @test all(interior(state.inputs.x) .== 5.0f0)
end

@testset "Namespaced inputs with SoilModel" begin
    NF = Float32
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing())
    strat = SoilGridsStratigraphy(NF)
    soil = Terrarium.SoilEnergyWaterCarbon(NF; strat)
    model = SoilModel(grid; soil)

    # prescribe texture for the organic horizon only; note that the fractions
    # must sum to unity in each grid cell
    sand = Field(grid, Ground(XY()))
    silt = Field(grid, Ground(XY()))
    clay = Field(grid, Ground(XY()))
    sand .= 0.25f0
    silt .= 0.5f0
    clay .= 0.25f0
    inputs = InputSources(
        InputSource(grid, sand; name = :horizon1 => :sand_fraction, domain = Ground()),
        InputSource(grid, silt; name = :horizon1 => :silt_fraction, domain = Ground()),
        InputSource(grid, clay; name = :horizon1 => :clay_fraction, domain = Ground()),
    )
    integrator = initialize(model; inputs)
    state = integrator.state
    @test Terrarium.namespace_names(state) == map(nameof, strat.horizons)
    ## inputs land only in the first horizon namespace
    @test all(interior(state.namespaces.horizon1.sand_fraction) .== 0.25f0)
    @test all(interior(state.namespaces.horizon1.silt_fraction) .== 0.5f0)
    @test all(interior(state.namespaces.horizon1.clay_fraction) .== 0.25f0)
    ## sibling horizons keep their defaults (sand defaults to one)
    @test all(interior(state.namespaces.horizon2.sand_fraction) .== 1.0f0)
    @test all(interior(state.namespaces.horizon3.sand_fraction) .== 1.0f0)
    @test all(interior(state.namespaces.horizon4.sand_fraction) .== 1.0f0)
    @test all(interior(state.namespaces.horizon5.sand_fraction) .== 1.0f0)
    @test all(interior(state.namespaces.horizon6.sand_fraction) .== 1.0f0)
end
