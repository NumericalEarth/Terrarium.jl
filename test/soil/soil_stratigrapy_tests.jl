using Terrarium
using Test

@testset "Soil porosity parameterizations" begin
    # Constant porosity
    porosity = ConstantSoilPorosity(Float64, mineral_porosity = 0.7, organic_porosity = 0.3)
    texture = SoilTexture(sand = 0.0, silt = 0.7, clay = 0.3)
    @test Terrarium.mineral_porosity(porosity, texture) == 0.7
    @test Terrarium.organic_porosity(porosity, texture) == 0.3
    # SURFEX porosity
    # TODO: should the hydraulic properties struct constructors also enforce parameter bounds?
    porosity = SoilPorositySURFEX(Float64)
    por0 = Terrarium.mineral_porosity(porosity, texture)
    @test por0 ≈ porosity.porosity_default
    for sand in 0.1:0.1:1.0
        silt = (1 - sand) * 0.7
        clay = (1 - sand) * 0.3
        por = Terrarium.mineral_porosity(porosity, SoilTexture(; sand, silt, clay))
        # test that increasing sand content decreases the mineral porosity;
        # we could also reproduce the calculation here, but that seems a bit redundant
        # and of course depends on the choice of parameters.
        @test 0 < por < por0    # Test construction
        horizon1 = ConstantSoilHorizon(Float64, :h1; texture = SoilTexture(Float64, sand = 0.5), thickness = 0.5)
        horizon2 = ConstantSoilHorizon(Float64, :h2; texture = SoilTexture(Float64, sand = 0.1), thickness = Inf)
        strat = SoilStratigraphy(Float64, horizon1, horizon2)

        @test length(strat) == 2

        # Test iteration
        h_iter = collect(strat)
        @test length(h_iter) == 2
        @test h_iter[1] == horizon1
        @test h_iter[2] == horizon2

        # Test variable generation
        vars = Terrarium.Variables(strat)
        @test length(vars.namespaces) == 2
        @test haskey(vars.namespaces, :h1)
        @test haskey(vars.namespaces, :h2)

        # Test soil_horizon
        grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 10))
        state = StateVariables(vars, grid)
        fields = Terrarium.get_fields(state, strat)
        horizon = Terrarium.soil_horizon(1, 1, 10, grid, fields, strat)
        texture = Terrarium.soil_texture(1, 1, grid, fields, horizon)
        @test texture == horizon1.texture
        horizon = Terrarium.soil_horizon(1, 1, 5, grid, fields, strat)
        texture = Terrarium.soil_texture(1, 1, grid, fields, horizon)
        @test texture == horizon2.texture

        # Mixed prescribed/constant stratigraphy
        horizon1 = PrescribedSoilHorizon(Float64, :h1)
        horizon2 = ConstantSoilHorizon(Float64, :h2; texture = SoilTexture(Float64, sand = 0.1), thickness = Inf)
        strat = SoilStratigraphy(Float64, horizon1, horizon2)
        vars = Terrarium.Variables(strat)
        state = StateVariables(vars, grid)
        fields = Terrarium.get_fields(state, strat)
        h1_texture = SoilTexture(Float64, sand = 0.5)
        set!(fields.h1, h1_texture)
        set!(fields.h1.thickness, 0.5)
        horizon = Terrarium.soil_horizon(1, 1, 10, grid, fields, strat)
        texture = Terrarium.soil_texture(1, 1, grid, fields.h1, horizon)
        @test texture == h1_texture
        horizon = Terrarium.soil_horizon(1, 1, 5, grid, fields, strat)
        texture = Terrarium.soil_texture(1, 1, grid, fields, horizon)
        @test texture == horizon2.texture
    end
end

@testset "Texture normalization" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 10), 10)
    sand = Field(grid, XY())
    set!(sand, 0.5)
    silt = Field(grid, XY())
    set!(silt, 0.4)
    clay = Field(grid, XY())
    set!(clay, 0.2) # violate bounds
    normalize_texture!(sand, silt, clay)
    # all entries sum to unity
    @test all(sand + silt + clay .≈ 1)
end

@testset "Soil horizons" begin
    # Constant horizon
    texture = SoilTexture(Float64)
    porosity = ConstantSoilPorosity(Float64)
    thickness = 0.5
    horizon = ConstantSoilHorizon(Float64, :ch; texture, porosity, thickness)
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 10))
    @test Terrarium.soil_thickness(1, 1, grid, (;), horizon) == thickness
    @test Terrarium.soil_texture(1, 1, grid, (;), horizon) == texture
    # Prescribed (input) horizon
    horizon = PrescribedSoilHorizon(Float64, :ph; porosity)
    vars = Terrarium.Variables(horizon)
    @test Tuple(map(Terrarium.varname, values(vars.inputs))) == (:sand_fraction, :silt_fraction, :clay_fraction, :thickness)
    fields = StateVariables(vars, grid)
    set!(fields.sand_fraction, texture.sand)
    set!(fields.silt_fraction, texture.silt)
    set!(fields.clay_fraction, texture.clay)
    set!(fields.thickness, thickness)
    thickness0 = Terrarium.soil_thickness(1, 1, grid, fields, horizon)
    @test thickness0 == thickness
    texture0 = Terrarium.soil_texture(1, 1, grid, fields, horizon)
    @test texture0 == texture
    set!(fields.thickness, 1)
    set!(fields.sand_fraction, 1)
    thickness1 = Terrarium.soil_thickness(1, 1, grid, fields, horizon)
    @test thickness1 == 1
    texture1 = Terrarium.soil_texture(1, 1, grid, fields, horizon)
    @test texture1.sand == 1
end

@testset "Soil stratigraphy" begin
    # Test construction
    horizon1 = ConstantSoilHorizon(Float64, :h1; texture = SoilTexture(Float64, sand = 0.5), thickness = 0.5)
    horizon2 = ConstantSoilHorizon(Float64, :h2; texture = SoilTexture(Float64, sand = 0.1), thickness = Inf)
    strat = SoilStratigraphy(Float64, horizon1, horizon2)

    @test length(strat) == 2

    # Test iteration
    h_iter = collect(strat)
    @test length(h_iter) == 2
    @test h_iter[1] == horizon1
    @test h_iter[2] == horizon2

    # Test variable generation
    vars = Terrarium.Variables(strat)
    @test length(vars.namespaces) == 2
    @test haskey(vars.namespaces, :h1)
    @test haskey(vars.namespaces, :h2)

    # Test soil_horizon
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 10))
    state = StateVariables(vars, grid)
    fields = Terrarium.get_fields(state, strat)
    horizon = Terrarium.soil_horizon(1, 1, 10, grid, fields, strat)
    texture = Terrarium.soil_texture(1, 1, grid, fields, horizon)
    @test texture == horizon1.texture
    horizon = Terrarium.soil_horizon(1, 1, 5, grid, fields, strat)
    texture = Terrarium.soil_texture(1, 1, grid, fields, horizon)
    @test texture == horizon2.texture

    # Mixed prescribed/constant stratigraphy
    horizon1 = PrescribedSoilHorizon(Float64, :h1, porosity = ConstantSoilPorosity(Float64, mineral_porosity = 0.3))
    horizon2 = ConstantSoilHorizon(Float64, :h2; texture = SoilTexture(Float64, sand = 0.1), thickness = Inf)
    strat = SoilStratigraphy(Float64, horizon1, horizon2)
    vars = Terrarium.Variables(strat)
    state = StateVariables(vars, grid)
    fields = Terrarium.get_fields(state, strat)
    h1_texture = SoilTexture(Float64, sand = 0.5)
    set!(fields.h1, h1_texture)
    set!(fields.h1.thickness, 0.5)
    horizon = Terrarium.soil_horizon(1, 1, 10, grid, fields, strat)
    texture = Terrarium.soil_texture(1, 1, grid, fields.h1, horizon)
    @test texture == h1_texture
    horizon = Terrarium.soil_horizon(1, 1, 5, grid, fields, strat)
    texture = Terrarium.soil_texture(1, 1, grid, fields, horizon)
    @test texture == horizon2.texture

    # Porosity
    bgc = ConstantSoilCarbonDensity(eltype(grid))
    @test Terrarium.porosity(1, 1, 10, grid, fields, strat, bgc) == horizon1.porosity.mineral_porosity

    # Organic fraction
    bgc = ConstantSoilCarbonDensity(eltype(grid), ρ_soc = 50.0)
    @test Terrarium.organic_fraction(1, 1, 10, grid, fields, strat, bgc) > 0

    # Type stability of kernel functions with heterogeneous horizon types;
    # this is required for GPU compatibility.
    @test @inferred(Terrarium.soil_texture(1, 1, 10, grid, fields, strat)) == h1_texture
    @test @inferred(Terrarium.soil_texture(1, 1, 5, grid, fields, strat)) == horizon2.texture
    @test @inferred(Terrarium.with_soil_horizon(Terrarium.soil_thickness, 1, 1, 10, grid, fields, strat)) == 0.5
    @test @inferred(Terrarium.with_soil_horizon(Terrarium.soil_thickness, 1, 1, 5, grid, fields, strat)) == Inf
end
