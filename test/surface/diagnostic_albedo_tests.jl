using Terrarium
using Test

@testset "DiagnosticAlbedo" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 3))
    albedo = DiagnosticAlbedo(NF)
    snow = SingleLayerSnow(NF)

    @testset "no snow component -> background values" begin
        state = StateVariables(Terrarium.Variables(Terrarium.variables(albedo)...), grid)
        compute_auxiliary!(state, grid, albedo, nothing)
        @test all(state.albedo .≈ albedo.background_albedo)
        @test all(state.emissivity .≈ albedo.background_emissivity)
    end

    @testset "blends background and snow by snow cover fraction" begin
        vars = Terrarium.Variables(Terrarium.variables(snow)..., Terrarium.variables(albedo)...)
        state = StateVariables(vars, grid)

        set!(state.snow_cover_fraction, 0.5)
        compute_auxiliary!(state, grid, albedo, nothing, snow)
        @test all(state.albedo .≈ 0.5 * albedo.background_albedo + 0.5 * snow.albedo.snow_albedo)
        @test all(state.emissivity .≈ 0.5 * albedo.background_emissivity + 0.5 * snow.albedo.snow_emissivity)

        # full snow cover recovers the snow endpoints; bare cover recovers the background
        set!(state.snow_cover_fraction, 1.0)
        compute_auxiliary!(state, grid, albedo, nothing, snow)
        @test all(state.albedo .≈ snow.albedo.snow_albedo)
        @test all(state.emissivity .≈ snow.albedo.snow_emissivity)

        set!(state.snow_cover_fraction, 0.0)
        compute_auxiliary!(state, grid, albedo, nothing, snow)
        @test all(state.albedo .≈ albedo.background_albedo)
    end

    @testset "blends background and vegetation by vegetated area fraction" begin
        vegetation = PrescribedVegetation(NF)
        traits = vegetation.traits
        vars = Terrarium.Variables(Terrarium.variables(vegetation)..., Terrarium.variables(albedo)...)
        state = StateVariables(vars, grid)

        # half of the maximum leaf area index gives a vegetated fraction of one half
        set!(state.leaf_area_index, traits.maximum_leaf_area_index / 2)
        compute_auxiliary!(state, grid, albedo, vegetation)
        @test all(state.albedo .≈ 0.5 * albedo.background_albedo + 0.5 * traits.albedo)
        @test all(state.emissivity .≈ 0.5 * albedo.background_emissivity + 0.5 * traits.emissivity)

        # leaf area beyond the maximum saturates at full vegetation cover
        set!(state.leaf_area_index, 2 * traits.maximum_leaf_area_index)
        compute_auxiliary!(state, grid, albedo, vegetation)
        @test all(state.albedo .≈ traits.albedo)
        @test all(state.emissivity .≈ traits.emissivity)

        # no leaf area recovers the background
        set!(state.leaf_area_index, 0.0)
        compute_auxiliary!(state, grid, albedo, vegetation)
        @test all(state.albedo .≈ albedo.background_albedo)
    end

    @testset "vegetated area fraction of VegetationCarbonCycle" begin
        traits = PlantTraits(NF)
        LAI_max = traits.maximum_leaf_area_index
        field(value) = set!(Field(grid, Terrarium.Canopy(XY())), value)

        # without vegetation dynamics, diagnosed from the relative leaf area and clamped to the unit interval
        vegetation = VegetationCarbonCycle(NF; vegetation_dynamics = nothing)
        @test Terrarium.vegetation_area_fraction(1, 1, grid, (; leaf_area_index = field(LAI_max / 4)), vegetation) ≈ 0.25
        @test Terrarium.vegetation_area_fraction(1, 1, grid, (; leaf_area_index = field(2 * LAI_max)), vegetation) ≈ 1

        # with vegetation dynamics, given by the prognostic coverage fraction
        vegetation = VegetationCarbonCycle(NF)
        fields = (; leaf_area_index = field(LAI_max), vegetation_area_fraction = field(0.3))
        @test Terrarium.vegetation_area_fraction(1, 1, grid, fields, vegetation) ≈ 0.3
    end
end
