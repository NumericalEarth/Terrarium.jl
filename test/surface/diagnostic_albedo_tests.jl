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
        @test all(state.albedo .≈ albedo.dry_ground_albedo)
        @test all(state.emissivity .≈ albedo.ground_emissivity)
    end

    @testset "ground albedo varies with soil moisture between dry and wet values" begin
        # loam texture; pure sand (the default) has a field capacity barely above the residual water content
        strat = HomogeneousSoilStratigraphy(NF; texture = SoilTexture(NF; sand = 0.4, silt = 0.4, clay = 0.2))
        soil = SoilEnergyWaterCarbon(NF; strat)
        constants = PhysicalConstants(NF)
        vars = Terrarium.Variables(Terrarium.variables(soil)..., Terrarium.variables(albedo)...)
        state = StateVariables(vars, grid)
        # unfrozen soil at the given saturation; initialize! sets the internal energy consistent with both
        function ground_albedo_at(saturation)
            set!(state.temperature, 5.0)
            set!(state.saturation_water_ice, saturation)
            Terrarium.initialize!(state, grid, soil, constants)
            compute_auxiliary!(state, grid, soil, constants)
            compute_auxiliary!(state, grid, albedo, nothing, nothing, soil)
            return interior(state.albedo)[1, 1, 1]
        end

        # saturated soil is above field capacity and dry soil below the residual water content
        @test ground_albedo_at(1.0) ≈ albedo.wet_ground_albedo
        @test ground_albedo_at(0.0) ≈ albedo.dry_ground_albedo

        # in between, the albedo lies within the dry and wet values and decreases monotonically as the soil wets
        αs = [ground_albedo_at(sat) for sat in 0.0:0.05:1.0]
        @test all(albedo.wet_ground_albedo .<= αs .<= albedo.dry_ground_albedo)
        @test all(diff(αs) .<= 0)
        # below field capacity (about half saturation for this loam), the albedo is strictly between the two
        @test albedo.wet_ground_albedo < ground_albedo_at(0.25) < albedo.dry_ground_albedo
    end

    @testset "blends background and snow by snow cover fraction" begin
        vars = Terrarium.Variables(Terrarium.variables(snow)..., Terrarium.variables(albedo)...)
        state = StateVariables(vars, grid)

        set!(state.snow_cover_fraction, 0.5)
        compute_auxiliary!(state, grid, albedo, nothing, snow)
        @test all(state.albedo .≈ 0.5 * albedo.dry_ground_albedo + 0.5 * snow.albedo.snow_albedo)
        @test all(state.emissivity .≈ 0.5 * albedo.ground_emissivity + 0.5 * snow.albedo.snow_emissivity)

        # full snow cover recovers the snow endpoints; bare cover recovers the background
        set!(state.snow_cover_fraction, 1.0)
        compute_auxiliary!(state, grid, albedo, nothing, snow)
        @test all(state.albedo .≈ snow.albedo.snow_albedo)
        @test all(state.emissivity .≈ snow.albedo.snow_emissivity)

        set!(state.snow_cover_fraction, 0.0)
        compute_auxiliary!(state, grid, albedo, nothing, snow)
        @test all(state.albedo .≈ albedo.dry_ground_albedo)
    end

    @testset "blends background and vegetation by vegetated area fraction" begin
        vegetation = PrescribedVegetation(NF)
        traits = vegetation.traits
        vars = Terrarium.Variables(Terrarium.variables(vegetation)..., Terrarium.variables(albedo)...)
        state = StateVariables(vars, grid)

        # half of the maximum leaf area index gives a vegetated fraction of one half
        set!(state.leaf_area_index, traits.maximum_leaf_area_index / 2)
        compute_auxiliary!(state, grid, albedo, vegetation)
        @test all(state.albedo .≈ 0.5 * albedo.dry_ground_albedo + 0.5 * traits.albedo)
        @test all(state.emissivity .≈ 0.5 * albedo.ground_emissivity + 0.5 * traits.emissivity)

        # leaf area beyond the maximum saturates at full vegetation cover
        set!(state.leaf_area_index, 2 * traits.maximum_leaf_area_index)
        compute_auxiliary!(state, grid, albedo, vegetation)
        @test all(state.albedo .≈ traits.albedo)
        @test all(state.emissivity .≈ traits.emissivity)

        # no leaf area recovers the background
        set!(state.leaf_area_index, 0.0)
        compute_auxiliary!(state, grid, albedo, vegetation)
        @test all(state.albedo .≈ albedo.dry_ground_albedo)
    end

    @testset "vegetated area fraction of PrognosticVegetation" begin
        traits = PlantTraits(NF)
        LAI_max = traits.maximum_leaf_area_index
        field(value) = set!(Field(grid, Terrarium.Canopy(XY())), value)

        # without vegetation dynamics, diagnosed from the relative leaf area and clamped to the unit interval
        vegetation = PrognosticVegetation(NF; vegetation_dynamics = nothing)
        @test Terrarium.vegetation_area_fraction(1, 1, grid, (; leaf_area_index = field(LAI_max / 4)), vegetation) ≈ 0.25
        @test Terrarium.vegetation_area_fraction(1, 1, grid, (; leaf_area_index = field(2 * LAI_max)), vegetation) ≈ 1

        # with vegetation dynamics, given by the prognostic coverage fraction
        vegetation = PrognosticVegetation(NF)
        fields = (; leaf_area_index = field(LAI_max), vegetation_area_fraction = field(0.3))
        @test Terrarium.vegetation_area_fraction(1, 1, grid, fields, vegetation) ≈ 0.3
    end
end
