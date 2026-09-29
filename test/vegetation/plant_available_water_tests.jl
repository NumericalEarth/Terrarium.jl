using Terrarium
using Terrarium: Variables, soil_moisture_limiting_factor
using Test

@testset "FieldCapacityLimitedPAW" begin
    # Check variables
    paw = FieldCapacityLimitedPAW()
    vars = Variables(paw)
    @test haskey(vars.auxiliary, :plant_available_water)
    @test haskey(vars.auxiliary, :soil_moisture_limiting_factor)
    @test haskey(vars.inputs, :root_fraction)

    # Initialize state variables
    grid = ColumnGrid(UniformSpacing(Δz = 0.2, N = 10))
    hydraulic_properties = ConstantSoilHydraulics(eltype(grid), unsat_hydraulic_cond = UnsatKLinear(eltype(grid)))
    hydrology = SoilHydrology(eltype(grid); hydraulic_properties)
    strat = HomogeneousSoilStratigraphy(eltype(grid), porosity = ConstantSoilPorosity(mineral_porosity = 0.5))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology, strat)
    soilvars = Variables(soil)
    state = StateVariables(merge(vars, soilvars), grid)
    set!(state.temperature, 10.0)

    # Soil moisture limiting factor
    ## use uniform root distribution
    Δz = zspacings(ground_domain(grid), Center(), Center(), Face())
    RF = set!(state.root_fraction, Δz / 2)
    ## set PAW to 50% in all layers
    PAW = set!(state.plant_available_water, 0.5)
    ## compute and check SMLF; should be approx. ∑ PAWᵢ * RFᵢ
    compute!(state.soil_moisture_limiting_factor)
    @test all(state.soil_moisture_limiting_factor .≈ sum(PAW * RF, dims = 3))

    # Check PAW calculations
    ## Case 1: Fully saturated
    por = 0.5
    liq = 1.0
    sat = 1.0
    θw = por * liq * sat
    θwp = hydraulic_properties.wilting_point
    θfc = hydraulic_properties.field_capacity
    set!(state.saturation_water_ice, sat)
    set!(state.liquid_water_fraction, liq)
    compute_auxiliary!(state, grid, paw, soil)
    @test all(state.plant_available_water .≈ 1)
    @test all(state.soil_moisture_limiting_factor .≈ 1)

    ## Case 2: Dry
    por = 0.5
    liq = 1.0
    sat = 0.0
    θw = por * liq * sat
    set!(state.saturation_water_ice, sat)
    set!(state.liquid_water_fraction, liq)
    compute_auxiliary!(state, grid, paw, soil)
    @test all(state.plant_available_water .≈ 0)
    @test all(state.soil_moisture_limiting_factor .≈ 0)

    ## Case 3: Frozen
    liq = 0.0
    sat = 1.0
    θw = por * liq * sat
    set!(state.saturation_water_ice, sat)
    set!(state.liquid_water_fraction, liq)
    compute_auxiliary!(state, grid, paw, soil)
    @test all(state.plant_available_water .≈ 0.0)
    @test all(state.soil_moisture_limiting_factor .≈ 0)

    # Case 4: Unsaturated
    liq = 1.0
    sat = 0.2
    θw = por * liq * sat
    set!(state.saturation_water_ice, sat)
    set!(state.liquid_water_fraction, liq)
    compute_auxiliary!(state, grid, paw, soil)
    @test all(state.plant_available_water .≈ 0.25)
    @test all(state.soil_moisture_limiting_factor .≈ sum(state.plant_available_water * state.root_fraction, dims = 3))
end
