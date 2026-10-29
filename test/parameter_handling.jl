using Terrarium
using Terrarium: parameters, getproperties
using Test

@testset "Collect parameters" begin
    # model setup
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_min = 0.05, Δz_max = 100.0, N = 100))
    soil = SoilEnergyWaterCarbon(eltype(grid))
    soil_params = vec(parameters(soil))
    @test length(soil_params) > 0
    @test all(map(Base.Fix1(haskey, soil_params), (:biogeochem, :energy, :hydrology)))
    @test haskey(soil_params, :strat)
    # check that collected parameters for thermal conductivities match the values in the struct
    @test all(map(==, soil_params.energy.thermal_properties.conductivities, getproperties(soil.energy.thermal_properties.conductivities)))
    model = SoilModel(grid; soil)
    model_params = vec(parameters(model))
    # increase all parameters by 5%, except the texture fractions, which must sum to one
    texture = getproperty(model_params.soil.strat.horizons, Symbol(1)).texture
    unperturbed_texture = copy(texture)
    model_params .*= 1.05
    texture .= unperturbed_texture
    integrator = initialize(model, model_params)
    @test all(vec(parameters(integrator.model)) .≈ model_params)
    # test reinitialization of integrator
    integrator = initialize(model)
    updated_integrator = initialize(integrator, model_params)
    @test all(vec(parameters(updated_integrator.model)) .≈ model_params)
end
