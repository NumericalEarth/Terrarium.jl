using Terrarium
using Terrarium: parameters, getproperties, ParameterEditing
using Test

@testset "Collect parameters" begin
    # model setup
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_min = 0.05, Δz_max = 100.0, N = 100))
    soil = SoilEnergyWaterCarbon(eltype(grid))
    soil_params = vec(parameters(soil))
    @test length(soil_params) > 0
    @test all(map(Base.Fix1(haskey, soil_params), (:biogeochem, :energy, :hydrology, :strat)))
    # check that collected parameters for thermal conductivities match the values in the struct
    @test all(map(==, soil_params.energy.thermal_properties.conductivities, getproperties(soil.energy.thermal_properties.conductivities)))
    model = SoilModel(grid; soil)
    model_params = vec(parameters(model))
    # increase all parameters by 5%
    model_params .*= 1.05
    integrator = initialize(model, model_params)
    @test all(vec(parameters(integrator.model)) .≈ model_params)
    # test reinitialization of integrator
    integrator = initialize(model)
    updated_integrator = initialize(integrator, model_params)
    @test all(vec(parameters(integrator.model)) .≈ model_params)
end

@testset "Regression: Reconstruct snow parameters" begin
    # `SingleLayerSnow` carries `NF` only via its supertype bound, so it needs an explicit
    # `constructorof` method for `reconstruct` to be able to rebuild it.
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_min = 0.05, Δz_max = 100.0, N = 100))
    snow = SingleLayerSnow(eltype(grid))
    snow_params = vec(parameters(snow))
    @test length(snow_params) > 0
    # round-trip with modified values
    updated_params = snow_params .* 1.05
    updated_snow = ParameterEditing.reconstruct(snow, updated_params)
    @test typeof(updated_snow) === typeof(snow)
    @test all(vec(parameters(updated_snow)) .≈ updated_params)
    # and as a component of a coupled model
    model = LandModel(grid; snow, vegetation = nothing)
    updated_model = ParameterEditing.reconstruct(model, ParameterEditing.ComponentVector(snow = updated_params))
    @test typeof(updated_model) === typeof(model)
    @test all(vec(parameters(updated_model.snow)) .≈ updated_params)
end
