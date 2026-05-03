using Terrarium
using Test

@testset "Prescribed skin temperature" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    skin_temperature = PrescribedSkinTemperature(eltype(grid))
    seb = SurfaceEnergyBalance(Float64; skin_temperature)
    model = SurfaceEnergyModel(grid, seb)
    state = initialize(model)
    @test hasproperty(state.inputs, :skin_temperature)
    set!(state.skin_temperature, 1.0)
    compute_auxiliary!(state, grid, skin_temperature)
    @test all(state.skin_temperature .≈ 1.0)
end

@testset "Implicit skin temperature" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    clock = Clock(time = 0.0)
    skin_temperature = ImplicitSkinTemperature()
    seb = SurfaceEnergyBalance(Float64; skin_temperature)
    model = SurfaceEnergyModel(grid, seb)
    state = initialize(model)
    @test !hasproperty(state.inputs, :skin_temperature)
    @test hasproperty(state.inputs, :ground_temperature)
    set!(state.surface_shortwave_down, 300.0) # sunny conditions
    set!(state.surface_longwave_down, 50.0)
    set!(state.specific_humidity, 0.002) # dry conditions
    set!(state.air_pressure, 101_325) # standard pressure
    set!(state.air_temperature, 10.0) # 10 °C
    set!(state.ground_temperature, 5.0) # 5 °C
    set!(state.windspeed, 1.0) # 1 m/s
    compute_auxiliary!(state, model)
    @test all(isfinite.(state.skin_temperature))
    @test all(state.sensible_heat_flux .< 0)
    # check that skin temperature converges for a large number of iterations
    Tskin_old = deepcopy(state.skin_temperature)
    resid = nothing
    balance = nothing
    for i in 1:10
        # compute fluxes
        Terrarium.compute_surface_energy_fluxes!(state, grid, seb, model.constants, model.atmosphere)
        # diagnose skin temperature
        Terrarium.update_skin_temperature!(state, grid, seb.skin_temperature)
        resid = maximum(abs.(state.skin_temperature - Tskin_old))
        balance = state.surface_net_radiation + state.latent_heat_flux + state.sensible_heat_flux - state.ground_heat_flux
        Tskin_old = deepcopy(state.skin_temperature)
        println("skin temperature at iteration $i: $(state.skin_temperature[1, 1])  residual: $resid  energy balance: $(balance[1, 1, 1])")
    end
    @test all(resid .< sqrt(eps()))
    @test all(abs.(balance) .< sqrt(eps()))
    # Cloudy and wet
    set!(state.surface_shortwave_down, 150.0)
    set!(state.surface_longwave_down, 50.0)
    set!(state.specific_humidity, 0.01)
    set!(state.air_pressure, 101_325) # standard pressure
    set!(state.air_temperature, 5.0) # 5 °C
    set!(state.ground_temperature, 10.0) # 10 °C
    set!(state.windspeed, 10.0) # 10 m/s
    set!(state.skin_temperature, 10.0) # initial skin temperature = ground temperature
    compute_auxiliary!(state, model)
    @test all(isfinite.(state.skin_temperature))
    @test all(state.sensible_heat_flux .> 0)
    # check that skin temperature converges for a large number of iterations
    Tskin_old = deepcopy(state.skin_temperature)
    for i in 1:10
        # compute fluxes
        Terrarium.compute_surface_energy_fluxes!(state, grid, seb, model.constants, model.atmosphere)
        # diagnose skin temperature
        Terrarium.update_skin_temperature!(state, grid, seb.skin_temperature)
        resid = maximum(abs.(state.skin_temperature - Tskin_old))
        balance = state.surface_net_radiation + state.latent_heat_flux + state.sensible_heat_flux - state.ground_heat_flux
        Tskin_old = deepcopy(state.skin_temperature)
        println("skin temperature at iteration $i: $(state.skin_temperature[1, 1])  residual: $resid  energy balance: $(balance[1, 1, 1])")
    end
    @test all(resid .< sqrt(eps()))
    @test all(abs.(balance) .< sqrt(eps()))
end
