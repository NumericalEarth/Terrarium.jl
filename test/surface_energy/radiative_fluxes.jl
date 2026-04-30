using Terrarium
using Test

@testset "Prescribed radiative fluxes" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    radiative_fluxes = PrescribedRadiativeFluxes(eltype(grid))
    seb = SurfaceEnergyBalance(Float64; radiative_fluxes)
    model = SurfaceEnergyModel(grid, seb)
    state = initialize(model)
    @test hasproperty(state.inputs, :surface_shortwave_up)
    @test hasproperty(state.inputs, :surface_longwave_up)
    set!(state.surface_shortwave_down, 100.0)
    set!(state.surface_longwave_down, 20.0)
    set!(state.surface_shortwave_up, 50.0)
    set!(state.surface_longwave_up, 5.0)
    compute_auxiliary!(state, grid, radiative_fluxes, seb, model.atmosphere)
    @test all(state.surface_net_radiation .≈ 50.0 - 100.0 + 5.0 - 20.0)
end

@testset "Diagnosed radiative fluxes" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    radiative_fluxes = DiagnosedRadiativeFluxes(eltype(grid))
    albedo = ConstantAlbedo(albedo = 0.5, emissivity = 0.9)
    seb = SurfaceEnergyBalance(Float64; radiative_fluxes, albedo)
    model = SurfaceEnergyModel(grid, seb)
    state = initialize(model)
    @test !hasproperty(state.inputs, :surface_shortwave_up)
    @test !hasproperty(state.inputs, :surface_longwave_up)
    @test hasproperty(state.auxiliary, :surface_shortwave_up)
    @test hasproperty(state.auxiliary, :surface_longwave_up)
    surface_shortwave_down = 100.0
    surface_longwave_down = 20.0
    set!(state.surface_shortwave_down, surface_shortwave_down)
    set!(state.surface_longwave_down, surface_longwave_down)
    compute_auxiliary!(state, grid, radiative_fluxes, seb, model.constants, model.atmosphere)
    @test all(state.surface_shortwave_up .≈ 0.5 * surface_shortwave_down)
    T_skin_K = Terrarium.celsius_to_kelvin(model.constants, 0.0)
    @test all(state.surface_longwave_up .≈ (1 - 0.9) * surface_longwave_down + Terrarium.stefan_boltzmann(model.constants, T_skin_K, 0.9))
    @test all(state.surface_net_radiation .≈ state.surface_shortwave_up - surface_shortwave_down + state.surface_longwave_up - surface_longwave_down)
end
