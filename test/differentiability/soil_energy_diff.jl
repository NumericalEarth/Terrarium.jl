using Terrarium
using Test

using Enzyme, FiniteDifferences
using FreezeCurves
using Statistics


function build_soil_energy_model(arch, ::Type{NF}) where {NF}
    grid = ColumnGrid(arch, Float64, ExponentialSpacing(N = 10))
    # initial conditions
    initializer = SoilInitializer(eltype(grid))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), NoFlow()))
    model = SoilModel(grid; soil, initializer)
    return model
end

function mean_soil_temperature_step!(integrator, Δt)

    timestep!(integrator, Δt)
    return mean(interior(integrator.state.temperature))
end

@testset "Soil energy: FreeWater freeze-thaw" begin
    # test liquid_water_fraction
    U = -1.0e7
    sat = 1.0
    por = 0.5
    Lθ = 3.34e8 * sat * por
    lwc_grad, = Enzyme.autodiff(Reverse, Terrarium.liquid_water_fraction, Active, Const(FreeWater()), Active(U), Const(Lθ))
    @test lwc_grad[2] ≈ 1 / Lθ
    ## check case where Lθ is zero
    Lθ = 0.0
    lwc_grad, = Enzyme.autodiff(Reverse, Terrarium.liquid_water_fraction, Active, Const(FreeWater()), Active(U), Const(Lθ))
    @test iszero(lwc_grad[2])

    # test energy_to_temperature
    ## case 1: frozen
    sat = 1.0
    por = 0.5
    Lθ = 3.34e8 * sat * por
    U = Lθ - 1.0e7
    C = 2.0e5
    inv_energy_grad, = Enzyme.autodiff(Reverse, Terrarium.energy_to_temperature, Active, Const(FreeWater()), Active(U), Const(Lθ), Const(C))
    @test inv_energy_grad[2] ≈ 1 / C
    ## case 2: phase change
    sat = 1.0
    por = 0.5
    Lθ = 3.34e8 * sat * por
    U = -Lθ / 2
    C = 2.0e5
    inv_energy_grad, = Enzyme.autodiff(Reverse, Terrarium.energy_to_temperature, Active, Const(FreeWater()), Active(U), Const(Lθ), Const(C))
    @test iszero(inv_energy_grad[2])
    ## case 3: thawed
    sat = 1.0
    por = 0.5
    Lθ = 3.34e8 * sat * por
    U = Lθ / 2
    C = 2.0e5
    inv_energy_grad, = Enzyme.autodiff(Reverse, Terrarium.energy_to_temperature, Active, Const(FreeWater()), Active(U), Const(Lθ), Const(C))
    @test inv_energy_grad[2] ≈ 1 / C
end

@testset "Soil energy model: timestep!" begin
    model = build_soil_energy_model(CPU(), Float64)
    integrator = initialize(model)
    dintegrator = make_zero(integrator)
    Δt = get_timestepper(integrator.model).Δt
    @time Enzyme.autodiff(set_runtime_activity(Reverse), mean_soil_temperature_step!, Active, Duplicated(integrator, dintegrator), Const(Δt))
    @test all(isfinite.(dintegrator.state.temperature))
end
