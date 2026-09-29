using Terrarium
using Test

import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.Diagnostics: cell_diffusion_timescale

# Recompute the thermal diffusion timescale independently from the public pointwise property
# functions, mirroring τ = Δz² C / κ, and return the per-cell minimum over the interior.
function reference_thermal_timescale(integrator)
    model = integrator.model
    soil = model.soil
    state = integrator.state
    energy = Terrarium.get_energy_balance(soil)
    hydrology = Terrarium.get_hydrology(soil)
    strat = Terrarium.get_stratigraphy(soil)
    bgc = Terrarium.get_biogeochemistry(soil)
    ground_grid = ground_domain(model.grid)
    fields = Terrarium.get_fields(state, energy, hydrology, strat, bgc)
    Nx, Ny, Nz = size(ground_grid)
    τmin = Inf
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        soilvol = Terrarium.soil_composition(i, j, k, ground_grid, fields, strat, hydrology, bgc)
        κ = Terrarium.compute_thermal_conductivity(energy.thermal_properties, soilvol)
        C = Terrarium.compute_heat_capacity(energy.thermal_properties, soilvol)
        Δz = Terrarium.Δzᵃᵃᶜ(i, j, k, ground_grid)
        τmin = min(τmin, Δz^2 * C / κ)
    end
    return τmin
end

@testset "cell_diffusion_timescale (SoilModel, NoFlow)" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 20))
    model = SoilModel(grid)
    integrator = initialize(model)

    τ = cell_diffusion_timescale(integrator)
    @test isfinite(τ)
    @test τ > 0
    @test typeof(τ) === Float64

    # NoFlow hydrology imposes no restriction, so the model timescale is the thermal one
    @test τ ≈ reference_thermal_timescale(integrator)

    # implied thermal diffusivity α = Δz²/τ should be a physically plausible soil value
    α = 0.1^2 / τ
    @test 1.0e-8 < α < 1.0e-4

    # land models have no advective restriction
    @test cell_advection_timescale(integrator) == Inf
end

@testset "cell_diffusion_timescale (SoilModel, RichardsEq)" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 20))
    hydrology = SoilHydrology(Float64; vertical_flow = RichardsEq())
    soil = Terrarium.SoilEnergyWaterCarbon(Float64; hydrology)
    model = SoilModel(grid; soil)
    integrator = initialize(model)

    τ_energy = cell_diffusion_timescale(integrator.state, grid, Terrarium.get_energy_balance(soil), soil, model.constants)
    τ_water = cell_diffusion_timescale(integrator.state, grid, Terrarium.get_hydrology(soil), soil, model.constants)
    @test isfinite(τ_energy) && τ_energy > 0
    @test isfinite(τ_water) && τ_water > 0

    # the coupled timescale is the minimum of the two process timescales
    τ = cell_diffusion_timescale(integrator)
    @test τ ≈ min(τ_energy, τ_water)
end

@testset "single-float32 numeric type" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 10))
    model = SoilModel(grid)
    integrator = initialize(model)
    τ = cell_diffusion_timescale(integrator)
    @test typeof(τ) === Float32
    @test isfinite(τ) && τ > 0
end

@testset "TimeStepWizard drives an adaptive Simulation" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 20))
    integrator = initialize(SoilModel(grid))
    sim = Simulation(integrator; Δt = 100.0, stop_iteration = 20)
    wizard = TimeStepWizard(diffusive_cfl = 0.4)
    sim.callbacks[:wizard] = Callback(wizard, IterationInterval(2))
    run!(sim)

    @test all(isfinite.(integrator.state.temperature))
    @test isfinite(sim.Δt) && sim.Δt > 0
    # the diffusive-CFL target step is bounded by diffusive_cfl * τ
    @test sim.Δt <= 0.4 * cell_diffusion_timescale(integrator) + 1
end
