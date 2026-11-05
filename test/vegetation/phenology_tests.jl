using Terrarium
using Terrarium: compute_phenology_factor, compute_leaf_area_index, compute_gdd_tendency,
    compute_auxiliary!, initialize, VegetationCarbonCycle, PALADYNPhenology,
    seconds_per_day
using Oceananigans: set!, interior
using Test

@testset "PALADYNPhenology phenology factor" begin
    phenol = PALADYNPhenology{Float64}() # defaults: T_gdd_base=5, gddcrit=200, range=5, f_deciduous=1
    T_warm = 20.0

    # green-up: no accumulated degree days -> no leaves
    @test compute_phenology_factor(phenol, 0.0, T_warm) == 0.0
    # linear ramp with gdd during growth
    @test compute_phenology_factor(phenol, 100.0, T_warm) ≈ 0.5
    # held at 1 once gdd exceeds gddcrit while warm
    @test compute_phenology_factor(phenol, 300.0, T_warm) == 1.0
    # senescence: linear decline in air temperature below the base temperature
    @test compute_phenology_factor(phenol, 300.0, 2.0) ≈ 0.4
    # fully dormant at T_gdd_base - T_senescence_range and below
    @test compute_phenology_factor(phenol, 300.0, 0.0) == 0.0
    # factor stays within [0, 1]
    for gdd in (0.0, 50.0, 250.0), T in (-5.0, 3.0, 25.0)
        ϕ = compute_phenology_factor(phenol, gdd, T)
        @test 0.0 ≤ ϕ ≤ 1.0
    end
end

@testset "PALADYNPhenology evergreen limit" begin
    phenol = PALADYNPhenology{Float64}(f_deciduous = 0.0)
    @test phenol.f_deciduous == 0.0
    # ϕ = 1 independent of gdd / temperature for an evergreen PFT
    @test compute_phenology_factor(phenol, 0.0, 20.0) == 1.0
    @test compute_phenology_factor(phenol, 0.0, -10.0) == 1.0
    # LAI = LAI_b when ϕ = 1
    @test compute_leaf_area_index(phenol, 1.0, 5.0) == 5.0
end

@testset "PALADYNPhenology GDD tendency" begin
    phenol = PALADYNPhenology{Float64}()
    spd = seconds_per_day(Float64)
    # warm: accumulates heat above base temperature (in degree-days per second)
    @test compute_gdd_tendency(phenol, 0.0, 20.0) ≈ (20.0 - 5.0) / spd
    # exactly at base temperature: no accumulation, no relaxation
    @test compute_gdd_tendency(phenol, 100.0, 5.0) == 0.0
    # cold: relaxes toward zero with the relaxation timescale
    @test compute_gdd_tendency(phenol, 300.0, 2.0) ≈ -300.0 / (30.0 * spd)
    # cold with empty accumulator: tendency vanishes
    @test compute_gdd_tendency(phenol, 0.0, -5.0) == 0.0
end

@testset "PALADYNPhenology in VegetationModel" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 5))
    model = VegetationModel(grid; vegetation = VegetationCarbonCycle(Float64; phenology = PALADYNPhenology(Float64)))
    integrator = initialize(model)
    state = integrator.state
    set!(state.air_temperature, 20.0)
    set!(state.balanced_leaf_area_index, 5.0)
    set!(state.growing_degree_days, 100.0)
    # run only the phenology auxiliary computation
    compute_auxiliary!(state, model.grid, model.vegetation.phenology, model.vegetation.carbon_dynamics, model.atmosphere)
    @test Array(interior(state.phenology_factor))[1] ≈ 0.5
    @test Array(interior(state.leaf_area_index))[1] ≈ 2.5
end
