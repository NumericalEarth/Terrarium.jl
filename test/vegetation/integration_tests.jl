using Terrarium
using Test

# The coupled `VegetationCarbonCycle.compute_auxiliary!` fuses the carbon-dynamics, phenology,
# photosynthesis, stomatal-conductance, and autotrophic-respiration auxiliaries into a single launch.
# These tests check that the fused kernel reproduces the per-process fan-out (the pre-fusion path) to
# machine precision.

# Auxiliary fields written by the fused kernel (carbon dynamics → phenology → photosynthesis →
# stomatal conductance → autotrophic respiration).
const VEGETATION_CARBON_CYCLE_AUXILIARIES = (
    :balanced_leaf_area_index,
    :phenology_factor, :leaf_area_index,
    :net_assimilation, :leaf_respiration, :gross_primary_production,
    :canopy_water_conductance,
    :autotrophic_respiration, :net_primary_production,
)

function vegetated_land(NF)
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    hydraulic_properties = ConstantSoilHydraulics(NF; swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(NF))
    hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(NF; hydrology)
    vegetation = VegetationCarbonCycle(NF)
    land = LandModel(grid; soil, vegetation)
    initializers = (
        temperature = (x, z) -> 15.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> min(1, 0.8 - 0.05 * z),
        carbon_vegetation = 0.5,
    )
    integrator = initialize(land; initializers)
    set!(integrator.state.rainfall, 1.0e-7)
    Terrarium.closure!(integrator.state, land)
    # Populate atmosphere / soil auxiliaries (vegetation reads air temperature, CO₂, radiation).
    compute_auxiliary!(integrator.state, land)
    return land, integrator.state
end

@testset "Fused vegetation carbon cycle auxiliary matches per-process fan-out" begin
    land, state = vegetated_land(Float64)
    grid = get_grid(land)
    vegetation = land.vegetation
    constants, atmos, soil = land.constants, land.atmosphere, land.soil

    # Fused path: a single launch over the coupled type (plus the root/PAW pre-stages it keeps).
    fused = deepcopy(state)
    compute_auxiliary!(fused, grid, vegetation, constants, atmos, soil)

    # Per-process path: the standalone launches the fan-out used to issue, in dependency order.
    per_process = deepcopy(state)
    compute_auxiliary!(per_process, grid, vegetation.root_distribution, soil)
    compute_auxiliary!(per_process, grid, vegetation.plant_available_water, soil)
    compute_auxiliary!(per_process, grid, vegetation.carbon_dynamics, vegetation.traits)
    compute_auxiliary!(per_process, grid, vegetation.phenology, vegetation.carbon_dynamics, atmos)
    compute_auxiliary!(per_process, grid, vegetation.photosynthesis, vegetation.stomatal_conductance,
                       vegetation.traits, constants, atmos)
    compute_auxiliary!(per_process, grid, vegetation.stomatal_conductance, vegetation.traits, constants, atmos)
    compute_auxiliary!(per_process, grid, vegetation.autotrophic_respiration, vegetation.carbon_dynamics,
                       vegetation.phenology, vegetation.traits, atmos)

    @testset "$(name)" for name in VEGETATION_CARBON_CYCLE_AUXILIARIES
        fused_vals = Array(interior(getproperty(fused, name)))
        per_process_vals = Array(interior(getproperty(per_process, name)))
        @test all(isfinite.(fused_vals))
        @test fused_vals ≈ per_process_vals
    end
end
