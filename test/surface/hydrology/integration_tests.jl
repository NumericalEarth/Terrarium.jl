using Terrarium
using Test

# The coupled `SurfaceHydrology.compute_auxiliary!` fuses the canopy-interception,
# evapotranspiration, and surface-runoff auxiliaries into a single launch. These tests check that the
# fused kernel reproduces the per-process fan-out (the pre-fusion path) to machine precision, and that
# the canopy evapotranspiration fluxes are scaled by the snow-free fraction (the snow coupling the
# per-process canopy-ET launch used to drop).

# Surface hydrology auxiliary fields written by the fused kernel (canopy + ET + runoff).
const SURFACE_HYDROLOGY_AUXILIARIES = (
    :canopy_water_interception, :canopy_water_removal, :saturation_canopy_water, :rainfall_ground,
    :ground_evaporation_conductance, :canopy_evaporation_conductance, :transpiration_conductance,
    :evaporation_canopy, :evaporation_ground, :transpiration,
    :surface_runoff, :infiltration,
)

function vegetated_snow_land(NF)
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
    vegetation = VegetationCarbonCycle(NF)
    snow = SingleLayerSnow(NF)
    land = LandModel(grid; soil, vegetation, snow)
    initializers = (
        temperature = (x, z) -> 2.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> min(1, 0.8 - 0.05 * z),
        carbon_vegetation = 0.1,
        snow_water_equivalent = 0.2,
        snow_temperature = -2.0,
    )
    integrator = initialize(land; initializers)
    set!(integrator.state.rainfall, 1.0e-7)
    Terrarium.closure!(integrator.state, land)
    # Populate atmosphere / soil / snow / vegetation auxiliaries (surface hydrology runs last).
    compute_auxiliary!(integrator.state, land)
    return land, integrator.state
end

@testset "Fused surface hydrology auxiliary matches per-process fan-out" begin
    land, state = vegetated_snow_land(Float64)
    grid = get_grid(land)
    hydrology = land.surface_hydrology
    constants, atmos, soil, vegetation, snow = land.constants, land.atmosphere, land.soil, land.vegetation, land.snow

    # Fused path: a single launch over the coupled type.
    fused = deepcopy(state)
    compute_auxiliary!(fused, grid, hydrology, constants, atmos, soil, vegetation, snow)

    # Per-process path: the three standalone launches the fan-out used to issue, in dependency order.
    per_process = deepcopy(state)
    compute_auxiliary!(per_process, grid, hydrology.canopy_interception, atmos)
    compute_auxiliary!(
        per_process, grid, hydrology.evapotranspiration, hydrology.canopy_interception,
        constants, atmos, soil, vegetation, snow
    )
    compute_auxiliary!(per_process, grid, hydrology.surface_runoff, hydrology.canopy_interception, soil, snow)

    @testset "$(name)" for name in SURFACE_HYDROLOGY_AUXILIARIES
        fused_vals = Array(interior(getproperty(fused, name)))
        per_process_vals = Array(interior(getproperty(per_process, name)))
        @test all(isfinite.(fused_vals))
        @test fused_vals ≈ per_process_vals
    end
end

@testset "Canopy ET fluxes are scaled by the snow-free fraction" begin
    # With a snow-covered surface, the canopy scheme's ground/canopy evaporation and transpiration must
    # be scaled by (1 − f_snow) — the coupling the per-process canopy-ET launch used to drop. Recomputing
    # the auxiliaries with the snow cover fraction forced to zero must recover the unscaled fluxes.
    land, state = vegetated_snow_land(Float64)
    grid = get_grid(land)
    hydrology = land.surface_hydrology
    constants, atmos, soil, vegetation, snow = land.constants, land.atmosphere, land.soil, land.vegetation, land.snow

    @test all(Array(interior(state.snow_cover_fraction)) .> 0)  # snow is present

    covered = deepcopy(state)
    compute_auxiliary!(covered, grid, hydrology, constants, atmos, soil, vegetation, snow)

    # Same state with the snow cover fraction zeroed: the fused kernel must then leave the ET fluxes
    # unscaled, i.e. the covered fluxes equal the bare fluxes times (1 − f_snow).
    bare = deepcopy(state)
    set!(bare.snow_cover_fraction, 0.0)
    compute_auxiliary!(bare, grid, hydrology, constants, atmos, soil, vegetation, snow)

    f = Array(interior(state.snow_cover_fraction))
    @test all(f .< 1)  # partially covered, so the scaling is nontrivial
    for name in (:evaporation_ground, :transpiration, :evaporation_canopy)
        covered_vals = Array(interior(getproperty(covered, name)))
        bare_vals = Array(interior(getproperty(bare, name)))
        @test all(isapprox.(covered_vals, (1 .- f) .* bare_vals; rtol = 1.0e-9))
    end
end
