using Terrarium
using Test

using Oceananigans.BoundaryConditions: BoundaryCondition, Flux

@testset "LandModel: Soil, no vegetation" begin
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    hydraulic_properties = ConstantSoilHydraulics(eltype(grid); swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(eltype(grid)))
    hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
    land = LandModel(grid; soil, vegetation = nothing)
    # Test vegetation = nothing results in bare ground evaporation and no canopy interception
    @test land.surface_hydrology.evapotranspiration isa Terrarium.BareGroundEvaporation
    @test land.surface_hydrology.canopy_interception isa Terrarium.NoCanopyInterception
    # Variably saturated with water table
    initializers = (
        temperature = (x, z) -> 5.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> min(1, 0.8 - 0.05 * z),
    )
    integrator = initialize(land; initializers)
    # Check that infiltration is correctly coupled to soil hydrology
    set!(integrator.state.infiltration, 1.0e-8)
    sat_top_bc = integrator.state.saturation_water_ice.boundary_conditions.top
    @test isa(sat_top_bc, BoundaryCondition{<:Flux})
    compute_boundary_conditions!(integrator.state, land)
    @test Array(interior(integrator.state.tendencies.saturation_water_ice))[1, 1, end] > 0
    # Check that ground heat flux is coupled to soil energy
    energy_top_bc = integrator.state.internal_energy.boundary_conditions.top
    @test isa(energy_top_bc, BoundaryCondition{<:Flux})
    @test energy_top_bc.condition == integrator.state.ground_heat_flux
    # Advance one timestep
    timestep!(integrator, 60.0)
    @test all(isfinite.(integrator.state.saturation_water_ice))
    @test all(isfinite.(integrator.state.internal_energy))
    @test all(isfinite.(integrator.state.ground_heat_flux))
end

@testset "LandModel: Coupled vegetation-soil" begin
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    hydraulic_properties = ConstantSoilHydraulics(eltype(grid); swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(eltype(grid)))
    hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
    vegetation = VegetationCarbonCycle(eltype(grid))
    land = LandModel(grid; soil, vegetation)
    @test land.surface_hydrology.evapotranspiration isa Terrarium.PALADYNCanopyEvapotranspiration
    @test land.surface_hydrology.canopy_interception isa Terrarium.PALADYNCanopyInterception
    # Variably saturated with water table
    initializers = (
        temperature = (x, z) -> 5.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> min(1, 0.8 - 0.05 * z),
        carbon_vegetation = 0.1,
    )
    integrator = initialize(land; initializers)
    set!(integrator.state.infiltration, 1.0e-8)
    sat_top_bc = integrator.state.saturation_water_ice.boundary_conditions.top
    @test isa(sat_top_bc, BoundaryCondition{<:Flux})
    compute_boundary_conditions!(integrator.state, land)
    @test Array(interior(integrator.state.tendencies.saturation_water_ice))[1, 1, end] > 0
    # Check that ground heat flux is coupled to soil energy
    energy_top_bc = integrator.state.internal_energy.boundary_conditions.top
    @test isa(energy_top_bc, BoundaryCondition{<:Flux})
    @test energy_top_bc.condition == integrator.state.ground_heat_flux
    # Advance one timestep
    timestep!(integrator, 60.0)
    @test all(isfinite.(integrator.state.saturation_water_ice))
    @test all(isfinite.(integrator.state.internal_energy))
    @test all(isfinite.(integrator.state.ground_heat_flux))
    @test all(isfinite.(integrator.state.carbon_vegetation))
    # TODO: also check ET and veg processes once they are working...
end

@testset "LandModel: Snow-coupled soil" begin
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    hydraulic_properties = ConstantSoilHydraulics(eltype(grid); swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(eltype(grid)))
    hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
    land = LandModel(grid; soil, snow = SingleLayerSnow(eltype(grid)), vegetation = nothing)
    @test land.snow isa SingleLayerSnow
    # Frozen soil beneath a cold snowpack
    initializers = (
        temperature = (x, z) -> -1.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> min(1, 0.8 - 0.05 * z),
        snow_water_equivalent = 0.2,
        snow_temperature = -5.0,
    )
    integrator = initialize(land; initializers)
    # With snow, the soil-top energy BC reads the blended `soil_heat_flux`, not `ground_heat_flux` directly
    energy_top_bc = integrator.state.internal_energy.boundary_conditions.top
    @test isa(energy_top_bc, BoundaryCondition{<:Flux})
    @test energy_top_bc.condition === integrator.state.soil_heat_flux
    # Advance one timestep
    timestep!(integrator, 60.0)
    # Cover fraction is diagnosed in compute_auxiliary! as f = W/(W + W_ref); with the default
    # half-coverage W_ref = 0.1 m, a 0.2 m SWE pack is majority-covered (f = 0.2/0.3 ≈ 0.67)
    @test all(integrator.state.snow_cover_fraction .> 0.6)
    @test all(isfinite.(integrator.state.snow_energy))
    @test all(isfinite.(integrator.state.snow_water_equivalent))
    @test all(isfinite.(integrator.state.snow_temperature))
    @test all(isfinite.(integrator.state.basal_heat_flux))
    @test all(isfinite.(integrator.state.soil_heat_flux))
    @test all(isfinite.(integrator.state.internal_energy))
end

@testset "LandModel: snow meltwater reaches the soil surface" begin
    # Water-conservation check for the snow→soil coupling (Option A): with a melting snowpack and no
    # rain, the meltwater outflow M_r is the sole surface water input, so the runoff scheme must
    # partition exactly M_r into infiltration + surface runoff (no drainage when the surface reservoir
    # is empty): infiltration + surface_runoff ≈ M_r.
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), RichardsEq()))
    snow = SingleLayerSnow(eltype(grid))
    land = LandModel(grid; soil, snow, vegetation = nothing)
    # Fully-liquid (melting) pack over unsaturated soil; no rainfall
    initializers = (
        temperature = (x, z) -> 1.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> 0.5,
        snow_water_equivalent = 0.2,
        snow_temperature = 0.0,
    )
    integrator = initialize(land; initializers)
    state = integrator.state
    set!(state.rainfall, 0.0)
    set!(state.snow_energy, 0.0) # set snow energy to zero -> fully melted (not realistic...)
    Terrarium.closure!(state, land)
    compute_auxiliary!(state, land)
    # meltwater outflow implied by the diagnosed liquid fraction
    θ_liq = Array(interior(state.snow_liquid_fraction))
    M_r = Terrarium.compute_meltwater_outflow.(Ref(snow.hydraulic_properties), θ_liq)
    @test all(θ_liq .≈ 1)                     # fully melted pack
    @test all(M_r .> 0)                       # meltwater draining
    infil = Array(interior(state.infiltration))
    runoff = Array(interior(state.surface_runoff))
    @test all(isapprox.(infil .+ runoff, M_r; rtol = 1.0e-6))
end

@testset "LandModel: latent flux partitioned between ground evaporation and sublimation" begin
    # The surface latent flux is split by snow-covered fraction: ground evaporation over (1 − f_snow),
    # snow sublimation over f_snow. The ground evaporation is scaled at its source by (1 − f_snow), so the
    # stored `evaporation_ground` must equal (1 − f_snow)·g·Δq(T_skin) — no double-counting with sublimation.
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), RichardsEq()))
    inits = (
        temperature = (x, z) -> 2.0 - 0.02 * z, saturation_water_ice = (x, z) -> 0.8,
        snow_water_equivalent = 0.5, snow_temperature = -2.0,
    )
    land = LandModel(grid; soil, snow = SingleLayerSnow(eltype(grid)), vegetation = nothing)
    integrator = initialize(land; initializers = inits)
    Terrarium.closure!(integrator.state, land)
    compute_auxiliary!(integrator.state, land)
    constants = land.constants
    f = Array(interior(integrator.state.snow_cover_fraction))
    E = Array(interior(integrator.state.evaporation_ground))
    g = Array(interior(integrator.state.ground_evaporation_conductance))
    Ts = Array(interior(integrator.state.skin_temperature))
    p = Array(interior(integrator.state.air_pressure))
    q_air = Array(interior(integrator.state.specific_humidity))
    ρ_air = Terrarium.air_density(1, 1, grid, integrator.state, land.atmosphere, constants)
    ρ_w = constants.material.density_water
    # unscaled bulk-aerodynamic ground evaporation g·Δq(T_skin); the stored flux is this scaled by (1 − f)
    Δq = Terrarium.specific_humidity_difference.(Ref(constants.thermodynamics), p, q_air, Ts)
    # ratio of air to water density
    r = ρ_air / ρ_w
    @test all(f .> 0.8)                                                # 0.5 m SWE -> f = 0.5/(0.5 + 0.1) ≈ 0.83 (W_ref = 0.1 m)
    @test all(isapprox.(E, (1 .- f) .* g .* Δq .* r; rtol = 1.0e-6))   # ground evaporation scaled by (1 − f_snow)
    @test all(isfinite.(interior(integrator.state.sublimation)))
    @test all(isfinite.(interior(integrator.state.latent_heat_flux)))
end

@testset "LandModel: thin snow over frozen soil stays stable" begin
    # Regression for the thin-snowpack instability: a vanishing snowpack over cold soil made the
    # snow→soil basal flux diverge (denominator ∝ d_snow → 0), driving the snow temperature far below
    # physical bounds and eventually a DomainError in the vapor-pressure closure. With the basal-flux
    # conduction depth floored at `min_conduction_thickness`, a thin pack must stay finite and bounded.
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), RichardsEq()))
    land = LandModel(grid; soil, snow = SingleLayerSnow(eltype(grid)), vegetation = nothing)
    # 1 mm SWE pack (d_snow ≈ 3 mm) over deeply frozen soil
    initializers = (
        temperature = (x, z) -> -10.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> min(1, 0.8 - 0.05 * z),
        snow_water_equivalent = 1.0e-3,
        snow_temperature = -10.0,
    )
    integrator = initialize(land; initializers)
    # step several times at a coupling-scale Δt; without the floor this blows up within a few steps
    for _ in 1:20
        timestep!(integrator, 300.0)
    end
    T_snow = interior(integrator.state.snow_temperature)
    @test all(isfinite.(T_snow))
    @test all(isfinite.(interior(integrator.state.skin_temperature)))
    @test all(isfinite.(interior(integrator.state.basal_heat_flux)))
    @test all(isfinite.(interior(integrator.state.internal_energy)))
    # snow temperature must remain physically bounded (never dive toward absolute zero)
    @test all(T_snow .>= -60)
    @test all(T_snow .<= 0)
end

@testset "LandModel: surface excess water pool is drained" begin
    # Regression test for the wiring gap that left the runoff-owned `surface_excess_water` pool with no
    # sink in the coupled model: `adjust_saturation_profile!` filled it every step but nothing drained it,
    # so it grew without bound. Now that `SurfaceHydrology.compute_tendencies!` calls the runoff tendency,
    # the pool must be drawn down over a `timestep!` at the surface drainage rate.
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), RichardsEq()))
    land = LandModel(grid; soil, vegetation = nothing)
    integrator = initialize(
        land; initializers = (
            temperature = (x, z) -> 5.0 - 0.02 * z,
            saturation_water_ice = (x, z) -> 0.5,
        )
    )
    state = integrator.state
    τ = land.surface_hydrology.surface_runoff.τ_r
    Δt = 60.0
    S₀ = 0.1
    set!(state.surface_excess_water, S₀)
    pools = Float64[Array(state.surface_excess_water)[1, 1, 1]]
    for _ in 1:5
        timestep!(integrator, Δt)
        push!(pools, Array(state.surface_excess_water)[1, 1, 1])
    end
    # The pool is monotonically drawn down (never grows) ...
    @test all(diff(pools) .< 0)
    # ... and follows the forward-Euler decay Sₙ = S₀ (1 − Δt/τ)ⁿ implied by ∂S∂t = −S/τ.
    analytic = [S₀ * (1 - Δt / τ)^n for n in 0:5]
    @test all(isapprox.(pools, analytic; rtol = 1.0e-9))
    @test all(isfinite.(state.saturation_water_ice))
end
