using Terrarium
using Test

using Oceananigans.BoundaryConditions: BoundaryCondition, Flux
using Oceananigans: Center, Face, znodes, zspacings
using Oceananigans.TimeSteppers: update_state!

import RingGrids

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
    # Check that ground evaporation is applied as a sink in the soil water tendency (issue #203).
    # Compute the tendencies once with zero and once with nonzero ground evaporation; the difference
    # in the top-layer saturation tendency must equal E / (Δz * porosity). The evaporation flux is set
    # after the boundary conditions since the SEB solve recomputes it from the skin temperature.
    state = integrator.state
    function top_saturation_tendency(E)
        Terrarium.reset_tendencies!(state)
        compute_boundary_conditions!(state, land)
        set!(state.evaporation_ground, E)
        compute_tendencies!(state, land)
        return Array(interior(state.tendencies.saturation_water_ice))[1, 1, end]
    end
    E = 2.0e-8 # m/s
    dsat_no_ET = top_saturation_tendency(0.0)
    dsat_ET = top_saturation_tendency(E)
    Δz_top = Terrarium.Δzᵃᵃᶜ(1, 1, grid.Nz, grid)
    strat = Terrarium.get_stratigraphy(soil)
    bgc = Terrarium.get_biogeochemistry(soil)
    por = Terrarium.porosity(1, 1, grid.Nz, grid, get_fields(state, strat, bgc), strat, bgc)
    @test dsat_ET < dsat_no_ET
    @test dsat_no_ET - dsat_ET ≈ E / (Δz_top * por)
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
    @test all(isfinite.(integrator.state.snow_basal_heat_flux))
    @test all(isfinite.(integrator.state.soil_heat_flux))
    @test all(isfinite.(integrator.state.internal_energy))
end

@testset "LandModel: snow meltwater reaches the soil surface" begin
    # Water-conservation check for the snow→soil coupling: with a melting snowpack and no rain, the
    # meltwater outflow `meltwater_flux` is the sole surface water input, so the runoff scheme must
    # partition exactly that flux into infiltration + surface runoff (no drainage when the surface
    # reservoir is empty).
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), RichardsEq()))
    snow = SingleLayerSnow(eltype(grid))
    # Fully-liquid (melting) pack over unsaturated soil; no rainfall
    initializers = (
        temperature = (x, z) -> 1.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> 0.5,
        snow_water_equivalent = 0.2,
        snow_temperature = 0.0,
    )
    function melting_land(timestepper)
        land = LandModel(grid; soil, snow, vegetation = nothing, timestepper)
        integrator = initialize(land; initializers)
        state = integrator.state
        set!(state.rainfall, 0.0)
        set!(state.snow_energy, 0.0) # set snow energy to zero -> fully melted (not realistic...)
        Terrarium.closure!(state, land)
        compute_auxiliary!(state, land)
        return land, integrator
    end
    surface_input(state) = Array(interior(state.infiltration)) .+ Array(interior(state.surface_runoff))

    # explicit: the outflow at the diagnosed liquid fraction
    land, integrator = melting_land(ForwardEuler(eltype(grid)))
    state = integrator.state
    θ_liq = Array(interior(state.snow_liquid_fraction))
    W_snow = Array(interior(state.snow_water_equivalent))
    M_r = Terrarium.compute_meltwater_outflow.(Ref(snow.hydraulic_properties), θ_liq, W_snow)
    @test all(θ_liq .≈ 1)                     # fully melted pack
    @test all(M_r .> 0)                       # meltwater draining
    @test all(isapprox.(surface_input(state), M_r; rtol = 1.0e-6))

    # implicit: the drainage of the last implicit step, which the surface hydrology receives in the next step
    land, integrator = melting_land(IMEX(ForwardEuler(eltype(grid); Δt = 900), BackwardEuler(eltype(grid))))
    state = integrator.state
    W⁰ = state.snow_water_equivalent[1, 1, 1]
    timestep!(integrator)
    M = Array(interior(state.meltwater_flux))
    @test all(M .> 0)
    @test all(M .* 900 .<= W⁰)                # drains no more than the pack holds
    @test all(isapprox.(surface_input(state), M; rtol = 1.0e-6))
end

@testset "LandModel: thin melting snowpack conserves water independently of Δt" begin
    # A thin, fully liquid pack must not release more water than it stores: the meltwater outflow is limited
    # by the stored excess liquid water (see `compute_meltwater_outflow`), so the water reaching the soil
    # surface (infiltration + surface runoff) never exceeds the initial snow water equivalent and does not
    # depend on the time step. Soil water is immobile (`NoFlow`) so that
    # the explicit Richards equation, which is unstable on this grid at these time steps, does not interfere.
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), NoFlow()))
    snow = SingleLayerSnow(eltype(grid))
    land = LandModel(grid; soil, snow, vegetation = nothing)
    W₀ = 0.005
    initializers = (
        temperature = (x, z) -> 1.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> 0.5,
        snow_water_equivalent = W₀,
        snow_temperature = 0.0,
    )
    duration = 6 * 3600.0
    released = map((450.0, 60.0)) do Δt
        integrator = initialize(land; initializers)
        state = integrator.state
        set!(state.rainfall, 0.0)
        set!(state.snow_energy, 0.0) # fully melted pack
        Terrarium.closure!(state, land)
        compute_auxiliary!(state, land)
        water_to_soil = 0.0
        for _ in 1:round(Int, duration / Δt)
            # fluxes applied by the forward Euler step from the current state
            water_to_soil += Δt * (only(interior(state.infiltration)) + only(interior(state.surface_runoff)))
            Terrarium.timestep!(integrator, Δt)
        end
        @test all(interior(state.snow_water_equivalent) .>= 0)
        water_to_soil
    end
    @test all(released .<= W₀ * (1 + 1.0e-6))  # no water is created
    @test all(released .> 0)                   # the pack drains
    @test isapprox(released[1], released[2]; rtol = 0.05)
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
    @test all(isfinite.(interior(integrator.state.snow_basal_heat_flux)))
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
    pools = Float64[Array(interior(state.surface_excess_water))[1, 1, 1]]
    for _ in 1:5
        timestep!(integrator, Δt)
        push!(pools, Array(interior(state.surface_excess_water))[1, 1, 1])
    end
    # The pool is monotonically drawn down (never grows) ...
    @test all(diff(pools) .< 0)
    # ... and follows the forward-Euler decay Sₙ = S₀ (1 − Δt/τ)ⁿ implied by ∂S∂t = −S/τ.
    analytic = [S₀ * (1 - Δt / τ)^n for n in 0:5]
    @test all(isapprox.(pools, analytic; rtol = 1.0e-9))
    @test all(isfinite.(state.saturation_water_ice))
end

# Shared setup for the snow boundary heat flux tests below: a snowpack of water equivalent `W₀` (m) at
# `T_snow` over warm soil (+5 °C near the top) under cold air, night-time radiation, and no precipitation. Soil water is
# immobile (`NoFlow`) so only the energy balance is exercised.
function cold_air_snow_simulation(W₀; T_snow = -5.0, T_air = -10.0)
    grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), NoFlow()))
    land = LandModel(grid; soil, snow = SingleLayerSnow(eltype(grid)), vegetation = nothing)
    initializers = (
        temperature = (x, z) -> 5.0 - 0.1 * z,
        saturation_water_ice = (x, z) -> 0.5,
        snow_water_equivalent = W₀,
        snow_temperature = T_snow,
    )
    integrator = initialize(land; initializers)
    state = integrator.state
    set!(state.air_temperature, T_air)
    set!(state.rainfall, 0.0)
    set!(state.snowfall, 0.0)
    # night-time radiation (no sunlight; cold sky) and moderate wind, so the skin stays near the air temperature
    set!(state.surface_shortwave_down, 0.0)
    set!(state.surface_longwave_down, 230.0)
    set!(state.windspeed, 2.0)
    Terrarium.closure!(state, land)
    compute_auxiliary!(state, land)
    return land, integrator
end

@testset "LandModel: thin cold snowpack over warm soil is not melted by the bare-ground flux" begin
    # Regression: the snow base used to receive the blended soil flux f·Q_base + (1 − f)·G, which for a thin
    # pack (f → 0) is the bare-ground flux G applied in full to the grid-cell-mean snow energy. Over warm
    # soil this melted a 1 mm pack within minutes even at subfreezing temperatures. With cell-mean boundary
    # fluxes (basal flux f·Q_base), the pack must persist and no meltwater may be produced.
    W₀ = 1.0e-3
    land, integrator = cold_air_snow_simulation(W₀)
    state = integrator.state
    Δt = 300.0
    for _ in 1:round(Int, 6 * 3600 / Δt)
        Terrarium.timestep!(integrator, Δt)
    end
    W = only(interior(state.snow_water_equivalent))
    @test isapprox(W, W₀; rtol = 0.05)
    @test all(interior(state.snow_liquid_fraction) .< 1.0e-6)
    @test all(isfinite.(interior(state.snow_temperature)))
end

@testset "LandModel: soil + snow column energy balance closes against the atmosphere-side flux" begin
    # The column (soil + snow) must lose exactly G₀, the conductive flux delivered to the skin and passed on
    # to the atmosphere. G₀ is not stored directly; it is reconstructed from the area-weighted conduction
    # G₀ = (1 − f)·G + f·S, with G = `ground_heat_flux` (per unit bare area), f = `snow_cover_fraction`, and
    # f·S = `snow_surface_heat_flux` (the cell-mean snow-top flux). Precipitation is zero, so there is no
    # precipitation enthalpy term. The snow energy tendency also contains the advective sublimation
    # correction Q_subl = ρ_w·L_sl·E_subl (applied while W > 0), which is not part of G₀ and is added to
    # the budget. The soil bottom boundary is the default (zero-flux) condition. Forward Euler `timestep!`
    # begins with `update_state!`, which recomputes the auxiliaries (including all fluxes) from the current
    # prognostic state, so `update_state!` is called here first and the fluxes are accumulated from the same
    # fields the step then applies. (Reading the fields left over from the end of the previous step would be
    # stale by one step, since the closure relations modify the prognostic state after the last evaluation.)
    for W₀ in (1.0e-3, 0.1)
        land, integrator = cold_air_snow_simulation(W₀)
        state = integrator.state
        grid = get_grid(land)
        ρ_w = land.constants.material.density_water
        L_sl = land.constants.thermodynamics.latent_heat_fusion
        Δz = diff(Array(znodes(Terrarium.ground_domain(grid), Face())))
        column_energy() = sum(vec(Array(interior(state.internal_energy))) .* Δz) + only(interior(state.snow_energy))
        E₀ = column_energy()
        flux_integral = 0.0
        subl_correction = 0.0
        Δt = 60.0
        for _ in 1:round(Int, 6 * 3600 / Δt)
            update_state!(integrator)
            f = only(interior(state.snow_cover_fraction))
            G = only(interior(state.ground_heat_flux))
            fS = only(interior(state.snow_surface_heat_flux))
            G₀ = (1 - f) * G + fS
            flux_integral += Δt * G₀
            W = only(interior(state.snow_water_equivalent))
            subl_correction += Δt * ρ_w * L_sl * only(interior(state.sublimation)) * (W > 0)
            Terrarium.timestep!(integrator, Δt)
        end
        ΔE = column_energy() - E₀
        expected = -flux_integral + subl_correction
        @test isapprox(ΔE, expected; rtol = 1.0e-9)
    end
end

# `ColumnGrid` is an alias for an Oceananigans `RectilinearGrid`, so the testsets above exercise
# Oceananigans' own grid metrics. `ColumnRingGrid` is a grid *wrapper* and must forward those
# metrics itself; when it did not, every `LandModel` on a ring grid failed on the first timestep
# with a `MethodError` on `Δxᶜᵃᵃ`, because the surface energy balance imposes a flux boundary
# condition and Oceananigans divides the flux by the cell area.
@testset "LandModel: ColumnRingGrid" begin
    rings = RingGrids.FullGaussianGrid(4)
    mask = RingGrids.Field(rings)
    mask .= 1
    grid = ColumnRingGrid(CPU(), Float64, ExponentialSpacing(Δz_max = 1.0, N = 10), rings, mask .> 0)
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology = SoilHydrology(eltype(grid), RichardsEq()))
    land = LandModel(grid; soil, vegetation = nothing)
    integrator = initialize(
        land; initializers = (
            temperature = (x, z) -> 5.0 - 0.02 * z,
            saturation_water_ice = (x, z) -> 0.5,
        )
    )
    state = integrator.state
    # The surface energy balance is coupled through a flux boundary condition, which is what needs
    # the cell metrics of the wrapped grid.
    @test isa(state.internal_energy.boundary_conditions.top, BoundaryCondition{<:Flux})
    # One column per unmasked ring grid point
    @test size(interior(state.temperature), 1) == sum(mask .> 0)

    timestep!(integrator, 60.0)
    @test all(isfinite.(state.temperature))
    @test all(isfinite.(state.saturation_water_ice))
    @test all(isfinite.(state.ground_heat_flux))
end

@testset "LandModel: dry soil under strong evaporation demand stays physical (ground resistance $(nameof(typeof(ground_resistance))))" for ground_resistance in (
        ConstantEvaporationResistanceFactor(Float64),
        SoilMoistureResistanceFactor(Float64),
    )
    # A sparsely vegetated column on the Somali coast (46.8°E, 4.7°N) in the dry season, where a coupled
    # SpeedyWeather simulation failed: SoilGrids texture of that column, and the forcing SpeedyWeather passed to it
    # (except for the air temperature, which is set to a typical value). The soil starts close to residual
    # saturation without rain, so ground evaporation (unlimited with `ConstantEvaporationResistanceFactor`) acts
    # on an almost dry Richards column. Over a few time steps, the soil must keep its saturation in the valid range,
    # lose exactly the water that evaporates, and keep physically plausible temperatures.
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(; N = 30))
    φ = 0.45
    texture = SoilTexture(NF; sand = 0.386, silt = 0.315, clay = 0.299)
    strat = HomogeneousSoilStratigraphy(NF; texture, porosity = ConstantSoilPorosity(NF; mineral_porosity = φ))
    biogeochem = ConstantSoilCarbonDensity(NF; ρ_soc = 0.0) # zero SOC -> pure mineral porosity
    soil = SoilEnergyWaterCarbon(NF; strat, biogeochem)
    evapotranspiration = PALADYNCanopyEvapotranspiration(NF; ground_resistance)
    surface_hydrology = SurfaceHydrology(NF; evapotranspiration)
    land = LandModel(grid; soil, vegetation = PrescribedVegetation(NF), surface_hydrology)
    T_air = 27.0
    integrator = initialize(land; initializers = (temperature = T_air, saturation_water_ice = 0.05))
    state = integrator.state
    set!(state.air_temperature, T_air)
    set!(state.specific_humidity, 0.0098)
    set!(state.windspeed, 9.0)
    set!(state.air_pressure, 98_931.0)
    set!(state.rainfall, 0.0)
    set!(state.surface_shortwave_down, 343.0)
    set!(state.surface_longwave_down, 321.0)
    set!(state.leaf_area_index, 0.0)
    compute_auxiliary!(state, land)

    Δz = Array(interior(zspacings(Terrarium.ground_domain(grid), Center(), Center(), Center())))[:]
    # soil water plus the surface pool that receives excess water from the top soil cell (m)
    column_water(state) = φ * sum(interior(state.saturation_water_ice)[1, 1, :] .* Δz) + only(interior(state.surface_excess_water))
    s_res = Terrarium.residual_saturation(Terrarium.get_hydraulic_properties(Terrarium.get_hydrology(soil)))
    W₀ = column_water(state)
    Δt = 450.0
    removed = 0.0
    Nt = 16 # number of time steps
    for _ in 1:Nt
        # recompute the fluxes from the current prognostic state, so that they are the ones the step applies
        update_state!(integrator)
        removed += Δt * (only(interior(state.evaporation_ground)) + only(interior(state.surface_runoff)))
        Terrarium.timestep!(integrator, Δt)
    end
    sat = interior(state.saturation_water_ice)[1, 1, :]
    @test all(isfinite, sat)
    @test all(isfinite, interior(state.internal_energy))
    @test all(s_res - 1.0e-6 .<= sat .<= 1)
    # no water is created or lost other than by evaporation and surface runoff
    @test column_water(state) ≈ W₀ - removed rtol = 1.0e-3
    # temperatures stays close to air temperature
    @test abs(only(interior(state.skin_temperature)) - T_air) < 20
    @test all(abs.(interior(state.temperature)[1, 1, :] .- T_air) .< 20)
end
