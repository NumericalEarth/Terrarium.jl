using Terrarium
using Thermodynamics
using Oceananigans
using Test

"""
    test_skin_temperature_solve!(state, grid, seb, model; kwargs...)

Iteratively solve for the implicit skin temperature until convergence by alternating between
computing surface energy fluxes and updating the skin temperature. All input values are set
inside this function from keyword arguments.

# Arguments
- `state`: initialized model state
- `grid`: column grid
- `seb`: [`SurfaceEnergyBalance`](@ref) containing an [`ImplicitSkinTemperature`](@ref)
- `model`: [`SurfaceEnergyModel`](@ref)

# Keyword arguments (all required unless noted)
- `surface_shortwave_down::Float64`: downward shortwave radiation (W/m²)
- `surface_longwave_down::Float64`: downward longwave radiation (W/m²)
- `air_pressure::Float64`: air pressure at the surface (Pa)
- `air_temperature::Float64`: air temperature (°C)
- `relative_humidity::Float64`: relative humidity (0–1)
- `ground_temperature::Float64`: ground temperature (°C)
- `windspeed::Float64`: wind speed at reference height (m/s)
- `κₛ::Float64`: ground thermal conductivity (W/(m·K)) — default 0.5
- `Cₕ::Float64`: aerodynamic heat transfer coefficient — default 2.0e-3
- `max_iterations::Int`: maximum iterations — default 20
- `tolerance::Float64`: convergence tolerance — default `sqrt(eps())`

# Returns
Named tuple with fields: `skin_temperature`, `residual`, `net_radiation`, `latent_heat_flux`,
`sensible_heat_flux`, `ground_heat_flux`.
"""
function test_skin_temperature_solve!(
        state,
        grid::Terrarium.AbstractLandGrid{NF},
        model::Terrarium.AbstractModel{NF};
        surface_shortwave_down,
        surface_longwave_down,
        air_pressure,
        air_temperature,
        relative_humidity,
        ground_temperature,
        windspeed,
        snow_water_equivalent = zero(NF),
        soil_moisture = one(NF),
        leaf_area_index = zero(NF),
        tolerance = NF(1.0e-5),
        Ts_min = NF(-100),
        Ts_max = NF(100)
    ) where {NF}
    set!(state.surface_shortwave_down, NF(surface_shortwave_down))
    set!(state.surface_longwave_down, NF(surface_longwave_down))
    set!(state.air_pressure, NF(air_pressure))
    set!(state.air_temperature, NF(air_temperature))
    thermodynamics_constants = model.constants.thermodynamics
    air_density = Thermodynamics.air_density(thermodynamics_constants, air_temperature + NF(273.15), air_pressure)
    specific_humidity_saturation = Terrarium.saturation_specific_humidity_vapor(thermodynamics_constants, air_temperature, air_density)
    set!(state.specific_humidity, NF(relative_humidity * specific_humidity_saturation))
    set!(state.windspeed, NF(windspeed))
    # Seed the implicit solve with a physical initial guess (the ground temperature),
    # mirroring the SEB initializer used in the coupled model.
    set!(state.skin_temperature, ground_temperature)
    set!(state.ground_temperature, NF(ground_temperature))
    # Conditional settings
    if hasproperty(state, :temperature)
        set!(state.temperature, NF(ground_temperature)) # set full temperature profile to ground_temperature
    end
    if hasproperty(state, :saturation_water_ice)
        set!(state.saturation_water_ice, NF(soil_moisture)) # same for soil moisture
    end
    if hasproperty(state, :snow_water_equivalent)
        set!(state.snow_water_equivalent, NF(snow_water_equivalent))
    end
    if hasproperty(state, :leaf_area_index)
        set!(state.leaf_area_index, NF(leaf_area_index))
    end

    # Update model state
    Terrarium.update_state!(state, model, InputSources(eltype(grid)))

    # Check that the skin temperature is finite and within plausible range
    @test all(isfinite.(state.skin_temperature)) && all(state.skin_temperature .> -100) && all(state.skin_temperature .< 100)

    # Check convergence in temperature space, mirroring exactly what `compute_skin_temperature_residual!`
    # computes: the atmosphere-side demand (R_net + H_s + H_l) inverted through the (unblended) ground and
    # snow-top conduction targets should reproduce the current skin temperature. Also check that the
    # *stored* `ground_heat_flux` is the explicit bare-ground conductive flux at that Ts (see the
    # `ImplicitSkinTemperature`-specific `compute_ground_heat_flux` in skin_temperature.jl).
    skinT = model.surface_energy_balance.skin_temperature
    snow = hasproperty(model, :snow) ? model.snow : nothing
    Tg, κg, Δzg = Terrarium.ground_thermal_interface(1, 1, grid, state, skinT)
    Ts = state.skin_temperature[1, 1]
    G = state.ground_heat_flux[1, 1]
    G_gradient = 2 * κg * (Tg - Ts) / Δzg
    R_net = state.surface_net_radiation[1, 1]
    H_s = state.sensible_heat_flux[1, 1]
    H_l = state.latent_heat_flux[1, 1]
    Ts_implicit = Terrarium.compute_skin_temperature(1, 1, grid, state, skinT, model.constants, snow)
    Ts_residual = Ts - Ts_implicit
    G_residual = G - G_gradient

    results = (
        skin_temperature = Ts,
        residual = Ts_residual,
        surface_net_radiation = R_net,
        latent_heat_flux = H_l,
        sensible_heat_flux = H_s,
        ground_heat_flux = G,
        ground_heat_flux_residual = G_residual,
    )

    G_check = abs(G_residual) < tolerance
    resid_check = abs(Ts_residual) < tolerance
    Ts_check = Ts_min < results.skin_temperature < Ts_max

    if !G_check || !resid_check
        inputs = (; surface_shortwave_down, surface_longwave_down, air_temperature, ground_temperature, air_pressure, relative_humidity, windspeed)
        @warn "SEB solve failed to converge for inputs $inputs; output: $results"
    end

    @test G_check && resid_check && Ts_check

    return results
end

@testset "Prescribed skin temperature" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    skin_temperature = PrescribedSkinTemperature(eltype(grid))
    seb = SurfaceEnergyBalance(Float64; skin_temperature)
    model = SurfaceEnergyModel(grid, surface_energy_balance = seb)
    state = StateVariables(model)
    @test hasproperty(state.inputs, :skin_temperature)
    set!(state.skin_temperature, 1.0)
    compute_auxiliary!(state, grid, skin_temperature)
    @test all(state.skin_temperature .≈ 1.0)
end

@testset "Prescribed skin temperature, full SEB" begin
    # Configuration used when the surface turbulent fluxes are computed by an external
    # coupler (e.g. NumericalEarth's Monin-Obukhov scheme) and prescribed to Terrarium:
    # the skin temperature and sensible/latent fluxes are inputs, radiation is diagnosed
    # locally, and the ground heat flux closes as the residual G = R_net + H_s + H_l.
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_max = 1.0, N = 20))
    NF = eltype(grid)
    seb = SurfaceEnergyBalance(
        NF;
        skin_temperature = PrescribedSkinTemperature(NF),
        turbulent_fluxes = PrescribedTurbulentFluxes(NF),
        radiative_fluxes = DiagnosedRadiativeFluxes(NF),
        albedo = DiagnosticAlbedo(NF)
    )
    soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
    land = LandModel(grid; soil, surface_energy_balance = seb, vegetation = nothing)
    integrator = initialize(
        land;
        initializers = (
            temperature = (x, z) -> 5.0 - 0.02 * z,
            saturation_water_ice = (x, z) -> 0.5,
        )
    )
    state = integrator.state

    # Prescribe the skin temperature and the turbulent heat fluxes.
    set!(state.skin_temperature, 10.0)
    set!(state.sensible_heat_flux, 20.0)
    set!(state.latent_heat_flux, 30.0)
    Terrarium.closure!(state, land)
    compute_auxiliary!(state, land)
    compute_boundary_conditions!(state, land)

    R_net = Array(interior(state.surface_net_radiation))
    G = Array(interior(state.ground_heat_flux))
    # Radiation is diagnosed from the prescribed skin temperature (so it is nonzero) and
    # the ground heat flux is the residual of the prescribed components.
    @test all(abs.(R_net) .> 0)
    @test all(isapprox.(G, R_net .+ 20 .+ 30; rtol = 1.0e-6))

    # The soil integrates against the prescribed-flux boundary condition.
    timestep!(integrator, 60.0)
    @test all(isfinite.(interior(state.internal_energy)))
    @test all(isfinite.(interior(state.ground_heat_flux)))
end

@testset "Implicit skin temperature: Smoke tests" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    solver = Terrarium.default_skin_temperature_solver(eltype(grid))
    skin_temperature = ImplicitSkinTemperature(eltype(grid); κₛ = 0.5, solver)
    aerodynamics = Terrarium.ConstantAerodynamics(Cₕ = 2.0e-3)
    atmosphere = Terrarium.PrescribedAtmosphere(Float64; aerodynamics = aerodynamics)
    seb = SurfaceEnergyBalance(Float64; skin_temperature)
    model = SurfaceEnergyModel(grid, surface_energy_balance = seb, atmosphere = atmosphere)

    @testset "Sunny and dry" begin
        # Sunny and dry in Bergen Norway (Figure 5.11, Shuttleworth 2012)
        state = StateVariables(model)
        results = test_skin_temperature_solve!(
            state, grid, model;
            surface_shortwave_down = 600.0,
            surface_longwave_down = 300.0,
            air_pressure = 101_325,
            air_temperature = 10.0,
            relative_humidity = 0.75,
            ground_temperature = 13.0,
            windspeed = 5.0,
        )
        @test all(results.sensible_heat_flux .> 0)
        @test all(results.latent_heat_flux .> 0)
        @test all(results.ground_heat_flux .< 0)
        @test all(results.surface_net_radiation .< 0)
    end

    @testset "Sunny and humid" begin
        state = StateVariables(model)
        results = test_skin_temperature_solve!(
            state, grid, model;
            surface_shortwave_down = 600.0,
            surface_longwave_down = 300.0,
            air_pressure = 101_325,
            air_temperature = 10.0,
            relative_humidity = 0.97,
            ground_temperature = 13.0,
            windspeed = 5.0,
        )
        @test all(results.sensible_heat_flux .> 0)
        @test all(results.latent_heat_flux .> 0)
        @test all(results.ground_heat_flux .< 0)
        @test all(results.surface_net_radiation .< 0)
    end

    @testset "Cloudy and dry" begin
        state = StateVariables(model)
        results = test_skin_temperature_solve!(
            state, grid, model;
            surface_shortwave_down = 150.0,
            surface_longwave_down = 200.0,
            air_pressure = 101_325,
            air_temperature = 5.0,
            relative_humidity = 0.6,
            ground_temperature = 10.0,
            windspeed = 10.0
        )
        @test all(results.sensible_heat_flux .< 0)
        @test all(results.latent_heat_flux .> 0)
        @test all(results.ground_heat_flux .> 0)
        @test all(results.surface_net_radiation .> 0)
    end

    @testset "Cloudy and humid" begin
        state = StateVariables(model)
        results = test_skin_temperature_solve!(
            state, grid, model;
            surface_shortwave_down = 150.0,
            surface_longwave_down = 200.0,
            air_pressure = 101_325,
            air_temperature = 5.0,
            relative_humidity = 0.9,
            ground_temperature = 10.0,
            windspeed = 10.0
        )
        @test all(results.sensible_heat_flux .> 0)
        @test all(results.latent_heat_flux .> 0)
        @test all(results.ground_heat_flux .> 0)
        @test all(results.surface_net_radiation .> 0)
    end
end

@testset "Implicit skin temperature: LandModel stress test" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.01))
    NF = eltype(grid)
    # Soil + snow + surface energy/hydrology + prescribed vegetation
    seb = SurfaceEnergyBalance(NF)
    soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
    snow = SingleLayerSnow(NF)
    vegetation = PrescribedVegetation(NF)
    model = LandModel(grid; soil, snow, vegetation, surface_energy_balance = seb)
    integrator = initialize(model)
    state = integrator.state

    @testset "Snow free, bare saturated soil" begin
        # Define a large range of conditions
        SW_range = NF.([50.0, 300, 1000.0])
        LW_range = NF.([100.0, 400.0])
        T_air_range = NF.([-40.0, 0.0, 10.0, 40.0])
        pres_range = NF.([90_000, 101_325, 105_000])
        humid_range = NF.([0.0, 0.5, 1.0])
        wind_range = NF.([0.1, 1.0, 10.0, 50.0])
        for config in Iterators.product(SW_range, LW_range, T_air_range, pres_range, humid_range, wind_range)
            SW, LW, T_air, pres, humid, wind = config
            test_state = deepcopy(state)
            results = test_skin_temperature_solve!(
                test_state, grid, model;
                surface_shortwave_down = SW,
                surface_longwave_down = LW,
                air_pressure = pres,
                air_temperature = T_air,
                relative_humidity = humid,
                ground_temperature = NF(0.95) * T_air - NF(1),
                windspeed = wind
            )
        end
    end

    @testset "Snow free, bare dry soil" begin
        # Define a large range of conditions
        SW_range = NF.([50.0, 300, 1000.0])
        LW_range = NF.([100.0, 400.0])
        T_air_range = NF.([-40.0, 0.0, 10.0, 40.0])
        pres_range = NF.([90_000, 101_325, 105_000])
        humid_range = NF.([0.0, 0.5, 1.0])
        wind_range = NF.([0.1, 1.0, 10.0, 50.0])
        for config in Iterators.product(SW_range, LW_range, T_air_range, pres_range, humid_range, wind_range)
            SW, LW, T_air, pres, humid, wind = config
            test_state = deepcopy(state)
            results = test_skin_temperature_solve!(
                test_state, grid, model;
                surface_shortwave_down = SW,
                surface_longwave_down = LW,
                air_pressure = pres,
                air_temperature = T_air,
                relative_humidity = humid,
                ground_temperature = NF(0.95) * T_air - NF(1),
                windspeed = wind,
                soil_moisture = NF(0)
            )
        end
    end

    @testset "Snow free, vegetated saturated soil" begin
        # Define a large range of conditions
        SW_range = NF.([50.0, 300, 1000.0])
        LW_range = NF.([100.0, 400.0])
        T_air_range = NF.([-40.0, 0.0, 10.0, 40.0])
        pres_range = NF.([90_000, 101_325, 105_000])
        humid_range = NF.([0.0, 0.5, 1.0])
        wind_range = NF.([0.1, 1.0, 10.0, 50.0])
        for config in Iterators.product(SW_range, LW_range, T_air_range, pres_range, humid_range, wind_range)
            SW, LW, T_air, pres, humid, wind = config
            test_state = deepcopy(state)
            results = test_skin_temperature_solve!(
                test_state, grid, model;
                surface_shortwave_down = SW,
                surface_longwave_down = LW,
                air_pressure = pres,
                air_temperature = T_air,
                relative_humidity = humid,
                ground_temperature = NF(0.8) * T_air - NF(1),
                windspeed = wind,
                leaf_area_index = NF(2)
            )
        end
    end

    @testset "Snow cover (partial)" begin
        # Define a large range of conditions
        SW_range = NF.([50.0, 300, 1000.0])
        LW_range = NF.([100.0, 400.0])
        T_air_range = NF.([-40.0, -10.0, 0.0])
        pres_range = NF.([90_000, 101_325, 105_000])
        humid_range = NF.([0.0, 0.5, 1.0])
        wind_range = NF.([0.1, 1.0, 10.0, 50.0])
        for config in Iterators.product(SW_range, LW_range, T_air_range, pres_range, humid_range, wind_range)
            SW, LW, T_air, pres, humid, wind = config
            test_state = deepcopy(state)
            results = test_skin_temperature_solve!(
                test_state, grid, model;
                surface_shortwave_down = SW,
                surface_longwave_down = LW,
                air_pressure = pres,
                air_temperature = T_air,
                relative_humidity = humid,
                ground_temperature = NF(0.9) * T_air - NF(1),
                windspeed = wind,
                snow_water_equivalent = NF(0.001)
            )
        end
    end

    @testset "Snow cover (full)" begin
        # Define a large range of conditions
        SW_range = NF.([50.0, 300, 1000.0])
        LW_range = NF.([100.0, 400.0])
        T_air_range = NF.([-40.0, -10.0, 0.0])
        pres_range = NF.([90_000, 101_325, 105_000])
        humid_range = NF.([0.0, 0.5, 1.0])
        wind_range = NF.([0.1, 1.0, 10.0, 50.0])
        for config in Iterators.product(SW_range, LW_range, T_air_range, pres_range, humid_range, wind_range)
            SW, LW, T_air, pres, humid, wind = config
            test_state = deepcopy(state)
            results = test_skin_temperature_solve!(
                test_state, grid, model;
                surface_shortwave_down = SW,
                surface_longwave_down = LW,
                air_pressure = pres,
                air_temperature = T_air,
                relative_humidity = humid,
                ground_temperature = NF(0.5) * T_air - NF(1),
                windspeed = wind,
                snow_water_equivalent = NF(0.1),
            )
        end
    end
end

# # For interactive testing
# grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.01))
# NF = eltype(grid)
# # Soil + snow + surface energy/hydrology + prescribed vegetation
# skinT = ImplicitSkinTemperature(NF, solver = Terrarium.RootSolver(NF; max_iterations = 100))
# seb = SurfaceEnergyBalance(NF, skin_temperature = skinT)
# soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
# snow = SingleLayerSnow(NF)
# vegetation = PrescribedVegetation(NF)
# model = LandModel(grid; soil, snow, vegetation, surface_energy_balance = seb)
# integrator = initialize(model)
# state = integrator.state
# SW = NF(300)
# LW = NF(400)
# pres = NF(101_325)
# T_air = NF(0.0)
# humid = NF(0.5)
# wind = NF(10.0)
# nf = NF(0.0)
# test_state = deepcopy(state)
# results = test_skin_temperature_solve!(
#     test_state, grid, model;
#     surface_shortwave_down = SW,
#     surface_longwave_down = LW,
#     air_pressure = pres,
#     air_temperature = T_air,
#     relative_humidity = humid,
#     ground_temperature = nf * T_air,
#     windspeed = wind,
#     snow_water_equivalent = NF(1.0)
# )
# @info "$(results.skin_temperature)"
