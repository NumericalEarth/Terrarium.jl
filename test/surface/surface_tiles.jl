using Terrarium
using Test

using Terrarium: InputVariable, Namespace, varname

# Variables of a tuple that are not namespaces, and the namespaces of a tuple, respectively
root_variables(vars::Tuple) = filter(var -> !isa(var, Namespace), vars)
variable_namespaces(vars::Tuple) = filter(var -> isa(var, Namespace), vars)

@testset "Surface tiles: variables(seb, snow)" begin
    NF = Float64
    seb = SurfaceEnergyBalance(NF)
    snow = SingleLayerSnow(NF)

    # Without snow, there are no tiles
    @test Terrarium.variables(seb, nothing) == Terrarium.variables(seb)

    vars = Terrarium.variables(seb, snow)
    namespaces = variable_namespaces(vars)
    @test map(varname, namespaces) == (:snow_free, :snow)
    # The root keeps the grid-cell skin temperature, fluxes, albedo, and emissivity, and adds the melt flux
    root_names = map(varname, root_variables(vars))
    for name in (:skin_temperature, :surface_shortwave_up, :surface_longwave_up, :surface_net_radiation, :sensible_heat_flux, :latent_heat_flux, :ground_heat_flux, :albedo, :emissivity, :snow_melt_energy_flux, :ground_temperature)
        @test name ∈ root_names
    end
    # Each tile owns the auxiliaries of the SEB but no input
    tile_names = map(varname, Terrarium.auxiliary_variables(Terrarium.variables(seb)))
    @test :skin_temperature ∈ tile_names
    @test :ground_temperature ∉ tile_names
    for ns in namespaces
        tile_vars = Terrarium.variables(ns)
        @test !any(var -> isa(var, InputVariable), tile_vars)
        @test map(varname, tile_vars) == tile_names
        @test :snow_melt_energy_flux ∉ map(varname, tile_vars)
    end

    # The prescribed SEB is owned by the coupler and is not tiled
    prescribed_seb = PrescribedSurfaceEnergyBalance(NF)
    @test Terrarium.variables(prescribed_seb, snow) == Terrarium.variables(prescribed_seb)
    @test isempty(variable_namespaces(Terrarium.variables(prescribed_seb, snow)))
end

@testset "Surface tiles: LandModel state" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_max = 1.0, N = 20))
    soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
    initializers = (
        temperature = (x, z) -> 1.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> 0.5,
        snow_water_equivalent = 0.05,
        snow_temperature = -3.0,
    )

    @testset "with snow" begin
        model = LandModel(grid; soil, snow = SingleLayerSnow(NF), vegetation = nothing)
        integrator = initialize(model; initializers)
        state = integrator.state
        @test :snow ∈ Terrarium.namespace_names(state)
        @test :snow_free ∈ Terrarium.namespace_names(state)
        @test all(iszero, interior(state.snow_melt_energy_flux))
        for tile in (state.snow, state.snow_free)
            @test hasproperty(tile, :skin_temperature)
            @test hasproperty(tile, :albedo)
            @test hasproperty(tile, :sensible_heat_flux)
            @test !hasproperty(tile, :ground_temperature)
            @test !hasproperty(tile, :snow_melt_energy_flux)
            # namespaced fields are distinct from the root fields
            @test tile.skin_temperature !== state.skin_temperature
        end
        # A tile namespace is itself a state from which the SEB auxiliaries can be retrieved
        tile_aux = Terrarium.auxiliary_fields(state.snow, model.surface_energy_balance)
        @test tile_aux.albedo === state.snow.albedo
        # Tile fields shadow the root fields of the same name and share the root inputs
        @test nameof(SnowCovered()) == :snow
        @test nameof(SnowFree()) == :snow_free
        tile_out = Terrarium.auxiliary_fields(state, SnowCovered(), model.surface_energy_balance)
        @test tile_out.skin_temperature === state.snow.skin_temperature
        @test tile_out.sensible_heat_flux === state.snow.sensible_heat_flux
        fields = get_fields(state, SnowCovered(), model.surface_energy_balance, model.atmosphere)
        @test fields.albedo === state.snow.albedo
        @test fields.skin_temperature === state.snow.skin_temperature
        @test fields.ground_temperature === state.ground_temperature
        @test fields.air_temperature === state.air_temperature
        # Time stepping is unaffected by the tile namespaces
        timestep!(integrator, 600.0)
        @test all(isfinite, interior(state.skin_temperature))
    end

    @testset "without snow" begin
        model = LandModel(grid; soil, snow = nothing, vegetation = nothing)
        vars = Terrarium.variables(model)
        @test !any(var -> isa(var, Namespace) && varname(var) ∈ (:snow, :snow_free), vars)
        @test :snow_melt_energy_flux ∉ map(varname, vars)
    end

    @testset "prescribed SEB with snow" begin
        model = LandModel(grid; soil, snow = SingleLayerSnow(NF), vegetation = nothing, surface_energy_balance = PrescribedSurfaceEnergyBalance(NF))
        vars = Terrarium.variables(model)
        @test !any(var -> isa(var, Namespace) && varname(var) ∈ (:snow, :snow_free), vars)
    end
end

@testset "Surface tiles: snow-free and snow-covered kernel functions" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_max = 1.0, N = 20))
    soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
    snow = SingleLayerSnow(NF)
    vegetation = PrescribedVegetation(NF)
    model = LandModel(grid; soil, snow, vegetation)
    initializers = (
        temperature = (x, z) -> 1.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> 0.6,
        snow_water_equivalent = 0.01,
        snow_temperature = -3.0,
    )
    integrator = initialize(model; initializers)
    state = integrator.state
    set!(state.leaf_area_index, 2.0)
    set!(state.surface_shortwave_down, 300.0)
    set!(state.surface_longwave_down, 280.0)
    set!(state.air_temperature, 2.0)
    timestep!(integrator, 600.0)
    # solve the SEB for the current state so that the stored fluxes and skin temperature are consistent
    compute_auxiliary!(state, model)
    compute_boundary_conditions!(state, model)

    seb = model.surface_energy_balance
    skinT = seb.skin_temperature
    tur = seb.turbulent_fluxes
    hydrology = model.surface_hydrology
    atmos = model.atmosphere
    constants = model.constants
    f = state.snow_cover_fraction[1, 1, 1]
    # partial snow cover, so that both tiles contribute to the blends
    @test 0 < f < 1

    @testset "skin temperature inverses" begin
        G₀ = state.surface_net_radiation[1, 1] + state.sensible_heat_flux[1, 1] + state.latent_heat_flux[1, 1]
        Tg, κg, Δzg = Terrarium.ground_thermal_interface(1, 1, grid, state, skinT)
        Tsnow, κsnow, dsnow = Terrarium.snow_thermal_interface(1, 1, grid, state, snow, constants)
        @test Tsnow ≈ state.snow_temperature[1, 1, 1]
        Ts_sf = Terrarium.compute_skin_temperature(1, 1, grid, state, SnowFree(), skinT, seb, constants, snow)
        Ts_sn = Terrarium.compute_skin_temperature(1, 1, grid, state, SnowCovered(), skinT, seb, constants, snow)
        # the snow-free tile ignores the snow component
        @test Terrarium.compute_skin_temperature(1, 1, grid, state, SnowFree(), skinT, seb, constants, nothing) == Ts_sf
        # snow-free tile: invert the ground conduction
        @test Ts_sf ≈ Tg - G₀ * Δzg / (2κg)
        # snow tile: invert the snow conduction alone, independently of the snow cover fraction
        @test Ts_sn ≈ Tsnow - G₀ * dsnow / (2κsnow)
        @test 2κsnow * (Tsnow - Ts_sn) / dsnow ≈ G₀
        # the blended inverse reproduces the area-weighted conduction relation of the single solve
        A = (1 - f) * 2κg / Δzg + f * 2κsnow / dsnow
        B = (1 - f) * 2κg / Δzg * Tg + f * 2κsnow / dsnow * Tsnow
        Ts_blend = Terrarium.compute_skin_temperature(1, 1, grid, state, skinT, seb, constants, snow)
        @test Ts_blend ≈ (B - G₀) / A
        @test Terrarium.compute_skin_temperature(1, 1, grid, state, skinT, seb, constants, nothing) == Ts_sf
        # the solved grid-cell skin temperature is the root of the blended relation
        @test state.skin_temperature[1, 1] ≈ Ts_blend atol = 1.0e-6
    end

    @testset "latent heat fluxes" begin
        ρₐ = Terrarium.air_density(1, 1, grid, state, atmos, constants)
        L_lv = constants.thermodynamics.latent_heat_vaporization
        L_sg = constants.thermodynamics.latent_heat_sublimation
        ρ_w = constants.material.density_water
        evtr = Terrarium.get_evapotranspiration(hydrology)
        Q_h = Terrarium.compute_surface_humidity_flux(1, 1, grid, state, evtr, constants, atmos)
        LE_sf = Terrarium.compute_latent_heat_flux(1, 1, grid, state, SnowFree(), tur, skinT, constants, atmos, hydrology, snow)
        LE_sn = Terrarium.compute_latent_heat_flux(1, 1, grid, state, SnowCovered(), tur, skinT, constants, atmos, hydrology, snow)
        # snow-free tile: evapotranspiration latent flux, unweighted
        @test LE_sf ≈ L_lv * ρₐ * Q_h
        # snow tile: sublimation latent flux, unweighted
        E_subl = Terrarium.compute_snow_sublimation_flux(1, 1, grid, state, snow, atmos, constants, skinT)
        @test LE_sn ≈ ρ_w * L_sg * E_subl
        # the grid-cell latent heat flux is their area-weighted blend
        @test Terrarium.compute_latent_heat_flux(1, 1, grid, state, tur, skinT, constants, atmos, hydrology, snow) ≈ (1 - f) * LE_sf + f * LE_sn
        @test Terrarium.compute_latent_heat_flux(1, 1, grid, state, tur, skinT, constants, atmos, hydrology, nothing) == LE_sf
        @test state.latent_heat_flux[1, 1] ≈ (1 - f) * LE_sf + f * LE_sn
        # without surface hydrology, the snow-free latent flux is the saturated bare-ground flux
        Tₛ = state.skin_temperature[1, 1]
        rₐ = Terrarium.aerodynamic_resistance(1, 1, grid, state, atmos)
        Δq = Terrarium.compute_specific_humidity_difference(1, 1, grid, state, atmos, constants, Tₛ)
        @test Terrarium.compute_latent_heat_flux(1, 1, grid, state, tur, skinT, constants, atmos) ≈ L_lv * ρₐ * Δq / rₐ
    end

    @testset "albedo and emissivity" begin
        albedo = seb.albedo
        α_sn, ϵ_sn = Terrarium.compute_albedo(1, 1, grid, state, SnowCovered(), albedo, vegetation, snow, soil)
        α_sf, ϵ_sf = Terrarium.compute_albedo(1, 1, grid, state, SnowFree(), albedo, vegetation, snow, soil)
        # snow tile: the snow albedo and emissivity alone
        @test α_sn == snow.albedo.snow_albedo
        @test ϵ_sn == snow.albedo.snow_emissivity
        # snow-free tile: ground and vegetation blend
        f_veg = Terrarium.vegetation_area_fraction(1, 1, grid, state, vegetation)
        @test 0 < f_veg < 1
        α_bg = Terrarium.compute_ground_albedo(1, 1, grid, state, albedo, soil)
        α_veg = Terrarium.compute_albedo(1, 1, grid, state, vegetation)
        ϵ_veg = Terrarium.compute_emissivity(1, 1, grid, state, vegetation)
        @test α_sf ≈ (1 - f_veg) * α_bg + f_veg * α_veg
        @test ϵ_sf ≈ (1 - f_veg) * albedo.ground_emissivity + f_veg * ϵ_veg
        # the grid-cell values are the area-weighted blend, and the root fields keep them
        α_cell, ϵ_cell = Terrarium.compute_albedo(1, 1, grid, state, albedo, vegetation, snow, soil)
        @test α_cell ≈ (1 - f) * α_sf + f * α_sn
        @test ϵ_cell ≈ (1 - f) * ϵ_sf + f * ϵ_sn
        @test Terrarium.compute_albedo(1, 1, grid, state, albedo, vegetation, nothing, soil) == (α_sf, ϵ_sf)
        compute_auxiliary!(state, grid, albedo, vegetation, snow, soil)
        @test state.albedo[1, 1] ≈ (1 - f) * α_sf + f * α_sn
        @test state.emissivity[1, 1] ≈ (1 - f) * ϵ_sf + f * ϵ_sn
        @test state.albedo[1, 1] ≈ (1 - f) * (1 - f_veg) * α_bg + (1 - f) * f_veg * α_veg + f * α_sn
    end
end
