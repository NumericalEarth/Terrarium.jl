using Terrarium
using Test

@testset "Snow energy closure" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 3))
    snow = SingleLayerSnow(NF)
    constants = PhysicalConstants(NF)
    closure = Terrarium.get_closure(snow)
    state = StateVariables(snow, grid)

    ρ_snow = Terrarium.snow_density(snow)
    ρ_w = constants.material.density_water
    L_f = constants.thermodynamics.latent_heat_fusion
    c_i = constants.thermodynamics.specific_heat_capacity_ice
    c_w = constants.thermodynamics.specific_heat_capacity_liquid_water
    W = NF(0.1)
    d_snow = W * ρ_w / ρ_snow
    Lθ = ρ_snow * L_f

    @testset "recovery: frozen (T < 0)" begin
        set!(state.snow_water_equivalent, W)
        T0 = NF(-5)
        U_v = T0 * ρ_snow * c_i - Lθ          # frozen: θ_liq = 0
        set!(state.snow_energy, U_v * d_snow)
        Terrarium.closure!(state, grid, closure, snow, constants)
        @test all(state.snow_temperature .≈ T0)
        @test all(state.snow_liquid_fraction .≈ 0)
    end

    @testset "phase change: T = 0, partial liquid" begin
        set!(state.snow_water_equivalent, W)
        set!(state.snow_energy, (-Lθ / 2) * d_snow)   # half-melted
        Terrarium.closure!(state, grid, closure, snow, constants)
        @test all(state.snow_temperature .≈ 0)
        @test all(state.snow_liquid_fraction .≈ 0.5)
    end

    @testset "temperature clipped at 0 for E > 0" begin
        set!(state.snow_water_equivalent, W)
        # E > 0: pack fully melted at 0°C plus a sensible excess (would give T = 2°C if unclipped)
        set!(state.snow_energy, NF(2) * ρ_snow * c_w * d_snow)
        Terrarium.closure!(state, grid, closure, snow, constants)
        @test all(state.snow_temperature .≈ 0)     # clipped at 0
        @test all(state.snow_liquid_fraction .≈ 1)
    end

    @testset "round-trip invclosure -> closure (T ≤ 0)" begin
        for T0 in (NF(-8), NF(-0.5), NF(0))
            set!(state.snow_water_equivalent, NF(0.05))
            set!(state.snow_temperature, T0)
            Terrarium.invclosure!(state, grid, closure, snow, constants)
            # scribble over the temperature to ensure the forward closure recomputes it
            set!(state.snow_temperature, NF(-999))
            Terrarium.closure!(state, grid, closure, snow, constants)
            @test all(state.snow_temperature .≈ T0)
        end
    end

    @testset "W -> 0 stays finite (no Inf/NaN)" begin
        set!(state.snow_water_equivalent, NF(0))
        set!(state.snow_energy, NF(0))
        Terrarium.closure!(state, grid, closure, snow, constants)
        @test all(isfinite.(state.snow_temperature))
        @test all(isfinite.(state.snow_liquid_fraction))
    end

    @testset "thin snow with residual energy: temperature stays bounded" begin
        # Regression for the unphysically cold snow temperature: the explicit dynamics can leave a
        # vanishing pack (tiny W) holding a non-negligible depth-integrated energy Ē. Recovering the
        # volumetric energy U = Ē/d_snow with only an eps offset then yields an enormous |U| and a
        # temperature thousands of degrees below zero. Flooring the depth at `min_conduction_thickness`
        # bounds the effective heat capacity so the recovered temperature stays physical.
        set!(state.snow_water_equivalent, NF(1.0e-6))   # d_snow ≈ 3e-6 m, far below the 5 mm floor
        set!(state.snow_energy, NF(-1.0e4))             # residual energy inconsistent with the tiny pack
        Terrarium.closure!(state, grid, closure, snow, constants)
        @test all(isfinite.(state.snow_temperature))
        # bounded by ≈ Ē/(C_snow·d_min); without the floor this would be several thousand °C below zero
        @test all(state.snow_temperature .>= -100)
        @test all(state.snow_temperature .<= 0)
    end

    @testset "initialize! sets snow_energy from prescribed temperature" begin
        set!(state.snow_water_equivalent, W)
        set!(state.snow_temperature, NF(-3))
        Terrarium.initialize!(state, grid, snow, constants)
        @test all(isfinite.(state.snow_energy))
        # frozen initial state: energy should be negative
        @test all(state.snow_energy .< 0)
    end

    @testset "basal heat flux: soil series resistance and thin-snow depth flooring" begin
        # Q_base = (T_soil − T_snow) / (Δz_soil/(2κ_soil) + max(d_snow, d_min)/(2κ_snow)): the snow-side
        # conduction thickness is floored at d_min (half the uppermost soil cell thickness) so the flux
        # stays bounded as the snowpack vanishes, and the soil's own half-cell resistance is included in
        # series rather than assumed negligible.
        landgrid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 5))
        soil = SoilEnergyWaterCarbon(NF)
        land = LandModel(landgrid; soil, snow, vegetation = nothing)

        function basal_flux_state(W_snow)
            initializers = (
                temperature = (x, z) -> NF(-10) - NF(0.01) * z,
                saturation_water_ice = (x, z) -> NF(0.8),
                snow_water_equivalent = W_snow,
                snow_temperature = NF(0),
            )
            integrator = initialize(land; initializers)
            Terrarium.closure!(integrator.state, land)
            compute_auxiliary!(integrator.state, land)
            return integrator.state
        end

        function expected_flux(state, W_snow)
            field_grid = Terrarium.get_field_grid(landgrid)
            k = field_grid.Nz
            strat = Terrarium.get_stratigraphy(soil)
            hydrology = Terrarium.get_hydrology(soil)
            bgc = Terrarium.get_biogeochemistry(soil)
            energy = Terrarium.get_energy_balance(soil)
            composition = Terrarium.soil_composition(1, 1, k, landgrid, state, strat, hydrology, bgc)
            κ_soil = Terrarium.compute_thermal_conductivity(energy.thermal_properties, composition)
            Δz_soil = Terrarium.Δzᵃᵃᶜ(1, 1, k, field_grid)
            d_min = Terrarium.min_snow_conduction_thickness(1, 1, landgrid, state, snow)
            ρ_snow = Terrarium.compute_snow_density(1, 1, landgrid, state, snow.density)
            κ_snow = Terrarium.compute_thermal_conductivity(snow, constants.material, ρ_snow)
            d_snow = max(W_snow * ρ_w / ρ_snow, d_min)
            T_soil = only(Array(interior(state.ground_temperature)))
            T_snow = only(Array(interior(state.snow_temperature)))
            R_soil = Δz_soil / (2κ_soil)
            R_snow = d_snow / (2κ_snow)
            return (T_soil - T_snow) / (R_soil + R_snow)
        end

        # thick snow: still finite and matches the series-resistance formula
        state_thick = basal_flux_state(NF(0.5))
        Q_thick = Terrarium.compute_snow_basal_heat_flux(1, 1, landgrid, state_thick, snow, soil, constants)
        @test isfinite(Q_thick)
        @test Q_thick ≈ expected_flux(state_thick, NF(0.5))

        # vanishing snow: flux stays finite (bounded by the floor) and still matches the formula
        for W_thin in (NF(0), NF(1.0e-9), NF(1.0e-6))
            state_thin = basal_flux_state(W_thin)
            Q_thin = Terrarium.compute_snow_basal_heat_flux(1, 1, landgrid, state_thin, snow, soil, constants)
            @test isfinite(Q_thin)
            @test Q_thin ≈ expected_flux(state_thin, W_thin)
        end
    end
end
