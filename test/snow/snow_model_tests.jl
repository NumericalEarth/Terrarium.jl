using Terrarium
using Test

@testset "Snow model (standalone)" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 3))
    model = SnowModel(grid)
    snow = model.snow
    constants = model.constants

    ρ_snow = Terrarium.snow_density(snow)
    ρ_w = constants.material.density_water
    L_f = constants.thermodynamics.latent_heat_fusion
    c_i = constants.thermodynamics.specific_heat_capacity_ice
    W0 = NF(0.1)
    d_snow = W0 * ρ_w / ρ_snow

    # Build a fresh state with all forcing zeroed; caller sets the prognostic/forcing of interest.
    function fresh_state()
        state = StateVariables(model)
        for f in (:snowfall, :rainfall, :air_temperature, :surface_heat_flux, :basal_heat_flux, :sublimation)
            set!(getproperty(state, f), 0)
        end
        return state
    end

    # Run the auxiliary + closure passes then the tendencies (as a timestep would).
    function step_tendencies!(state)
        compute_auxiliary!(state, model)
        Terrarium.closure!(state, model)
        compute_tendencies!(state, model)
        return state
    end

    @testset "frozen, no forcing: zero tendencies" begin
        state = fresh_state()
        set!(state.snow_water_equivalent, W0)
        set!(state.snow_temperature, NF(-5))
        Terrarium.initialize!(state, model.grid, snow, constants)   # invclosure: T, W -> E
        step_tendencies!(state)
        @test all(state.snow_liquid_fraction .≈ 0)                  # frozen
        @test all(state.tendencies.snow_water_equivalent .≈ 0)      # no melt, no precip/sublimation
        @test all(state.tendencies.snow_energy .≈ 0)                # no fluxes
    end

    @testset "snowfall accumulates SWE and advects cold" begin
        state = fresh_state()
        set!(state.snow_water_equivalent, W0)
        set!(state.snow_temperature, NF(-5))
        Terrarium.initialize!(state, model.grid, snow, constants)
        P_s = NF(1.0e-6)
        set!(state.snowfall, P_s)
        set!(state.air_temperature, NF(-2))
        step_tendencies!(state)
        @test all(state.tendencies.snow_water_equivalent .≈ P_s)    # dW/dt = snowfall
        # fresh snow is ice: relative to liquid water at 0°C (U = 0) it advects the fusion deficit −L_f
        # plus sensible heat c_i·T_air for T_air < 0
        @test all(state.tendencies.snow_energy .≈ ρ_w * P_s * (c_i * NF(-2) - L_f))
    end

    @testset "snowfall onto bare ground (W=0) advects cold, not zero" begin
        # Regression test: the conductive-term gate in `compute_snow_energy_tendency` must not also
        # gate `Q_prcp`, or the very first step of accumulation onto bare ground (W read as 0 before
        # this step's mass increment) gets zero recorded energy input despite the snow being cold.
        state = fresh_state()
        set!(state.snow_water_equivalent, zero(NF))
        set!(state.snow_temperature, NF(-8))
        Terrarium.initialize!(state, model.grid, snow, constants)
        P_s = NF(1.0e-6)
        set!(state.snowfall, P_s)
        set!(state.air_temperature, NF(-8))
        step_tendencies!(state)
        @test all(state.tendencies.snow_water_equivalent .≈ P_s)
        # Same advected-cold formula as the W0 > 0 case above: gating must not silently zero this out.
        @test all(state.tendencies.snow_energy .≈ ρ_w * P_s * (c_i * NF(-8) - L_f))
        @test all(state.tendencies.snow_energy .< 0)  # cold snow carries negative enthalpy in, not zero
    end

    @testset "fully melted pack drains (SWE + energy decrease)" begin
        state = fresh_state()
        set!(state.snow_water_equivalent, W0)
        set!(state.snow_energy, NF(1000))                            # E > 0 -> θ_liq = 1 (fully melted)
        step_tendencies!(state)
        @test all(state.snow_liquid_fraction .≈ 1)
        M_r = snow.hydraulic_properties.saturated_conductivity      # S* = 1 at θ_liq = 1
        @test all(state.tendencies.snow_water_equivalent .≈ -M_r)   # meltwater drains
        # meltwater is liquid at 0°C (U = 0 reference), so it carries no enthalpy: energy is conserved
        @test all(isapprox.(interior(state.tendencies.snow_energy), 0; atol = 1.0e-9))
    end

    @testset "no drainage below capillary retention" begin
        state = fresh_state()
        set!(state.snow_water_equivalent, W0)
        Lθ = ρ_snow * L_f
        # choose θ_liq just below L_c via a phase-change energy: θ_liq = 1 + U_v/Lθ
        θ_target = snow.hydraulic_properties.capillary_retention / 2
        set!(state.snow_energy, (θ_target - 1) * Lθ * d_snow)
        step_tendencies!(state)
        @test all(0 .< state.snow_liquid_fraction .< snow.hydraulic_properties.capillary_retention)
        @test all(state.tendencies.snow_water_equivalent .≈ 0)      # retained, no outflow
    end

    @testset "basal flux warms, surface flux cools (positive upward)" begin
        state = fresh_state()
        set!(state.snow_water_equivalent, W0)
        set!(state.snow_temperature, NF(-5))                        # frozen -> no melt
        Terrarium.initialize!(state, model.grid, snow, constants)
        set!(state.basal_heat_flux, NF(10))                         # soil -> snow: gain
        set!(state.surface_heat_flux, NF(30))                       # snow -> sky: loss
        step_tendencies!(state)
        @test all(state.tendencies.snow_energy .≈ NF(10) - NF(30))  # dE/dt = Q_base - Q_top
        @test all(state.tendencies.snow_water_equivalent .≈ 0)
    end

    @testset "sublimation removes SWE" begin
        state = fresh_state()
        set!(state.snow_water_equivalent, W0)
        set!(state.snow_temperature, NF(-5))
        Terrarium.initialize!(state, model.grid, snow, constants)
        E_subl = NF(2.0e-7)
        set!(state.sublimation, E_subl)
        step_tendencies!(state)
        @test all(state.tendencies.snow_water_equivalent .≈ -E_subl)
    end

    @testset "conservation: draining meltwater carries no enthalpy" begin
        # A partially melted pack (θ_liq above capillary retention, no other forcing) drains meltwater.
        # Meltwater is liquid water at 0 °C, which is the zero-enthalpy reference (U = 0) of the FreeWater
        # closure, so draining it removes mass but no energy: dW < 0 while dE = 0. (An ice-referenced
        # meltwater flux would instead give dE = ρ_w·L_f·dW, spuriously refreezing the snowpack.)
        state = fresh_state()
        set!(state.snow_water_equivalent, W0)
        Lθ = ρ_snow * L_f
        θ_target = NF(0.5)  # > default capillary retention (0.05) -> outflow
        set!(state.snow_energy, (θ_target - 1) * Lθ * d_snow)  # U_v = (θ-1)·Lθ -> θ_liq = θ_target
        step_tendencies!(state)
        dW = Array(interior(state.tendencies.snow_water_equivalent))
        dE = Array(interior(state.tendencies.snow_energy))
        @test all(dW .< 0)          # meltwater drains from the snowpack
        @test all(isapprox.(dE, 0; atol = 1.0e-9))   # but carries no enthalpy: energy is conserved
    end

    @testset "conservation: snowfall accretes at the fresh-snow enthalpy" begin
        # Fresh snow falling at air temperature `T_air` onto a pack at `T_air` must accrete without
        # changing the intensive state (temperature). Equivalently, the accreting snow carries the snowpack's
        # specific enthalpy, so the energy tendency per unit mass equals the snowpack's E/W: dE/dW = E/W.
        # (With an ice-referenced precip flux, fresh snow would carry ρ_w·c_i·T_air, missing the −ρ_w·L_f
        # ice deficit, and the snowpack would spuriously warm toward 0 °C.)
        state = fresh_state()
        T_air = NF(-8)
        set!(state.snow_water_equivalent, W0)
        set!(state.snow_temperature, T_air)
        Terrarium.initialize!(state, model.grid, snow, constants)   # invclosure: (T_air, W0) -> E
        E0 = Array(interior(state.snow_energy))
        set!(state.snowfall, NF(1.0e-6))
        set!(state.air_temperature, T_air)
        step_tendencies!(state)
        dW = Array(interior(state.tendencies.snow_water_equivalent))
        dE = Array(interior(state.tendencies.snow_energy))
        @test all(dW .> 0)                       # pack accumulates
        @test all(dE ./ dW .≈ E0 ./ W0)          # accreting snow carries the snowpack's specific enthalpy
    end

    @testset "conservation: sublimation nets the vaporization enthalpy" begin
        # The surface energy balance folds the full sublimation enthalpy ρ_w·L_s·E_subl into Q_top
        # (surface_heat_flux), but the departing mass leaves the snowpack as ice, whose enthalpy relative to
        # liquid water at 0 °C is −L_f. The advective correction +ρ_w·L_f·E_subl therefore leaves a net
        # pack loss of exactly ρ_w·L_v·E_subl, the vaporization enthalpy carried by the departing vapor.
        # (Without the correction the snowpack would lose the full L_s and spuriously overcool.)
        state = fresh_state()
        set!(state.snow_water_equivalent, W0)
        set!(state.snow_temperature, NF(-5))
        Terrarium.initialize!(state, model.grid, snow, constants)
        L_s = constants.thermodynamics.latent_heat_sublimation
        L_v = constants.thermodynamics.latent_heat_vaporization
        E_subl = NF(2.0e-7)
        set!(state.sublimation, E_subl)
        set!(state.surface_heat_flux, ρ_w * L_s * E_subl)   # SEB latent contribution to Q_top
        step_tendencies!(state)
        dE = Array(interior(state.tendencies.snow_energy))
        @test all(dE .≈ -ρ_w * L_v * E_subl)                # net loss = vaporization enthalpy only
    end
end
