using Terrarium
using Terrarium:
    humidity_flux,
    transpiration_conductance,
    canopy_evaporation_conductance,
    ground_evaporation_conductance,
    ground_evaporation_resistance_factor
using Test

# ==============================================================================
# Constructors
# ==============================================================================

@testset "BareGroundEvaporation constructor" begin
    # Default construction
    bg = BareGroundEvaporation(Float64)
    @test bg.ground_resistance isa SoilMoistureResistanceFactor

    # Custom ground resistance
    crf = ConstantEvaporationResistanceFactor(Float64)
    bg2 = BareGroundEvaporation(Float64, ground_resistance = crf)
    @test bg2.ground_resistance === crf
end

# ==============================================================================
# humidity_flux
#
# All evapotranspiration components share the functional form Qh = Δq · g, so a
# single set of tests covers ground/canopy evaporation and transpiration for
# both schemes.
# ==============================================================================

@testset "humidity_flux" begin
    @testset "$(nameof(typeof(evtr)))" for evtr in (
            BareGroundEvaporation(Float64),
            PALADYNCanopyEvapotranspiration(Float64),
        )
        g = 0.02  # m/s (typical conductance)

        # Flux is zero when there is no humidity difference
        @test iszero(humidity_flux(evtr, 0.0, g))

        # Flux is positive and equals Δq · g for positive Δq
        Δq = 0.001
        Qh = humidity_flux(evtr, Δq, g)
        @test Qh > 0
        @test Qh ≈ Δq * g

        # Flux scales linearly with Δq
        @test humidity_flux(evtr, 2Δq, g) ≈ 2Qh

        # Flux scales linearly with conductance
        @test humidity_flux(evtr, Δq, g / 2) ≈ Qh / 2

        # Negative Δq gives negative flux (condensation/deposition)
        @test humidity_flux(evtr, -Δq, g) ≈ -Qh

        # Numerical stability at extreme values
        @test isfinite(humidity_flux(evtr, Δq, 1.0e-12))
        @test isfinite(humidity_flux(evtr, Δq, 1.0e5))
        @test isfinite(humidity_flux(evtr, 1.0e-15, 1.0e5))
        @test isfinite(humidity_flux(evtr, 1.0, 1.0e5))

        # Type stability (Float32)
        Qh_f32 = @inferred humidity_flux(evtr, Float32(Δq), Float32(g))
        @test typeof(Qh_f32) == Float32
    end
end

# ==============================================================================
# Conductance functions
# ==============================================================================

@testset "transpiration_conductance" begin
    canopy_ET = PALADYNCanopyEvapotranspiration(Float64)

    # g_trp = 1 / (rₐ + rₐ_can) where rₐ_can = 1/g_stm
    g_stm = 0.1
    rₐ = 100.0
    rₛ = 1 / g_stm
    g_trp = transpiration_conductance(canopy_ET, rₐ, g_stm)
    @test g_trp ≈ 1 / (rₐ + rₛ)

    # Zero stomatal conductance results in zero transpiration conductance
    g_trp_zero = transpiration_conductance(canopy_ET, rₐ, 0.0)
    @test isfinite(g_trp_zero)
    @test iszero(g_trp_zero)

    # Very large stomatal conductance approaches 1/rₐ
    @test transpiration_conductance(canopy_ET, rₐ, 1.0e6) ≈ 1 / rₐ

    # Decreases with increasing aerodynamic resistance
    @test transpiration_conductance(canopy_ET, 100.0, g_stm) <
        transpiration_conductance(canopy_ET, 50.0, g_stm)

    # Increases with increasing stomatal conductance
    @test transpiration_conductance(canopy_ET, rₐ, 0.5) >
        transpiration_conductance(canopy_ET, rₐ, 0.01)

    # Type stability (Float32)
    canopy_ET_f32 = PALADYNCanopyEvapotranspiration(Float32)
    g_f32 = transpiration_conductance(canopy_ET_f32, Float32(100.0), Float32(0.1))
    @test typeof(g_f32) == Float32
end

@testset "canopy_evaporation_conductance" begin
    canopy_ET = PALADYNCanopyEvapotranspiration(Float64)

    # g_can = f_can / rₐ
    rₐ = 50.0
    @test canopy_evaporation_conductance(canopy_ET, 1.0, rₐ) ≈ 1.0 / rₐ

    # Zero canopy saturation gives zero conductance
    @test iszero(canopy_evaporation_conductance(canopy_ET, 0.0, rₐ))

    # Increases linearly with f_can
    f_can = 0.2
    rₐ = 50.0
    @test 2 * canopy_evaporation_conductance(canopy_ET, f_can, rₐ) ≈
        canopy_evaporation_conductance(canopy_ET, 2 * f_can, rₐ)

    # Decreases with increasing rₐ
    @test canopy_evaporation_conductance(canopy_ET, 1.0, 100.0) <
        canopy_evaporation_conductance(canopy_ET, 1.0, 50.0)

    # Type stability (Float32)
    canopy_ET_f32 = PALADYNCanopyEvapotranspiration(Float32)
    g_f32 = canopy_evaporation_conductance(canopy_ET_f32, Float32(1.0), Float32(50.0))
    @test typeof(g_f32) == Float32
end

@testset "ground_evaporation_resistance_factor" begin
    # Constant resistance factor (returns its factor regardless of state)
    crf = ConstantEvaporationResistanceFactor(Float64)
    @test ground_evaporation_resistance_factor(nothing, nothing, nothing, nothing, crf) == 1.0

    crf_half = ConstantEvaporationResistanceFactor(0.5)
    @test ground_evaporation_resistance_factor(nothing, nothing, nothing, nothing, crf_half) == 0.5

    # Soil moisture resistance factor (direct call with θw, θfc, θres)
    smrf = SoilMoistureResistanceFactor(Float64)
    θfc = 0.4   # field capacity
    θres = 0.15 # residual water content
    β_moist = ground_evaporation_resistance_factor(smrf, 0.3, θfc, θres)
    @test 0 < β_moist ≤ 1.0

    # β increases as soil wets toward field capacity
    @test ground_evaporation_resistance_factor(smrf, 0.38, θfc, θres) > β_moist

    # Saturated soil gives β = 1
    @test ground_evaporation_resistance_factor(smrf, 0.5, θfc, θres) == 1.0

    # Residual water content gives β = 0
    @test ground_evaporation_resistance_factor(smrf, θres, θfc, θres) == 0.0

    # Below residual water content (e.g. frozen top layer), β stays zero instead of growing again
    @test ground_evaporation_resistance_factor(smrf, 0.0, θfc, θres) == 0.0
    @test ground_evaporation_resistance_factor(smrf, θres / 2, θfc, θres) == 0.0

    # β is monotonically nondecreasing in θw over the whole range
    βs = [ground_evaporation_resistance_factor(smrf, θw, θfc, θres) for θw in 0.0:0.01:0.6]
    @test all(diff(βs) .>= 0)
    @test all(0 .<= βs .<= 1)

    # Degenerate case θfc = θres: β switches from zero to one at θres
    @test ground_evaporation_resistance_factor(smrf, 0.1, θres, θres) == 0.0
    @test ground_evaporation_resistance_factor(smrf, 0.2, θres, θres) == 1.0

    # Float32 type stability
    β32 = @inferred ground_evaporation_resistance_factor(SoilMoistureResistanceFactor(Float32), 0.3f0, 0.4f0, 0.15f0)
    @test β32 isa Float32
end

# ==============================================================================
# Integration tests
# ==============================================================================

@testset "Canopy ET flux" begin
    canopy_ET = PALADYNCanopyEvapotranspiration(Float64)

    rₐ = 100.0
    rₐ_can = 200.0
    g_stm = 0.1
    f_can = 0.8
    β = 0.7
    Δq_skin = 0.01    # humidity difference at skin temperature
    Δq_ground = 0.005 # humidity difference at ground temperature

    # Step 1: conductances
    g_trp = transpiration_conductance(canopy_ET, rₐ, g_stm)
    g_can = canopy_evaporation_conductance(canopy_ET, f_can, rₐ)
    g_gnd = ground_evaporation_conductance(canopy_ET, β, rₐ, rₐ_can)

    # Step 2: partitioned fluxes (kinematic humidity fluxes)
    Qh_trp = humidity_flux(canopy_ET, Δq_skin, g_trp)
    Qh_can = humidity_flux(canopy_ET, Δq_skin, g_can)
    Qh_gnd = humidity_flux(canopy_ET, Δq_ground, g_gnd)

    # All components are positive (evaporation)
    @test Qh_trp > 0
    @test Qh_can > 0
    @test Qh_gnd > 0

    # Check that evaporation from the canopy is higher than from the ground
    @test Qh_can > Qh_gnd

    # Dry soil (β = 0) suppresses ground evaporation
    @test iszero(ground_evaporation_conductance(canopy_ET, zero(β), rₐ, rₐ_can))
    @test iszero(humidity_flux(canopy_ET, Δq_ground, 0.0))

    # Saturated canopy evaporates more than partially wet canopy
    g_can_sat = canopy_evaporation_conductance(canopy_ET, 1.0, rₐ)
    @test humidity_flux(canopy_ET, Δq_skin, g_can_sat) > Qh_can
end

@testset "Bare ground ET flux" begin
    bg = BareGroundEvaporation(Float64)

    rₐ = 50.0
    Δq = 0.001

    # Test humidity_flux for bare ground
    β = 0.5
    g = β / rₐ
    @test humidity_flux(bg, Δq, g) ≈ g * Δq
end

# ==============================================================================
# Physical consistency
# ==============================================================================

@testset "Physical consistency" begin
    canopy_ET = PALADYNCanopyEvapotranspiration(Float64)
    Δq = 0.01

    # ET increases with wind speed (lower rₐ → higher conductance → higher flux)
    g_trp_low_wind = transpiration_conductance(canopy_ET, 200.0, 0.1)
    g_trp_high_wind = transpiration_conductance(canopy_ET, 50.0, 0.1)
    @test g_trp_high_wind > g_trp_low_wind
    @test humidity_flux(canopy_ET, Δq, g_trp_high_wind) >
        humidity_flux(canopy_ET, Δq, g_trp_low_wind)

    # ET increases with stomatal conductance
    g_trp_closed = transpiration_conductance(canopy_ET, 100.0, 0.01)
    g_trp_open = transpiration_conductance(canopy_ET, 100.0, 1.0)
    @test humidity_flux(canopy_ET, Δq, g_trp_open) >
        humidity_flux(canopy_ET, Δq, g_trp_closed)

    # Canopy evaporation increases with canopy water content
    g_can_dry = canopy_evaporation_conductance(canopy_ET, 0.1, 50.0)
    g_can_wet = canopy_evaporation_conductance(canopy_ET, 0.9, 50.0)
    @test humidity_flux(canopy_ET, Δq, g_can_wet) >
        humidity_flux(canopy_ET, Δq, g_can_dry)
end

# ==============================================================================
# Boundary conditions
# ==============================================================================

@testset "No vegetation scenario" begin
    # Without vegetation, canopy ET should vanish
    canopy_ET = PALADYNCanopyEvapotranspiration(Float64)
    Δq = 0.01

    # Zero stomatal conductance → effectively zero transpiration
    g_trp_no_veg = transpiration_conductance(canopy_ET, 100.0, 0.0)
    @test iszero(g_trp_no_veg)
    @test humidity_flux(canopy_ET, Δq, g_trp_no_veg) ≈ 0 atol = 1.0e-6

    # Zero canopy saturation → zero canopy evaporation
    g_can_no_veg = canopy_evaporation_conductance(canopy_ET, 0.0, 50.0)
    @test iszero(g_can_no_veg)
    @test iszero(humidity_flux(canopy_ET, Δq, g_can_no_veg))
end

@testset "Nighttime scenario (negative VPD)" begin
    canopy_ET = PALADYNCanopyEvapotranspiration(Float64)

    # At night the humidity difference can be negative (condensation)
    Δq_night = -0.001
    g_trp = transpiration_conductance(canopy_ET, 100.0, 0.1)
    g_can = canopy_evaporation_conductance(canopy_ET, 0.5, 50.0)

    Qh_trp_night = humidity_flux(canopy_ET, Δq_night, g_trp)
    Qh_can_night = humidity_flux(canopy_ET, Δq_night, g_can)
    @test Qh_trp_night < 0
    @test Qh_can_night < 0

    # Magnitude matches the corresponding daytime flux
    @test Qh_trp_night ≈ -humidity_flux(canopy_ET, -Δq_night, g_trp)
    @test Qh_can_night ≈ -humidity_flux(canopy_ET, -Δq_night, g_can)
end
