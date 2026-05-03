using Terrarium
using Terrarium: forcing, compute_volumetric_water_content_tendency, hydraulic_conductivity
using Test

using FreezeCurves
using Oceananigans

@testset "Hydraulic properties (constant)" begin
    # For prescribed hydraulic properties, just check that the returned values
    # match what was set.
    hydraulic_props = ConstantSoilHydraulics(
        Float64;
        sat_hydraulic_cond = 1.0e-6,
        field_capacity = 0.1,
        wilting_point = 0.02,
    )
    @test saturated_hydraulic_conductivity(hydraulic_props) == hydraulic_props.sat_hydraulic_cond
    @test field_capacity(hydraulic_props) == hydraulic_props.field_capacity
    @test wilting_point(hydraulic_props) == hydraulic_props.wilting_point
end

@testset "Hydraulic properties (SURFEX)" begin
    hydraulic_props = SoilHydraulicsSURFEX(Float64)
    # check that wilting point is equal to zero when there is no clay
    wp0 = wilting_point(hydraulic_props, SoilTexture(sand = 0.5, silt = 0.5, clay = 0.0))
    @test iszero(wp0)
    for clay in 0.1:0.1:1.0
        sand = (1 - clay) * 0.7
        silt = (1 - clay) * 0.3
        wp = wilting_point(hydraulic_props, SoilTexture(; sand, silt, clay))
        @test 0 < wp < 1
    end

    # check that field capacity is equal to zero when there is no clay
    fc0 = field_capacity(hydraulic_props, SoilTexture(sand = 0.5, silt = 0.5, clay = 0.0))
    @test iszero(fc0)
    for clay in 0.1:0.1:1.0
        sand = (1 - clay) * 0.7
        silt = (1 - clay) * 0.3
        fc = field_capacity(hydraulic_props, SoilTexture(; sand, silt, clay))
        @test 0 < fc < 1
    end
end

@testset "Unsaturated hydraulic conductivity (linear)" begin
    hydraulics = ConstantSoilHydraulics(Float64; unsat_hydraulic_cond = UnsatKLinear(Float64))

    # saturated case
    soil = SoilVolume()
    K = hydraulic_conductivity(hydraulics, soil)
    @test K ≈ hydraulics.sat_hydraulic_cond

    # unsaturated
    soil = SoilVolume(saturation = 0.5)
    K = hydraulic_conductivity(hydraulics, soil)
    @test 0 < K < hydraulics.sat_hydraulic_cond

    # dry
    soil = SoilVolume(saturation = 0.0)
    K = hydraulic_conductivity(hydraulics, soil)
    @test iszero(K)

    # frozen
    soil = SoilVolume(liquid = 0.0)
    K = hydraulic_conductivity(hydraulics, soil)
    @test iszero(K)
end

@testset "Unsaturated hydraulic conductivity (Van Genuchten)" begin
    hydraulics = ConstantSoilHydraulics(Float64; swrc = VanGenuchten(), unsat_hydraulic_cond = UnsatKVanGenuchten(Float64))

    # saturated case
    soil = SoilVolume()
    K = hydraulic_conductivity(hydraulics, soil)
    @test K ≈ hydraulics.sat_hydraulic_cond

    # unsaturated
    soil = SoilVolume(saturation = 0.5)
    K = hydraulic_conductivity(hydraulics, soil)
    @test 0 < K < hydraulics.sat_hydraulic_cond

    # dry
    soil = SoilVolume(saturation = 0.0)
    K = hydraulic_conductivity(hydraulics, soil)
    @test iszero(K)

    # frozen
    soil = SoilVolume(liquid = 0.0)
    K = hydraulic_conductivity(hydraulics, soil)
    @test iszero(K)
end

@testset "SoilHydrology: adjust_saturation_profile" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 100))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    hydraulic_properties = ConstantSoilHydraulics(Float64; swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(Float64))
    hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
    state = initialize(hydrology, grid)

    # Case 1: Oversaturation at surface
    set!(state.saturation_water_ice, (x, z) -> max(1.1 + z, 1.0))
    ∫sat_excess = Field(Integral(state.saturation_water_ice - 1, dims = 3))
    compute!(∫sat_excess)
    Terrarium.adjust_saturation_profile!(state, grid, hydrology)
    @test all(state.saturation_water_ice .≈ 1)
    @test all(state.surface_excess_water .≈ ∫sat_excess)

    # Case 2: Undersaturation at surface
    set!(state.saturation_water_ice, (x, z) -> min(-0.1 - z, 1.0))
    ∫sat = Field(Integral(state.saturation_water_ice, dims = 3))
    ∫sat_deficit = Field(Integral(state.saturation_water_ice, dims = 3, condition = state.saturation_water_ice .< 0))
    ∫sat₀ = compute!(∫sat)[1, 1, 1]
    compute!(∫sat_deficit)
    Terrarium.adjust_saturation_profile!(state, grid, hydrology)
    ∫sat₁ = compute!(∫sat)[1, 1, 1]
    @test all(state.saturation_water_ice .>= hydraulic_properties.residual)
    @test ∫sat₁ ≈ ∫sat₀

    # Case 3: Completely dry with negative saturation near surface
    set!(state.saturation_water_ice, (x, z) -> min(-0.1 - z, 0.0))
    Terrarium.adjust_saturation_profile!(state, grid, hydrology)
    @test all(state.saturation_water_ice .≈ hydraulic_properties.residual)
end

@testset "SoilHydrology: Richardson-Richards' equation" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 100))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    hydraulic_properties = ConstantSoilHydraulics(Float64; swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(Float64))
    hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
    initializer = DefaultInitializer(eltype(grid))
    model = SoilModel(grid; soil, initializer)
    # Fully saturated, steady state
    initializers = (saturation_water_ice = (x, z) -> 1.0,)
    integrator = initialize(model, ForwardEuler(); initializers)
    state = integrator.state
    # check that initial water table depth is correctly calculated from initial condition
    @test all(isapprox.(state.water_table, 0, atol = 1.0e-12))
    # also check that pressure head is zero everywhere
    @test all(isapprox.(state.pressure_head, 0, atol = 1.0e-12))
    compute_auxiliary!(state, model)
    # check that all hydraulic conductivities are finite and equal to K_sat
    @test all(isfinite.(state.hydraulic_conductivity))
    @test all(state.hydraulic_conductivity .≈ hydraulic_properties.sat_hydraulic_cond)
    compute_tendencies!(state, model)
    # check that all tendencies are zero
    @test all(iszero.(state.tendencies.saturation_water_ice))
    # check timestep!
    timestep!(integrator)
    @test all(state.saturation_water_ice .≈ 1)

    # Variably saturated with water table
    initializers = (saturation_water_ice = (x, z) -> min(1, 0.5 - 0.1 * z),)
    integrator = initialize(model, ForwardEuler(); initializers)
    state = integrator.state
    water_table = state.water_table
    hydraulic_cond = state.hydraulic_conductivity
    saturation = state.saturation_water_ice
    # check that initial water table depth is correctly calculated from initial condition
    @test all(water_table .≈ -5.0)
    # also check that pressure head is negative
    @test all(state.pressure_head .< 0)
    compute_auxiliary!(state, model)
    # check that all hydraulic conductivities are finite and positive
    @test all(isfinite.(hydraulic_cond))
    @test all(hydraulic_cond .> 0)
    compute_tendencies!(state, model)
    # check that all tendencies are finite
    @test all(isfinite.(state.tendencies.saturation_water_ice))
    # do timestep! and compute total water mass
    ∫sat₀ = Field(Integral(saturation, dims = 3))
    compute!(∫sat₀)
    Δt = 60.0
    timestep!(integrator, Δt)
    ∫sat₁ = Field(Integral(saturation, dims = 3))
    compute!(∫sat₁)
    # check saturation levels are all finite and valid
    @test all(isfinite.(saturation))
    @test all(0 .<= saturation .<= 1)
    # check mass conservation
    @test ∫sat₀[1, 1, 1] ≈ ∫sat₁[1, 1, 1]
    # run for one simulation hour and check that mass is still conserved
    run!(integrator; period = Hour(1), Δt)
    ∫sat₂ = Field(Integral(saturation, dims = 3))
    compute!(∫sat₂)
    @test all(isfinite.(saturation))
    @test all(0 .<= saturation .<= 1)
    @test ∫sat₀[1, 1, 1] ≈ ∫sat₁[1, 1, 1] ≈ ∫sat₂[1, 1, 1]
end

@testset "Soil moisture forcing (source/sink)" begin
    mutable struct ForcingValue{NF}
        value::NF
    end

    Nz = 10
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = Nz))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    porosity = ConstantSoilPorosity(eltype(grid))
    strat = HomogeneousStratigraphy(eltype(grid); porosity)
    hydraulic_properties = ConstantSoilHydraulics(eltype(grid); swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(Float64))
    forcing_value = ForcingValue(0.0)
    vwc_forcing = Forcing(parameters = forcing_value, discrete_form = true) do i, j, k, grid, clock, fields, params
        params.value
    end
    hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties, vwc_forcing)
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology, strat)
    model = SoilModel(grid; soil)
    # Variably saturated with water table
    initializers = (
        temperature = 10.0, # positive soil temperature
        saturation_water_ice = 1.0, # fully saturated
    )
    integrator = initialize(model, ForwardEuler(); initializers)
    state = integrator.state
    fields = get_fields(state, hydrology)
    # check that forcing_ET is zero when no latent heat flux is supplied
    @test iszero(forcing(1, 1, Nz, grid, state.clock, fields, vwc_forcing, hydrology))
    # negative flux
    forcing_value.value = -0.01
    @test forcing(1, 1, Nz, grid, state.clock, fields, vwc_forcing, hydrology) == forcing_value.value
    # positive flux
    forcing_value.value = 0.01
    @test forcing(1, 1, Nz, grid, state.clock, fields, vwc_forcing, hydrology) == forcing_value.value
    # check tendency calculation
    dθdt = compute_volumetric_water_content_tendency(1, 1, Nz, grid, state.clock, fields, hydrology, model.constants, nothing)
    @test dθdt == forcing_value.value
    # take one timestep and check that water was removed
    dt = 60.0
    forcing_value.value = -1.0e-5
    timestep!(integrator, dt)
    @test state.saturation_water_ice[1, 1, Nz] .≈ 1 + forcing_value.value * dt / porosity.mineral_porosity
end
