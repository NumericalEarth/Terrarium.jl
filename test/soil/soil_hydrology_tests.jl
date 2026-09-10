using Terrarium
using Terrarium: forcing, compute_volumetric_water_content_tendency, hydraulic_conductivity, Variables, auxiliary, get_closure
using Test

using FreezeCurves
using Oceananigans

@testset "Hydraulic properties (constant)" begin
    # For prescribed hydraulic properties, just check that the returned values
    # match what was set.
    hydraulic_props = ConstantSoilHydraulics(
        Float64;
        saturated_conductivity = 1.0e-6,
        field_capacity = 0.1,
        wilting_point = 0.02,
    )
    @test saturated_hydraulic_conductivity(hydraulic_props) == hydraulic_props.saturated_conductivity
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

    # check that field capacity is equal to the minimum when there is no clay
    fc0 = field_capacity(hydraulic_props, SoilTexture(sand = 0.5, silt = 0.5, clay = 0.0))
    @test fc0 ≈ hydraulic_props.field_capacity_min
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
    soil = SoilComposition()
    K = hydraulic_conductivity(hydraulics, soil)
    @test K ≈ hydraulics.saturated_conductivity

    # unsaturated
    soil = SoilComposition(saturation = 0.5)
    K = hydraulic_conductivity(hydraulics, soil)
    @test 0 < K < hydraulics.saturated_conductivity

    # dry
    soil = SoilComposition(saturation = 0.0)
    K = hydraulic_conductivity(hydraulics, soil)
    @test iszero(K)

    # frozen
    soil = SoilComposition(liquid = 0.0)
    K = hydraulic_conductivity(hydraulics, soil)
    @test iszero(K)
end

@testset "Unsaturated hydraulic conductivity (Van Genuchten)" begin
    hydraulics = ConstantSoilHydraulics(Float64; swrc = VanGenuchten(), unsat_hydraulic_cond = UnsatKVanGenuchten(Float64))

    # saturated case
    soil = SoilComposition()
    K = hydraulic_conductivity(hydraulics, soil)
    @test K ≈ hydraulics.saturated_conductivity

    # unsaturated
    soil = SoilComposition(saturation = 0.5)
    K = hydraulic_conductivity(hydraulics, soil)
    @test 0 < K < hydraulics.saturated_conductivity

    # dry
    soil = SoilComposition(saturation = 0.0)
    K = hydraulic_conductivity(hydraulics, soil)
    @test iszero(K)

    # frozen
    soil = SoilComposition(liquid = 0.0)
    K = hydraulic_conductivity(hydraulics, soil)
    @test iszero(K)
end

@testset "SoilHydrology: adjust_saturation_profile" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 100))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    hydraulic_properties = ConstantSoilHydraulics(Float64; swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(Float64))
    hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
    state = StateVariables(hydrology, grid)

    # Case 1: Oversaturation at surface. Standalone soil hydrology (no runoff process to receive it)
    # discards excess water reaching the surface, so only the clamped saturation profile is checked here.
    set!(state.saturation_water_ice, (x, z) -> max(1.1 + z, 1.0))
    Terrarium.adjust_saturation_profile!(state, grid, hydrology)
    @test all(state.saturation_water_ice .≈ 1)

    # Case 1b: With a runoff process passed as an optional dependency, the excess water removed from the
    # oversaturated surface is routed into the runoff-owned `surface_excess_water` pool instead of discarded.
    runoff = DirectSurfaceRunoff(Float64)
    coupled_state = StateVariables(merge(Variables(hydrology), Variables(runoff)), grid)
    set!(coupled_state.saturation_water_ice, (x, z) -> max(1.1 + z, 1.0))
    ∫sat_excess = Field(Integral(coupled_state.saturation_water_ice - 1, dims = 3))
    compute!(∫sat_excess)
    Terrarium.adjust_saturation_profile!(coupled_state, grid, hydrology, runoff)
    @test all(coupled_state.saturation_water_ice .≈ 1)
    @test all(coupled_state.surface_excess_water .≈ ∫sat_excess)

    # Case 2: Undersaturation at surface
    set!(state.saturation_water_ice, (x, z) -> min(-0.1 - z, 1.0))
    ∫sat = Field(Integral(state.saturation_water_ice, dims = 3))
    ∫sat_deficit = Field(Integral(state.saturation_water_ice, dims = 3, condition = state.saturation_water_ice .< 0))
    ∫sat₀ = compute!(∫sat)[1, 1, 1]
    compute!(∫sat_deficit)
    Terrarium.adjust_saturation_profile!(state, grid, hydrology)
    ∫sat₁ = compute!(∫sat)[1, 1, 1]
    @test all(state.saturation_water_ice .>= hydraulic_properties.residual_saturation)
    @test ∫sat₁ ≈ ∫sat₀

    # Case 3: Completely dry with negative saturation near surface
    set!(state.saturation_water_ice, (x, z) -> min(-0.1 - z, 0.0))
    Terrarium.adjust_saturation_profile!(state, grid, hydrology)
    @test all(state.saturation_water_ice .≈ hydraulic_properties.residual_saturation)
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
    integrator = initialize(model; initializers)
    state = integrator.state
    # check that initial water table depth is correctly calculated from initial condition
    @test all(isapprox.(state.water_table, 0, atol = 1.0e-12))
    # also check that pressure head is zero everywhere
    @test all(isapprox.(state.pressure_head, 0, atol = 1.0e-12))
    compute_auxiliary!(state, model)
    # check that all hydraulic conductivities are finite and equal to K_sat
    @test all(isfinite.(state.hydraulic_conductivity))
    @test all(state.hydraulic_conductivity .≈ hydraulic_properties.saturated_conductivity)
    compute_tendencies!(state, model)
    # check that all tendencies are zero
    @test all(iszero.(state.tendencies.saturation_water_ice))
    # check timestep!
    timestep!(integrator)
    @test all(state.saturation_water_ice .≈ 1)

    # Variably saturated with water table
    initializers = (saturation_water_ice = (x, z) -> min(1, 0.5 - 0.1 * z),)
    integrator = initialize(model; initializers)
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

@testset "SoilHydrology: saturation_infiltration_bc" begin
    # `saturation_water_ice` is dimensionless (VWC / porosity), so a physical infiltration flux
    # (m/s of water depth) must be normalized by porosity before it can drive that field's top
    # boundary condition; see `saturation_infiltration_bc`, `InfiltrationFlux`, and `porosity_top`.
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 10))
    mineral_porosity = 0.4
    porosity_scheme = ConstantSoilPorosity(eltype(grid); mineral_porosity)
    strat = HomogeneousSoilStratigraphy(eltype(grid); porosity = porosity_scheme)
    biogeochem = ConstantSoilCarbonDensity(eltype(grid); ρ_soc = 0.0) # zero SOC -> pure mineral porosity

    # Unit check: the discrete BC function itself equals -infiltration / porosity pointwise. `fields`
    # only needs an (empty) `:soil` namespace entry since the default `ConstantSoilHorizon` declares
    # no variables of its own; `clock` is unused by this particular condition.
    fgrid = get_field_grid(grid)
    infiltration = Field{Center, Center, Nothing}(fgrid)
    set!(infiltration, 2.0e-7)
    fields = (; infiltration, soil = (;))
    parameters = (; strat, bgc = biogeochem)
    value = Terrarium.saturation_infiltration_bc(1, 1, fgrid, nothing, fields, parameters)
    @test value ≈ -2.0e-7 / mineral_porosity

    # End-to-end check, at the `SoilHydrology` level (no full model needed): the top-cell
    # `saturation_water_ice` tendency contributed by `InfiltrationFlux` scales by exactly
    # `1/porosity`.
    function top_saturation_tendency(mineral_porosity)
        grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
        strat = HomogeneousSoilStratigraphy(eltype(grid); porosity = ConstantSoilPorosity(eltype(grid); mineral_porosity))
        bgc = ConstantSoilCarbonDensity(eltype(grid))
        hydrology = SoilHydrology(eltype(grid), RichardsEq())
        infiltration_var = auxiliary(:infiltration, XY(), units = u"m/s", desc = "Infiltration flux")
        # `strat`'s own variables (the per-horizon namespace `porosity_top` reaches into) must be part
        # of the state too, not just `hydrology`'s.
        vars = merge(Variables(hydrology), Variables(strat), Variables((infiltration_var,)))
        state = StateVariables(vars, grid; boundary_conditions = InfiltrationFlux(strat, bgc))
        soil = SoilEnergyWaterCarbon(eltype(grid); strat, hydrology, biogeochem = bgc)
        set!(state.saturation_water_ice, (x, z) -> min(1, 0.8 - 0.05 * z))
        set!(state.infiltration, 2.0e-7)
        Terrarium.closure!(state, grid, get_closure(hydrology), hydrology, soil)
        fill!(parent(state.tendencies.saturation_water_ice), 0)
        compute_boundary_conditions!(state, grid, hydrology, strat, bgc)
        return Array(interior(state.tendencies.saturation_water_ice))[1, 1, end]
    end

    tendency_lo_porosity = top_saturation_tendency(0.4)
    tendency_hi_porosity = top_saturation_tendency(0.8)
    @test !iszero(tendency_hi_porosity)
    @test tendency_lo_porosity ≈ 2 * tendency_hi_porosity
end

@testset "Soil moisture forcing (source/sink)" begin
    mutable struct ForcingValue{NF}
        value::NF
    end

    Nz = 10
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = Nz))
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    porosity = ConstantSoilPorosity(eltype(grid))
    strat = HomogeneousSoilStratigraphy(eltype(grid); porosity)
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
    integrator = initialize(model; initializers)
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
