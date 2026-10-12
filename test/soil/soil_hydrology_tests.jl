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
    # oversaturated surface is routed into the runoff-owned `surface_excess_water` pool instead of discarded,
    # as a water depth (excess saturation times porosity, integrated over depth).
    runoff = DirectSurfaceRunoff(Float64)
    mineral_porosity = 0.4
    strat = HomogeneousSoilStratigraphy(Float64; porosity = ConstantSoilPorosity(Float64; mineral_porosity))
    bgc = ConstantSoilCarbonDensity(Float64; ρ_soc = 0.0) # zero SOC -> pure mineral porosity
    coupled_state = StateVariables(merge(Variables(hydrology), Variables(runoff), Variables(strat)), grid)
    set!(coupled_state.saturation_water_ice, (x, z) -> max(1.1 + z, 1.0))
    ∫sat_excess = Field(Integral(coupled_state.saturation_water_ice - 1, dims = 3))
    compute!(∫sat_excess)
    Terrarium.adjust_saturation_profile!(coupled_state, grid, hydrology, runoff, strat, bgc)
    @test all(coupled_state.saturation_water_ice .≈ 1)
    @test all(coupled_state.surface_excess_water .≈ mineral_porosity .* ∫sat_excess)

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
    ground_grid = ground_domain(grid)
    infiltration = Field{Center, Center, Nothing}(ground_grid)
    set!(infiltration, 2.0e-7)
    fields = (; infiltration, soil = (;))
    parameters = (; strat, bgc = biogeochem)
    value = Terrarium.saturation_infiltration_bc(1, 1, ground_grid, nothing, fields, parameters)
    @test value ≈ -2.0e-7 / mineral_porosity

    # End-to-end check, at the `SoilHydrology` level (no full model needed): the top-cell
    # `saturation_water_ice` tendency contributed by `InfiltrationFlux` scales by exactly
    # `1/porosity`.
    function top_saturation_tendency(mineral_porosity)
        grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 50))
        strat = HomogeneousSoilStratigraphy(eltype(grid); porosity = ConstantSoilPorosity(eltype(grid); mineral_porosity))
        bgc = ConstantSoilCarbonDensity(eltype(grid))
        hydrology = SoilHydrology(eltype(grid), RichardsEq())
        soil = SoilEnergyWaterCarbon(eltype(grid); strat, hydrology, biogeochem = bgc)
        infiltration_var = auxiliary(:infiltration, Terrarium.Ground(Terrarium.Top()), units = u"m/s", desc = "Infiltration flux")
        # state of the full soil (the hydrology closure moves the heat of redistributed water), plus the
        # infiltration input of the boundary condition
        vars = merge(Variables(soil), Variables((infiltration_var,)))
        state = StateVariables(vars, grid; boundary_conditions = InfiltrationFlux(strat, bgc))
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

# Implicit Richards equation (modified Picard iteration)
module SoilHydrologyImplicitRichardsTests

using Terrarium
using Test

using Oceananigans.BoundaryConditions: GradientBoundaryCondition, ValueBoundaryCondition, fill_halo_regions!
using Terrarium: merge_boundary_conditions, update_state!, cell_diffusion_timescale, reset_tendencies!,
    get_hydrology, get_stratigraphy, get_biogeochemistry, get_energy_balance, get_swrc, default_timestepping, implicit_operator

module ImplicitRichardsTestUtils

    using Terrarium
    using Terrarium: merge_boundary_conditions, Δzᵃᵃᶜ, get_hydrology, get_stratigraphy, get_biogeochemistry

    const POROSITY = 0.4

    # Richards-equation soil with a van Genuchten retention curve and a homogeneous mineral porosity
    function richards_soil(NF; saturated_conductivity = 1.0e-5, swrc = VanGenuchten(α = 2.0, n = 2.0))
        hydraulic_properties = ConstantSoilHydraulics(NF; swrc, saturated_conductivity, unsat_hydraulic_cond = UnsatKVanGenuchten(NF))
        hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
        strat = HomogeneousSoilStratigraphy(NF; porosity = ConstantSoilPorosity(NF; mineral_porosity = POROSITY))
        biogeochem = ConstantSoilCarbonDensity(NF; ρ_soc = 0.0)
        return SoilEnergyWaterCarbon(NF; hydrology, strat, biogeochem)
    end

    implicit_timestepper(NF; iterations = 2) =
        IMEX(ForwardEuler(NF), BackwardEuler(NF; solver = Picard(NF; iterations)))

    # Soil column with a constant infiltration flux `infiltration` (m/s, positive downward) at the top and,
    # by default, an impermeable bottom.
    function build_water_column(
            grid, timestepper; saturation, temperature = 5.0, infiltration = 0.0,
            extra_boundary_conditions = (;), soil_kwargs = (;)
        )
        NF = eltype(grid)
        model = SoilModel(grid; soil = richards_soil(NF; soil_kwargs...), timestepper)
        # flux boundary condition on saturation (positive upward), normalized by the porosity
        boundary_conditions = merge_boundary_conditions(InfiltrationFlux(NF(-infiltration / POROSITY)), extra_boundary_conditions)
        return initialize(model; boundary_conditions, initializers = (; temperature, saturation_water_ice = saturation))
    end

    cell_thicknesses(grid) = [Δzᵃᵃᶜ(1, 1, k, ground_domain(grid)) for k in 1:ground_domain(grid).Nz]

    function porosities(integrator)
        (; state, model) = integrator
        strat, bgc = get_stratigraphy(model.soil), get_biogeochemistry(model.soil)
        fields = get_fields(state, strat, bgc)
        grid = get_grid(model)
        return [Terrarium.porosity(1, 1, k, grid, fields, strat, bgc) for k in 1:ground_domain(grid).Nz]
    end

    saturation_profile(integrator) = copy(interior(integrator.state.saturation_water_ice)[1, 1, :])

    # column-integrated water and ice content ∑ φ s Δz (m)
    column_water(integrator) = sum(porosities(integrator) .* saturation_profile(integrator) .* cell_thicknesses(get_grid(integrator.model)))

    # first moment depth (m, positive) of the water added to the column relative to the profile `s⁰`
    function infiltrated_water_depth(integrator, s⁰)
        grid = get_grid(integrator.model)
        Δz = cell_thicknesses(grid)
        z = collect(znodes(ground_domain(grid), Center()))
        Δθ = porosities(integrator) .* (saturation_profile(integrator) .- s⁰) .* Δz
        return -sum(z .* Δθ) / sum(Δθ)
    end

    function run_for!(integrator, Δt, duration)
        nsteps = round(Int, duration / Δt)
        @assert nsteps * Δt ≈ duration
        for _ in 1:nsteps
            timestep!(integrator, Δt)
        end
        return integrator
    end

    dividing_dt(duration, Δt_max) = duration / ceil(duration / Δt_max)

end

using .ImplicitRichardsTestUtils: richards_soil, implicit_timestepper, build_water_column, cell_thicknesses, porosities,
    saturation_profile, column_water, infiltrated_water_depth, run_for!, dividing_dt, POROSITY

@testset "Implicit Richards: explicit tendencies, timescale, and boundary conditions" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.1, N = 10))
    integrator = build_water_column(grid, implicit_timestepper(NF); saturation = 0.6)
    # without boundary fluxes and sources, the explicit tendency of an implicitly routed saturation is zero
    update_state!(integrator.state, integrator.model, integrator.inputs; compute_tendencies = true)
    @test all(iszero, interior(integrator.state.tendencies.saturation_water_ice))
    # ... while the explicit tendency holds the interior Darcy flux divergence
    explicit = build_water_column(grid, ForwardEuler(NF); saturation = 0.6)
    update_state!(explicit.state, explicit.model, explicit.inputs; compute_tendencies = true)
    @test any(!iszero, interior(explicit.state.tendencies.saturation_water_ice))
    # cell diffusion timescale: implicit Richards imposes no restriction
    @test isfinite(cell_diffusion_timescale(explicit))
    @test cell_diffusion_timescale(integrator) == Inf
    # value boundary conditions on the pressure head are rejected; gradient conditions are supported
    model = SoilModel(grid; soil = richards_soil(NF), timestepper = implicit_timestepper(NF))
    value_bc = (pressure_head = (bottom = ValueBoundaryCondition(NF(-1)),),)
    @test_throws ArgumentError initialize(model; boundary_conditions = value_bc, initializers = (saturation_water_ice = 0.6,))
    gradient_bc = (pressure_head = (bottom = GradientBoundaryCondition(NF(1)),),)
    @test initialize(model; boundary_conditions = gradient_bc, initializers = (saturation_water_ice = 0.6,)) isa ModelIntegrator
    # the same value condition is fine under explicit time stepping
    @test initialize(SoilModel(grid; soil = richards_soil(NF)); boundary_conditions = value_bc, initializers = (saturation_water_ice = 0.6,)) isa ModelIntegrator
    # the minimum moisture capacity of the implicit solve is a property of the hydraulic properties
    @test Terrarium.minimum_moisture_capacity(ConstantSoilHydraulics(NF)) == 1.0e-5
    @test Terrarium.minimum_moisture_capacity(SoilHydraulicsSURFEX(NF; minimum_moisture_capacity = 1.0e-4)) == 1.0e-4
end

@testset "Implicit Richards: explicit water outflow timescale" begin
    NF = Float64
    Δz = 0.05
    grid = ColumnGrid(CPU(), NF, UniformSpacing(; Δz, N = 20))
    # draining sandy column: gravity drainage (unit head gradient) at the bottom, infiltration at the top
    drainage = (pressure_head = (bottom = GradientBoundaryCondition(NF(1)),),)
    soil_kwargs = (saturated_conductivity = 1.0e-4, swrc = VanGenuchten(α = 3.5, n = 3.0))
    infiltration = 2.0e-5
    s = 0.6
    s_res = Terrarium.residual_saturation(ConstantSoilHydraulics(NF))
    function outflow_timescale(integrator)
        (; state, model) = integrator
        update_state!(state, model, integrator.inputs; compute_tendencies = true)
        class = timestepping(state, Val(:saturation_water_ice))
        return Terrarium.water_outflow_timescale(state, get_grid(model), get_hydrology(model.soil), model.soil, model.constants, nothing, class)
    end
    # implicit: only the drainage through the bottom face, q = -K, is explicit
    implicit = build_water_column(grid, implicit_timestepper(NF); saturation = s, infiltration, extra_boundary_conditions = drainage, soil_kwargs)
    τ = outflow_timescale(implicit)
    K = interior(implicit.state.hydraulic_conductivity)[1, 1, :]
    @test isfinite(τ)
    @test τ ≈ POROSITY * (s - s_res) * Δz / min(K[1], K[2])
    # implicit heat advection imposes no limit, so the outflow limit is the advective timescale of the model
    @test Terrarium.cell_advection_timescale(implicit) ≈ τ
    # explicit: the interior gravity fluxes balance, and the top cell loses K - infiltration
    explicit = build_water_column(grid, ForwardEuler(NF); saturation = s, infiltration, extra_boundary_conditions = drainage, soil_kwargs)
    @test outflow_timescale(explicit) ≈ POROSITY * (s - s_res) * Δz / (min(K[end - 1], K[end]) - infiltration)
    # no explicit outflow without drainage
    closed = build_water_column(grid, implicit_timestepper(NF); saturation = s, infiltration, soil_kwargs)
    @test outflow_timescale(closed) == Inf
    @test Terrarium.cell_advection_timescale(closed) == Inf
end


@testset "Implicit Richards: linear consistency with dense backward Euler" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.05, Δz_max = 0.3, N = 12))
    Nz = 12
    saturation = 0.6
    integrator = build_water_column(grid, implicit_timestepper(NF; iterations = 1); saturation)
    (; state, model) = integrator
    soil = model.soil
    hydrology, strat, bgc = get_hydrology(soil), get_stratigraphy(soil), get_biogeochemistry(soil)
    Δt = 50 * cell_diffusion_timescale(build_water_column(grid, ForwardEuler(NF); saturation))
    # hydraulic conductivity at tⁿ (the liquid water fraction is set by the energy closure after the
    # hydrology is initialized, so the conductivity is only consistent after `compute_auxiliary!`)
    compute_auxiliary!(state, model)
    s⁰ = saturation_profile(integrator)
    Ψ⁰ = copy(interior(state.pressure_head)[1, 1, :])
    # explicit saturation tendency G(Ψ) of the Darcy flux at fixed hydraulic conductivity, which is linear in Ψ
    function explicit_tendency(Ψ)
        interior(state.pressure_head)[1, 1, :] .= Ψ
        reset_tendencies!(state)
        compute_boundary_conditions!(state, grid, hydrology, strat, bgc)
        compute_tendencies!(state, grid, hydrology, soil, model.constants, nothing, Explicit())
        return copy(interior(state.tendencies.saturation_water_ice)[1, 1, :])
    end
    G⁰ = explicit_tendency(Ψ⁰)
    ε = 1.0e-4
    L = zeros(NF, Nz, Nz)
    for j in 1:Nz
        Ψ = copy(Ψ⁰)
        Ψ[j] += ε
        L[:, j] .= (explicit_tendency(Ψ) .- G⁰) ./ ε
    end
    @test maximum(abs, G⁰ .- L * Ψ⁰) < 1.0e-6 * maximum(abs, G⁰)
    interior(state.pressure_head)[1, 1, :] .= Ψ⁰
    # specific moisture capacity C = ∂θ/∂ψ by central differences of the retention curve
    swrc = get_swrc(hydrology)
    ψm = inv(swrc)(saturation * POROSITY; θsat = POROSITY)
    δ = 1.0e-6
    C = (swrc(ψm + δ; θsat = POROSITY) - swrc(ψm - δ; θsat = POROSITY)) / (2δ)
    # linearly implicit backward Euler (C - Δt φ L) Ψ¹ = C Ψ⁰, then s¹ = s⁰ + Δt L Ψ¹
    A = -Δt * POROSITY .* L
    for k in 1:Nz
        A[k, k] += C
    end
    Ψ_dense = A \ (C .* Ψ⁰)
    s_dense = s⁰ .+ Δt .* (L * Ψ_dense)
    timestep!(integrator, Δt)
    s_implicit = saturation_profile(integrator)
    relative_error = maximum(abs, s_implicit .- s_dense) / maximum(abs, s_dense .- s⁰)
    @info "Implicit Richards linear consistency" Δt maximum(abs, s_dense .- s⁰) relative_error
    @test relative_error < 1.0e-8
    # column water is conserved (no boundary fluxes)
    @test sum(POROSITY .* (s_implicit .- s⁰) .* cell_thicknesses(grid)) ≈ 0 atol = 1.0e-14
end

@testset "Implicit Richards: mass conservation with infiltration (SoilModel)" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.02, Δz_max = 0.3, N = 20))
    Δt = 3600.0
    infiltration = 2.0e-6
    for iterations in (2, 3)
        integrator = build_water_column(grid, implicit_timestepper(NF; iterations); saturation = (x, z) -> 0.4 - 0.1 * z, infiltration)
        conserved = true
        for _ in 1:48
            W⁰ = column_water(integrator)
            timestep!(integrator, Δt)
            conserved &= isapprox(column_water(integrator) - W⁰, Δt * infiltration; rtol = 1.0e-9)
        end
        s = saturation_profile(integrator)
        @test conserved
        @test all(isfinite, s)
        @test all(0 .<= s .<= 1)
        # the infiltrating water has wetted the surface
        @test s[end] > 0.6
    end
end

@testset "Implicit Richards: mass conservation with surface saturation (LandModel)" begin
    NF = Float64
    Δt = 3600.0
    for iterations in (2, 3)
        grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.05, Δz_max = 0.5, N = 20))
        land = LandModel(grid; soil = richards_soil(NF), vegetation = nothing, timestepper = implicit_timestepper(NF; iterations))
        initializers = (
            temperature = (x, z) -> 10.0 - 0.02 * z,
            saturation_water_ice = (x, z) -> min(1.0, 0.97 - 0.02 * z),
            rainfall = 2.0e-5,
            air_temperature = 15.0,
        )
        integrator = initialize(land; initializers)
        (; state) = integrator
        @test timestepping(state, Val(:saturation_water_ice)) === Implicit()
        @test timestepping(state, Val(:internal_energy)) === Implicit()
        Δz = cell_thicknesses(grid)
        φ = porosities(integrator)
        conserved = true
        routed = 0.0
        for _ in 1:24
            W⁰ = column_water(integrator)
            S⁰ = interior(state.surface_excess_water)[1, 1, 1]
            timestep!(integrator, Δt)
            # explicit tendencies at tⁿ: soil boundary fluxes and sinks, and the drainage of the excess pool
            G_soil = sum(φ .* interior(state.tendencies.saturation_water_ice)[1, 1, :] .* Δz)
            G_pool = interior(state.tendencies.surface_excess_water)[1, 1, 1]
            ΔW = column_water(integrator) - W⁰
            ΔS = interior(state.surface_excess_water)[1, 1, 1] - S⁰
            conserved &= isapprox(ΔW + ΔS, Δt * (G_soil + G_pool); rtol = 1.0e-9, atol = 1.0e-12)
            routed += ΔS - Δt * G_pool
        end
        @info "Implicit Richards LandModel surface excess (M = $iterations)" routed
        @test conserved
        # excess water reaching the saturated surface was routed into the surface pool
        @test routed > 0
        s = saturation_profile(integrator)
        @test all(isfinite, s)
        @test all(0 .<= s .<= 1)
        @test all(isfinite, interior(state.temperature))
    end
end

@testset "Implicit Richards: infiltration into a dry column (Celia et al., 1990)" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.02, N = 50))
    duration = 86400.0
    kwargs = (saturation = 0.3, infiltration = 3.0e-6)
    s⁰ = fill(0.3, 50)
    # explicit reference with a step well below the diffusive limit of the wetted column
    wet = build_water_column(grid, ForwardEuler(NF); saturation = 0.9)
    τ_wet = cell_diffusion_timescale(wet)
    τ_dry = cell_diffusion_timescale(build_water_column(grid, ForwardEuler(NF); kwargs...))
    reference = build_water_column(grid, ForwardEuler(NF); kwargs...)
    run_for!(reference, dividing_dt(duration, τ_wet / 4), duration)
    d_reference = infiltrated_water_depth(reference, s⁰)
    s_reference = saturation_profile(reference)
    Δt = 3600.0
    for iterations in (2, 3)
        integrator = run_for!(build_water_column(grid, implicit_timestepper(NF; iterations); kwargs...), Δt, duration)
        d = infiltrated_water_depth(integrator, s⁰)
        relative_error = abs(d - d_reference) / d_reference
        saturation_error = maximum(abs, saturation_profile(integrator) .- s_reference)
        @info "Implicit Richards infiltration (M = $iterations)" Δt τ_dry τ_wet Δt / τ_wet d d_reference relative_error saturation_error
        @test Δt > 50 * τ_wet
        @test relative_error < 0.05
        @test saturation_error < 0.1
    end
end

@testset "Implicit Richards: stability near saturation" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.02, N = 50))
    # gravity drainage at the bottom (unit gradient of the total head)
    drainage = (pressure_head = (bottom = GradientBoundaryCondition(NF(1)),),)
    kwargs = (saturation = 0.98, infiltration = 5.0e-6, extra_boundary_conditions = drainage)
    τ = cell_diffusion_timescale(build_water_column(grid, ForwardEuler(NF); kwargs...))
    Δt = 1800.0
    integrator = build_water_column(grid, implicit_timestepper(NF); kwargs...)
    bounded = true
    for _ in 1:200
        timestep!(integrator, Δt)
        s = saturation_profile(integrator)
        bounded &= all(isfinite, s) && all(0 .<= s .<= 1)
    end
    @info "Implicit Richards near saturation" τ Δt / τ extrema(saturation_profile(integrator))
    @test Δt > 100 * τ
    @test bounded
    @test all(isfinite, interior(integrator.state.pressure_head))
end

@testset "Implicit Richards: frozen saturated column" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.05, N = 20))
    # frozen (zero hydraulic conductivity) and saturated (zero moisture capacity) below, thawed above
    integrator = build_water_column(
        grid, implicit_timestepper(NF);
        saturation = (x, z) -> z < -0.5 ? 1.0 : 0.7, temperature = (x, z) -> z < -0.5 ? -2.0 : 2.0, infiltration = 2.0e-7
    )
    conserved = true
    for _ in 1:48
        W⁰ = column_water(integrator)
        timestep!(integrator, 3600.0)
        conserved &= isapprox(column_water(integrator) - W⁰, 3600.0 * 2.0e-7; rtol = 1.0e-9)
    end
    s = saturation_profile(integrator)
    @test conserved
    @test all(isfinite, s)
    @test all(0 .<= s .<= 1)
end

@testset "Implicit Richards: SoilModel and LandModel with both variables implicit" begin
    NF = Float64
    Δt = 1800.0
    configurations = (
        bare = (
            snow = nothing,
            initializers = (temperature = (x, z) -> 5.0 - 0.02 * z, saturation_water_ice = (x, z) -> min(1.0, 0.7 - 0.05 * z), rainfall = 1.0e-6, air_temperature = 10.0),
        ),
        snow = (
            snow = SingleLayerSnow(NF),
            initializers = (
                temperature = (x, z) -> -1.0 - 0.02 * z,
                saturation_water_ice = (x, z) -> min(1.0, 0.7 - 0.05 * z),
                snow_water_equivalent = 0.2,
                snow_temperature = -5.0,
                air_temperature = -2.0,
            ),
        ),
    )
    for (name, config) in pairs(configurations)
        grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_max = 1.0, N = 30))
        land = LandModel(grid; soil = richards_soil(NF), snow = config.snow, vegetation = nothing, timestepper = implicit_timestepper(NF))
        integrator = initialize(land; initializers = config.initializers)
        (; state) = integrator
        Δz = cell_thicknesses(grid)
        φ = porosities(integrator)
        Cw = get_energy_balance(land.soil).thermal_properties.heat_capacities.water
        energy_conserved = true
        water_conserved = true
        for _ in 1:24
            W⁰ = column_water(integrator)
            S⁰ = interior(state.surface_excess_water)[1, 1, 1]
            E⁰ = sum(interior(state.internal_energy)[1, 1, :] .* Δz)
            T_top = interior(state.temperature)[1, 1, end]
            timestep!(integrator, Δt)
            # the implicit interior fluxes cancel, so the column changes by Δt times the explicit tendencies at tⁿ,
            # minus the excess water routed to the surface pool and its heat, at the temperature of the top cell at tⁿ
            G_energy = sum(interior(state.tendencies.internal_energy)[1, 1, :] .* Δz)
            G_soil = sum(φ .* interior(state.tendencies.saturation_water_ice)[1, 1, :] .* Δz)
            G_pool = interior(state.tendencies.surface_excess_water)[1, 1, 1]
            routed = interior(state.surface_excess_water)[1, 1, 1] - S⁰ - Δt * G_pool
            ΔE = sum(interior(state.internal_energy)[1, 1, :] .* Δz) - E⁰
            ΔW = column_water(integrator) - W⁰ + interior(state.surface_excess_water)[1, 1, 1] - S⁰
            energy_conserved &= isapprox(ΔE, Δt * G_energy - Cw * T_top * routed; rtol = 1.0e-9, atol = 1.0e-6 * Δt)
            water_conserved &= isapprox(ΔW, Δt * (G_soil + G_pool); rtol = 1.0e-9, atol = 1.0e-12)
        end
        @test energy_conserved
        @test water_conserved
        @test all(isfinite, interior(state.internal_energy))
        @test all(isfinite, interior(state.temperature))
        @test all(0 .<= interior(state.saturation_water_ice) .<= 1)
        if !isnothing(config.snow)
            @test all(isfinite, interior(state.snow_energy))
            @test all(isfinite, interior(state.snow_water_equivalent))
        end
    end
    # standalone SoilModel with both variables implicit
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.02, Δz_max = 0.5, N = 20))
    model = SoilModel(grid; soil = richards_soil(NF), timestepper = implicit_timestepper(NF))
    boundary_conditions = merge_boundary_conditions(InfiltrationFlux(NF(-1.0e-6 / POROSITY)), SoilHeatFlux(NF(-10)))
    integrator = initialize(model; boundary_conditions, initializers = (temperature = 1.0, saturation_water_ice = 0.6))
    @test timestepping(integrator.state, Val(:internal_energy)) === Implicit()
    @test timestepping(integrator.state, Val(:saturation_water_ice)) === Implicit()
    for _ in 1:48
        timestep!(integrator, Δt)
    end
    @test all(isfinite, interior(integrator.state.temperature))
    @test all(0 .<= interior(integrator.state.saturation_water_ice) .<= 1)
end

end # module SoilHydrologyImplicitRichardsTests
