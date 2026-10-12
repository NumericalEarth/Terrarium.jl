using Terrarium
using Test

using Terrarium: getproperties
using SpecialFunctions: erfc

import Oceananigans.BoundaryConditions: ValueBoundaryCondition, FluxBoundaryCondition, NoFluxBoundaryCondition, regularize_field_boundary_conditions

@testset "Thermal properties" begin
    thermal_props = SoilThermalProperties(Float64)

    # check that all necessary properties are defined
    @test all(map(∈(propertynames(thermal_props.conductivities)), (:water, :ice, :air, :mineral, :organic)))
    @test all(map(∈(propertynames(thermal_props.heat_capacities)), (:water, :ice, :air, :mineral, :organic)))

    # check that all default values are valid
    @test all(>(0), getproperties(thermal_props.conductivities))
    @test all(>(0), getproperties(thermal_props.heat_capacities))

    # sanity checks for bulk thermal properties
    @test Terrarium.compute_thermal_conductivity(thermal_props, SoilComposition(porosity = 1.0, saturation = 1.0, liquid = 1.0)) ≈ thermal_props.conductivities.water
    @test Terrarium.compute_thermal_conductivity(thermal_props, SoilComposition(porosity = 1.0, saturation = 1.0, liquid = 0.0)) ≈ thermal_props.conductivities.ice
    @test Terrarium.compute_thermal_conductivity(thermal_props, SoilComposition(porosity = 1.0, saturation = 0.0, liquid = 0.0)) ≈ thermal_props.conductivities.air
    # pure mineral soil: the bulk mineral conductivity is the quartz-weighted geometric mean of the
    # quartz and non-quartz endpoints, recovered at the pure-sand and pure-clay limits respectively
    @test Terrarium.compute_thermal_conductivity(thermal_props, SoilComposition(porosity = 0.0, saturation = 0.0, solid = MineralOrganic(texture = SoilTexture(:sand), organic = 0.0))) ≈ thermal_props.conductivities.quartz
    @test Terrarium.compute_thermal_conductivity(thermal_props, SoilComposition(porosity = 0.0, saturation = 0.0, solid = MineralOrganic(texture = SoilTexture(:clay), organic = 0.0))) ≈ thermal_props.conductivities.mineral
    @test Terrarium.compute_thermal_conductivity(thermal_props, SoilComposition(porosity = 0.0, saturation = 0.0, solid = MineralOrganic(organic = 1.0))) ≈ thermal_props.conductivities.organic
end

@testset "Soil energy: initialize!" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing())
    energy = SoilThermodynamics(eltype(grid))
    soil = SoilEnergyWaterCarbon(eltype(grid); energy)
    constants = PhysicalConstants(eltype(grid))
    state = StateVariables(soil, grid)
    # Test initialization with 0°C temperature
    set!(state.temperature, 0.0)
    Terrarium.initialize!(state, grid, energy, soil, constants)
    @test all(state.liquid_water_fraction .≈ 1.0)
    @test all(state.internal_energy .≈ 0.0)
    # Test initialization with positive temperature
    set!(state.temperature, 1.0) # initialize temperature to zero
    Terrarium.initialize!(state, grid, energy, soil, constants)
    @test all(state.liquid_water_fraction .≈ 1.0)
    @test all(state.internal_energy .> 0.0)
    # Test initialization with negative temperature
    set!(state.temperature, -1.0) # initialize temperature to zero
    Terrarium.initialize!(state, grid, energy, soil, constants)
    @test all(state.liquid_water_fraction .≈ 0.0)
    @test all(state.internal_energy .< 0.0)
end

@testset "Soil energy: compute_tendencies!" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    energy = SoilThermodynamics(eltype(grid))
    soil = SoilEnergyWaterCarbon(eltype(grid); energy)
    constants = PhysicalConstants(eltype(grid))
    state = StateVariables(soil, grid)
    set!(state.temperature, (x, z) -> 0.0 - 0.01 * z)
    Terrarium.initialize!(state, grid, energy, soil, constants)
    compute_tendencies!(state, grid, energy, soil)
    @test all(isfinite.(state.tendencies.internal_energy))
end

@testset "Soil energy: closure!" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    energy = SoilThermodynamics(eltype(grid))
    soil = SoilEnergyWaterCarbon(eltype(grid); energy)
    constants = PhysicalConstants(eltype(grid))
    state = StateVariables(soil, grid)
    set!(state.internal_energy, 1.0e6)
    Terrarium.closure!(state, grid, energy.closure, energy, soil, constants)
    @test all(state.temperature .> 0)
    @test all(state.liquid_water_fraction .≈ 1.0)
end

# Analytical solution to the 1D heat equation with diffusivity α
# and periodic upper boundary with mean T₀, amplitude A, and period P.
function heat_conduction_linear_periodic_ub(T₀, A, P, α)
    T(z, t) = T₀ + A * exp(-z * sqrt(π / (α * P))) * sin(2π * t / P - z * sqrt(π / (α * P)))
    return T
end

# Analytical solution to the 1D heat equation with diffusivity α
# under a step change of the upper boundary condition at steady state.
function heat_conduction_linear_step_ub(ΔTₛ, α)
    ΔT(z, t) = ΔTₛ * erfc(z / (2 * sqrt(α * t)))
    return ΔT
end

@testset "Heat diffusion with periodic upper BC" begin
    # parameters
    T₀ = 2.0
    A = 1.0
    P = 24 * 3600
    k = 2.0
    c = 1.0e6
    α = k / c
    T_sol = heat_conduction_linear_periodic_ub(T₀, A, P, α)

    # model setup
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_min = 0.05, Δz_max = 100.0, N = 100))
    # set carbon content to zero so the soil has only a mineral constituent
    biogeochem = ConstantSoilCarbonDensity(ρ_soc = 0.0)
    # set porosity to zero to remove influence of pore space;
    # this is just a hack to configure the model to simulate heat conduction in a fully solid medium
    soil_porosity = ConstantSoilPorosity(mineral_porosity = 0.0)
    # use a pure-clay (non-quartz) texture so the bulk solid conductivity equals the `mineral`
    # endpoint, isolating the analytical conduction comparison from the texture weighting
    strat = HomogeneousSoilStratigraphy(Float64; texture = SoilTexture(Float64, :clay), porosity = soil_porosity)
    # set thermal properties
    thermal_properties = SoilThermalProperties(
        eltype(grid);
        conductivities = SoilThermalConductivities(mineral = k),
        heat_capacities = SoilHeatCapacities(mineral = c),
    )
    energy = SoilThermodynamics(eltype(grid); thermal_properties)
    soil = SoilEnergyWaterCarbon(eltype(grid); energy, strat, biogeochem)
    initializer = DefaultInitializer(eltype(grid))
    model = SoilModel(grid; soil, initializer)
    # periodic upper boundary temperature
    upperbc(z, t) = T₀ + A * sin(2π * t / P)
    bcs = PrescribedSurfaceTemperature(:Tsurf, upperbc)
    # temperature initial condition
    initializers = (temperature = (x, z) -> T_sol(-z, 0.0), saturation_water_ice = 0.0)
    integrator = initialize(model; initializers, boundary_conditions = bcs)
    # TODO: Rewrite this part once we have a proper output handling system
    Ts_buf = [deepcopy(integrator.state.temperature)]
    ts = [0.0]
    Δt = 60.0
    # run for one hour, saving every time step
    while current_time(integrator) < 2 * P
        timestep!(integrator, Δt)
        push!(Ts_buf, deepcopy(integrator.state.temperature))
        push!(ts, current_time(integrator))
    end

    z_centers = znodes(integrator.state.temperature)
    Ts = reduce(hcat, Ts_buf)[1, :, :]
    Ts_target = T_sol.(reshape(-z_centers, 1, :), reshape(ts, :, 1))
    relative_error = abs.((Ts .- Ts_target) ./ Ts_target)
    @show maximum(relative_error)
    @test maximum(relative_error) < 0.1
end

@testset "Step heat diffusion" begin
    # Temperatures should be > 0 to avoid effects from phase change
    T₀ = 1.0
    T₁ = 2.0
    # model setup
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_min = 0.01, Δz_max = 100.0, N = 100))
    # temperature initial condition
    initializer = SoilInitializer(eltype(grid), energy = ConstantSoilTemperature(T₀))
    # set carbon content to zero so the soil has only a mineral constituent
    biogeochem = ConstantSoilCarbonDensity(ρ_soc = 0.0)
    # set porosity to zero to remove influence of pore space;
    # this is just a hack to configure the model to simulate heat conduction in a fully solid medium
    soil_porosity = ConstantSoilPorosity(mineral_porosity = 0.0)
    # use a pure-clay (non-quartz) texture so the bulk solid conductivity equals the `mineral`
    # endpoint, isolating the analytical conduction comparison from the texture weighting
    strat = HomogeneousSoilStratigraphy(Float64; texture = SoilTexture(Float64, :clay), porosity = soil_porosity)
    soil = SoilEnergyWaterCarbon(eltype(grid); strat, biogeochem)
    model = SoilModel(grid; soil, initializer)
    # constant upper boundary temperature set to T₁
    bcs = (temperature = (top = ValueBoundaryCondition(T₁),),)
    integrator = initialize(model, boundary_conditions = bcs)
    # TODO: Rewrite this part once we have a proper output handling system
    Ts_buf = [deepcopy(integrator.state.temperature)]
    ts = [0.0]
    Δt = 10.0
    # run for 24 hours, saving every time step
    while current_time(integrator) < 24 * 3600
        timestep!(integrator, Δt)
        push!(Ts_buf, deepcopy(integrator.state.temperature))
        push!(ts, current_time(integrator))
    end

    soil_thermal_props = soil.energy.thermal_properties
    k = soil_thermal_props.conductivities.mineral
    c = soil_thermal_props.heat_capacities.mineral
    α = k / c
    z_centers = znodes(integrator.state.temperature)
    ΔT_sol = heat_conduction_linear_step_ub(T₁ - T₀, α)
    Ts = reduce(hcat, Ts_buf)[1, :, :]
    Ts_target = T₀ .+ ΔT_sol.(reshape(-z_centers, 1, :), reshape(ts, :, 1))
    relative_error = abs.((Ts .- Ts_target) ./ Ts_target)
    @show maximum(relative_error)

    # Check error at last time step
    last_error_threshold = 1.0e-3
    @test maximum(relative_error[end, :]) < last_error_threshold

    # Check total error over all time steps
    max_error_threshold = 0.1
    @test maximum(relative_error) < max_error_threshold
end

# Implicit heat conduction (backward Euler with the Picard solver)
module SoilEnergyImplicitHeatConductionTests

using Terrarium
using Test

using Oceananigans.BoundaryConditions: GradientBoundaryCondition
using Terrarium: merge_boundary_conditions, update_state!, closure!, cell_diffusion_timescale, compute_implicit_conductivity!,
    get_energy_balance, get_hydrology, get_stratigraphy, get_biogeochemistry

module ImplicitHeatConductionTestUtils

    using Terrarium
    using Terrarium: merge_boundary_conditions, Δzᵃᵃᶜ

    # Soil column under flux boundary conditions: `top_flux` (W/m², positive upward, i.e. out of the soil)
    # and `bottom_flux` (W/m², positive upward, i.e. into the soil).
    function build_heat_column(grid, timestepper; temperature, top_flux = 0.0, bottom_flux = 0.0, saturation = 0.8, extra_boundary_conditions = (;))
        NF = eltype(grid)
        model = SoilModel(grid; timestepper)
        boundary_conditions = merge_boundary_conditions(SoilHeatFlux(NF(top_flux)), GeothermalHeatFlux(NF(bottom_flux)), extra_boundary_conditions)
        initializers = (; temperature, saturation_water_ice = NF(saturation))
        return initialize(model; boundary_conditions, initializers)
    end

    implicit_timestepper(NF; iterations = 1) =
        IMEX(ForwardEuler(NF), BackwardEuler(NF; solver = Picard(NF; iterations)))

    # vertical cell thicknesses of the (single) column
    cell_thicknesses(grid) = [Δzᵃᵃᶜ(1, 1, k, ground_domain(grid)) for k in 1:ground_domain(grid).Nz]

    # column-integrated internal energy ∑ U Δz (J/m²)
    column_energy(integrator) =
        sum(interior(integrator.state.internal_energy)[1, 1, :] .* cell_thicknesses(get_grid(integrator.model)))

    # whether temperature and liquid water fraction are given by the energy closure of the internal energy
    function closure_consistent(integrator)
        state = integrator.state
        T = copy(interior(state.temperature))
        F = copy(interior(state.liquid_water_fraction))
        Terrarium.closure!(state, integrator.model)
        return T == interior(state.temperature) && F == interior(state.liquid_water_fraction)
    end

    # equivalent frozen thickness ∑ (1 - F) Δz (m), a smooth measure of the freezing front depth
    frozen_thickness(integrator) =
        sum((1 .- interior(integrator.state.liquid_water_fraction)[1, 1, :]) .* cell_thicknesses(get_grid(integrator.model)))

    column_temperature(integrator) = copy(interior(integrator.state.temperature)[1, 1, :])

    function run_for!(integrator, Δt, duration)
        nsteps = round(Int, duration / Δt)
        @assert nsteps * Δt ≈ duration
        for _ in 1:nsteps
            timestep!(integrator, Δt)
        end
        return integrator
    end

    # time step that divides `duration` and is no larger than `Δt_max`
    dividing_dt(duration, Δt_max) = duration / ceil(duration / Δt_max)

end

using .ImplicitHeatConductionTestUtils: build_heat_column, implicit_timestepper, column_energy, frozen_thickness,
    column_temperature, run_for!, dividing_dt, closure_consistent, cell_thicknesses

@testset "Implicit heat conduction: linear consistency with dense backward Euler" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.05, Δz_max = 0.5, N = 12))
    # thawed, smooth initial profile and zero boundary fluxes: the operator is linear in U
    temperature = (x, z) -> 5.0 + 2.0 * exp(z)
    explicit = build_heat_column(grid, ForwardEuler(NF); temperature)
    Δt = 10 * cell_diffusion_timescale(explicit)
    U⁰ = copy(interior(explicit.state.internal_energy)[1, 1, :])
    Nz = length(U⁰)
    # explicit tendency G(U) of the full heat conduction operator
    function explicit_tendency(U)
        state = explicit.state
        interior(state.internal_energy)[1, 1, :] .= U
        closure!(state, explicit.model)
        update_state!(state, explicit.model, explicit.inputs; compute_tendencies = true)
        return copy(interior(state.tendencies.internal_energy)[1, 1, :])
    end
    # dense operator L by finite differences of the (linear) tendency
    G⁰ = explicit_tendency(U⁰)
    ε = 1.0e4
    L = zeros(NF, Nz, Nz)
    for j in 1:Nz
        U = copy(U⁰)
        U[j] += ε
        L[:, j] .= (explicit_tendency(U) .- G⁰) ./ ε
    end
    @test maximum(abs, G⁰ .- L * U⁰) < 1.0e-6 * maximum(abs, G⁰)
    # backward Euler (I - Δt L) Uⁿ⁺¹ = Uⁿ
    A = -Δt .* L
    for k in 1:Nz
        A[k, k] += 1
    end
    U_dense = A \ U⁰
    # under implicit routing, the explicit tendency holds only the (here zero) boundary fluxes
    integrator = build_heat_column(grid, implicit_timestepper(NF); temperature)
    update_state!(integrator.state, integrator.model, integrator.inputs; compute_tendencies = true)
    @test all(iszero, interior(integrator.state.tendencies.internal_energy))
    for iterations in (1, 2)
        integrator = build_heat_column(grid, implicit_timestepper(NF; iterations); temperature)
        @test interior(integrator.state.internal_energy)[1, 1, :] ≈ U⁰
        timestep!(integrator, Δt)
        U_implicit = interior(integrator.state.internal_energy)[1, 1, :]
        @test maximum(abs, U_implicit .- U_dense) / maximum(abs, U_dense .- U⁰) < 1.0e-8
    end
end

@testset "Implicit heat conduction: energy conservation with flux boundary conditions" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.02, Δz_max = 0.5, N = 20))
    Δt = 3600.0
    bottom_flux = 0.05
    # (initial temperature, top flux) for a thawed column and for a column with a freezing front
    cases = (thawed = (5.0, -20.0), freezing = (0.5, 80.0))
    for (case, (T₀, top_flux)) in pairs(cases), iterations in (1, 3)
        integrator = build_heat_column(grid, implicit_timestepper(NF; iterations); temperature = T₀, top_flux, bottom_flux)
        net_flux = bottom_flux - top_flux
        conserved = true
        for _ in 1:48
            E⁰ = column_energy(integrator)
            timestep!(integrator, Δt)
            ΔE = column_energy(integrator) - E⁰
            conserved &= isapprox(ΔE, Δt * net_flux; rtol = 1.0e-9)
        end
        @test conserved
        @test all(isfinite, interior(integrator.state.temperature))
        # temperature and liquid water fraction on exit are the closure of the new internal energy
        @test closure_consistent(integrator)
        if case == :freezing
            # the freezing front has entered the column
            @test minimum(interior(integrator.state.liquid_water_fraction)) < 1
        end
    end
end

@testset "Implicit heat conduction: stability at 100× the explicit limit" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.01, N = 50))
    T_min, T_max = 1.0, 5.0
    temperature = (x, z) -> z > -0.25 ? T_min : T_max
    τ = cell_diffusion_timescale(build_heat_column(grid, ForwardEuler(NF); temperature))
    Δt = 100 * τ
    integrator = build_heat_column(grid, implicit_timestepper(NF); temperature)
    for _ in 1:200
        timestep!(integrator, Δt)
    end
    T = interior(integrator.state.temperature)
    @test all(isfinite, T)
    # backward Euler with an M-matrix satisfies a discrete maximum principle
    @test all(T_min - 1.0e-8 .<= T .<= T_max + 1.0e-8)
    # the initial step has diffused substantially
    @test maximum(T) - minimum(T) < 0.5 * (T_max - T_min)
end

@testset "Implicit heat conduction: first-order accuracy for a step in the top flux" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.02, N = 50))
    duration = 86400.0
    kwargs = (temperature = 10.0, top_flux = 20.0)
    reference = build_heat_column(grid, ForwardEuler(NF); kwargs...)
    Δt_reference = dividing_dt(duration, cell_diffusion_timescale(reference) / 10)
    T_reference = column_temperature(run_for!(reference, Δt_reference, duration))
    errors = map((3600.0, 1800.0)) do Δt
        integrator = run_for!(build_heat_column(grid, implicit_timestepper(NF); kwargs...), Δt, duration)
        maximum(abs, column_temperature(integrator) .- T_reference)
    end
    @info "Implicit heat conduction step-flux errors (K) at Δt = 3600 s, 1800 s" errors
    # the surface has cooled by about 1 K, so 0.05 K is a 5% error of the response
    @test errors[1] < 0.05
    # first-order convergence in Δt
    @test 1.5 < errors[1] / errors[2] < 2.5
end

@testset "Implicit heat conduction: freezing front with FreeWater" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.02, N = 50))
    duration = 4 * 86400.0
    kwargs = (temperature = 1.0, top_flux = 60.0)
    reference = build_heat_column(grid, ForwardEuler(NF); kwargs...)
    Δt_reference = dividing_dt(duration, cell_diffusion_timescale(reference) / 10)
    run_for!(reference, Δt_reference, duration)
    d_reference = frozen_thickness(reference)
    T_reference = column_temperature(reference)
    @test d_reference > 0.1
    # tolerances on the maximum temperature error (K) for M = 1 and M = 2 Picard iterations; the
    # lagged apparent heat capacity of the linearly implicit step (M = 1) overestimates the cooling of
    # cells that start to freeze within a step, which further iterations largely remove
    temperature_tolerances = (0.35, 0.03)
    for (iterations, temperature_tolerance) in zip((1, 2), temperature_tolerances)
        integrator = run_for!(build_heat_column(grid, implicit_timestepper(NF; iterations); kwargs...), 3600.0, duration)
        d = frozen_thickness(integrator)
        relative_error = abs(d - d_reference) / d_reference
        temperature_error = maximum(abs, column_temperature(integrator) .- T_reference)
        @info "Implicit heat conduction freezing front (M = $iterations)" d d_reference relative_error temperature_error
        @test relative_error < 0.01
        @test temperature_error < temperature_tolerance
    end
end

@testset "Implicit heat conduction: unsupported boundary conditions" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.1, N = 10))
    for boundary_conditions in (PrescribedSurfaceTemperature(:T_ub, NF(1)), PrescribedBottomTemperature(:T_lb, NF(1)))
        model = SoilModel(grid; timestepper = implicit_timestepper(NF))
        @test_throws ArgumentError initialize(model; boundary_conditions)
        # the same boundary conditions are fine under explicit time stepping
        @test initialize(SoilModel(grid); boundary_conditions) isa ModelIntegrator
    end
    # gradient boundary conditions on temperature are supported
    gradient_bc = (temperature = (bottom = GradientBoundaryCondition(NF(-0.03)),),)
    model = SoilModel(grid; timestepper = implicit_timestepper(NF))
    @test initialize(model; boundary_conditions = gradient_bc) isa ModelIntegrator
end

@testset "Implicit heat conduction: energy conservation with a gradient boundary condition" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.02, Δz_max = 0.5, N = 20))
    Δt = 3600.0
    top_flux = 20.0
    # ∂T/∂z (K/m) at the bottom boundary; z is positive upward, so T increases with depth and the implied
    # diffusive flux -κ ∂T/∂z through the bottom face is positive (upward, i.e. into the column)
    gradient = -0.03
    extra_boundary_conditions = (temperature = (bottom = GradientBoundaryCondition(NF(gradient)),),)
    for iterations in (1, 2)
        integrator = build_heat_column(grid, implicit_timestepper(NF; iterations); temperature = 5.0, top_flux, extra_boundary_conditions)
        (; state, model) = integrator
        soil = model.soil
        energy, hydrology = get_energy_balance(soil), get_hydrology(soil)
        strat, bgc = get_stratigraphy(soil), get_biogeochemistry(soil)
        conserved = true
        bottom_fluxes = Float64[]
        for _ in 1:24
            E⁰ = column_energy(integrator)
            # flux implied by the gradient condition at tⁿ, with the thermal conductivity at the bottom face
            # from the same kernel function (face interpolation of `compute_thermal_conductivity`)
            fields = get_fields(state, energy, hydrology, strat, bgc)
            κ = Field{Center, Center, Face}(ground_domain(get_grid(model)))
            compute_implicit_conductivity!(κ, 1, 1, 1, get_grid(model), fields, energy, hydrology, strat, bgc)
            κ_bottom = κ[1, 1, 1]
            bottom_flux = -κ_bottom * gradient
            push!(bottom_fluxes, bottom_flux)
            timestep!(integrator, Δt)
            ΔE = column_energy(integrator) - E⁰
            conserved &= isapprox(ΔE, Δt * (bottom_flux - top_flux); rtol = 1.0e-9)
        end
        @test conserved
        @test all(>(0), bottom_fluxes)
        @test all(isfinite, interior(state.temperature))
    end
end

# Coupled LandModel configurations (cf. test/coupled_models/land_model_tests.jl) with the default
# `NoFlow` soil hydrology, so that the soil column energy changes only through the soil heat flux at the
# top and the geothermal heat flux at the bottom.
@testset "Implicit heat conduction: LandModel with and without snow" begin
    NF = Float64
    Δt = 1800.0
    geothermal_flux = 0.05
    configurations = (
        bare = (
            snow = nothing,
            initializers = (temperature = (x, z) -> 5.0 - 0.02 * z, saturation_water_ice = 0.8),
        ),
        snow = (
            snow = SingleLayerSnow(NF),
            initializers = (
                temperature = (x, z) -> -1.0 - 0.02 * z,
                saturation_water_ice = 0.8,
                snow_water_equivalent = 0.2,
                snow_temperature = -5.0,
            ),
        ),
    )
    for (name, config) in pairs(configurations)
        grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_max = 1.0, N = 30))
        land = LandModel(grid; snow = config.snow, vegetation = nothing, timestepper = implicit_timestepper(NF))
        @test get_hydrology(land.soil).vertical_flow isa NoFlow
        integrator = initialize(land; initializers = config.initializers, boundary_conditions = GeothermalHeatFlux(NF(geothermal_flux)))
        state = integrator.state
        @test timestepping(state, Val(:internal_energy)) === Implicit()
        conserved = true
        for _ in 1:24
            E⁰ = column_energy(integrator)
            timestep!(integrator, Δt)
            # `soil_heat_flux` (positive upward) still holds the value at tⁿ that entered the step
            soil_heat_flux = interior(state.soil_heat_flux)[1, 1, 1]
            ΔE = column_energy(integrator) - E⁰
            conserved &= isapprox(ΔE, Δt * (geothermal_flux - soil_heat_flux); rtol = 1.0e-9, atol = 1.0e-6 * Δt)
        end
        @test conserved
        @test all(isfinite, interior(state.internal_energy))
        @test all(isfinite, interior(state.temperature))
        @test all(isfinite, interior(state.soil_heat_flux))
        if !isnothing(config.snow)
            @test all(isfinite, interior(state.snow_energy))
            @test all(isfinite, interior(state.snow_water_equivalent))
        end
    end
end

end # module SoilEnergyImplicitHeatConductionTests

# Advection of heat by the soil water flow, explicit and implicit
module SoilEnergyAdvectionTests

using Terrarium
using Test

using Oceananigans.BoundaryConditions: GradientBoundaryCondition
using Terrarium: merge_boundary_conditions, Δzᵃᵃᶜ, get_hydrology, get_stratigraphy, get_biogeochemistry, get_energy_balance,
    advective_heat_flux, compute_heat_capacity, soil_composition

module EnergyAdvectionTestUtils

    using Terrarium
    using Terrarium: merge_boundary_conditions, Δzᵃᵃᶜ, get_hydrology, get_stratigraphy, get_biogeochemistry, get_energy_balance

    const POROSITY = 0.4

    # `air = false` sets the heat capacity of air to zero, so that water replacing air in the pores carries
    # exactly the heat that it adds to the bulk heat capacity (the heat of the displaced air is not tracked)
    function richards_soil(NF; saturated_conductivity = 1.0e-4, air = true)
        hydraulic_properties = ConstantSoilHydraulics(
            NF; swrc = VanGenuchten(α = 2.0, n = 2.0), saturated_conductivity, unsat_hydraulic_cond = UnsatKVanGenuchten(NF)
        )
        hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
        strat = HomogeneousSoilStratigraphy(NF; porosity = ConstantSoilPorosity(NF; mineral_porosity = POROSITY))
        biogeochem = ConstantSoilCarbonDensity(NF; ρ_soc = 0.0)
        heat_capacities = air ? SoilHeatCapacities(NF) : SoilHeatCapacities(NF; air = 0.0)
        energy = SoilThermodynamics(NF; thermal_properties = SoilThermalProperties(NF; heat_capacities))
        return SoilEnergyWaterCarbon(NF; energy, hydrology, strat, biogeochem)
    end

    implicit_timestepper(NF; iterations = 2) =
        IMEX(ForwardEuler(NF), BackwardEuler(NF; solver = Picard(NF; iterations)))

    # gravity drainage at the bottom (unit gradient of the total head)
    gravity_drainage(NF) = (pressure_head = (bottom = GradientBoundaryCondition(NF(1)),),)

    function build_column(grid, timestepper; temperature, saturation, infiltration = 0.0, extra_boundary_conditions = (;), soil = richards_soil(eltype(grid)))
        NF = eltype(grid)
        model = SoilModel(grid; soil, timestepper)
        boundary_conditions = merge_boundary_conditions(InfiltrationFlux(NF(-infiltration / POROSITY)), extra_boundary_conditions)
        integrator = initialize(model; boundary_conditions, initializers = (; temperature, saturation_water_ice = saturation))
        # the hydraulic conductivity depends on the liquid water fraction, which is set by the energy closure
        # after the hydrology is initialized
        compute_auxiliary!(integrator.state, model)
        return integrator
    end

    # steady downward flow through a uniformly saturated column: infiltration equal to the face
    # conductivity and gravity drainage at the bottom
    function build_steady_flow_column(grid, timestepper; temperature, saturation = 0.7)
        NF = eltype(grid)
        probe = build_column(grid, ForwardEuler(NF); temperature, saturation)
        K = interior(probe.state.hydraulic_conductivity)[1, 1, 2]
        integrator = build_column(grid, timestepper; temperature, saturation, infiltration = K, extra_boundary_conditions = gravity_drainage(NF))
        return integrator, K
    end

    cell_thicknesses(grid) = [Δzᵃᵃᶜ(1, 1, k, ground_domain(grid)) for k in 1:ground_domain(grid).Nz]

    column_energy(integrator) = sum(interior(integrator.state.internal_energy)[1, 1, :] .* cell_thicknesses(get_grid(integrator.model)))

    column_temperature(integrator) = copy(interior(integrator.state.temperature)[1, 1, :])

    # bulk volumetric heat capacity of every cell
    function heat_capacities(integrator)
        (; state, model) = integrator
        soil = model.soil
        energy, hydrology, strat, bgc = get_energy_balance(soil), get_hydrology(soil), get_stratigraphy(soil), get_biogeochemistry(soil)
        fields = get_fields(state, energy, hydrology, strat, bgc)
        grid = get_grid(model)
        return map(1:ground_domain(grid).Nz) do k
            composition = Terrarium.soil_composition(1, 1, k, grid, fields, strat, hydrology, bgc)
            Terrarium.compute_heat_capacity(energy.thermal_properties, composition)
        end
    end

end

using .EnergyAdvectionTestUtils: richards_soil, implicit_timestepper, gravity_drainage, build_column, build_steady_flow_column,
    cell_thicknesses, column_energy, column_temperature, heat_capacities, POROSITY

@testset "Energy advection: advective heat flux" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.1, N = 10))
    # immobile water carries no heat
    integrator = initialize(SoilModel(grid); initializers = (temperature = 5.0, saturation_water_ice = 0.5))
    soil = integrator.model.soil
    energy, hydrology = get_energy_balance(soil), get_hydrology(soil)
    fields = get_fields(integrator.state, energy, hydrology)
    @test advective_heat_flux(1, 1, 5, grid, fields, energy, hydrology) == 0
    @test Terrarium.cell_advection_timescale(integrator) == Inf
    # gravity-driven downward flow (q < 0) carries the temperature of the upper (upwind) cell
    integrator = build_column(grid, ForwardEuler(NF); temperature = (x, z) -> 10.0 + 10 * z, saturation = 0.7)
    (; state, model) = integrator
    # the advective heat flux uses the water flux of the step, which is computed with the tendencies
    Terrarium.update_state!(state, model, integrator.inputs; compute_tendencies = true)
    soil = model.soil
    energy, hydrology, strat, bgc = get_energy_balance(soil), get_hydrology(soil), get_stratigraphy(soil), get_biogeochemistry(soil)
    fields = get_fields(state, energy, hydrology, strat, bgc)
    k = 5
    q = Terrarium.darcy_flux(1, 1, k, ground_domain(grid), state.pressure_head, state.hydraulic_conductivity)
    @test q < 0
    Cw = energy.thermal_properties.heat_capacities.water
    T = interior(state.temperature)[1, 1, :]
    @test advective_heat_flux(1, 1, k, grid, fields, energy, hydrology, strat, bgc) ≈ Cw * T[k] * q
    # the advective timescale is that of the thermal front, C Δz / (C_w |q|)
    τ = Terrarium.cell_advection_timescale(integrator)
    @test isfinite(τ)
    @test τ > 0
    # the water flux of the step is the Darcy flux at tⁿ, stored with the tendencies; no flux through the
    # closed boundary faces
    @test interior(state.water_flux)[1, 1, k] == q
    @test interior(state.water_flux)[1, 1, 1] == 0
    @test interior(state.water_flux)[1, 1, end] == 0
    # implicit advection of heat imposes no advective restriction
    implicit = build_column(grid, implicit_timestepper(NF); temperature = (x, z) -> 10.0 + 10 * z, saturation = 0.7)
    @test Terrarium.cell_advection_timescale(implicit) == Inf
end

@testset "Energy advection: uniform temperature under steady flow ($name)" for (name, timestepper) in
    (("explicit", ForwardEuler(Float64)), ("implicit", implicit_timestepper(Float64)))
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.05, N = 20))
    T₀ = 7.0
    integrator, K = build_steady_flow_column(grid, timestepper; temperature = T₀)
    @test K > 0
    s₀ = copy(interior(integrator.state.saturation_water_ice))
    # explicit advection of heat is subject to the advective CFL limit, implicit advection is not; larger implicit
    # steps are limited by the explicit drainage flux at tⁿ, which then overdraws the bottom cell
    probe = build_steady_flow_column(grid, ForwardEuler(NF); temperature = T₀)[1]
    τ_advection = Terrarium.cell_advection_timescale(probe)
    Δt = name == "explicit" ? min(Terrarium.cell_diffusion_timescale(integrator), τ_advection) / 2 : 2 * τ_advection
    @test name == "explicit" || Δt > 10 * Terrarium.cell_diffusion_timescale(probe)
    for _ in 1:48
        timestep!(integrator, Δt)
    end
    T = column_temperature(integrator)
    # water flows through the column at constant saturation and carries its heat along
    @test maximum(abs, interior(integrator.state.saturation_water_ice) .- s₀) < 1.0e-10
    @test maximum(abs, T .- T₀) < 1.0e-10
end

@testset "Energy advection: energy conservation of warm infiltration ($name)" for (name, timestepper) in
    (("explicit", ForwardEuler(Float64)), ("implicit", implicit_timestepper(Float64)))
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.02, Δz_max = 0.2, N = 20))
    infiltration = 5.0e-6
    # warm surface layer over a cold column, impermeable and insulated bottom, no conductive flux at the top
    integrator = build_column(grid, timestepper; temperature = (x, z) -> z > -0.1 ? 15.0 : 2.0, saturation = 0.5, infiltration)
    Cw = get_energy_balance(integrator.model.soil).thermal_properties.heat_capacities.water
    Δt = name == "explicit" ? Terrarium.cell_diffusion_timescale(integrator) / 2 : 1800.0
    conserved = true
    for _ in 1:24
        E⁰ = column_energy(integrator)
        T_top = column_temperature(integrator)[end]
        timestep!(integrator, Δt)
        # the infiltrating water carries the temperature of the top cell (upwind) at tⁿ
        conserved &= isapprox(column_energy(integrator) - E⁰, Δt * Cw * T_top * infiltration; rtol = 1.0e-9)
    end
    @test conserved
    @test all(isfinite, interior(integrator.state.temperature))
end

@testset "Energy advection: thermal front velocity" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.02, N = 100))
    T_b, ΔT = 2.0, 5.0
    temperature = (x, z) -> T_b + ΔT * exp(-((z + 0.5) / 0.1)^2)
    z = collect(znodes(ground_domain(grid), Center()))
    centroid(integrator) = (T = column_temperature(integrator) .- T_b; sum(z .* T) / sum(T))
    # about 0.6 m of front displacement, so that the pulse stays well inside the column
    duration = 4 * 3600.0
    # explicit reference within the advective and diffusive limits
    reference, K = build_steady_flow_column(grid, ForwardEuler(NF); temperature)
    τ_advection = Terrarium.cell_advection_timescale(reference)
    Δt_reference = duration / ceil(Int, duration / (min(τ_advection, Terrarium.cell_diffusion_timescale(reference)) / 2))
    # implicit upwind advection at advective Courant numbers well above one
    integrator, _ = build_steady_flow_column(grid, implicit_timestepper(NF); temperature)
    Δt = duration / floor(Int, duration / (5 * τ_advection))
    C = heat_capacities(integrator)
    Cw = get_energy_balance(integrator.model.soil).thermal_properties.heat_capacities.water
    θ = POROSITY * 0.7
    z⁰ = centroid(integrator)
    for (run, step) in ((reference, Δt_reference), (integrator, Δt))
        for _ in 1:round(Int, duration / step)
            timestep!(run, step)
        end
    end
    velocity_reference = (z⁰ - centroid(reference)) / duration
    velocity = (z⁰ - centroid(integrator)) / duration
    thermal_velocity = Cw * K / C[50]
    pore_velocity = K / θ
    profile_error = maximum(abs, column_temperature(integrator) .- column_temperature(reference)) / ΔT
    @info "Energy advection thermal front" Δt / τ_advection velocity velocity_reference thermal_velocity pore_velocity profile_error
    # the front moves downward at the advective velocity of heat, C_w q / C, not at the pore velocity q / θ
    @test velocity_reference ≈ thermal_velocity rtol = 0.05
    @test velocity ≈ velocity_reference rtol = 0.05
    # implicit upwind advection is more diffusive than the explicit reference, but stays close to it
    @test profile_error < 0.15
end

@testset "Energy advection: freeze–thaw infiltration with both variables implicit" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.02, N = 50))
    infiltration = 2.0e-8
    # warm thawed surface layer over soil just below the freezing point
    integrator = build_column(
        grid, implicit_timestepper(NF); temperature = (x, z) -> z > -0.1 ? 5.0 : -0.5, saturation = 0.6, infiltration,
        soil = richards_soil(NF; saturated_conductivity = 1.0e-5)
    )
    Cw = get_energy_balance(integrator.model.soil).thermal_properties.heat_capacities.water
    Δt = 3600.0
    energy_conserved = true
    water_conserved = true
    bounded = true
    W(integrator) = sum(POROSITY .* interior(integrator.state.saturation_water_ice)[1, 1, :] .* cell_thicknesses(grid))
    for _ in 1:96
        E⁰ = column_energy(integrator)
        W⁰ = W(integrator)
        T_top = column_temperature(integrator)[end]
        timestep!(integrator, Δt)
        # the column energy is dominated by the latent heat of the frozen soil, so allow for its roundoff
        energy_conserved &= isapprox(column_energy(integrator) - E⁰, Δt * Cw * T_top * infiltration; rtol = 1.0e-9, atol = 1.0e-13 * abs(E⁰))
        water_conserved &= isapprox(W(integrator) - W⁰, Δt * infiltration; rtol = 1.0e-9)
        s = interior(integrator.state.saturation_water_ice)
        bounded &= all(isfinite, interior(integrator.state.temperature)) && all(0 .<= s .<= 1)
    end
    F = interior(integrator.state.liquid_water_fraction)[1, 1, :]
    @info "Energy advection freeze–thaw infiltration" extrema(column_temperature(integrator)) extrema(F)
    @test energy_conserved
    @test water_conserved
    @test bounded
end

# Columns of 50 cells of 2 cm with van Genuchten soils: a sandy column with infiltration (a wetting front),
# with and without gravity drainage, and a nearly saturated column with infiltration that is filled at once.
const ADVECTION_COLUMNS = (
    sandy_drainage = (saturated_conductivity = 1.0e-4, saturation = 0.2, infiltration = 2.0e-5, drainage = true),
    sandy_closed = (saturated_conductivity = 1.0e-4, saturation = 0.2, infiltration = 2.0e-5, drainage = false),
    wet_closed = (saturated_conductivity = 1.0e-5, saturation = 0.95, infiltration = 5.0e-6, drainage = false),
)

function build_advection_column(column, timestepper; temperature, air = true)
    NF = Float64
    grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.02, N = 50))
    soil = richards_soil(NF; column.saturated_conductivity, air)
    extra_boundary_conditions = column.drainage ? gravity_drainage(NF) : (;)
    return build_column(grid, timestepper; temperature, column.saturation, column.infiltration, extra_boundary_conditions, soil)
end

@testset "Energy advection: uniform temperature with moving water ($name)" for (name, column) in pairs(ADVECTION_COLUMNS)
    NF = Float64
    T₀ = 5.0
    # with the heat capacity of air set to zero, heat moves exactly with the water: infiltration, wetting
    # fronts, drainage, and the redistribution and removal of excess water leave a uniform temperature unchanged
    runs = (
        (ForwardEuler(NF), 1.0, 3 * 3600.0),
        ((implicit_timestepper(NF), Δt, 86400.0) for Δt in (300.0, 1800.0, 3600.0, 10800.0))...,
    )
    for (timestepper, Δt, duration) in runs
        integrator = build_advection_column(column, timestepper; temperature = T₀, air = false)
        deviation = 0.0
        for _ in 1:round(Int, duration / Δt)
            timestep!(integrator, Δt)
            deviation = max(deviation, maximum(abs, column_temperature(integrator) .- T₀))
        end
        s = interior(integrator.state.saturation_water_ice)
        @info "Energy advection uniform temperature ($name)" nameof(typeof(timestepper)) Δt deviation extrema(s)
        @test deviation < 1.0e-6
    end
end

@testset "Energy advection: energy budget with infiltration, drainage, and surface excess ($name)" for (name, column) in pairs(ADVECTION_COLUMNS)
    NF = Float64
    temperature = (x, z) -> 5.0 + 3.0 * cos(8z)
    for (timestepper, Δt, nsteps) in ((ForwardEuler(NF), 1.0, 3 * 3600), (implicit_timestepper(NF), 1800.0, 48))
        integrator = build_advection_column(column, timestepper; temperature)
        (; state) = integrator
        grid = get_grid(integrator.model)
        Δz = cell_thicknesses(grid)
        Cw = get_energy_balance(integrator.model.soil).thermal_properties.heat_capacities.water
        implicit = timestepping(state, Val(:internal_energy)) === Implicit()
        water(integrator) = sum(POROSITY .* interior(integrator.state.saturation_water_ice)[1, 1, :] .* Δz)
        closed = true
        excess = 0.0
        for _ in 1:nsteps
            E⁰ = column_energy(integrator)
            W⁰ = water(integrator)
            T_top = column_temperature(integrator)[end]
            timestep!(integrator, Δt)
            # explicit tendencies at tⁿ: boundary heat and water fluxes, including infiltration and, under explicit
            # time stepping of the energy, drainage
            G_energy = sum(interior(state.tendencies.internal_energy)[1, 1, :] .* Δz)
            G_water = sum(POROSITY .* interior(state.tendencies.saturation_water_ice)[1, 1, :] .* Δz)
            # excess water removed at the surface, with the heat of the top cell at tⁿ
            removed = Δt * G_water - (water(integrator) - W⁰)
            # under implicit time stepping of the energy, the heat of the drainage water leaves at the temperature
            # of the bottom cell at tⁿ⁺¹
            q_bottom = interior(state.water_flux)[1, 1, 1]
            drained_heat = implicit ? Δt * Cw * min(q_bottom, 0.0) * column_temperature(integrator)[1] : 0.0
            ΔE = column_energy(integrator) - E⁰
            closed &= isapprox(ΔE, Δt * G_energy + drained_heat - Cw * T_top * removed; rtol = 1.0e-9, atol = 1.0e-9 * abs(E⁰))
            excess += removed
        end
        @info "Energy advection budget ($name)" nameof(typeof(timestepper)) excess
        @test closed
        # the closed columns fill and shed excess water within the day of the implicit runs
        @test column.drainage || !implicit || excess > 0
    end
end

@testset "Energy advection: sandy column with drainage stays bounded at large time steps" begin
    NF = Float64
    temperature = (x, z) -> 5.0 + 3.0 * cos(8z)
    # at much larger steps, the explicit drainage flux at tⁿ empties the column in a single step (see the known
    # limitations of the implicit Richards step), and the maximum principle is no longer guaranteed
    for Δt in (1800.0, 3600.0)
        integrator = build_advection_column(ADVECTION_COLUMNS.sandy_drainage, implicit_timestepper(NF); temperature)
        T_min, T_max = extrema(column_temperature(integrator))
        bounded = true
        for _ in 1:round(Int, 5 * 86400.0 / Δt)
            timestep!(integrator, Δt)
            T = column_temperature(integrator)
            # discrete maximum principle, up to the small effect of the heat of the air displaced by the water
            bounded &= all(isfinite, T) && T_min - 1.0e-2 <= minimum(T) && maximum(T) <= T_max + 1.0e-2
        end
        @test bounded
    end
end

end # module SoilEnergyAdvectionTests
