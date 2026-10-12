using Terrarium
using Test

import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.Diagnostics: cell_diffusion_timescale

# Recompute the thermal diffusion timescale independently from the public pointwise property
# functions, mirroring τ = Δz² C / κ, and return the per-cell minimum over the interior.
function reference_thermal_timescale(integrator)
    model = integrator.model
    soil = model.soil
    state = integrator.state
    energy = Terrarium.get_energy_balance(soil)
    hydrology = Terrarium.get_hydrology(soil)
    strat = Terrarium.get_stratigraphy(soil)
    bgc = Terrarium.get_biogeochemistry(soil)
    ground_grid = ground_domain(model.grid)
    fields = Terrarium.get_fields(state, energy, hydrology, strat, bgc)
    Nx, Ny, Nz = size(ground_grid)
    τmin = Inf
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        soilvol = Terrarium.soil_composition(i, j, k, ground_grid, fields, strat, hydrology, bgc)
        κ = Terrarium.compute_thermal_conductivity(energy.thermal_properties, soilvol)
        C = Terrarium.compute_heat_capacity(energy.thermal_properties, soilvol)
        Δz = Terrarium.Δzᵃᵃᶜ(i, j, k, ground_grid)
        τmin = min(τmin, Δz^2 * C / κ)
    end
    return τmin
end

@testset "cell_diffusion_timescale (SoilModel, NoFlow)" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 20))
    model = SoilModel(grid; timestepper = ForwardEuler(Float64))
    integrator = initialize(model)

    τ = cell_diffusion_timescale(integrator)
    @test isfinite(τ)
    @test τ > 0
    @test typeof(τ) === Float64

    # NoFlow hydrology imposes no restriction, so the model timescale is the thermal one
    @test τ ≈ reference_thermal_timescale(integrator)

    # implied thermal diffusivity α = Δz²/τ should be a physically plausible soil value
    α = 0.1^2 / τ
    @test 1.0e-8 < α < 1.0e-4

    # land models have no advective restriction
    @test cell_advection_timescale(integrator) == Inf
end

@testset "cell_diffusion_timescale (SoilModel, RichardsEq)" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 20))
    hydrology = SoilHydrology(Float64; vertical_flow = RichardsEq())
    soil = Terrarium.SoilEnergyWaterCarbon(Float64; hydrology)
    model = SoilModel(grid; soil, timestepper = ForwardEuler(Float64))
    integrator = initialize(model)

    τ_energy = cell_diffusion_timescale(integrator.state, grid, Terrarium.get_energy_balance(soil), soil, model.constants)
    τ_water = cell_diffusion_timescale(integrator.state, grid, Terrarium.get_hydrology(soil), soil, model.constants)
    @test isfinite(τ_energy) && τ_energy > 0
    @test isfinite(τ_water) && τ_water > 0

    # the coupled timescale is the minimum of the two process timescales
    τ = cell_diffusion_timescale(integrator)
    @test τ ≈ min(τ_energy, τ_water)
end

@testset "single-float32 numeric type" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 10))
    model = SoilModel(grid; timestepper = ForwardEuler(Float32))
    integrator = initialize(model)
    τ = cell_diffusion_timescale(integrator)
    @test typeof(τ) === Float32
    @test isfinite(τ) && τ > 0
end

@testset "TimeStepWizard drives an adaptive Simulation" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 20))
    integrator = initialize(SoilModel(grid; timestepper = ForwardEuler(Float64)))
    sim = Simulation(integrator; Δt = 100.0, stop_iteration = 20)
    wizard = TimeStepWizard(diffusive_cfl = 0.4)
    sim.callbacks[:wizard] = Callback(wizard, IterationInterval(2))
    run!(sim)

    @test all(isfinite.(integrator.state.temperature))
    @test isfinite(sim.Δt) && sim.Δt > 0
    # the diffusive-CFL target step is bounded by diffusive_cfl * τ
    @test sim.Δt <= 0.4 * cell_diffusion_timescale(integrator) + 1
end

# The `TimeStepWizard` under IMEX time stepping: implicit terms impose no limit, the remaining explicit terms do
module IMEXWizardTests

    using Terrarium
    using Test

    using Oceananigans.BoundaryConditions: GradientBoundaryCondition
    using Terrarium: AbstractIMEX, AbstractTimeStepper, AbstractVariable, Explicit, Implicit, merge_boundary_conditions,
        get_energy_balance, get_hydrology, get_stratigraphy, get_biogeochemistry, cell_advection_timescale, cell_diffusion_timescale

    const POROSITY = 0.4

    # IMEX that integrates `saturation_water_ice` implicitly but `internal_energy` explicitly
    struct ExplicitEnergyIMEX{NF, E <: AbstractTimeStepper{NF}, I <: AbstractTimeStepper{NF}} <: AbstractIMEX{NF}
        explicit::E
        implicit::I
    end

    Terrarium.timestepping(::AbstractVariable{:internal_energy}, ::SoilModel, ::ExplicitEnergyIMEX) = Explicit()

    implicit_solver(NF) = BackwardEuler(NF; solver = Picard(NF; iterations = 2))

    implicit_timestepper(NF) = IMEX(ForwardEuler(NF), implicit_solver(NF))

    function richards_soil(NF; saturated_conductivity = 1.0e-5, swrc = VanGenuchten(α = 2.0, n = 2.0))
        hydraulic_properties = ConstantSoilHydraulics(NF; swrc, saturated_conductivity, unsat_hydraulic_cond = UnsatKVanGenuchten(NF))
        hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
        strat = HomogeneousSoilStratigraphy(NF; porosity = ConstantSoilPorosity(NF; mineral_porosity = POROSITY))
        biogeochem = ConstantSoilCarbonDensity(NF; ρ_soc = 0.0)
        return SoilEnergyWaterCarbon(NF; hydrology, strat, biogeochem)
    end

    sandy_soil(NF) = richards_soil(NF; saturated_conductivity = 1.0e-4, swrc = VanGenuchten(α = 3.5, n = 3.0))

    # sandy column with infiltration (m/s, positive downward) at the top and, if `drainage`, gravity drainage at the bottom
    function sandy_column(grid, timestepper; infiltration = 2.0e-5, saturation = 0.6, drainage = true)
        NF = eltype(grid)
        bottom = drainage ? (pressure_head = (bottom = GradientBoundaryCondition(NF(1)),),) : (;)
        boundary_conditions = merge_boundary_conditions(InfiltrationFlux(NF(-infiltration / POROSITY)), bottom)
        model = SoilModel(grid; soil = sandy_soil(NF), timestepper)
        return initialize(model; boundary_conditions, initializers = (temperature = 5.0, saturation_water_ice = saturation))
    end

    # run a `Simulation` for `stop_iteration` steps with the `wizard` called at every iteration
    function run_with_wizard!(integrator; Δt, wizard, stop_iteration)
        sim = Simulation(integrator; Δt, stop_iteration)
        sim.callbacks[:wizard] = Callback(wizard, IterationInterval(1))
        run!(sim)
        return sim
    end

    @testset "TimeStepWizard is not throttled by implicit heat conduction (SoilModel)" begin
        NF = Float64
        grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.02, Δz_max = 0.5, N = 20))
        τ_explicit = cell_diffusion_timescale(initialize(SoilModel(grid; timestepper = ForwardEuler(NF))))
        integrator = initialize(SoilModel(grid; timestepper = implicit_timestepper(NF)))
        @test cell_diffusion_timescale(integrator) == Inf
        @test cell_advection_timescale(integrator) == Inf
        max_Δt = 6 * 3600.0
        wizard = TimeStepWizard(diffusive_cfl = 0.5, max_change = 1.5, max_Δt = max_Δt)
        sim = run_with_wizard!(integrator; Δt = 60.0, wizard, stop_iteration = 30)
        @info "IMEX SoilModel wizard" sim.Δt τ_explicit sim.Δt / τ_explicit
        # only `max_Δt` limits the step, far beyond the explicit diffusive limit
        @test sim.Δt == max_Δt
        @test sim.Δt > 10 * τ_explicit
        @test all(isfinite, interior(integrator.state.temperature))
    end

    @testset "TimeStepWizard is not throttled by implicit soil terms (LandModel)" begin
        NF = Float64
        grid = ColumnGrid(CPU(), NF, ExponentialSpacing(Δz_min = 0.02, Δz_max = 1.0, N = 30))
        initializers = (
            temperature = (x, z) -> 5.0 - 0.02 * z,
            saturation_water_ice = (x, z) -> min(1.0, 0.7 - 0.05 * z),
            rainfall = 1.0e-6,
            air_temperature = 10.0,
        )
        build(timestepper) = initialize(LandModel(grid; soil = richards_soil(NF), snow = nothing, vegetation = nothing, timestepper); initializers)
        # explicit heat conduction limit (explicit Richards is unbounded near saturation)
        explicit = build(ForwardEuler(NF))
        soil = explicit.model.soil
        τ_explicit = cell_diffusion_timescale(explicit.state, grid, get_energy_balance(soil), soil, explicit.model.constants)
        integrator = build(implicit_timestepper(NF))
        @test cell_diffusion_timescale(integrator) == Inf
        max_Δt = 3600.0
        wizard = TimeStepWizard(cfl = 0.2, diffusive_cfl = 0.5, max_change = 2.0, max_Δt = max_Δt)
        sim = run_with_wizard!(integrator; Δt = 60.0, wizard, stop_iteration = 48)
        τ_advection = cell_advection_timescale(integrator)
        @info "IMEX LandModel wizard" sim.Δt τ_explicit sim.Δt / τ_explicit τ_advection
        # the step is limited only by `max_Δt` or by the explicit terms
        @test sim.Δt ≈ min(max_Δt, 0.2 * τ_advection)
        @test sim.Δt > 10 * τ_explicit
        @test all(isfinite, interior(integrator.state.temperature))
        @test all(0 .<= interior(integrator.state.saturation_water_ice) .<= 1)
    end

    @testset "TimeStepWizard is limited by explicit heat advection under implicit water" begin
        NF = Float64
        grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.1, N = 20))
        # closed bottom, so that no explicit water outflow limits the step
        integrator = sandy_column(grid, ExplicitEnergyIMEX(ForwardEuler(NF), implicit_solver(NF)); drainage = false)
        @test timestepping(integrator.state, Val(:internal_energy)) === Explicit()
        @test timestepping(integrator.state, Val(:saturation_water_ice)) === Implicit()
        max_Δt = 6 * 3600.0
        wizard = TimeStepWizard(cfl = 0.2, diffusive_cfl = 0.5, max_change = 1.5, max_Δt = max_Δt)
        sim = run_with_wizard!(integrator; Δt = 60.0, wizard, stop_iteration = 30)
        soil = integrator.model.soil
        energy, hydrology, strat, bgc = get_energy_balance(soil), get_hydrology(soil), get_stratigraphy(soil), get_biogeochemistry(soil)
        τ_heat = cell_advection_timescale(integrator.state, grid, energy, hydrology, strat, bgc)
        τ_diffusion = cell_diffusion_timescale(integrator)
        @info "IMEX wizard with explicit heat advection" sim.Δt τ_heat τ_diffusion cell_advection_timescale(integrator)
        # the advective limit of the explicit heat advection is the binding limit of the model
        @test isfinite(τ_heat)
        @test cell_advection_timescale(integrator) == τ_heat
        @test 0.2 * τ_heat < 0.5 * τ_diffusion
        @test sim.Δt ≈ 0.2 * τ_heat
        @test all(isfinite, interior(integrator.state.temperature))
    end

    @testset "TimeStepWizard keeps a draining sandy column bounded (IMEX)" begin
        NF = Float64
        grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.05, N = 20))
        τ_explicit = cell_diffusion_timescale(sandy_column(grid, ForwardEuler(NF)))
        integrator = sandy_column(grid, implicit_timestepper(NF))
        bounded = Ref(true)
        check_bounds(sim) = bounded[] &= all(0 .<= interior(integrator.state.saturation_water_ice) .<= 1) &&
            all(isfinite, interior(integrator.state.temperature))
        wizard = TimeStepWizard(cfl = 0.2, diffusive_cfl = 0.5, max_change = 1.5, max_Δt = 86400.0)
        sim = Simulation(integrator; Δt = 60.0, stop_time = 86400.0)
        steps = Float64[]
        sim.callbacks[:wizard] = Callback(wizard, IterationInterval(1))
        sim.callbacks[:record] = Callback(sim -> (push!(steps, sim.Δt); check_bounds(sim)), IterationInterval(1))
        run!(sim)
        τ_outflow = cell_advection_timescale(integrator)
        @info "IMEX wizard, draining sandy column" τ_explicit extrema(steps) length(steps) τ_outflow sim.Δt
        @test bounded[]
        @test integrator.clock.time ≈ 86400.0
        # the explicit drainage through the bottom face, not the diffusive limit, bounds the step
        @test isfinite(τ_outflow)
        @test sim.Δt ≈ 0.2 * τ_outflow
    end

end # module IMEXWizardTests
