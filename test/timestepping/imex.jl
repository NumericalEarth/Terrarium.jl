using Terrarium
using Test

using Terrarium: AbstractGrid, AbstractIMEX, AbstractTimeStepper, AbstractVariable, EmptyCache, Explicit, Implicit, prognostic, XY

module IMEXTestTypes

    using Terrarium
    using Terrarium: AbstractGrid, AbstractIMEX, AbstractTimeStepper, AbstractVariable, Implicit, prognostic, XY

    # A mock implicit timestepper used only to verify IMEX routing. Its update is deliberately distinct
    # from forward Euler (u += 2·∂u∂t·Δt) so we can tell which sub-stepper integrated which variable.
    # It declares the `Implicit()` timestepping trait so IMEX routes it to the implicit cache slot.
    struct MockImplicit{NF} <: AbstractTimeStepper{NF}
        Δt::NF
    end
    MockImplicit(::Type{NF}; Δt = 300.0) where {NF} = MockImplicit{NF}(NF(Δt))

    Terrarium.timestepping(::MockImplicit) = Implicit()
    Terrarium.default_dt(ts::MockImplicit) = ts.Δt

    # MockImplicit "solves" the implicit system by adding one more explicit increment, so a variable routed
    # to it ends at u = uⁿ + 2·Δt·∂u∂t (the IMEX explicit stage already applied the first increment).
    function Terrarium.implicit_step!(integrator, ts::MockImplicit, Δt, names::Tuple)
        state = integrator.state
        for name in names
            state.prognostic[name] .+= state.tendencies[name] .* Δt
        end
        return nothing
    end

    # Global counter of tendency evaluations, used to check that each IMEX step calls `update_state!` once.
    const TENDENCY_EVALUATIONS = Ref(0)

    # Minimal two-variable model with constant unit tendencies. Both variables default to the `Explicit`
    # timestepping class; specific routing under an IMEX timestepper is declared via `timestepping`
    @kwdef struct TwoVarModel{NF, Grid <: AbstractGrid{NF}, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
        grid::Grid
        initializer = DefaultInitializer(eltype(grid))
        timestepper::TS = ForwardEuler(eltype(grid))
    end

    Terrarium.variables(::TwoVarModel) = (
        prognostic(:a, XY()),
        prognostic(:b, XY()),
    )
    Terrarium.compute_auxiliary!(state, ::TwoVarModel) = nothing
    function Terrarium.compute_tendencies!(state, ::TwoVarModel)
        TENDENCY_EVALUATIONS[] += 1
        set!(state.tendencies.a, 1.0)
        set!(state.tendencies.b, 1.0)
        return nothing
    end

    # Under any IMEX timestepper, integrate `:b` implicitly while `:a` keeps the default `Explicit` class.
    Terrarium.timestepping(::AbstractVariable{:b}, ::TwoVarModel, ::AbstractIMEX) = Implicit()

    # Give the implicitly routed variables an implicit operator, one tridiagonal operator per variable solved
    # in sequence (without defining the implicit process hooks for them), so that the `BackwardEuler` cache
    # can be allocated for this model.
    Terrarium.implicit_operator(model::TwoVarModel, state, names::Tuple{Vararg{Symbol}}) =
        Terrarium.SplitOperator(NamedTuple{names}(map(_ -> TridiagonalOperator(ground_domain(get_grid(model))), names)))

    # A second model identical to `TwoVarModel` but with the routing flipped, used to exercise the
    # per-model nature of `timestepping`: here `:a` is integrated implicitly and `:b` explicitly.
    @kwdef struct FlippedModel{NF, Grid <: AbstractGrid{NF}, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
        grid::Grid
        initializer = DefaultInitializer(eltype(grid))
        timestepper::TS = ForwardEuler(eltype(grid))
    end

    Terrarium.variables(::FlippedModel) = (
        prognostic(:a, XY()),
        prognostic(:b, XY()),
    )
    Terrarium.compute_auxiliary!(state, ::FlippedModel) = nothing
    function Terrarium.compute_tendencies!(state, ::FlippedModel)
        set!(state.tendencies.a, 1.0)
        set!(state.tendencies.b, 1.0)
        return nothing
    end

    Terrarium.timestepping(::AbstractVariable{:a}, ::FlippedModel, ::AbstractIMEX) = Implicit()
end

@testset "Single timestepper integrates all prognostic variables" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    model = IMEXTestTypes.TwoVarModel(grid) # default: single ForwardEuler
    integrator = initialize(model)
    # a single stateless timestepper gets an EmptyCache
    @test integrator.state.timestepper_cache isa EmptyCache
    Δt = default_dt(integrator)
    timestep!(integrator)
    # both variables are stepped by forward Euler regardless of their declared class
    @test all(interior(integrator.state.a) .≈ Δt)
    @test all(interior(integrator.state.b) .≈ Δt)
end

@testset "Single Heun gets a HeunCache" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    model = IMEXTestTypes.TwoVarModel(grid; timestepper = Heun(Float64))
    integrator = initialize(model)
    @test integrator.state.timestepper_cache isa Terrarium.HeunCache
    Δt = default_dt(integrator)
    timestep!(integrator)
    # constant tendency ⇒ Heun matches forward Euler: u += 1·Δt
    @test all(interior(integrator.state.a) .≈ Δt)
    @test all(interior(integrator.state.b) .≈ Δt)
end

@testset "IMEX routes by the timestepping class" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    timestepper = IMEX(ForwardEuler(Float64), IMEXTestTypes.MockImplicit(Float64))
    model = IMEXTestTypes.TwoVarModel(grid; timestepper)
    integrator = initialize(model)
    cache = integrator.state.timestepper_cache
    # the IMEXCache holds the resolved per-variable classes (in prognostic order) as a type parameter
    @test cache isa Terrarium.IMEXCache
    @test Terrarium.timestepping(cache) == (Explicit(), Implicit())   # (:a, :b)
    @test cache.explicit isa EmptyCache && cache.implicit isa EmptyCache
    Δt = default_dt(integrator)
    timestep!(integrator)
    @test all(interior(integrator.state.a) .≈ Δt)       # :a → Explicit → forward Euler: a += 1·Δt
    @test all(interior(integrator.state.b) .≈ 2 * Δt)   # :b → Implicit → mock: b += 2·Δt
end

@testset "IMEX evaluates tendencies once per step" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    timestepper = IMEX(ForwardEuler(Float64), IMEXTestTypes.MockImplicit(Float64))
    model = IMEXTestTypes.TwoVarModel(grid; timestepper)
    integrator = initialize(model)
    Δt = default_dt(integrator)
    IMEXTestTypes.TENDENCY_EVALUATIONS[] = 0
    timestep!(integrator, Δt; finalize = false)
    @test IMEXTestTypes.TENDENCY_EVALUATIONS[] == 1
    timestep!(integrator, Δt; finalize = false)
    @test IMEXTestTypes.TENDENCY_EVALUATIONS[] == 2
    # the clock is ticked exactly once per step
    @test integrator.clock.iteration == 2
    @test integrator.clock.time ≈ 2 * Δt
end

@testset "timestepping is resolved per model" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    # same IMEX timestepper, but the FlippedModel routes :a implicitly and :b explicitly
    timestepper = IMEX(ForwardEuler(Float64), IMEXTestTypes.MockImplicit(Float64))
    model = IMEXTestTypes.FlippedModel(grid; timestepper)
    integrator = initialize(model)
    cache = integrator.state.timestepper_cache
    @test Terrarium.timestepping(cache) == (Implicit(), Explicit())   # (:a, :b)
    Δt = default_dt(integrator)
    timestep!(integrator)
    @test all(interior(integrator.state.a) .≈ 2 * Δt)   # :a → Implicit → mock: a += 2·Δt
    @test all(interior(integrator.state.b) .≈ Δt)       # :b → Explicit → forward Euler: b += 1·Δt
end

@testset "timestepping(state, var) returns the resolved class" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    # IMEX: resolved from the IMEXCache type parameter
    timestepper = IMEX(ForwardEuler(Float64), IMEXTestTypes.MockImplicit(Float64))
    state = initialize(IMEXTestTypes.TwoVarModel(grid; timestepper)).state
    @test @inferred(timestepping(state, Val(:a))) === Explicit()
    @test @inferred(timestepping(state, Val(:b))) === Implicit()
    @test timestepping(state, prognostic(:b, XY()).var) === Implicit()
    # names that are not routed prognostic variables are explicit
    @test @inferred(timestepping(state, Val(:not_a_variable))) === Explicit()
    # any non-IMEX cache: all variables explicit
    state = initialize(IMEXTestTypes.TwoVarModel(grid)).state
    @test @inferred(timestepping(state, Val(:a))) === Explicit()
    @test @inferred(timestepping(state, Val(:b))) === Explicit()
end

@testset "Heun is rejected as the explicit partner of an IMEX" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    timestepper = IMEX(Heun(Float64), IMEXTestTypes.MockImplicit(Float64))
    model = IMEXTestTypes.TwoVarModel(grid; timestepper)
    @test_throws ArgumentError initialize(model)
end

@testset "IMEX validates the order of its sub-steppers" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    timestepper = IMEX(BackwardEuler(Float64), ForwardEuler(Float64))
    err = try
        initialize(IMEXTestTypes.TwoVarModel(grid; timestepper))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("explicit sub-stepper", err.msg)
end

@testset "BackwardEuler" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 4))
    @test_throws ArgumentError Picard(Float64; iterations = 0)
    ts = BackwardEuler(Float64)
    @test timestepping(ts) === Implicit()
    # BackwardEuler has no time step of its own; the IMEX takes Δt from its explicit partner
    @test_throws ArgumentError default_dt(ts)
    @test default_dt(IMEX(ForwardEuler(Float64; Δt = 600.0), ts)) == 600.0
    @test ts.solver isa Picard
    @test ts.solver.iterations == 2

    # BackwardEuler allocates its cache for the implicitly routed variables only (here `:b`)
    timestepper = IMEX(ForwardEuler(Float64), BackwardEuler(Float64))
    integrator = initialize(IMEXTestTypes.TwoVarModel(grid; timestepper))
    cache = integrator.state.timestepper_cache.implicit
    @test cache isa Terrarium.BackwardEulerCache
    @test keys(cache.previous_state) == (:b,)
    @test keys(cache.workspaces) == (:b,)
    workspace = cache.workspaces.b
    @test workspace isa Terrarium.ImplicitSolverWorkspace
    @test size(workspace.rhs) == size(integrator.state.b)
    @test size(workspace.explicit_stage) == size(integrator.state.b)
    @test workspace.rhs !== workspace.explicit_stage
    # one operator for all implicitly routed variables, as returned by the block query `implicit_operator`;
    # the fallback splits it into the per-variable operators
    @test cache.operator isa Terrarium.SplitOperator
    @test cache.operator isa AbstractLinearOperator
    @test keys(cache.operator.operators) == (:b,)
    # `:b` has no closure, so its linear system is solved for `:b` itself
    @test workspace.iteration == Terrarium.Iteration(:b)
    @test Terrarium.targetname(workspace.iteration) == :b
    @test !Terrarium.isfinal(workspace.iteration)
    @test workspace.iteration.number == 1
    operator = cache.operator.operators.b
    @test operator isa TridiagonalOperator
    @test operator isa AbstractLinearOperator
    @test location(operator.diagonal_coefficient) == (Center, Center, Center)
    @test location(operator.face_conductivity) == (Center, Center, Face)
    @test size(operator.diagonal_coefficient) == (1, 1, 4)
    @test operator.tridiagonal_solver isa Terrarium.BatchedTridiagonalSolver
    @test operator.tridiagonal_solver.b isa Terrarium.ImplicitDiffusionDiagonal
    @test Terrarium.get_cache(integrator.state.timestepper_cache, timestepper.implicit) === cache
    # the cache can be adapted (e.g. for kernel arguments or device transfer)
    @test adapt(Array, cache) isa Terrarium.BackwardEulerCache
    @test adapt(Array, cache).operator isa Terrarium.SplitOperator
    @test adapt(Array, cache).operator.operators.b isa TridiagonalOperator
    @test adapt(Array, cache).workspaces.b isa Terrarium.ImplicitSolverWorkspace
    @test adapt(Array, cache).workspaces.b.iteration === workspace.iteration
    # `:b` has an implicit operator but defines no implicit process hooks, so the `linearize!` fallback throws
    err = try
        timestep!(integrator)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("linearize!", err.msg)
    # without an `implicit_operator` method, a routed variable fails early at `initialize`
    err = try
        initialize(IMEXTestTypes.FlippedModel(grid; timestepper))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("no implicit operator", err.msg)

    # BackwardEuler cannot be used on its own
    @test_throws ArgumentError initialize(IMEXTestTypes.TwoVarModel(grid; timestepper = BackwardEuler(Float64)))
end

module IMEXVectorTypes

    using Terrarium
    using Terrarium: AbstractGrid, AbstractIMEX, AbstractVariable, Implicit, prognostic, XY
    using StaticArrays: SVector

    # A model whose vector-valued variable `:u` is routed implicitly, which the scalar implicit solvers reject
    @kwdef struct VecIMEXModel{NF, Grid <: AbstractGrid{NF}, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
        grid::Grid
        initializer = DefaultInitializer(eltype(grid))
        timestepper::TS = ForwardEuler(eltype(grid))
    end

    Terrarium.variables(::VecIMEXModel{NF}) where {NF} = (
        prognostic(:u, XY(); eltype = SVector{3, NF}),
        prognostic(:s, XY()),
    )
    Terrarium.compute_auxiliary!(state, ::VecIMEXModel) = nothing
    Terrarium.compute_tendencies!(state, ::VecIMEXModel) = nothing
    Terrarium.timestepping(::AbstractVariable{:u}, ::VecIMEXModel, ::AbstractIMEX) = Implicit()
end

@testset "BackwardEuler rejects non-scalar element types" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    timestepper = IMEX(ForwardEuler(Float64), BackwardEuler(Float64))
    err = try
        initialize(IMEXVectorTypes.VecIMEXModel(grid; timestepper))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("non-scalar", err.msg)
end

@testset "SoilModel routes internal energy implicitly under IMEX" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 10))
    timestepper = IMEX(ForwardEuler(Float64), BackwardEuler(Float64))
    model = SoilModel(grid; timestepper)
    integrator = initialize(model)
    state = integrator.state
    @test timestepping(state, Val(:internal_energy)) === Implicit()
    @test keys(state.timestepper_cache.implicit.workspaces) == (:internal_energy,)
    @test keys(state.timestepper_cache.implicit.previous_state) == (:internal_energy,)
    # the operator is a `SplitOperator` with the heat conduction operator, solved for the temperature
    operator = state.timestepper_cache.implicit.operator
    @test operator isa Terrarium.SplitOperator
    @test keys(operator.operators) == (:internal_energy,)
    @test operator.operators.internal_energy isa TridiagonalOperator
    @test map(w -> w.iteration, state.timestepper_cache.implicit.workspaces) == (internal_energy = Terrarium.Iteration(:internal_energy; target = :temperature),)
    # implicitly integrated heat conduction imposes no diffusive time step restriction
    @test isinf(Terrarium.cell_diffusion_timescale(integrator))
    # with a single explicit timestepper the diffusive restriction is finite
    integrator = initialize(SoilModel(grid; timestepper = ForwardEuler(Float64)))
    @test timestepping(integrator.state, Val(:internal_energy)) === Explicit()
    @test isfinite(Terrarium.cell_diffusion_timescale(integrator))
end

@testset "Default timesteppers" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 10))
    # Richards soil and land models default to IMEX with the step taken from the explicit partner
    richards_soil = SoilEnergyWaterCarbon(Float64; hydrology = SoilHydrology(Float64, RichardsEq()))
    for (model, Δt) in ((SoilModel(grid; soil = richards_soil), 3600), (LandModel(grid), 900))
        @test model.timestepper isa IMEX{Float64, ForwardEuler{Float64}, <:BackwardEuler{Float64}}
        @test Terrarium.default_dt(model.timestepper) == Δt
    end
    # all other models, including a soil model without water flow (the default `NoFlow`), keep forward Euler
    for model in (SoilModel(grid), SnowModel(grid), VegetationModel(grid), SurfaceEnergyModel(grid))
        @test model.timestepper isa ForwardEuler{Float64}
    end
end

# Routing of the soil variables of a `SoilModel` under `IMEX` and the structure of the implicit cache and operators
module IMEXSoilRoutingTests

    using Terrarium
    using Test

    using Terrarium: merge_boundary_conditions, implicit_operator, default_timestepping, get_hydrology

    const POROSITY = 0.4

    function richards_soil(NF)
        hydraulic_properties = ConstantSoilHydraulics(NF; swrc = VanGenuchten(α = 2.0, n = 2.0), saturated_conductivity = 1.0e-5, unsat_hydraulic_cond = UnsatKVanGenuchten(NF))
        hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
        strat = HomogeneousSoilStratigraphy(NF; porosity = ConstantSoilPorosity(NF; mineral_porosity = POROSITY))
        biogeochem = ConstantSoilCarbonDensity(NF; ρ_soc = 0.0)
        return SoilEnergyWaterCarbon(NF; hydrology, strat, biogeochem)
    end

    implicit_timestepper(NF) = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver = Picard(NF; iterations = 2)))

    function build_water_column(grid, timestepper; saturation)
        model = SoilModel(grid; soil = richards_soil(eltype(grid)), timestepper)
        return initialize(model; initializers = (temperature = 5.0, saturation_water_ice = saturation))
    end

    @testset "IMEX routing and implicit operators of the soil variables" begin
        NF = Float64
        grid = ColumnGrid(CPU(), NF, UniformSpacing(Δz = 0.1, N = 10))
        @test default_timestepping(SoilHydrology(NF, NoFlow())) === Explicit()
        @test default_timestepping(SoilHydrology(NF, RichardsEq())) === Implicit()
        # NoFlow: saturation is not prognostic, so it is never routed, and only energy is implicit
        model = SoilModel(grid; timestepper = implicit_timestepper(NF))
        @test timestepping(Terrarium.prognostic(:saturation_water_ice, Terrarium.Ground(Terrarium.XYZ())), model, model.timestepper) === Explicit()
        integrator = initialize(model)
        @test timestepping(integrator.state, Val(:saturation_water_ice)) === Explicit()
        @test timestepping(integrator.state, Val(:internal_energy)) === Implicit()
        @test_throws ArgumentError implicit_operator(model, integrator.state, (:saturation_water_ice,))
        # RichardsEq: saturation is routed to the implicit sub-stepper
        integrator = build_water_column(grid, implicit_timestepper(NF); saturation = 0.6)
        @test timestepping(integrator.state, Val(:saturation_water_ice)) === Implicit()
        @test implicit_operator(integrator.model, integrator.state, (:saturation_water_ice,)).operators.saturation_water_ice isa TridiagonalOperator
        # the implicit timestepper's cache holds one `SplitOperator` with one operator per implicitly routed
        # variable, applied in the order set by the model (water, then energy)
        cache = integrator.state.timestepper_cache.implicit
        @test cache.operator isa Terrarium.SplitOperator
        operators = cache.operator.operators
        @test keys(operators) == (:saturation_water_ice, :internal_energy)
        # the order is set by the model, not by the order of the names
        reversed = implicit_operator(integrator.model, integrator.state, (:internal_energy, :saturation_water_ice))
        @test keys(reversed.operators) == (:saturation_water_ice, :internal_energy)
        # with water flow, the heat operator advects heat with the water flux; the Richards operator does not advect
        @test operators.internal_energy.face_velocity isa Field
        @test isnothing(operators.saturation_water_ice.face_velocity)
        # each variable is solved for the variable of its closure
        @test map(w -> w.iteration, cache.workspaces) == (
            internal_energy = Terrarium.Iteration(:internal_energy; target = :temperature),
            saturation_water_ice = Terrarium.Iteration(:saturation_water_ice; target = :pressure_head),
        )
        @test Terrarium.varname(Terrarium.get_closure(get_hydrology(integrator.model.soil))) === :pressure_head
        @test Terrarium.varname(integrator.model.soil.energy.closure) === :temperature
        @test all(operator -> operator isa TridiagonalOperator, operators)
        @test operators.internal_energy !== operators.saturation_water_ice
        @test operators.internal_energy.face_conductivity !== operators.saturation_water_ice.face_conductivity
    end

end # module IMEXSoilRoutingTests
