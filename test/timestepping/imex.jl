using Terrarium
using Test

using Terrarium: AbstractLandGrid, AbstractIMEX, AbstractTimeStepper, AbstractVariable, EmptyCache, Explicit, Implicit, prognostic, XY

module IMEXTestTypes

    using Terrarium
    using Terrarium: AbstractLandGrid, AbstractIMEX, AbstractTimeStepper, AbstractVariable, Implicit, prognostic, XY

    # A mock implicit timestepper used only to verify IMEX routing. Its update is deliberately distinct
    # from forward Euler (u += 2·∂u∂t·Δt) so we can tell which sub-stepper integrated which variable.
    # It declares the `Implicit()` timestepping trait so IMEX routes it to the implicit cache slot.
    struct MockImplicit{NF} <: AbstractTimeStepper{NF}
        Δt::NF
    end
    MockImplicit(::Type{NF}; Δt = 300.0) where {NF} = MockImplicit{NF}(NF(Δt))

    Terrarium.timestepping(::MockImplicit) = Implicit()
    Terrarium.default_dt(ts::MockImplicit) = ts.Δt
    Terrarium.is_adaptive(::MockImplicit) = false

    # MockImplicit is just a wrong version of ForwardEuler for testing IMEX routing
    function Terrarium.timestep!(integrator, ts::MockImplicit, Δt, names::Tuple)
        Terrarium.update_state!(integrator, compute_tendencies = true)
        state = integrator.state
        for name in names
            state.prognostic[name] .+= 2 .* state.tendencies[name] .* Δt
        end
        return nothing
    end

    # Minimal two-variable model with constant unit tendencies. Both variables default to the `Explicit`
    # timestepping class; specific routing under an IMEX timestepper is declared via `timestepping`
    @kwdef struct TwoVarModel{NF, Grid <: AbstractLandGrid{NF}, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
        grid::Grid
        initializer = DefaultInitializer(eltype(grid))
        timestepper::TS = ForwardEuler(eltype(grid))
    end

    Terrarium.variables(::TwoVarModel) = (prognostic(:a, XY()), prognostic(:b, XY()))
    Terrarium.compute_auxiliary!(state, ::TwoVarModel) = nothing
    function Terrarium.compute_tendencies!(state, ::TwoVarModel)
        set!(state.tendencies.a, 1.0)
        set!(state.tendencies.b, 1.0)
        return nothing
    end

    # Under any IMEX timestepper, integrate `:b` implicitly while `:a` keeps the default `Explicit` class.
    Terrarium.timestepping(::AbstractVariable{:b}, ::TwoVarModel, ::AbstractIMEX) = Implicit()

    # A second model identical to `TwoVarModel` but with the routing flipped, used to exercise the
    # per-model nature of `timestepping`: here `:a` is integrated implicitly and `:b` explicitly.
    @kwdef struct FlippedModel{NF, Grid <: AbstractLandGrid{NF}, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
        grid::Grid
        initializer = DefaultInitializer(eltype(grid))
        timestepper::TS = ForwardEuler(eltype(grid))
    end

    Terrarium.variables(::FlippedModel) = (prognostic(:a, XY()), prognostic(:b, XY()))
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

@testset "timestepping is resolved per model; Heun explicit cache" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = 1))
    # same IMEX timestepper, but the FlippedModel routes :a implicitly and :b explicitly
    timestepper = IMEX(Heun(Float64), IMEXTestTypes.MockImplicit(Float64))
    model = IMEXTestTypes.FlippedModel(grid; timestepper)
    integrator = initialize(model)
    cache = integrator.state.timestepper_cache
    @test Terrarium.timestepping(cache) == (Implicit(), Explicit())   # (:a, :b)
    @test cache.explicit isa Terrarium.HeunCache   # Heun's cache lives in the explicit slot
    Δt = default_dt(integrator)
    timestep!(integrator)
    @test all(interior(integrator.state.a) .≈ 2 * Δt)   # :a → Implicit → mock: a += 2·Δt
    @test all(interior(integrator.state.b) .≈ Δt)       # :b → Explicit → Heun: b += 1·Δt
end
