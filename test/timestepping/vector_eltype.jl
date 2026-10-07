using Terrarium
using StaticArrays: SVector
using Test

# Variables with a custom `eltype` (here a per-cell SVector), mixed with scalar variables,
# integrated by ForwardEuler: du/dt = -k .* u with k = (0.1, 0.2, 0.3).

@kwdef struct VecModel{NF, Grid <: Terrarium.AbstractGrid{NF}, I, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
    grid::Grid
    initializer::I = DefaultInitializer(eltype(grid))
    timestepper::TS = ForwardEuler(eltype(grid))
end

Terrarium.variables(::VecModel{NF}) where {NF} = (
    Terrarium.prognostic(:u, Terrarium.XY(); eltype = SVector{3, NF}),
    Terrarium.prognostic(:s, Terrarium.XY()),
)

Terrarium.compute_auxiliary!(state, model::VecModel) = nothing

function Terrarium.compute_tendencies!(state, model::VecModel{NF}) where {NF}
    k = SVector{3, NF}(0.1, 0.2, 0.3)
    state.tendencies.u[1, 1, 1] = -k .* state.prognostic.u[1, 1, 1]
    state.tendencies.s[1, 1, 1] = -0.5 * state.prognostic.s[1, 1, 1]
    return nothing
end

@testset "Variables with custom eltype" begin
    @test Base.eltype(Terrarium.prognostic(:u, XY())) === nothing
    pv = Terrarium.prognostic(:u, XY(); eltype = SVector{3, Float64})
    @test Base.eltype(pv) === SVector{3, Float64}
    @test Base.eltype(pv.tendency) === SVector{3, Float64} # tendency inherits eltype
    @test Base.eltype(Terrarium.auxiliary(:a, XY(); eltype = SVector{2, Float64})) === SVector{2, Float64}
    @test Base.eltype(Terrarium.input(:i, XY(); eltype = SVector{2, Float64})) === SVector{2, Float64}
    # same name/dims/units but different eltype are not the same variable
    @test Terrarium.var(:x, XY()) != Terrarium.var(:x, XY(); eltype = SVector{2, Float64})
end

@testset "ForwardEuler with SVector prognostic" begin
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(N = 1))
    model = VecModel(grid)
    integrator = initialize(model)
    state = integrator.state
    @test eltype(state.prognostic.u) === SVector{3, Float64}
    @test eltype(state.tendencies.u) === SVector{3, Float64}
    @test eltype(state.prognostic.s) === Float64
    # reset! must zero vector-valued fields without broadcasting the SVector
    reset!(state)
    @test state.prognostic.u[1, 1, 1] == zero(SVector{3, Float64})
    set!(state.prognostic.u, Ref(SVector(1.0, 2.0, 3.0)))
    set!(state.prognostic.s, 4.0)
    dt = default_dt(integrator)
    timestep!(integrator)
    @test state.prognostic.u[1, 1, 1] ≈ SVector(1.0, 2.0, 3.0) .* (1 .- dt .* SVector(0.1, 0.2, 0.3))
    @test state.prognostic.s[1, 1, 1] ≈ 4.0 * (1 - 0.5 * dt)
end
