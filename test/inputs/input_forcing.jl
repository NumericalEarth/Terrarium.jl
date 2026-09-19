module ForcingInputTest

using Terrarium
using Terrarium: AbstractLandGrid, XY, XYZ, Ground, prognostic, input
using Test

DEFAULT_NF = Float32

@kwdef struct TestModel{NF, Grid <: AbstractLandGrid{NF}} <: Terrarium.AbstractModel{NF, Grid}
    grid::Grid
    initializer = DefaultInitializer(eltype(grid))
    timestepper = ForwardEuler(eltype(grid))
end

Terrarium.variables(model::TestModel) = (
    prognostic(:x, Ground(XYZ())),
    input(:F, Ground(XY())),
)

Terrarium.compute_auxiliary!(state, model::TestModel) = nothing

function Terrarium.compute_tendencies!(state, model::TestModel)
    # set tendency to forcing term
    return state.tendencies.x .= state.F
end

@testset "Forcing input" begin
    grid = ColumnGrid(CPU(), DEFAULT_NF, ExponentialSpacing())
    model = TestModel(grid)
    F = Field(grid, Ground(XY()))
    F_in = InputSource(grid, F; name = :F, domain = Ground())
    @test isa(F_in, FieldInputSource)
    integrator = initialize(model, inputs = F_in)
    # check that state variable is defined
    @test hasproperty(integrator.state.inputs, :F)
end

@testset "Forcing input with time series" begin
    grid = ColumnGrid(CPU(), DEFAULT_NF, ExponentialSpacing())
    model = TestModel(grid)
    t_F = 0:0.1:1
    F = FieldTimeSeries(grid, Ground(XY()), t_F)
    F.data .= ones(size(F))
    F_in = InputSource(F; name = :F, domain = Ground())
    @test isa(F_in, FieldTimeSeriesInputSource)
    integrator = initialize(model, inputs = F_in)
    # check initial values
    @test all(integrator.state.x .≈ 0)
    @test all(integrator.state.F .≈ 1)
    # advance one timestep and check updated values
    timestep!(integrator, 0.1)
    @test all(integrator.state.x .≈ 0.1)
    @test all(integrator.state.F .≈ 1)
end

end
