using Terrarium
using Terrarium: StateVariables, prognostic, auxiliary, var, input, fill_halo_regions!, merge_boundary_conditions, varname, vardims, XY, XYZ
using Test

@testset "Field BC configuration" begin
    Nz = 10
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = Nz))
    vars = variables(prognostic(:x, XYZ()), auxiliary(:y, XYZ()), input(:c, XY()))
    clock = Clock(time = 0.0)
    upperbc = ValueBoundaryCondition(1.0)
    lowerbc = FluxBoundaryCondition(-0.01)
    bc1 = (x = (top = upperbc, bottom = lowerbc),)
    state = StateVariables(vars, grid; clock, boundary_conditions = bc1)
    @test state.x.boundary_conditions.top == upperbc
    @test state.x.boundary_conditions.bottom == lowerbc
    set!(state.x, 0.5)
    fill_halo_regions!(state.x, state)
    # check that halo matches boundary value that makes the face equal to 1
    @test state.x[1, 1, Nz + 1] == 1.5
    bc2 = (y = (top = ValueBoundaryCondition(var(:c, XY())), bottom = ValueBoundaryCondition(0.0)),)
    merged_bcs = merge_boundary_conditions(bc1, bc2)
    @test hasproperty(merged_bcs, :x)
    @test hasproperty(merged_bcs, :y)
    state = StateVariables(vars, grid; clock, boundary_conditions = merged_bcs)
    # check that we can set the input variable to modify the boundary condition
    set!(state.c, 1.0)
    fill_halo_regions!(state.y, state)
    @test state.x[1, 1, Nz + 1] == 2.0
end
