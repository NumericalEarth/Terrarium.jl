using Terrarium
using Terrarium: StateVariables, prognostic, auxiliary, var, input, fill_halo_regions!, merge_boundary_conditions, varname, vardims, XY, XYZ
using Oceananigans.BoundaryConditions: GradientBoundaryCondition
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
    fill_halo_regions!(state.x, state)
    fill_halo_regions!(state.y, state)
    @test state.x[1, 1, Nz + 1] == 2.0
end

@testset "One-sided vertical boundary conditions" begin
    # A condition given on one vertical side only must leave the other side at the Oceananigans default
    # (no flux for a vertically bounded, center-located field), not `nothing`: a `nothing` condition means
    # the halo is never filled and stays zero, which acts as a spurious boundary temperature.
    Nz = 10
    grid = ColumnGrid(CPU(), Float64, UniformSpacing(Δz = 0.1, N = Nz))
    clock = Clock(time = 0.0)
    vars = variables(prognostic(:x, XYZ()))
    gradient_bc = (x = (bottom = GradientBoundaryCondition(0.0),),)
    value_bc = (x = (bottom = ValueBoundaryCondition(2.0),),)
    flux_bc = (x = (top = FluxBoundaryCondition(0.0),),)
    for (bcs, given, other) in ((gradient_bc, :bottom, :top), (value_bc, :bottom, :top), (flux_bc, :top, :bottom))
        state = StateVariables(vars, grid; clock, boundary_conditions = bcs)
        fbcs = state.x.boundary_conditions
        @test !isnothing(getproperty(fbcs, given))
        @test !isnothing(getproperty(fbcs, other))
        set!(state.x, 5.0)
        fill_halo_regions!(state.x, state)
        # unspecified side: no-flux halo copies the adjacent interior value
        halo_index = other == :top ? Nz + 1 : 0
        interior_index = other == :top ? Nz : 1
        @test state.x[1, 1, halo_index] == state.x[1, 1, interior_index] == 5.0
    end
    # the specified side keeps its own halo semantics (zero gradient: halo equals the interior value)
    state = StateVariables(vars, grid; clock, boundary_conditions = gradient_bc)
    set!(state.x, 5.0)
    fill_halo_regions!(state.x, state)
    @test state.x[1, 1, 0] == 5.0

    # soil column: a uniform temperature with zero flux through both boundaries must not evolve
    for bcs in (
            (temperature = (bottom = GradientBoundaryCondition(0.0),),),
            PrescribedBottomTemperature(:T_bottom, 5.0),
            SoilHeatFlux(0.0),
        )
        model = SoilModel(grid)
        integrator = initialize(model; boundary_conditions = bcs, initializers = (temperature = 5.0, saturation_water_ice = 0.5))
        timestep!(integrator, 1.0)
        state = integrator.state
        @test !isnothing(state.temperature.boundary_conditions.top)
        @test !isnothing(state.temperature.boundary_conditions.bottom)
        @test state.temperature[1, 1, Nz + 1] == 5.0
        @test state.tendencies.internal_energy[1, 1, Nz] == 0.0
        @test all(==(0.0), interior(state.tendencies.internal_energy))
    end
end
