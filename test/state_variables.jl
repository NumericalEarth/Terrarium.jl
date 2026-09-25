using Terrarium
using Test

using Terrarium: AbstractLandGrid, VarDims, XY, XYZ, prognostic, auxiliary, input, namespace

DEFAULT_NF = Float32

module StateVariablesTestTypes

    using Terrarium
    using Terrarium: AbstractLandGrid, VarDims, XY, XYZ, prognostic, auxiliary, input, namespace

    using Test

    @kwdef struct SubModel{NF, Grid <: AbstractLandGrid{NF}} <: Terrarium.AbstractModel{NF, Grid}
        grid::Grid
        initializer = DefaultInitializer(eltype(grid))
        timestepper = ForwardEuler(eltype(grid))
    end

    Terrarium.variables(model::SubModel) = (
        # duplicate naming allowed in new namesapce
        auxiliary(:auxvar2D, XY()),
        # inputs are handled "globally" (i.e. all inputs with a given name refer to the same field)
        input(:forcing, XY()),
    )

    @kwdef struct TestModel{NF, Grid <: AbstractLandGrid{NF}, Sub} <: Terrarium.AbstractModel{NF, Grid}
        grid::Grid
        submodel::Sub = SubModel(; grid)
        initializer = DefaultInitializer(eltype(grid))
        timestepper = ForwardEuler(eltype(grid))
    end

    struct TestClosure <: Terrarium.AbstractClosureRelation end

    Terrarium.variables(::TestClosure) = (
        auxiliary(:closurevar, XYZ()),
    )

    Terrarium.variables(model::TestModel) = (
        prognostic(:progvar3D, XYZ()),
        prognostic(:cprogvar3D, XYZ(), closure = TestClosure()),
        prognostic(:progvar2D, XY()),
        auxiliary(:auxvar3D, XYZ()),
        auxiliary(:auxvar2D, XY()),
        input(:forcing, XY()),
        namespace(:submodel, variables(model.submodel)),
    )
end

@testset "State variable initialization" begin
    grid = ColumnGrid(CPU(), DEFAULT_NF, ExponentialSpacing(N = 10))
    model = StateVariablesTestTypes.TestModel(grid)
    state = StateVariables(model)
    # Check that all prognostic variables are defined correctly
    @test hasproperty(state, :progvar3D) && isa(state.prognostic.progvar3D, Field{Center, Center, Center})
    @test hasproperty(state, :progvar2D) && isa(state.prognostic.progvar2D, Field{Center, Center, Nothing})
    @test hasproperty(state, :cprogvar3D) && isa(state.prognostic.cprogvar3D, Field{Center, Center, Center})
    # Check that all tendencies are defined correctly
    @test hasproperty(state.tendencies, :progvar3D) && isa(state.tendencies.progvar3D, Field{Center, Center, Center})
    @test hasproperty(state.tendencies, :progvar2D) && isa(state.tendencies.progvar2D, Field{Center, Center, Nothing})
    # Check that tendency for prognostic variable with closure relation is defined correctly
    @test hasproperty(state.tendencies, :cprogvar3D)
    @test !hasproperty(state.tendencies, :closurevar)
    # Check that all auxiliary variables are defined correctly
    @test hasproperty(state, :auxvar3D) && isa(state.auxvar3D, Field{Center, Center, Center})
    @test hasproperty(state, :auxvar2D) && isa(state.auxvar2D, Field{Center, Center, Nothing})
    # Check that all input variables are defined correctly
    @test hasproperty(state, :forcing) && isa(state.auxvar2D, Field{Center, Center, Nothing})
    # Check submodel namespace
    @test hasproperty(state, :submodel) && isa(state.submodel, StateVariables)
    @test hasproperty(state.submodel, :auxvar2D) && isa(state.submodel.auxvar2D, Field{Center, Center, Nothing})
end

@testset "State variable utilities" begin
    grid = ColumnGrid(CPU(), DEFAULT_NF, ExponentialSpacing(N = 10))
    model = StateVariablesTestTypes.TestModel(grid)
    state = StateVariables(model)
    for input in state.inputs
        set!(input, 1)
    end

    # test fill!
    fill!(state, 2)
    @test all(map(field -> all(field .== 2), state.prognostic))
    @test all(map(field -> all(field .== 2), state.auxiliary))
    @test all(map(field -> all(field .== 2), state.tendencies))
    @test all(map(field -> all(field .== 1), state.inputs)) # inputs should not be modified

    # test copyto!
    state2 = StateVariables(model)
    copyto!(state2, state)
    @test all(map(field -> all(field .== 2), state2.prognostic))
    @test all(map(field -> all(field .== 2), state2.auxiliary))
    @test all(map(field -> all(field .== 2), state2.tendencies))
    @test all(map(field -> all(field .== 1), state2.inputs))

    # test field getters
    fields = get_fields(state, :progvar3D, :submodel => (:auxvar2D,))
    @test hasproperty(fields, :progvar3D)
    @test hasproperty(fields, :submodel)
    @test hasproperty(fields.submodel, :auxvar2D)
    fields = @inferred get_fields(state, model)
    @test hasproperty(fields, :submodel)
    prog_fields = @inferred Terrarium.prognostic_fields(state, model)
    @test length(prog_fields) == 3 # note that namespaces are NOT included
    @test hasproperty(prog_fields, :progvar3D)
    @test hasproperty(prog_fields, :cprogvar3D)
    @test hasproperty(prog_fields, :progvar2D)
    aux_fields = @inferred Terrarium.auxiliary_fields(state, model)
    @test length(aux_fields) == 2 # note that namespaces are NOT included
    @test hasproperty(aux_fields, :auxvar3D)
    @test hasproperty(aux_fields, :auxvar2D)
    input_fields = @inferred Terrarium.input_fields(state, model)
    @test length(input_fields) == 1 # note that namespaces are NOT included
    @test hasproperty(input_fields, :forcing)
end
