using Terrarium
using Terrarium: Variables, prognostic, auxiliary, input, tendency, namespace, prognostic_variables, auxiliary_variables, input_variables
using Test
using Oceananigans: Center, Face
using Unitful
using Terrarium: Canopy, Ground, Surface

@testset "varpath" begin
    @test varpath(:x) == (:x,)
    @test varpath(:ns => :x) == (:ns, :x)
    @test varpath(:a => :b => :c) == (:a, :b, :c)
    @test varpath((:ns, :x)) == (:ns, :x)
end

@testset "VarDims types" begin
    # Test XYZ dimensions
    xyz = XYZ()
    @test xyz.x isa Center
    @test xyz.y isa Center
    @test xyz.z isa Center

    xyz_custom = XYZ(x = Face(), y = Center(), z = Face())
    @test xyz_custom.x isa Face
    @test xyz_custom.y isa Center
    @test xyz_custom.z isa Face

    # Test XY dimensions
    xy = XY()
    @test xy.x isa Center
    @test xy.y isa Center

    xy_custom = XY(x = Face(), y = Center())
    @test xy_custom.x isa Face
    @test xy_custom.y isa Center
end

@testset "Variable construction" begin
    # Test basic Variable creation
    var = Terrarium.var(:temperature, Ground(XYZ()), u"K")
    @test Terrarium.varname(var) == :temperature
    @test Terrarium.vardims(var) isa XYZ
    @test Terrarium.varunits(var) == u"K"

    # Test with default units (NoUnits)
    var_no_units = Terrarium.var(:pressure, Ground(XY()))
    @test Terrarium.varname(var_no_units) == :pressure
    @test Terrarium.varunits(var_no_units) == Terrarium.NoUnits

    # Test convenience constructor
    var_convenience = Terrarium.var(:saturation, Ground(XYZ(x = Center(), y = Center(), z = Face())))
    @test Terrarium.varname(var_convenience) == :saturation
    @test Terrarium.vardims(var_convenience).z isa Face
end

@testset "PrognosticVariable construction" begin
    # Test basic prognostic variable
    prog = prognostic(:temperature, Ground(XYZ()); units = u"K")
    @test Terrarium.hasclosure(prog) == false
    @test isa(prog.closure, Nothing)
    @test isa(prog.tendency, Terrarium.AuxiliaryVariable)
    @test Terrarium.varname(prog) == :temperature
    @test Terrarium.varname(prog.tendency) == :temperature

    # Test with closure relation
    struct MockClosure <: Terrarium.AbstractClosureRelation end
    prog_with_closure = prognostic(:enthalpy, Ground(XYZ()); units = u"J", closure = MockClosure())
    @test Terrarium.hasclosure(prog_with_closure) == true
    @test isa(prog_with_closure.closure, MockClosure)

    # Test with bounds and description
    prog_full = prognostic(
        :moisture,
        XY();
        units = u"kg",
        bounds = Terrarium.UnitInterval,
        desc = "Soil moisture content"
    )
    @test prog_full.desc == "Soil moisture content"
end

@testset "AuxiliaryVariable construction" begin
    # Test basic auxiliary variable without constructor
    aux = auxiliary(:pressure, Ground(XYZ()); units = u"Pa")
    @test Terrarium.varname(aux) == :pressure
    @test isa(aux.ctor, Nothing)

    # Test with bounds
    aux_bounded = auxiliary(:saturation, Ground(XYZ()); units = u"K", bounds = Terrarium.UnitInterval)
    @test aux_bounded.bounds == Terrarium.UnitInterval
end

@testset "InputVariable construction" begin
    # Test basic input variable without default
    inp = input(:radiation, Canopy(XYZ()); units = u"W/m^2")
    @test Terrarium.varname(inp) == :radiation
    @test isa(inp.default, Nothing)

    # Test with numeric default
    inp_default_num = input(:precipitation, Surface(XY()); default = 0.01, units = u"m/s")
    @test inp_default_num.default ≈ 0.01

    # Test with function default
    f(t) = sin(t)
    inp_default_func = input(:forcing, Ground(XYZ()); default = f, units = u"K")
    @test inp_default_func.default(π / 2) ≈ 1.0
end

@testset "tendency helper" begin
    # Test that tendency creates appropriate auxiliary variable
    temp_var = Terrarium.var(:temperature, Ground(XYZ()), u"K")
    tend = tendency(temp_var)

    @test Terrarium.varname(tend) == :temperature
    @test Terrarium.vardims(tend) isa XYZ
    # Check units are K/s (upreferred)
    @test Terrarium.varunits(tend) == u"K/s"
end

@testset "Namespace construction" begin
    # Test basic namespace creation
    vars = Variables(prognostic(:temp, Ground(XYZ()); units = u"K"))
    ns = namespace(:soil, vars)

    @test Terrarium.varname(ns) == :soil
    @test isa(ns.vars.prognostic[:temp], Terrarium.PrognosticVariable)

    # Test namespace from tuple of variables
    ns_from_tuple = namespace(
        :atmosphere,
        (auxiliary(:humidity, Ground(XY()); units = u"kg"), prognostic(:pressure, Ground(XYZ()); units = u"Pa"))
    )
    @test Terrarium.varname(ns_from_tuple) == :atmosphere
    @test Terrarium.varname((prognostic_variables(ns_from_tuple.vars)[1])) == :pressure
    @test Terrarium.varname((auxiliary_variables(ns_from_tuple.vars)[1])) == :humidity
end

@testset "Variables construction" begin
    # Test basic Variables creation from tuple
    vars = Variables(
        prognostic(:temperature, Ground(XYZ()); units = u"K"),
        auxiliary(:pressure, Ground(XY()); units = u"Pa"),
        input(:radiation, Canopy(XYZ()); default = 100.0, units = u"W/m^2")
    )

    @test length(vars.prognostic) == 1
    @test length(vars.auxiliary) == 1
    @test length(vars.inputs) == 1
    @test Terrarium.varname(first(values(vars.prognostic))) == :temperature

    # Test with namespaces
    vars_with_ns = Variables(
        prognostic(:temp, Ground(XYZ()); units = u"K"),
        namespace(:soil, (prognostic(:moisture, Ground(XY()); units = u"kg"),))
    )

    @test length(vars_with_ns.namespaces) == 1
    @test Terrarium.varname(first(values(vars_with_ns.namespaces))) == :soil

    # Test automatic merging of namespaces with same name
    vars_merged = Variables(
        namespace(:atmosphere, (prognostic(:temp, Ground(XYZ()); units = u"K"),)),
        namespace(:atmosphere, (auxiliary(:pressure, Ground(XY()); units = u"Pa"),))
    )

    @test length(vars_merged.namespaces) == 1
    ns = first(values(vars_merged.namespaces))
    @test haskey(ns.vars.prognostic, :temp)
    @test haskey(ns.vars.auxiliary, :pressure)
end

@testset "Variables duplicate detection" begin
    # Test that duplicate variable names raise an error
    @test_throws ErrorException Variables(
        prognostic(:temperature, Ground(XYZ()); units = u"K"),
        auxiliary(:temperature, Ground(XY()); units = u"Pa")  # Same name!
    )

    # Test that duplicate namespace names are merged (not an error)
    vars = Variables(
        namespace(:soil, (prognostic(:temp, Ground(XYZ()); units = u"K"),)),
        namespace(:soil, (auxiliary(:pressure, Ground(XY()); units = u"Pa"),))
    )
    @test length(vars.namespaces) == 1
    @test haskey(vars.namespaces, :soil)
end

@testset "Variable helper functions" begin
    # Test prognostic_variables
    vars = Variables(
        prognostic(:temp, Ground(XYZ()); units = u"K"),
        prognostic(:moisture, Ground(XY()); units = u"kg"),
        auxiliary(:pressure, Ground(XYZ()); units = u"Pa"),
        input(:radiation, Canopy(XYZ()); default = 100.0)
    )

    prog_vars = prognostic_variables(vars)
    @test length(prog_vars) == 2
    @test Terrarium.varname(first(prog_vars)) in [:temp, :moisture]

    # Test auxiliary_variables
    aux_vars = auxiliary_variables(vars)
    @test length(aux_vars) == 1
    @test Terrarium.varname(first(aux_vars)) == :pressure

    # Test input_variables
    input_vars = input_variables(vars)
    @test length(input_vars) == 1
    @test Terrarium.varname(first(input_vars)) == :radiation
end

@testset "Variable show methods" begin
    # Test that Variable shows correctly
    var = Terrarium.var(:temperature, Ground(XYZ()), u"K")
    io = IOBuffer()
    show(io, MIME"text/plain"(), var)
    str = String(take!(io))
    @test occursin("temperature", str)
    @test occursin("K", str)

    # Test Variable without units
    var_no_units = Terrarium.var(:pressure, Ground(XY()))
    io = IOBuffer()
    show(io, MIME"text/plain"(), var_no_units)
    str = String(take!(io))
    @test occursin("pressure", str)
end

@testset "Variables show methods" begin
    vars = Variables(
        prognostic(:temp, Ground(XYZ()); units = u"K"),
        auxiliary(:pressure, Ground(XY()); units = u"Pa")
    )

    io = IOBuffer()
    show(io, vars)
    str = String(take!(io))

    @test occursin("Variables", str)
    @test occursin("Prognostic", str)
    @test occursin("Auxiliary", str)
    @test occursin("temp", str)
    @test occursin("pressure", str)
end

@testset "merge Variables" begin
    vars1 = Variables(
        prognostic(:temp, Ground(XYZ()); units = u"K")
    )

    vars2 = Variables(
        auxiliary(:pressure, Ground(XY()); units = u"Pa")
    )

    merged = merge(vars1, vars2)

    @test length(merged.prognostic) == 1
    @test length(merged.auxiliary) == 1
    @test haskey(merged.prognostic, :temp)
    @test haskey(merged.auxiliary, :pressure)
end

@testset "Namespace merging in Variables" begin
    # same-named namespaces should merge their variables
    vars = Variables(
        Terrarium.namespace(:ns, (Terrarium.input(:a, Ground(XY())),)),
        Terrarium.namespace(:ns, (Terrarium.input(:b, Ground(XY())),)),
    )
    @test Tuple(keys(vars.namespaces)) == (:ns,)
    @test Tuple(keys(vars.namespaces[:ns].vars.inputs)) == (:a, :b)

    # nested namespaces merge recursively
    nested = Variables(
        Terrarium.namespace(:outer, (Terrarium.namespace(:inner, (Terrarium.input(:a, Ground(XY())),)),)),
        Terrarium.namespace(:outer, (Terrarium.namespace(:inner, (Terrarium.input(:b, Ground(XY())),)),)),
    )
    @test Tuple(keys(nested.namespaces)) == (:outer,)
    @test Tuple(keys(nested.namespaces[:outer].namespaces[:inner].inputs)) == (:a, :b)
end

@testset "with_scope" begin
    var = input(:temperature, Ground(XYZ()); units = u"K")

    # Test root scope
    result = Terrarium.with_scope((), var)
    @test result === var

    # Test single namespace
    ns = Terrarium.with_scope((:soil,), var)
    @test length(ns.vars) == 1
    @test Terrarium.varname(ns) == :soil
    @test Terrarium.varname(first(ns.vars)) == :temperature

    # Test nested namespaces
    ns1 = Terrarium.with_scope((:atmosphere, :boundary), var)
    @test Terrarium.varname(ns1) == :atmosphere
    ns2 = first(ns1.vars)
    @test Terrarium.varname(ns2) == :boundary
    @test Terrarium.varname(first(ns2.vars)) == :temperature
end
