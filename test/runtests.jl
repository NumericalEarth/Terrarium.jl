using Terrarium
using Test

FLAG_ENZYME_TESTS = "enzyme" in ARGS ? true : false
MAIN_TESTS = !FLAG_ENZYME_TESTS ? true : false

if FLAG_ENZYME_TESTS
    @testset "Enzyme" begin
        include("differentiability/soil_energy_diff.jl")
        include("differentiability/soil_hydrology_diff.jl")
        include("differentiability/vegetation_model_diff.jl")
        include("differentiability/snow_model_diff.jl")
        include("differentiability/prescribed_seb_diff.jl")
        include("checkpointing/autodiff.jl")
    end
end

if MAIN_TESTS
    @testset "Grids" begin
        include("grids.jl")
    end

    @testset "Solvers" begin
        include("solvers/test_solvers.jl")
    end

    @testset "State variables" begin
        include("state_variables.jl")
    end

    @testset "Boundary conditions" begin
        include("boundary_conditions.jl")
    end

    @testset "Inputs" begin
        include("inputs/inputs.jl")
        include("inputs/namespaced_inputs.jl")
        include("inputs/input_forcing.jl")
        include("inputs/raster_inputs.jl")
    end

    @testset "Timestepping" begin
        include("timestepping/explicit_step.jl")
        include("timestepping/run_simulation.jl")
        include("timestepping/heun.jl")
        include("timestepping/imex.jl")
        include("timestepping/adaptive_timestep.jl")
    end

    @testset "Thermodynamics" begin
        include("thermodynamics_test.jl")
    end

    @testset "Soil model and processes" begin
        include("soil/soil_model_tests.jl")
    end

    @testset "Snow processes" begin
        include("snow/snow_properties_tests.jl")
        include("snow/snow_energy_tests.jl")
        include("snow/snow_model_tests.jl")
    end

    @testset "Vegetation model and processes" begin
        include("vegetation/vegetation_model_tests.jl")
    end

    @testset "Surface energy balance" begin
        include("surface/seb_tests.jl")
        include("surface/diagnostic_albedo_tests.jl")
    end

    @testset "Surface hydrology" begin
        include("surface/hydrology/surface_hydrology_tests.jl")
    end

    @testset "Coupled models" begin
        include("coupled_models/land_model_tests.jl")
    end

    @testset "Utilities" begin
        include("utils.jl")
    end
end
