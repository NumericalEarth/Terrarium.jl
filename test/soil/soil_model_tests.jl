using Terrarium
using Test

@testset "Soil composition" begin
    include("soil_composition_tests.jl")
end

@testset "Soil stratigraphy" begin
    include("soil_stratigrapy_tests.jl")
end

@testset "Soil energy" begin
    include("soil_energy_tests.jl")
end

@testset "Soil Stefan problem" begin
    include("soil_freeze_thaw_tests.jl")
end

@testset "Soil hydrology" begin
    include("soil_hydrology_tests.jl")
end
