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

@testset "Soil hydrology" begin
    include("soil_hydrology_tests.jl")
end

@testset "Soil fused kernels" begin
    include("soil_fused_kernel_tests.jl")
end
