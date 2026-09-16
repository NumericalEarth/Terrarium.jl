using Terrarium
using Test

@testset "Photosynthesis" begin
    include("photosynthesis_tests.jl")
end

@testset "Stomatal conductance" begin
    include("stomatal_conductance_tests.jl")
end

@testset "Autotrophic respiration" begin
    include("autotrophic_respiration_tests.jl")
end

@testset "Vegetation carbon dynamics" begin
    include("carbon_dynamics_tests.jl")
end

@testset "Vegetation dynamics" begin
    include("vegetation_dynamics_tests.jl")
end

@testset "Phenology" begin
    include("phenology_tests.jl")
end

@testset "Root distribution" begin
    include("root_distribution_tests.jl")
end

@testset "Plant Available Water" begin
    include("plant_available_water_tests.jl")
end

@testset "Vegetation carbon cycle integration" begin
    include("integration_tests.jl")
end
