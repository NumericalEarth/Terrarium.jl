using Terrarium
using Test

@testset "SEB: Albedo" begin
    include("albedo.jl")
end

@testset "SEB: Radiative fluxes" begin
    include("radiative_fluxes.jl")
end

@testset "SEB: Turbulent fluxes" begin
    include("turbulent_fluxes.jl")
end

@testset "SEB: Skin temperature" begin
    include("skin_temperature.jl")
end

@testset "SEB: Ground heat flux" begin
    include("ground_heat_flux.jl")
end
