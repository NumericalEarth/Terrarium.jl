using Terrarium
using Terrarium: ConstantAlbedo, PrescribedAlbedo, albedo, emissivity
using Test

using Oceananigans

@testset "Constant albedo" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing())
    albd = ConstantAlbedo(albedo = 0.4, emissivity = 0.8)
    @test albedo(1, 1, nothing, grid, albd) == 0.4
    @test emissivity(1, 1, nothing, grid, albd) == 0.8
end

@testset "PrescribedAlbedo" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing())
    albd = PrescribedAlbedo(eltype(grid))
    state = (
        inputs = (
            albedo = set!(Field(grid, Terrarium.Ground(XY())), 0.4),
            emissivity = set!(Field(grid, Terrarium.Ground(XY())), 0.8),
        )
    )
    @test albedo(1, 1, grid, state, albd) == 0.4
    @test emissivity(1, 1, grid, state, albd) == 0.8
end
