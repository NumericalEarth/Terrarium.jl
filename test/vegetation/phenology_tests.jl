using Terrarium
using Terrarium: compute_f_deciduous, compute_phenology_factor, compute_LAI
using Test

@testset "f_deciduous test" begin
    phenol = PALADYNPhenology{Float64}()
    # For now, test that f_deciduous = 0
    f_deciduous = compute_f_deciduous(phenol)
    @test f_deciduous == 0.0
end

@testset "phen test" begin
    phenol = PALADYNPhenology{Float64}()
    # For now, test that phen = 1
    phen = compute_phenology_factor(phenol)
    @test phen == 1.0
end

@testset "LAI test" begin
    phenol = PALADYNPhenology{Float64}()
    LAI_b = 5.0  # Mock value
    LAI = compute_LAI(phenol, LAI_b)
    # For now, test that LAI = LAI_b (since f_deciduous=0 and phen=1)
    @test LAI == LAI_b
end
