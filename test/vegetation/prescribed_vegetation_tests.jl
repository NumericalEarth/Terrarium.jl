using Terrarium
using Terrarium: Variables, compute_auxiliary!, vegetation_area_fraction
using Test

@testset "PrescribedVegetation variables" begin
    vars = Variables(PrescribedVegetation(Float64))
    # the leaf area index is a prescribed input, and there is no phenology factor
    @test haskey(vars.inputs, :leaf_area_index)
    @test !haskey(vars.auxiliary, :phenology_factor)
    @test !haskey(vars.inputs, :phenology_factor)
    @test isempty(vars.prognostic)
    # PrescribedVegetation has no phenology component
    @test !hasfield(typeof(PrescribedVegetation(Float64)), :phenology)
    @test_throws MethodError PrescribedVegetation(Float64; phenology = PALADYNPhenology(Float64))
end

@testset "Prescribed leaf area index drives photosynthesis" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 5))
    traits = PlantTraits(Float64; maximum_leaf_area_index = 4.0)
    vegetation = PrescribedVegetation(Float64; traits)
    model = VegetationModel(grid; vegetation)
    integrator = initialize(model)
    state = integrator.state
    set!(state.air_temperature, 20.0) # °C
    set!(state.surface_shortwave_down, 400.0) # W/m²
    GPP = map((0.0, 1.0, 2.0, 4.0)) do LAI
        set!(state.leaf_area_index, LAI)
        compute_auxiliary!(state, model)
        return Array(interior(state.gross_primary_production))[1]
    end
    # no leaves, no photosynthesis; GPP increases (and saturates) with the prescribed LAI
    @test GPP[1] == 0
    @test all(>(0), GPP[2:end])
    @test issorted(GPP)
    # the vegetated area fraction is the leaf area relative to its maximum, clamped to [0, 1]
    for (LAI, fraction) in ((1.0, 0.25), (2.0, 0.5), (6.0, 1.0))
        set!(state.leaf_area_index, LAI)
        @test vegetation_area_fraction(1, 1, grid, (; leaf_area_index = state.leaf_area_index), vegetation) ≈ fraction
    end
end
