using Terrarium
using Terrarium: Variables, compute_auxiliary!, vegetation_area_fraction
using Test

# names of all variables (prognostic, tendency, auxiliary, and input) declared by `vars`
variable_names(vars::Variables) = union(keys(vars.prognostic), keys(vars.tendencies), keys(vars.auxiliary), keys(vars.inputs))

@testset "PrescribedVegetation variables" begin
    vegetation = PrescribedVegetation(Float64)
    vars = Variables(vegetation)
    names = variable_names(vars)

    # the leaf area index is a prescribed input
    @test haskey(vars.inputs, :leaf_area_index)
    @test !haskey(vars.auxiliary, :leaf_area_index)
    # there is nothing to integrate in time
    @test isempty(vars.prognostic)
    @test isempty(vars.tendencies)

    # all variables of the included processes are present
    included = (
        vegetation.photosynthesis,
        vegetation.stomatal_conductance,
        vegetation.root_distribution,
        vegetation.plant_available_water,
    )
    included_names = union((variable_names(Variables(process)) for process in included)...)
    @testset "$name present" for name in sort(collect(included_names))
        @test name in names
    end

    # all variables that belong only to processes of prognostic vegetation are absent
    excluded = (
        PALADYNPhenology(Float64),
        PALADYNCarbonDynamics(Float64),
        PALADYNVegetationDynamics(Float64),
        PALADYNAutotrophicRespiration(Float64),
    )
    excluded_names = setdiff(union((variable_names(Variables(process)) for process in excluded)...), included_names, (:leaf_area_index,))
    @test :phenology_factor in excluded_names
    @testset "$name absent" for name in sort(collect(excluded_names))
        @test !(name in names)
    end

    # PrescribedVegetation has no phenology component
    @test !hasfield(typeof(vegetation), :phenology)
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
