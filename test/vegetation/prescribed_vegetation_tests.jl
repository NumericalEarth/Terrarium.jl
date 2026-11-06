using Terrarium
using Terrarium: Variables, vegetation_area_fraction
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
    @test isempty(setdiff(included_names, names))

    # all variables that belong only to processes of prognostic vegetation are absent
    excluded = (
        PALADYNPhenology(Float64),
        PALADYNCarbonDynamics(Float64),
        PALADYNVegetationDynamics(Float64),
        PALADYNAutotrophicRespiration(Float64),
    )
    excluded_names = setdiff(union((variable_names(Variables(process)) for process in excluded)...), included_names, (:leaf_area_index,))
    @test isempty(intersect(excluded_names, names))

    # PrescribedVegetation has no phenology component
    @test !hasfield(typeof(vegetation), :phenology)
    @test_throws MethodError PrescribedVegetation(Float64; phenology = PALADYNPhenology(Float64))
end

@testset "Prescribed leaf area index under time stepping" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 5))
    traits = PlantTraits(Float64; maximum_leaf_area_index = 4.0)
    model = VegetationModel(grid; vegetation = PrescribedVegetation(Float64; traits))
    LAI = 3.0
    # the leaf area index is prescribed either by an input source or by setting the input field directly
    lai_field = Field(grid, XY())
    set!(lai_field, LAI)
    lai_source = InputSource(grid, lai_field; name = :leaf_area_index, domain = Terrarium.Canopy())
    @testset "$label" for (label, inputs) in (("input source", InputSources(lai_source)), ("set!", InputSources(Float64)))
        integrator = initialize(model; inputs)
        state = integrator.state
        label == "set!" && set!(state.leaf_area_index, LAI)
        set!(state.air_temperature, 20.0) # °C
        set!(state.surface_shortwave_down, 400.0) # W/m²
        for _ in 1:24
            timestep!(integrator, 3600.0)
            # the prescribed value is neither integrated nor overwritten by the vegetation processes
            @test all(interior(state.leaf_area_index) .== LAI)
        end
    end
end

@testset "Vegetated area fraction from prescribed leaf area index" begin
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 5))
    traits = PlantTraits(Float64; maximum_leaf_area_index = 4.0)
    vegetation = PrescribedVegetation(Float64; traits)
    model = VegetationModel(grid; vegetation)
    state = initialize(model).state
    # the vegetated area fraction is the leaf area relative to its maximum, clamped to [0, 1]
    for (LAI, fraction) in ((0.0, 0.0), (1.0, 0.25), (2.0, 0.5), (6.0, 1.0))
        set!(state.leaf_area_index, LAI)
        @test vegetation_area_fraction(1, 1, grid, (; leaf_area_index = state.leaf_area_index), vegetation) ≈ fraction
    end
end
