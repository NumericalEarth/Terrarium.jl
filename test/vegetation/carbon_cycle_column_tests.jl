using Terrarium
using Terrarium: Variables, compute_λ_NPP, compute_Λ_loc
using Test

# Time-integrated checks of the vegetation carbon cycle in a single standalone column
# (no soil coupling, so the soil moisture limiting factor is one).

const SECONDS_PER_DAY = 86400.0

column_value(field) = Array(interior(field))[1]

# names of all variables (prognostic, tendency, auxiliary, and input) declared by `vars`
variable_names(vars::Variables) = union(keys(vars.prognostic), keys(vars.tendencies), keys(vars.auxiliary), keys(vars.inputs))

@testset "VegetationCarbonCycle variables" begin
    vegetation = VegetationCarbonCycle(Float64)
    vars = Variables(vegetation)
    components = (
        vegetation.photosynthesis,
        vegetation.stomatal_conductance,
        vegetation.autotrophic_respiration,
        vegetation.phenology,
        vegetation.carbon_dynamics,
        vegetation.vegetation_dynamics,
        vegetation.root_distribution,
        vegetation.plant_available_water,
    )
    # all variables of all component processes are present
    component_names = union((variable_names(Variables(component)) for component in components)...)
    @test isempty(setdiff(component_names, variable_names(vars)))
    # the vegetation carbon pool, the vegetated area fraction, and the growing degree days are prognostic
    @test issetequal(keys(vars.prognostic), (:carbon_vegetation, :vegetation_area_fraction, :growing_degree_days))
    # and the leaf area index is computed from them rather than prescribed
    @test haskey(vars.auxiliary, :leaf_area_index)
    @test !haskey(vars.inputs, :leaf_area_index)
end

@testset "Vegetation carbon budget closure" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 5))
    vegetation = VegetationCarbonCycle(NF)
    model = VegetationModel(grid; vegetation)
    initializers = (carbon_vegetation = 5.0, vegetation_area_fraction = 0.5, growing_degree_days = 300.0)
    integrator = initialize(model; initializers, inputs = InputSources(NF))
    state = integrator.state
    set!(state.air_temperature, 15.0)
    set!(state.surface_shortwave_down, 300.0)
    Δt = SECONDS_PER_DAY
    C_veg_initial = column_value(state.carbon_vegetation)
    budget = 0.0
    # the auxiliaries in the state are those of the current state; timestep! updates them after the step
    Terrarium.compute_auxiliary!(state, model)
    for _ in 1:30
        LAI_b = column_value(state.balanced_leaf_area_index)
        NPP = column_value(state.net_primary_production)
        λ_NPP = compute_λ_NPP(vegetation.carbon_dynamics, vegetation.traits, LAI_b)
        Λ_loc = compute_Λ_loc(vegetation.carbon_dynamics, vegetation.traits, LAI_b)
        # Willeit & Ganopolski (2016), Eq. 72: the share λ_NPP of NPP goes to spreading, the rest to the
        # carbon pool, and litterfall leaves it
        budget += ((1 - λ_NPP) * NPP - Λ_loc) * Δt
        timestep!(integrator, Δt)
    end
    @test column_value(state.carbon_vegetation) - C_veg_initial ≈ budget rtol = 1.0e-10
end

@testset "Leaf area index changes under time stepping" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 5))
    model = VegetationModel(grid; vegetation = VegetationCarbonCycle(NF))
    # spring green-up: warm air and no accumulated growing degree days yet
    initializers = (carbon_vegetation = 5.0, vegetation_area_fraction = 0.5, growing_degree_days = 0.0)
    integrator = initialize(model; initializers)
    state = integrator.state
    set!(state.air_temperature, 15.0)
    set!(state.surface_shortwave_down, 300.0)
    Terrarium.compute_auxiliary!(state, model)
    LAI = [column_value(state.leaf_area_index)]
    for _ in 1:10
        timestep!(integrator, SECONDS_PER_DAY)
        push!(LAI, column_value(state.leaf_area_index))
    end
    @test all(isfinite, LAI)
    # the leaves come out as growing degree days accumulate
    @test LAI[1] == 0
    @test LAI[end] > 0
    @test issorted(LAI) && allunique(LAI[2:end])
end
