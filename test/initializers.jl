using Terrarium
using Terrarium: RingGrids, parameters, variables, input_variables, varname
using Oceananigans.Grids: LatitudeLongitudeGrid, Bounded
using Test

@testset "Initializers declare input variables" begin
    NF = Float32
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 10))
    model = SoilModel(grid)
    # the default initializer contributes no variables
    @test :initial_surface_temperature ∉ map(varname, input_variables(variables(model)))
    model = SoilModel(grid; initializer = SoilInitializer(NF))
    names = map(varname, input_variables(variables(model)))
    @test :initial_surface_temperature ∈ names
    @test :geothermal_heat_flux ∈ names
    @test :vadose_zone_saturation ∈ names
    @test :water_table_depth ∈ names
end

@testset "Scalar QuasiThermalSteadyState and SaturationWaterTable" begin
    NF = Float32
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 10))
    T₀ = NF(-1)
    Qgeo = NF(0.05)
    energy = QuasiThermalSteadyState(NF; T₀, Qgeo)
    hydrology = SaturationWaterTable(NF; water_table_depth = NF(1), vadose_zone_saturation = NF(0.5))
    model = SoilModel(grid; initializer = SoilInitializer(NF; energy, hydrology))
    integrator = initialize(model)
    state = integrator.state
    z = znodes(state.temperature)
    @test interior(state.temperature)[1, 1, :] ≈ T₀ .- Qgeo .* z
    @test all(interior(state.initial_surface_temperature) .== T₀)
    @test all(interior(state.geothermal_heat_flux) .== Qgeo)
    sat = interior(state.saturation_water_ice)[1, 1, :]
    @test all(sat[z .<= -1] .== 1)
    @test all(sat[z .> -1] .== NF(0.5))
    # re-initialization reproduces the same state
    T_before = copy(interior(state.temperature))
    Terrarium.initialize!(integrator)
    @test interior(state.temperature) == T_before
end

@testset "Column-varying initial values" begin
    NF = Float32
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 5), 3)
    # function of the column coordinate
    energy = QuasiThermalSteadyState(NF; T₀ = x -> NF(x), Qgeo = zero(NF))
    integrator = initialize(SoilModel(grid; initializer = SoilInitializer(NF; energy)))
    T = interior(integrator.state.temperature)
    @test T[:, 1, end] ≈ interior(integrator.state.initial_surface_temperature)[:, 1, 1]
    @test T[1, 1, end] < T[2, 1, end] < T[3, 1, end]
    # Field with the same values
    T₀_field = Field(grid, Terrarium.Ground(XY()))
    set!(T₀_field, x -> NF(x))
    energy = QuasiThermalSteadyState(NF; T₀ = T₀_field, Qgeo = zero(NF))
    integrator_field = initialize(SoilModel(grid; initializer = SoilInitializer(NF; energy)))
    @test interior(integrator_field.state.temperature) == T
    # fields are not parameters
    @test :T₀ ∉ keys(vec(parameters(integrator_field.model)).initializer.energy)
end

@testset "InputSource overrides the geothermal heat flux default" begin
    NF = Float32
    rings = RingGrids.FullGaussianGrid(4)
    grid = ColumnRingGrid(CPU(), NF, UniformSpacing(Δz = NF(0.5), N = 4), rings)
    model = SoilModel(grid; initializer = SoilInitializer(NF; energy = QuasiThermalSteadyState(NF; T₀ = zero(NF), Qgeo = NF(0.02))))
    Qgeo_field = Field(grid, Terrarium.Ground(XY()))
    set!(Qgeo_field, NF(0.1))
    source = InputSource(grid, Qgeo_field; name = :geothermal_heat_flux, domain = Terrarium.Ground(), units = u"W/m^2")
    integrator = initialize(model; inputs = InputSources(source))
    @test all(interior(integrator.state.geothermal_heat_flux) .== NF(0.1))
    z = znodes(integrator.state.temperature)
    @test interior(integrator.state.temperature)[1, 1, :] ≈ -NF(0.1) .* z
    # mismatched units are reported as a conflict
    bad_source = InputSource(grid, Qgeo_field; name = :geothermal_heat_flux, domain = Terrarium.Ground())
    @test_throws ErrorException initialize(model; inputs = InputSources(bad_source))
end

@testset "LatitudinalClimatology" begin
    NF = Float32
    climatology = LatitudinalClimatology(NF; T_equator = NF(10), ΔT = NF(40))
    @test climatology(0) == NF(10)
    @test climatology(90) ≈ NF(-30)
    @test climatology(-90) ≈ NF(-30)

    rings = RingGrids.FullGaussianGrid(4)
    grid = ColumnRingGrid(CPU(), NF, UniformSpacing(Δz = NF(0.5), N = 4), rings)
    @test length(λnodes(grid)) == length(φnodes(grid)) == size(grid, 1)
    model = SoilModel(grid; initializer = SoilInitializer(NF; energy = QuasiThermalSteadyState(NF; T₀ = climatology)))
    integrator = initialize(model)
    T_surface = interior(integrator.state.initial_surface_temperature)[:, 1, 1]
    @test T_surface ≈ climatology.(φnodes(grid))

    # also as a direct field initializer
    integrator_direct = initialize(SoilModel(grid); initializers = (temperature = climatology,))
    @test interior(integrator_direct.state.temperature)[:, 1, 1] ≈ climatology.(φnodes(grid))

    # parameters are exposed and re-evaluated on reconstruction
    ps = vec(parameters(model))
    @test haskey(ps.initializer.energy.T₀, :T_equator)
    ps.initializer.energy.T₀.T_equator = NF(30)
    integrator = initialize(integrator, ps)
    @test interior(integrator.state.initial_surface_temperature)[:, 1, 1] ≈ LatitudinalClimatology(NF; T_equator = NF(30)).(φnodes(grid))

    # masked grid: only active columns are evaluated
    mask = convert.(Bool, ones(rings))
    mask[1:5] .= false
    masked_grid = ColumnRingGrid(CPU(), NF, UniformSpacing(Δz = NF(0.5), N = 4), rings, mask)
    integrator_masked = initialize(SoilModel(masked_grid; initializer = SoilInitializer(NF; energy = QuasiThermalSteadyState(NF; T₀ = climatology))))
    @test interior(integrator_masked.state.initial_surface_temperature)[:, 1, 1] ≈ climatology.(φnodes(masked_grid))

    # LatitudeLongitudeGrid
    llg = LatitudeLongitudeGrid(CPU(), NF; size = (2, 3, 5), longitude = (0, 10), latitude = (0, 60), z = (-1, 0), topology = (Bounded, Bounded, Bounded))
    integrator_llg = initialize(SoilModel(llg; initializer = SoilInitializer(NF; energy = QuasiThermalSteadyState(NF; T₀ = climatology))))
    @test interior(integrator_llg.state.initial_surface_temperature)[1, :, 1] ≈ climatology.(φnodes(llg, Center()))
end
