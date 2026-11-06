using Terrarium
using Terrarium: compute_λ_NPP, compute_Λ_loc
using Oceananigans.OutputReaders: Cyclical
using Test

# Time-integrated checks of the vegetation carbon cycle in a single standalone column
# (no soil coupling, so the soil moisture limiting factor is one).

const SECONDS_PER_DAY = 86400.0

column_value(field) = Array(interior(field))[1]

@testset "Vegetation carbon budget closure" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 5))
    vegetation = PrognosticVegetation(NF)
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

@testset "Multi-year seasonal column run" begin
    NF = Float64
    grid = ColumnGrid(CPU(), NF, ExponentialSpacing(N = 5))
    model = VegetationModel(grid; vegetation = PrognosticVegetation(NF))
    # Daily climatology of a cold-temperate site: air temperature between -7 and 17 °C and daily mean
    # shortwave radiation between 30 and 270 W/m², repeated every 365 days.
    year = 365 * SECONDS_PER_DAY
    times = 0.0:SECONDS_PER_DAY:(year - SECONDS_PER_DAY)
    seasonal(t, mean, amplitude) = mean + amplitude * sin(2π * (t / year - 0.3))
    air_temperature = FieldTimeSeries(grid, XY(), times; time_indexing = Cyclical())
    shortwave = FieldTimeSeries(grid, XY(), times; time_indexing = Cyclical())
    for (n, t) in enumerate(times)
        set!(air_temperature[n], seasonal(t, 5.0, 12.0))
        set!(shortwave[n], seasonal(t, 150.0, 120.0))
    end
    inputs = InputSources(
        InputSource(air_temperature; name = :air_temperature, units = u"°C"),
        InputSource(shortwave; name = :surface_shortwave_down, units = u"W/m^2"),
    )
    initializers = (carbon_vegetation = 5.0, vegetation_area_fraction = 0.5)
    integrator = initialize(model; inputs, initializers)
    state = integrator.state
    for _ in 1:3
        annual_GPP = 0.0
        annual_Ra = 0.0
        ϕ_min, ϕ_max = Inf, -Inf
        C_veg_min, ν_min, ν_max = Inf, Inf, -Inf
        for _ in 1:365
            timestep!(integrator, SECONDS_PER_DAY)
            annual_GPP += column_value(state.gross_primary_production) * SECONDS_PER_DAY
            annual_Ra += column_value(state.autotrophic_respiration) * SECONDS_PER_DAY
            ϕ = column_value(state.phenology_factor)
            ϕ_min, ϕ_max = min(ϕ_min, ϕ), max(ϕ_max, ϕ)
            C_veg_min = min(C_veg_min, column_value(state.carbon_vegetation))
            ν = column_value(state.vegetation_area_fraction)
            ν_min, ν_max = min(ν_min, ν), max(ν_max, ν)
        end
        @test isfinite(C_veg_min) && C_veg_min > 0
        @test 0 ≤ ν_min && ν_max ≤ 1
        # the deciduous phenology completes an annual cycle
        @test ϕ_min < 0.05 && ϕ_max > 0.95
        # annual GPP (kgC/m²/yr) and the ratio of autotrophic respiration to GPP are in the ranges
        # reported for temperate and boreal forests
        @test 0.3 < annual_GPP < 2.0
        @test 0.3 < annual_Ra / annual_GPP < 0.7
    end
end
