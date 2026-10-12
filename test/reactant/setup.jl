# Model registry for the CPU-vs-Reactant correctness suite.
#
# Add a new tested configuration by adding a `build_model(::Val{:name}, arch, NF)` method that
# returns a NamedTuple `(; model, boundary_conditions, initializers, Δt)`, optionally with an
# `inputs` entry holding an `InputSource` (e.g. a time-varying `FieldTimeSeries`). `arch` is the
# ONLY thing that differs between the CPU and Reactant runs — everything else is identical.

import RingGrids

using Oceananigans.OutputReaders: Cyclical

# --- generic helpers --------------------------------------------------------------------

function build_integrator(v::Val, arch, NF)
    cfg = build_model(v, arch, NF)
    inputs = hasproperty(cfg, :inputs) ? cfg.inputs : InputSources(NF)
    return Terrarium.initialize(
        cfg.model;
        inputs,
        boundary_conditions = cfg.boundary_conditions,
        initializers = cfg.initializers
    )
end

cpu_dt(v::Val, NF) = build_model(v, CPU(), NF).Δt

# --- :soil_heat_column — minimal single-column SoilModel (heat conduction) --------------

function build_model(::Val{:soil_heat_column}, arch, NF)
    grid = ColumnGrid(arch, NF, UniformSpacing(Δz = NF(0.2), N = 10))
    model = SoilModel(grid)
    # constant surface temperature; default (zero-flux) bottom boundary
    bcs = PrescribedSurfaceTemperature(:T_ub, NF(1))
    # linear initial temperature profile with depth (z ≤ 0)
    inits = (temperature = (x, z) -> NF(-1) - NF(0.05) * z,)
    return (; model, boundary_conditions = bcs, initializers = inits, Δt = NF(600))
end

# --- :soil_heat_column_stretched — single-column SoilModel on an ExponentialSpacing grid -
# Same physics as :soil_heat_column but with array-valued (stretched) vertical coordinates,
# which exercise the upstream fix (Oceananigans ≥ 0.110.9) that lets array z trace under
# Reactant.
function build_model(::Val{:soil_heat_column_stretched}, arch, NF)
    grid = ColumnGrid(arch, NF, ExponentialSpacing(Δz_min = NF(0.05), Δz_max = NF(1), N = 10))
    model = SoilModel(grid)
    bcs = PrescribedSurfaceTemperature(:T_ub, NF(1))
    inits = (temperature = (x, z) -> NF(-1) - NF(0.05) * z,)
    return (; model, boundary_conditions = bcs, initializers = inits, Δt = NF(600))
end

# --- :soil_heat_column_implicit — :soil_heat_column with implicit heat conduction ---------
# Same column as :soil_heat_column, integrated with `IMEX(ForwardEuler, BackwardEuler)` so that the
# interior heat conduction runs through the `Picard` solver with a `TridiagonalOperator` (two Picard
# iterations, so the in-step energy closure is traced as well). Value boundary conditions on
# temperature are not supported by the implicit operator, so the surface is cooled by a constant
# `SoilHeatFlux` instead of a prescribed surface temperature. The step is several times the explicit
# diffusive limit; the column stays thawed so the comparison does not hinge on the exact step at which
# a cell enters the phase change interval, which CPU and XLA need not agree on.

function build_model(::Val{:soil_heat_column_implicit}, arch, NF)
    grid = ColumnGrid(arch, NF, UniformSpacing(Δz = NF(0.2), N = 10))
    solver = Picard(NF; iterations = 2)
    timestepper = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver))
    model = SoilModel(grid; timestepper)
    # constant heat loss at the surface (W/m², positive upward); default (zero-flux) bottom boundary
    bcs = SoilHeatFlux(NF(20))
    inits = (temperature = (x, z) -> NF(5) - NF(0.05) * z,)
    return (; model, boundary_conditions = bcs, initializers = inits, Δt = NF(3600))
end

# --- :soil_heat_column_implicit_freezing — implicit heat conduction across the freezing point -
# As :soil_heat_column_implicit, but the column starts just above the freezing point (0.5 °C) under a
# strong surface cooling flux, so that a freezing front enters the column (within the 100 test steps the
# top cell freezes completely and the cell below starts to freeze) and the regularized apparent
# heat capacity of the `FreeWater` freeze curve (switched on for cells in phase change) is exercised
# under XLA, together with the in-step energy closure of the two Picard iterations.

function build_model(::Val{:soil_heat_column_implicit_freezing}, arch, NF)
    grid = ColumnGrid(arch, NF, UniformSpacing(Δz = NF(0.2), N = 10))
    solver = Picard(NF; iterations = 2)
    timestepper = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver))
    model = SoilModel(grid; timestepper)
    bcs = SoilHeatFlux(NF(80))
    # water must be present for a phase change (the default initial saturation is zero)
    inits = (temperature = NF(0.5), saturation_water_ice = NF(0.8))
    return (; model, boundary_conditions = bcs, initializers = inits, Δt = NF(3600))
end

# --- :soil_richards_column_implicit — implicit Richards flow and heat conduction -----------
# Single-column SoilModel with Richards-equation hydrology (Brooks–Corey retention curve) under
# `IMEX(ForwardEuler, BackwardEuler)`, so that both `internal_energy` and `saturation_water_ice` are
# integrated by the `Picard` solver (two Picard iterations: the modified Picard iteration
# of the Richards equation with the in-step hydrology closure, and the implicit heat conduction), and
# the energy tendency carries the explicit advection of heat by the Darcy flux. Water infiltrates
# through a constant `InfiltrationFlux` at the top (impermeable bottom). The column is thawed and, with a
# low saturated conductivity, stays unsaturated within the 100 test steps (saturation 0.35 to 0.46): once
# water accumulates to saturation at the impermeable base, whether a cell is exactly saturated switches
# the water table and the hydrostatic head, and the run becomes sensitive to roundoff (Float32 and
# Float64 CPU runs then already differ by about 1 K), which CPU and XLA need not agree on.

function build_model(::Val{:soil_richards_column_implicit}, arch, NF)
    grid = ColumnGrid(arch, NF, UniformSpacing(Δz = NF(0.1), N = 20))
    # The Brooks–Corey retention curve (the default) is used since the inverse of the FreezeCurves
    # van Genuchten curve is not type stable for Float32 (one branch promotes to Float64), which
    # leaves a stack allocation in the saturation–pressure closure kernel that XLA cannot raise.
    hydraulic_properties = ConstantSoilHydraulics(NF; saturated_conductivity = NF(1.0e-6))
    hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(NF; hydrology)
    solver = Picard(NF; iterations = 2)
    timestepper = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver))
    model = SoilModel(grid; soil, timestepper)
    # infiltration as a flux of saturation (m/s, positive upward) and a constant surface heat loss (W/m²)
    bcs = merge(InfiltrationFlux(NF(-2.0e-7)), SoilHeatFlux(NF(10)))
    inits = (
        temperature = (x, z) -> NF(5) - NF(0.05) * z,
        saturation_water_ice = (x, z) -> NF(0.3) - NF(0.05) * z,
    )
    return (; model, boundary_conditions = bcs, initializers = inits, Δt = NF(3600))
end

# --- :soil_heat_global — global ColumnRingGrid SoilModel (heat conduction) --------------
# Mirrors examples/simulations/soil_heat_global.jl, but with a synthetic all-land mask
# (no NetCDF inputs in CI) and a purely functional, smooth surface BC (no array gather).

function build_model(::Val{:soil_heat_global}, arch, NF)
    rings = RingGrids.FullGaussianGrid(4)                 # small: 4 nlat_half
    grid = ColumnRingGrid(arch, NF, UniformSpacing(Δz = NF(0.2), N = 20), rings)  # default all-active mask
    model = SoilModel(grid)

    # smooth surface forcing as a function of column index x and time t (no indexed lookup);
    # the closure must capture only isbits values (no NF::Type!) to be a valid kernel argument
    amplitude = NF(5)
    ω = NF(2π) / NF(24 * 3600)
    surface_temperature(x, t) = amplitude * sin(ω * t) * cos(x)
    bcs = PrescribedSurfaceTemperature(:T_ub, surface_temperature)

    # smooth, deterministic initial temperature profile
    inits = (temperature = (x, z) -> NF(2) * cos(NF(x)) - NF(0.05) * z,)
    return (; model, boundary_conditions = bcs, initializers = inits, Δt = NF(600))
end

# --- :snow_column — minimal single-column standalone SnowModel --------------------------
# Exercises the snow closure (energy↔temperature) and the mass/energy tendencies under Reactant.
# Boundary heat fluxes and SWE/temperature are prescribed as constant input/initial fields.

function build_model(::Val{:snow_column}, arch, NF)
    grid = ColumnGrid(arch, NF, UniformSpacing(Δz = NF(0.2), N = 10))
    model = SnowModel(grid)
    # frozen pack with steady conductive gain at the base and loss at the surface (no melt)
    inits = (
        snow_water_equivalent = NF(0.3),
        snow_temperature = NF(-5),
        basal_heat_flux = NF(2),
        surface_heat_flux = NF(10),
    )
    return (; model, boundary_conditions = (;), initializers = inits, Δt = NF(600))
end

# --- :land_soil_snow — coupled LandModel: soil (Richards + heat) + snow, no vegetation ---
# *Coupled* configuration
#
# Cold winter conditions: a frozen, unsaturated soil column under a shallow snowpack. The pack is
# kept below freezing (air temperature < 0) so the 100-step comparison does not hinge on the exact
# step at which a melt threshold is crossed, which CPU and XLA need not agree on.
#
# TODO: Currently this only works with the `NewtonsMethod` solver.

function build_model(::Val{:land_soil_snow}, arch, NF)
    grid = ColumnGrid(arch, NF, ExponentialSpacing(Δz_min = NF(0.05), Δz_max = NF(1), N = 10))
    swrc = VanGenuchten(α = NF(2), n = NF(2))
    hydraulic_properties = ConstantSoilHydraulics(NF; swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(NF))
    hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(NF; hydrology)
    # Use a `NewtonSolver` with fixed iterations for Reactant
    skin_temperature = Terrarium.ImplicitSkinTemperature(NF; solver = Terrarium.NewtonSolver(NF; iterations = 5))
    seb = SurfaceEnergyBalance(NF; skin_temperature)
    model = LandModel(grid; soil, snow = SingleLayerSnow(NF), vegetation = nothing, surface_energy_balance = seb)
    # Soil BCs are set up by `LandModel` itself (ground/soil heat flux and infiltration coupling),
    # so only the initial state and the prescribed atmospheric inputs are specified here.
    inits = (
        temperature = (x, z) -> NF(-1) - NF(0.02) * z,
        saturation_water_ice = (x, z) -> min(NF(1), NF(0.8) - NF(0.05) * z),
        snow_water_equivalent = NF(0.2),
        snow_temperature = NF(-5),
        air_temperature = NF(-2),
    )
    return (; model, boundary_conditions = (;), initializers = inits, Δt = NF(600))
end

# --- :vegetation_column — standalone VegetationModel with PrescribedVegetation -----------
# Static (constant) prescribed LAI; see :vegetation_column_lai_cycle for the time-varying input.

function build_model(::Val{:vegetation_column}, arch, NF)
    grid = ColumnGrid(arch, NF, ExponentialSpacing(Δz_min = NF(0.05), Δz_max = NF(1), N = 10))
    vegetation = PrescribedVegetation(NF)
    model = VegetationModel(grid; vegetation)
    inits = (
        leaf_area_index = NF(3),
        air_temperature = NF(20),
    )
    return (; model, boundary_conditions = (;), initializers = inits, Δt = NF(600))
end

# --- :vegetation_column_lai_cycle — as :vegetation_column, with a time-varying LAI input -
# The leaf area index is a cyclical `FieldTimeSeries` input, so every step interpolates the
# series at the traced `clock.time` from *inside* the compiled loop (the `FieldTimeSeriesInputSource`
# path used by the LAI climatology in `examples/simulations/vegetation_global.jl`). The cycle is
# deliberately short (four snapshots, 12 h period) so that the 100 test steps of 600 s cross
# several snapshot boundaries and wrap around the period at least once.

function build_model(::Val{:vegetation_column_lai_cycle}, arch, NF)
    grid = ColumnGrid(arch, NF, ExponentialSpacing(Δz_min = NF(0.05), Δz_max = NF(1), N = 10))
    vegetation = PrescribedVegetation(NF)
    model = VegetationModel(grid; vegetation)
    # snapshots every 3 h; `Cyclical()` infers the period 4 × 3 h = 12 h from the spacing
    times = 0.0:(3 * 3600.0):(9 * 3600.0)
    lai_fts = FieldTimeSeries(grid, XY(), times; time_indexing = Cyclical())
    for (n, lai) in enumerate((1, 4, 2, 3))
        set!(lai_fts[n], NF(lai))
    end
    inputs = InputSource(lai_fts; name = :leaf_area_index)
    inits = (air_temperature = NF(20),)
    return (; model, boundary_conditions = (;), initializers = inits, inputs, Δt = NF(600))
end

# --- :land_default — the default coupled LandModel: soil + snow + vegetation ------------
# `LandModel(grid)` with every component at its default (`VegetationCarbonCycle` with PALADYN
# phenology/carbon/dynamics, `SoilEnergyWaterCarbon` with Richards-equation hydrology and SURFEX
# hydraulics, `SingleLayerSnow`, default surface energy balance and hydrology), except the
# skin-temperature solver: the default `RootSolver` iterates to a tolerance, and that convergence-tested
# loop cannot be raised to StableHLO (see docs/dev/2026-08/2026-08-04-PLAN_reactant_coupled_land_model.md),
# so the fixed-iteration `NewtonSolver` is used as in :land_soil_snow.
#
# Cold conditions as in :land_soil_snow (no melt, so the comparison does not hinge on the exact step at
# which a threshold is crossed); the vegetation carbon pool and area fraction start nonzero so the
# vegetation tendencies are active rather than identically zero.

function build_model(::Val{:land_default}, arch, NF)
    grid = ColumnGrid(arch, NF, ExponentialSpacing(Δz_min = NF(0.05), Δz_max = NF(1), N = 10))
    skin_temperature = Terrarium.ImplicitSkinTemperature(NF; solver = Terrarium.NewtonSolver(NF; iterations = 5))
    seb = SurfaceEnergyBalance(NF; skin_temperature)
    model = LandModel(grid; surface_energy_balance = seb)
    inits = (
        temperature = (x, z) -> NF(-1) - NF(0.02) * z,
        saturation_water_ice = (x, z) -> min(NF(1), NF(0.8) - NF(0.05) * z),
        snow_water_equivalent = NF(0.2),
        snow_temperature = NF(-5),
        air_temperature = NF(-2),
        carbon_vegetation = NF(2),          # kgC/m²
        vegetation_area_fraction = NF(0.5),
    )
    return (; model, boundary_conditions = (;), initializers = inits, Δt = NF(600))
end
