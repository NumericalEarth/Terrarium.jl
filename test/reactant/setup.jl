# Model registry for the CPU-vs-Reactant correctness suite.
#
# Add a new tested configuration by adding a `build_model(::Val{:name}, arch, NF)` method that
# returns a NamedTuple `(; model, boundary_conditions, initializers, Δt)`. `arch` is the ONLY
# thing that differs between the CPU and Reactant runs — everything else is identical.

import RingGrids

# --- generic helpers --------------------------------------------------------------------

function build_integrator(v::Val, arch, NF)
    cfg = build_model(v, arch, NF)
    return Terrarium.initialize(
        cfg.model;
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
# Static (constant) prescribed LAI: the time-varying LAI climatology goes through
# `InputSource` temporal interpolation, which does not trace under Reactant yet.

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
