# Model configuration registry for the Terrarium benchmark suite.
#
# A configuration is one `build_model(::Val{:name}, arch, NF; nlat_half, nz, model_kwargs)` method
# returning `(; model, boundary_conditions, initializers, Δt)`. The architecture and the resolution
# (`nlat_half` horizontally, `nz` vertically) are the ONLY things that vary between benchmark runs;
# everything else is fixed by the configuration so that numbers from different architectures and
# resolutions stay comparable.
#
# This mirrors the registry pattern used by `test/reactant/setup.jl`.

import RingGrids

"""
All configuration names known to the benchmark suite, in the order they appear in tables.
"""
const CONFIGURATIONS = (:land, :land_no_vegetation, :soil, :soil_heat)

"""
Horizontal grid for a given `nlat_half`: a full Gaussian grid with `2 * nlat_half` latitude rings.
"""
benchmark_rings(nlat_half::Integer) = RingGrids.FullGaussianGrid(nlat_half)

"""
Number of columns (horizontal grid points) of the benchmark grid at `nlat_half`. The benchmark grids
use an all-land mask, so every grid point of the ring grid is an active column: `8 * nlat_half^2`.
"""
ncolumns(nlat_half::Integer) = RingGrids.get_npoints(benchmark_rings(nlat_half))

"""
Approximate horizontal resolution in degrees latitude at the given `nlat_half`.
"""
resolution_degrees(nlat_half::Integer) = 180 / (2 * nlat_half)

"""
Benchmark grid: `nlat_half` sets the horizontal resolution, `nz` the number of soil layers.

The vertical spacing grows quasi-exponentially from 5 cm at the surface to 1 m at the bottom, as in
`test/coupled_models/land_model_tests.jl`. Note that the two endpoints are fixed, so a larger `nz`
both refines the profile and deepens the column.
"""
function benchmark_grid(arch, ::Type{NF}; nlat_half::Integer, nz::Integer) where {NF}
    spacing = ExponentialSpacing(Δz_min = NF(0.05), Δz_max = NF(1), N = nz)
    return ColumnRingGrid(arch, NF, spacing, benchmark_rings(nlat_half))
end

# Resolve a NamedTuple of per-run model kwargs.
# Convention: a `Type` or `Function` value is called with `NF` — every Terrarium component has an `T(NF)`
# constructor — so a suite can write `model_kwargs = (timestepper = Heun,)`. Everything else is
# passed through unchanged.
_resolve_kwarg_value(v::Type, ::Type{NF}) where {NF} = v(NF)
_resolve_kwarg_value(v::Function, ::Type{NF}) where {NF} = v(NF)
_resolve_kwarg_value(v, ::Type) = v
function resolve_model_kwargs(nt::NamedTuple, ::Type{NF}) where {NF}
    return NamedTuple{keys(nt)}(map(v -> _resolve_kwarg_value(v, NF), values(nt)))
end

"""
Model kwargs of the coupled land configurations, merged in before the per-run `model_kwargs`: the
skin temperature is solved with the fixed-iteration `NewtonSolver` on every architecture.

Reactant needs it, because the default solver (`RootSolver`) iterates to a tolerance, and that
convergence-tested loop cannot be raised to StableHLO (`test/reactant/setup.jl` does the same). Native
runs use it too, so that all architectures integrate the same physics; on native architectures the two
solvers cost the same (see `docs/dev/plans/2026-10-17-1-native_vs_reactant_performance_gap.md`).
"""
function land_solver_kwargs(::Type{NF}) where {NF}
    skin_temperature = ImplicitSkinTemperature(NF; solver = Terrarium.NewtonSolver(NF; iterations = 5))
    return (surface_energy_balance = SurfaceEnergyBalance(NF; skin_temperature),)
end

"""
Implicit-explicit timestepper of the land and soil configurations: `ForwardEuler` for the explicit terms
and `BackwardEuler` with two `Picard` iterations for the stiff vertical operators (soil heat conduction,
Richards flow, and, in the land model, snow meltwater drainage). Spelled out rather than taken from the
model defaults, so that a change of default does not silently change what is benchmarked. The fixed
iteration count also keeps the implicit step free of convergence-tested loops, as Reactant requires.
"""
imex_timestepper(::Type{NF}) where {NF} = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver = Picard(NF; iterations = 2)))

"""
Initial state shared by the coupled land configurations: a warm, variably saturated soil column
under a thin cold snowpack. `carbon_vegetation` is only meaningful with vegetation enabled.
"""
function land_initializers(::Type{NF}; vegetation::Bool) where {NF}
    ## hoisted constants — see the Reactant note at the top of this file
    T_surface = NF(5)
    lapse_rate = NF(0.02)
    saturation_surface = NF(0.8)
    saturation_gradient = NF(0.05)
    saturated = one(NF)
    common = (
        temperature = (x, z) -> T_surface - lapse_rate * z,
        saturation_water_ice = (x, z) -> min(saturated, saturation_surface - saturation_gradient * z),
        snow_water_equivalent = NF(0.2),
        snow_temperature = NF(-2),
    )
    return vegetation ? merge(common, (carbon_vegetation = NF(0.1),)) : common
end

# --- :land — fully coupled land model ---------------------------------------------------
# Vegetation carbon cycle + soil energy/water/carbon with Richards equation + single-layer snow, on
# top of the default surface energy balance, surface hydrology and prescribed atmosphere, integrated
# with `imex_timestepper` (implicit soil heat conduction, Richards flow, and snow meltwater drainage).
# This is the headline configuration: the cross-architecture overview table is its resolution sweep.
#
# Δt = 900 s, the `LandModel` default: with the soil integrated implicitly, the step is limited by the
# explicit snow heat conduction, and the coupled model raises a DomainError in the turbulent-flux
# thermodynamics (log of a negative saturation) for time steps of roughly half an hour and larger.

function build_model(::Val{:land}, arch, ::Type{NF}; nlat_half::Integer, nz::Integer, model_kwargs = (;)) where {NF}
    grid = benchmark_grid(arch, NF; nlat_half, nz)

    texture = SoilTexture(NF; sand = NF(0.4), clay = NF(0.2))
    strat = HomogeneousSoilStratigraphy(NF; texture)
    soil = SoilEnergyWaterCarbon(NF; strat, hydrology = SoilHydrology(NF, RichardsEq()))

    vegetation_dynamics = PALADYNVegetationDynamics(NF)
    vegetation = PrognosticVegetation(NF; vegetation_dynamics)

    snow = SingleLayerSnow(NF)
    defaults = merge(land_solver_kwargs(NF), (timestepper = imex_timestepper(NF),))
    kwargs = merge(defaults, resolve_model_kwargs(model_kwargs, NF))
    model = LandModel(grid; soil, snow, vegetation, kwargs...)
    initializers = land_initializers(NF; vegetation = true)
    return (; model, boundary_conditions = (;), initializers, Δt = NF(900))
end

# --- :land_no_vegetation — same, without the canopy ---------------------------------------
# `vegetation = nothing` also selects `BareGroundEvaporation` and `NoCanopyInterception`, so the
# pair (:land, :land_no_vegetation) isolates the cost of the vegetation and canopy path.

function build_model(::Val{:land_no_vegetation}, arch, ::Type{NF}; nlat_half::Integer, nz::Integer, model_kwargs = (;)) where {NF}
    grid = benchmark_grid(arch, NF; nlat_half, nz)

    texture = SoilTexture(NF; sand = NF(0.4), clay = NF(0.2))
    strat = HomogeneousSoilStratigraphy(NF; texture)
    soil = SoilEnergyWaterCarbon(NF; strat, hydrology = SoilHydrology(NF, RichardsEq()))

    snow = SingleLayerSnow(NF)
    defaults = merge(land_solver_kwargs(NF), (timestepper = imex_timestepper(NF),))
    kwargs = merge(defaults, resolve_model_kwargs(model_kwargs, NF))
    model = LandModel(grid; soil, snow, vegetation = nothing, kwargs...)
    initializers = land_initializers(NF; vegetation = false)
    return (; model, boundary_conditions = (;), initializers, Δt = NF(900))
end

# --- :soil — stand-alone soil with Richards flow --------------------------------------------
# The soil column of the land configurations (same texture, stratigraphy, and Richards-equation
# hydrology) as a `SoilModel` without surface processes, integrated with `imex_timestepper`. Isolates
# the cost of the implicit soil heat and water solve from that of the surface and snow physics.
#
# Driven by a smooth, time-varying surface heat flux and no infiltration, so that the soil water only
# redistributes from its initial profile (impermeable bottom). Value boundary conditions on temperature
# are not supported by the implicit heat conduction operator, hence a flux rather than a prescribed
# surface temperature as in :soil_heat.
#
# Δt = 3600 s, the `SoilModel` default under IMEX: the implicit soil solve is not bound by the explicit
# diffusive limit, and the longer step it allows is part of what this configuration measures.

function build_model(::Val{:soil}, arch, ::Type{NF}; nlat_half::Integer, nz::Integer, model_kwargs = (;)) where {NF}
    grid = benchmark_grid(arch, NF; nlat_half, nz)

    texture = SoilTexture(NF; sand = NF(0.4), clay = NF(0.2))
    strat = HomogeneousSoilStratigraphy(NF; texture)
    soil = SoilEnergyWaterCarbon(NF; strat, hydrology = SoilHydrology(NF, RichardsEq()))
    kwargs = merge((timestepper = imex_timestepper(NF),), resolve_model_kwargs(model_kwargs, NF))
    model = SoilModel(grid; soil, kwargs...)

    ## smooth diurnal surface heat flux (W/m², positive upward) as a function of column index x and
    ## time t; the closure must capture only isbits values to be a valid kernel argument under Reactant
    amplitude = NF(50)
    ω = NF(2π) / NF(24 * 3600)
    surface_heat_flux(x, t) = amplitude * sin(ω * t) * cos(x)
    boundary_conditions = SoilHeatFlux(surface_heat_flux)

    initializers = land_initializers(NF; vegetation = false)
    initializers = (; initializers.temperature, initializers.saturation_water_ice)
    return (; model, boundary_conditions, initializers, Δt = NF(3600))
end

# --- :soil_heat — minimal soil heat conduction (Reactant reference) ------------------------
# Mirrors `test/reactant/setup.jl`'s `:soil_heat_global`: the only configuration currently proven to
# compile under Reactant. It is not part of the headline comparison — it exists so that an empty
# Reactant column can be attributed to the land physics rather than to the benchmark harness, which
# is also why it keeps the default sandy texture, with immobile soil water (`NoFlow`),
# rather than the loam of the land configurations. It is integrated explicitly with `ForwardEuler`
# (spelled out, as for the IMEX configurations), so that it also serves as the explicit reference
# for the cost of the soil energy tendencies.

function build_model(::Val{:soil_heat}, arch, ::Type{NF}; nlat_half::Integer, nz::Integer, model_kwargs = (;)) where {NF}
    grid = benchmark_grid(arch, NF; nlat_half, nz)
    soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, NoFlow()))
    kwargs = merge((timestepper = ForwardEuler(NF),), resolve_model_kwargs(model_kwargs, NF))
    model = SoilModel(grid; soil, kwargs...)

    ## smooth surface forcing as a function of column index x and time t; the closure must capture
    ## only isbits values to be a valid kernel argument under Reactant
    amplitude = NF(5)
    ω = NF(2π) / NF(24 * 3600)
    surface_temperature(x, t) = amplitude * sin(ω * t) * cos(x)
    boundary_conditions = PrescribedSurfaceTemperature(:T_ub, surface_temperature)

    T₀ = NF(2)
    lapse_rate = NF(0.05)
    initializers = (temperature = (x, z) -> T₀ * cos(NF(x)) - lapse_rate * z,)
    ## Δt = 600 s: the explicit diffusive limit of the thinnest (5 cm) layer of this dry sandy soil;
    ## the cell diffusion timescale is about 1150 s, and ForwardEuler goes unstable at Δt = 900 s
    return (; model, boundary_conditions, initializers, Δt = NF(600))
end

"""
    build_model(name::Symbol, arch, NF; nlat_half, nz, model_kwargs)

Convenience wrapper dispatching on the configuration `name`.
"""
function build_model(name::Symbol, arch, ::Type{NF}; kwargs...) where {NF}
    name in CONFIGURATIONS || error("Unknown configuration `$name`. Known: $(join(CONFIGURATIONS, ", ")).")
    return build_model(Val(name), arch, NF; kwargs...)
end
