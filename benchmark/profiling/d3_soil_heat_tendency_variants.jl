# D3: per-cell cost of the soil heat tendency kernel, baseline vs variants.
# usage: julia --project=. d3_soil_heat_tendency_variants.jl <V0|V1|V2|V3|V4> <cpu|gpu> <nlat_half> <outfile>
# V0 baseline; V1 parameter-only conductivities hoisted (passed as compile-time constants, which is
# equivalent to an isbits kernel argument); V2 sqrt instead of x^(1/2); V3 κ stored once per cell and
# interpolated to faces; V4 = V1 + V3.
const VARIANT, ARCHNAME, NLAT, OUTFILE = ARGS[1], ARGS[2], parse(Int, ARGS[3]), ARGS[4]
ARCHNAME == "gpu" && using CUDA
include(joinpath(@__DIR__, "common.jl"))
using Oceananigans.Operators: ℑzᵃᵃᶠ, ∂zᵃᵃᶠ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: CenterField
using KernelAbstractions: @kernel, @index

const ARCH = ARCHNAME == "gpu" ? GPU() : CPU()
integrator, Δt = setup(:soil_heat, ARCH; nlat_half = NLAT)
const STATE = integrator.state
const MODEL = integrator.model
const GRID = get_grid(MODEL)
const ENERGY, SOIL, CONSTANTS = MODEL.soil.energy, MODEL.soil, MODEL.constants
Terrarium.compute_auxiliary!(STATE, MODEL)
Terrarium.compute_boundary_conditions!(STATE, MODEL)
sync(ARCH)

const TEND = STATE.tendencies.internal_energy
energy_tendency!() = Terrarium.compute_tendencies!(STATE, GRID, ENERGY, SOIL, CONSTANTS)
function tendency_values()
    fill!(parent(TEND), 0)
    energy_tendency!()
    sync(ARCH)
    return Array(interior(TEND))
end
function time_tendency()
    energy_tendency!()
    sync(ARCH)
    b = @benchmark (energy_tendency!(); sync(ARCH)) evals = 1 seconds = 4
    return median(b).time, b.allocs
end

const NCELLS = prod(size(interior(TEND)))
baseline = tendency_values()
t0, a0 = time_tendency()

const PROPS = ENERGY.thermal_properties
if VARIANT in ("V1", "V4")
    PROPS.conductivity_weighting isa Terrarium.InverseQuadratic || error("V1 assumes InverseQuadratic, got $(typeof(PROPS.conductivity_weighting))")
    strat = Terrarium.get_stratigraphy(SOIL)
    length(strat.horizons) == 1 || error("V1 assumes a single soil horizon")
    c = PROPS.conductivities
    κm = Terrarium.mineral_thermal_conductivity(c, strat.horizons[1].texture)
    sq = map(sqrt, (; water = c.water, ice = c.ice, air = c.air, organic = c.organic, mineral = κm))
    @eval const SQ = $sq
    @eval @inline function Terrarium.compute_thermal_conductivity(props::Terrarium.SoilThermalProperties, soil::Terrarium.SoilComposition)
        f = Terrarium.volumetric_fractions(soil)
        s = SQ.water * f.water + SQ.ice * f.ice + SQ.air * f.air + SQ.organic * f.organic + SQ.mineral * f.mineral
        return s * s
    end
end
if VARIANT == "V2"
    @eval (f::Terrarium.InverseQuadratic)(xs, weights) = sum(Terrarium.fastmap((x, w) -> sqrt(x) * w, xs, weights))^2
end
if VARIANT in ("V3", "V4")
    const KF = CenterField(STATE.internal_energy.grid)
    @kernel function kappa_kernel!(κ, grid, fields, energy, hydrology, strat, bgc)
        i, j, k = @index(Global, NTuple)
        @inbounds κ[i, j, k] = Terrarium.compute_thermal_conductivity(i, j, k, Terrarium.ground_domain(grid), fields, energy, hydrology, strat, bgc)
    end
    @eval function Terrarium.compute_tendencies!(state, grid, energy::Terrarium.SoilThermodynamics, soil::Terrarium.AbstractSoil, args...)
        procs = (Terrarium.get_hydrology(soil), Terrarium.get_stratigraphy(soil), Terrarium.get_biogeochemistry(soil))
        tendencies = Terrarium.tendency_fields(state, energy)
        fields = Terrarium.get_fields(state, energy, procs...)
        Oceananigans.launch!(grid, Terrarium.XYZ, kappa_kernel!, KF, fields, energy, procs...)
        fill_halo_regions!(KF)
        fields_κ = merge(fields, (; κ_center = KF))
        Oceananigans.launch!(grid, Terrarium.XYZ, Terrarium.compute_tendencies_kernel!, tendencies, fields_κ, energy, procs...)
        return nothing
    end
    @eval Base.@propagate_inbounds function Terrarium.diffusive_heat_flux(i, j, k, grid, fields, energy::Terrarium.AbstractThermodynamics, args...)
        κ = ℑzᵃᵃᶠ(i, j, k, grid, fields.κ_center)
        return -κ * ∂zᵃᵃᶠ(i, j, k, grid, fields.temperature)
    end
end

if VARIANT == "V0"
    t1, a1, maxrel_all, maxrel_int = t0, a0, 0.0, 0.0
else
    variant = Base.invokelatest(tendency_values)
    t1, a1 = Base.invokelatest(time_tendency)
    scale = maximum(abs, baseline)
    maxrel_all = maximum(abs, variant .- baseline) / scale
    maxrel_int = maximum(abs, variant[:, :, 2:(end - 1)] .- baseline[:, :, 2:(end - 1)]) / scale
end

open(OUTFILE, "a") do io
    say(io, @sprintf("| %s | %s | %d | %d | %.1f | %.1f | %.2f× | %d | %.2e | %.2e | %.3e | %d |",
        VARIANT, ARCHNAME, ncolumns(NLAT), NCELLS, t0 / NCELLS, t1 / NCELLS, t0 / t1, a1, maxrel_all, maxrel_int, maximum(abs, baseline), count(!iszero, baseline)))
end
