using Terrarium
using Terrarium: SoilEnergyWaterCarbon, RichardsEq, SoilHydrology
using Terrarium: ConstantSoilHydraulics, VanGenuchten, UnsatKVanGenuchten
using Terrarium: reset_tendencies!
using Test

using Oceananigans

# The coupled `SoilEnergyWaterCarbon` computes its auxiliaries and tendencies with a single fused
# kernel each. These tests check that the fused path reproduces the pre-fusion per-process path
# (calling `compute_auxiliary!` / `compute_tendencies!` on each sub-process in turn) to machine
# precision, for both `RichardsEq` (water + energy) and `NoFlow` (energy only) hydrology.

# Snapshot every auxiliary and tendency field as a plain `Array`. `collect` is used rather than
# `Array` since some auxiliaries (e.g. `ground_temperature`) are `ZReducedField`s, for which
# `Array` fails with a `DimensionMismatch`.
snapshot(state) = (;
    (name => collect(getproperty(state.auxiliary, name)) for name in propertynames(state.auxiliary))...,
    (name => collect(getproperty(state.tendencies, name)) for name in propertynames(state.tendencies))...,
)

function run_fused_equivalence_test(soil, grid; initializers)
    model = SoilModel(grid; soil, initializer = DefaultInitializer(eltype(grid)))
    integrator = initialize(model; initializers)
    state = integrator.state
    constants = model.constants

    # Fused path: one launch per stage, dispatching on the coupled soil type.
    reset_tendencies!(state)
    compute_auxiliary!(state, grid, soil, constants)
    compute_tendencies!(state, grid, soil, constants)
    fused = snapshot(state)

    # Per-process path (pre-fusion): call each sub-process host method in turn.
    reset_tendencies!(state)
    for process in (soil.hydrology, soil.biogeochem, soil.energy)
        compute_auxiliary!(state, grid, process, soil, constants)
    end
    for process in (soil.hydrology, soil.biogeochem, soil.energy)
        compute_tendencies!(state, grid, process, soil, constants)
    end
    per_process = snapshot(state)

    # Every auxiliary and tendency must agree bit-for-bit between the two paths.
    for name in propertynames(state.auxiliary)
        @test fused[name] == per_process[name]
    end
    for name in propertynames(state.tendencies)
        @test fused[name] == per_process[name]
    end

    # Sanity: the tendencies are actually non-trivial (guards against a vacuous all-zeros match).
    @test any(!iszero, fused.internal_energy)

    return nothing
end

@testset "SoilEnergyWaterCarbon: fused kernels match per-process" begin
    @testset "RichardsEq (water + energy)" begin
        grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 50))
        swrc = VanGenuchten(α = 2.0, n = 2.0)
        hydraulic_properties = ConstantSoilHydraulics(Float64; swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(Float64))
        hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
        soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
        initializers = (
            saturation_water_ice = (x, z) -> min(1, 0.6 - 0.05 * z),
            temperature = (x, z) -> 10.0 + 2.0 * z,
        )
        run_fused_equivalence_test(soil, grid; initializers)
    end

    @testset "NoFlow (energy only)" begin
        grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 30))
        soil = SoilEnergyWaterCarbon(eltype(grid))  # default hydrology is NoFlow
        initializers = (
            temperature = (x, z) -> 5.0 + z,
            saturation_water_ice = 0.5,
        )
        run_fused_equivalence_test(soil, grid; initializers)
    end
end
