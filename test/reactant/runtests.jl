# CPU-vs-Reactant correctness tests for Terrarium.
#
# Run from the repo root with the dedicated environment (Reactant is NOT a dependency of the
# main test suite):
#
#     julia --project=test/reactant test/reactant/runtests.jl

# Reactant raising requires @inbounds elision: with --check-bounds=yes (as e.g. Pkg.test
# forces by default) every kernel retains its bounds-check throw paths, which lower to
# llvm.intr.trap in the GPU pipeline and cannot be raised to StableHLO — every compile
# would fail with an opaque MLIR error. Fail fast with a readable message instead.
# (Checked before any `using` so we do not first recompile the whole stack.)
if Base.JLOptions().check_bounds == 1  # 0 = auto (default), 1 = yes, 2 = no
    error(
        "The Reactant correctness tests cannot run with forced bounds checking " *
            "(--check-bounds=yes): bounds checks make every kernel un-raisable under " *
            "Reactant. Run directly with default flags: " *
            "julia --project=test/reactant test/reactant/runtests.jl"
    )
end

using Terrarium
using Reactant
using CUDA   # required by Reactant's KernelAbstractions integration, even on CPU
using Test

# Tolerances: XLA reorders floating-point ops, so exact equality is not expected
const DEFAULT_NF = Float32
const NSTEPS = 100
const RTOL = 1.0e-3
const ATOL = 1.0e-6

include("correctness.jl")
include("setup.jl")

@testset "Terrarium CPU vs Reactant" begin
    test_model(:soil_heat_column)
    test_model(:soil_heat_column_stretched)   # array-valued (ExponentialSpacing) vertical coordinates
    test_model(:soil_heat_global)
    test_model(:snow_column)                  # standalone single-layer snow (closure + tendencies)
    test_model(:land_soil_snow)
    test_model(:vegetation_column)            # standalone PrescribedVegetation, constant LAI
end

include("autodiff.jl")
