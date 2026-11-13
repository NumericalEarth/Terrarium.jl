using Terrarium
using CUDA

arch = GPU()
# Define a simple grid with 1 column
grid = ColumnGrid(arch, ExponentialSpacing(Δz_max = 1.0, N = 30))
# Set up Richards model for soil hydrology
swrc = VanGenuchten(α = 2.0, n = 2.0)
hydraulic_properties = ConstantSoilHydraulics(eltype(grid); swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(eltype(grid)))
hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
vegetation = PrognosticVegetation(eltype(grid))
# Construct coupled model
land = LandModel(grid; soil, vegetation)
# Variably saturated with water table at roughly 5 m depth
initializers = (
    saturation_water_ice = (x, z) -> min(1, 0.5 - 0.1 * z),
    C_veg = 0.1,
)
integrator = @time initialize(land; initializers);

using BenchmarkTools

# Benchmark adapt for full state
res1 = @benchmark adapt(CUDA.KernelAdaptor(), ($integrator.state))
# Extract fields for soil hydrology
fields = get_fields(integrator.state, hydrology, soil.biogeochem)
tendencies = Terrarium.tendency_fields(integrator.state, hydrology)
res2 = @benchmark adapt(CUDA.KernelAdaptor(), ($tendencies, $fields))
