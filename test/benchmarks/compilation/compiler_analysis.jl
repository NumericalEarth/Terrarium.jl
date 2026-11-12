using SnoopCompileCore

using Oceananigans

invs = @snoop_invalidations using Terrarium;

using SnoopCompile, AbstractTrees, ProfileView
trees = invalidation_trees(invs);
for (i, tree) in enumerate(trees)
    println("=== INVALIDATION TREE $i/$(length(trees)) ===")
    display(tree)
end

# check if any of the invalidated methods are from Terrarium
filtered_trees = filter(tree -> tree.method.module == Terrarium, trees)
@assert length(filtered_trees) == 0

function build_model(arch = CPU())
    # Define a simple grid with 1 column
    grid = ColumnGrid(arch, ExponentialSpacing(Δz_max = 1.0, N = 30))
    # Set up Richards model for soil hydrology
    swrc = VanGenuchten(α = 2.0, n = 2.0)
    hydraulic_properties = ConstantSoilHydraulics(eltype(grid); swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(eltype(grid)))
    hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
    soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
    vegetation = PrognosticVegetation(eltype(grid))
    # Construct coupled model
    land = LandModel(grid; soil, vegetation, initializer)
    return land
end

vegsoil = build_model()
# Variably saturated with water table at roughly 5 m depth
initializers = (saturation_water_ice = (x, z) -> min(1, 0.5 - 0.1 * z),)
tinf = @snoop_inference initialize(vegsoil; initializers)
# print_tree(tinf, maxdepth=100)
ProfileView.view(flamegraph(tinf))
Δt = 60.0
@time timestep!(integrator, Δt)
