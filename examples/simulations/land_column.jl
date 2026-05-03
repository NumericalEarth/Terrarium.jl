using Terrarium
using CUDA

arch = CPU()
# Define a simple grid with 1 column
grid = ColumnGrid(arch, ExponentialSpacing(Δz_max = 1.0, N = 30))
# Set up Richards model for soil hydrology
swrc = VanGenuchten(α = 2.0, n = 2.0)
hydraulic_properties = ConstantSoilHydraulics(eltype(grid); swrc, unsat_hydraulic_cond = UnsatKVanGenuchten(eltype(grid)))
hydrology = SoilHydrology(eltype(grid), RichardsEq(); hydraulic_properties)
soil = SoilEnergyWaterCarbon(eltype(grid); hydrology)
vegetation = VegetationCarbon(eltype(grid))
# Construct coupled model
land = LandModel(grid; soil, vegetation)
# Variably saturated with water table at roughly 5 m depth
initializers = (
    temperature = 15.0,
    saturation_water_ice = (x, z) -> min(1, 0.5 - 0.1 * z),
    carbon_vegetation = 0.5,
)
integrator = @time initialize(land, ForwardEuler(); initializers);
# manually set atmospheric inputs to different values
set!(integrator.state.windspeed, 1.0) # 1 m/s
set!(integrator.state.specific_humidity, 1.0e-4) # kg/kg
Δt = 60.0
@time timestep!(integrator, Δt)
@show integrator.state.latent_heat_flux[1, 1, 1]
