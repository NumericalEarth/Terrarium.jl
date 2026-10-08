using Terrarium

dyear = 60.0 * 60.0 * 24.0 * 365.0
yr_to_sec = 1 / dyear
const SIMSTOP = 2 * dyear

grid = ColumnGrid(CPU(), Float64, UniformSpacing(N = 2))
#carbon model
soc_resp = Terrarium.SoilCarbonRespiration(eltype(grid))
soc_transp = Terrarium.SoilCarbonTransport(eltype(grid))
biogeochem = OnePoolSoilCarbon(eltype(grid); transport = soc_transp, respiration = soc_resp)
soil = SoilEnergyWaterCarbon(eltype(grid); biogeochem) # coupled soil processes
model = SoilModel(grid; soil) # soil model
display(variables(model))
initializers = (; density_soc = 0.0, temperature = 20.0, saturation_water_ice = 1.0)

t_F = 0:dyear:SIMSTOP

litter = FieldTimeSeries(grid, XY(), t_F)
litter.data .= clamp.(0.5 / dyear .+ 0.005 / dyear .* randn(size(litter)), 0, 100)

inputs = InputSources(
    InputSource(grid, litter; units = u"kg/m^2/s", name = :litter),
)

bc = Terrarium.LitterfallFlux(biogeochem)
integrator = initialize(model; inputs, initializers, boundary_conditions = bc)

sim = Simulation(integrator; stop_time = SIMSTOP, Δt = 3600)

using Oceananigans: TimeInterval, JLD2Writer
using Oceananigans.Units: seconds

const INTERVAL = Day(365)
output_file = tempname()
sim.output_writers[:snapshots] = JLD2Writer(
    integrator,
    (density_soc = integrator.state.density_soc,);
    filename = output_file,
    overwrite_files = true,
    schedule = TimeInterval(Second(INTERVAL).value)
)

run!(sim)

fts = FieldTimeSeries(output_file, "density_soc")

density_soc = fts[end]

using CairoMakie
fig = Figure(size = (600, 400))
ax = CairoMakie.Axis(fig[1, 1])
plot!(ax, Day(0):Day(INTERVAL):Day(Second(SIMSTOP)), [fts[i][1, 1, 1] for i in 1:length(fts)])
plot!(ax, Day(0):Day(INTERVAL):Day(Second(SIMSTOP)), [fts[i][1, 1, 2] for i in 1:length(fts)])
fig
