using Terrarium

using ArgParse
using BenchmarkTools
using CSV, DataFrames
using CUDA
using Dates
using Rasters, NCDatasets
using Statistics

import RingGrids

s = ArgParseSettings()
@add_arg_table! s begin
    "--device", "-d"
    default = "cpu"
    help = "Device to run benchmark on: 'cpu' or 'gpu'"
    "--samples", "-s"
    arg_type = Int
    default = 10
    help = "Number of benchmark samples"
    "--prefix", "-p"
    default = "File prefix (including path) for output data"
    default = "outputs/benchmarks/soil_heat_hydrology_benchmark"
end
parsed_args = parse_args(ARGS, s)
nsamples = parsed_args["samples"]
prefix = parsed_args["prefix"]
device = parsed_args["device"]
arch = if device == "gpu"
    @assert CUDA.functional() "No GPU available!"
    GPU()
elseif device == "cpu"
    CPU()
else
    error("unrecognized device $(parsed_args["device"])")
end

function set_up_model(arch, ::Type{NF}, ring_grid::RingGrids.AbstractGrid) where {NF}
    grid = ColumnRingGrid(arch, NF, ExponentialSpacing(N = 30), ring_grid)
    # Initial conditions
    initializer = SoilInitializer(eltype(grid))
    energy = SoilEnergyBalance(NF)
    hydrology = SoilHydrology(NF, RichardsEq())
    # Periodic surface temperature with annual cycle
    T_ub = PrescribedTemperature((x, t) -> 30 * sin(2π * t / (24 * 3600 * 365)))
    boundary_conditions = SoilBoundaryConditions(eltype(grid), energy, hydrology, top = T_ub)
    model = SoilModel(grid; initializer, boundary_conditions, energy, hydrology)
    state = initialize(model)
    return state
end

# quick test
rg = RingGrids.FullGaussianGrid(8)
state = set_up_model(arch, Float32, rg)
timestep!(state, 60.0)

nrings_options = [2^i for i in 1:10]

data = []
for nrings in nrings_options
    rg = RingGrids.FullGaussianGrid(nrings)
    npoints = RingGrids.get_npoints(rg)
    @info "Running benchmark for grid:\n $rg"
    state = set_up_model(arch, Float32, rg)
    bench = @benchmark run!(state, period = Hour(1), Δt = 60.0) samples = nsamples
    times = bench.times ./ 1.0e6
    mid_time = median(times)
    min_time = minimum(times)
    push!(data, (; min_time, mid_time, nrings, npoints)) # push times in milliseconds
    @show bench
end

df = DataFrame(data)
@show df

filename = "$(prefix)_$device_nthreads=$(Threads.nthreads()).csv"
mkpath(dirname(filename))
CSV.write(filename, df)

exit()

# Interactive plotting

using DataFrames, CSV
using CairoMakie, GeoMakie
using CairoMakie.GeometryBasics
using RingGrids

# GLMakie.activate!(inline = true)

cpu_data = DataFrame(CSV.File("outputs/benchmarks/soil_heat_hydrology_benchmark_cpu_nthreads=32.csv"))
gpu_data = DataFrame(CSV.File("outputs/benchmarks/soil_heat_hydrology_benchmark_gpu_nthreads=32.csv"))

ring_grid_5deg = FullGaussianGrid(14)
ring_grid_3deg = FullGaussianGrid(24)
ring_grid_1deg = FullGaussianGrid(72)
ring_grid_halfdeg = FullGaussianGrid(72 * 2)
ring_grid_qtrdeg = FullGaussianGrid(72 * 4)
ring_grid_10km = FullGaussianGrid(72 * 10)
ring_grid_1km = FullGaussianGrid(72 * 100)
ref_grids = [ring_grid_5deg, ring_grid_1deg, ring_grid_halfdeg, ring_grid_qtrdeg, ring_grid_10km]
ref_npoints = [get_npoints(rg) for rg in ref_grids]

Makie.with_theme(fontsize = 18) do
    let fig = Figure(size = (800, 400))
        ax = Axis(
            fig[1, 1],
            title = "Terrarium.jl GPU vs. CPU scaling",
            ylabel = "Simulated years per day (SYPD)",
            xlabel = "Number of grid cells",
            xscale = log10,
            yscale = log10
        )
        # data is in ms / sim hr x 1 d / (1000*24*3600 ms) x 24 sim hr / sim day ->
        cpu_sypd = 1000 * 24 * 3600 ./ (24 * cpu_data.mid_time)
        gpu_sypd = 1000 * 24 * 3600 ./ (24 * gpu_data.mid_time)
        scatterlines!(ax, cpu_data.npoints, cpu_sypd, label = "CPU")
        scatterlines!(ax, gpu_data.npoints, gpu_sypd, label = "GPU")
        # Draw vertical reference lines and place a short annotation next to each one.
        # Choose human-friendly labels for the reference grids.
        ref_labels = ["5°", "1°", "0.5°", "0.25°", "0.1°"]
        ys_max = maximum(vcat(cpu_sypd, gpu_sypd))
        for (x, lbl) in zip(ref_npoints, ref_labels)
            vlines!(ax, [x], linestyle = :dash, color = :black)
            # place label slightly to the right of the line at the top of the plotted range
            text!(ax, GeometryBasics.Point2(x * 1.2, 200); text = lbl, align = (:left, :top), fontsize = 18)
        end
        axislegend(ax)
        Makie.save("plots/terrarium_cpu_vs_gpu_scaling.svg", fig)
        Makie.save("plots/terrarium_cpu_vs_gpu_scaling.pdf", fig)
        fig
    end
end

using SpeedyWeather, CUDA

# healpix_globe_32 = RingGrids.globe(RingGrids.HEALPixGrid, 32, interactive=false)
# healpix_globe_64 = RingGrids.globe(RingGrids.HEALPixGrid, 64, interactive=false)

grid = SpectralGrid(HEALPixGrid(128))
model = PrimitiveDryModel(grid)
sim = initialize!(model)
run!(sim, period = Day(10))
temp_hi = sim.diagnostic_variables.grid.temp_grid[:, 8]
temp_lo = zeros(HEALPixGrid(32))
RingGrids.interpolate!(temp_lo, temp_hi)
globe(temp_hi, interactive = false)
globe(temp_lo, interactive = false)
