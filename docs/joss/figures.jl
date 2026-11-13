# Paper figures for the coupled SpeedyWeather-Terrarium simulation.
#
# Figure 1: architecture diagram, created externally (not by this script).
# Figure 2: SoilModel performance — SYPD vs grid resolution for CPU, GPU and
#           Reactant, plotted from `benchmark/assets/benchmark_results.json`
#           (suite `bench202`, the `:soil_heat` SoilModel sweep) produced by
#           `benchmark/manual_benchmarking.jl`. The benchmarked SoilModel solves
#           heat conduction only (immobile soil water, i.e. no hydrology).
# Figure 3: land-surface climatology, 2x3 panels.
#           top:    skin temperature, GPP, max annual snow depth
#           bottom: evapotranspiration, evaporative fraction, albedo
#
# All maps use a linear equirectangular (plate carrée) projection because
# `heatmap!` warps only the corners of its textured quad, so a nonlinear
# projection would skew the field relative to the per-vertex coastlines.
#
# Run from the repository root with
#     julia --project=docs/joss docs/joss/figures.jl

using Terrarium: ThermodynamicConstants
using Rasters, NCDatasets
using CairoMakie, GeoMakie
using JSON3
using Statistics
using Dates

# Output of the coupled run produced by `run_terrarium_speedy_simulation.jl`,
# which also writes Terrarium's `skin_temperature`. Defaults to the latest run in
# `outputs/` (the highest-numbered `run_*` directory with an `output.nc`); pass a
# run directory name (e.g. `run_0003`) as the first argument to use another run.
outputs_dir = joinpath(@__DIR__, "outputs")
run_name = if isempty(ARGS)
    runs = filter(readdir(outputs_dir)) do name
        startswith(name, "run_") && isfile(joinpath(outputs_dir, name, "output.nc"))
    end
    @assert !isempty(runs) "no run_* directory with an output.nc found in $outputs_dir"
    last(sort(runs))
else
    first(ARGS)
end
run_dir = joinpath(outputs_dir, run_name)
@info "Plotting the coupled simulation output in $run_dir"
assets_dir = joinpath(@__DIR__, "assets")
mkpath(assets_dir)

outputs = RasterStack(joinpath(run_dir, "output.nc"), lazy = true)

# The first snapshot is the uninitialized initial condition, so every time
# reduction below drops it.
times = Array(dims(outputs[:shf], Ti))
# Snow years run from October to September, so that each contains one full
# snow season in both hemispheres (the simulation starts on October 1st).
snow_years = year.(times .- Month(9))

# Latent heat of vaporization (J/kg), taken from Terrarium's thermodynamic
# constants, used to convert the surface humidity flux (kg/s/m²) into a latent
# heat flux (W/m²) for the evaporative fraction.
const latent_heat_vaporization = ThermodynamicConstants(Float64).latent_heat_vaporization

"""
    to_map_order(lon, lat, data)

Shift longitudes from 0..360 to -180..180 and sort latitudes ascending, so the
field can be plotted with `heatmap!` on a `GeoAxis`. The data is rolled by half
the grid so that it stays aligned with the shifted longitudes.
"""
function to_map_order(lon, lat, data)
    if maximum(lon) > 180 # longitudes are given on 0..360
        lon = lon .- 180
        data = circshift(data, (div(length(lon), 2), 0))
    end
    if !issorted(lat) # output latitudes run north to south
        lat = reverse(lat)
        data = reverse(data; dims = 2)
    end
    return lon, lat, data
end

"""
    time_mean(var)

Time mean of a `Raster` along `Ti`, dropping the uninitialized first snapshot.
The result keeps every dimension except time.
"""
function time_mean(var)
    nd = ndims(var)
    mean_ = mean(var[ntuple(_ -> :, nd - 1)..., 2:end]; dims = Ti)
    return replace(Array(dropdims(mean_; dims = Ti)), missing => NaN32)
end

"""
    annual_max_mean(var)

For each complete snow year (October to September, see `snow_years`), take the
maximum over time, then average those annual maxima. Snow years that the output
does not cover completely (e.g. a final snapshot on October 1st) are excluded.
`var` must be 2D in space with `Ti` as its last dimension. The uninitialized
first snapshot is excluded.
"""
function annual_max_mean(var)
    nd = ndims(var)
    @assert nd == 3 "annual_max_mean expects a (lon, lat, time) Raster"
    arr = replace(Array(var[:, :, 2:end]), missing => NaN32)
    yr = snow_years[2:end]
    t = times[2:end]
    # a snow year is complete if its snapshots span (almost) a full year
    complete_years = filter(y -> last(t[yr .== y]) - first(t[yr .== y]) >= Day(360), unique(yr))
    @assert !isempty(complete_years) "the output does not cover a complete snow year (October to September)"
    annual_maxima = [dropdims(maximum(arr[:, :, yr .== y]; dims = 3); dims = 3) for y in complete_years]
    return sum(annual_maxima) / length(annual_maxima)
end

"""
    robust_colorrange(data; bounds = (0.01, 0.99))

Quantile-based color range so localized spikes do not wash out the map.
"""
function robust_colorrange(data; bounds = (0.01, 0.99))
    finite = data[isfinite.(data)]
    range = quantile(finite, collect(bounds))
    range[1] == range[2] && (range = extrema(finite))
    return (Float64(range[1]), Float64(range[2]))
end

# Latitude range (degrees) shown on the maps; Antarctica is cropped out since its
# ice sheet dominates or adds noise to several of the land-surface panels.
const MAP_LATITUDES = (-60, 90)

"""
    map_panel!(fig, panel, cb, data; title, units, colormap, colorrange, width = 14, heatmap_kwargs...)

Draw one map with coastlines and its own colorbar. `panel` and `cb` are layout
positions for the axis and colorbar respectively; `width` is the colorbar width
in pixels. Further keyword arguments (e.g. `lowclip`) are passed to `heatmap!`.
"""
function map_panel!(fig, panel, cb, data; title, units = "", colormap, colorrange, width = 14, heatmap_kwargs...)
    lon, lat, data = to_map_order(Array(dims(outputs[:shf], X)), Array(dims(outputs[:shf], Y)), data)
    # `tellwidth/tellheight = false` keeps the aspect-driven size suggestion out of
    # the layout solver, which otherwise collapses the cells to wildly unequal
    # sizes; the aspect then just letterboxes the map inside its (even) cell.
    ax = GeoAxis(
        fig[panel...]; dest = "+proj=longlat +datum=WGS84", title,
        aspect = 360 / (MAP_LATITUDES[2] - MAP_LATITUDES[1]), limits = ((-180, 180), MAP_LATITUDES),
        tellwidth = false, tellheight = false, titlesize = 14, titlegap = 2
    )
    hm = heatmap!(ax, lon, lat, data; colormap, colorrange, heatmap_kwargs...)
    lines!(ax, GeoMakie.coastlines(); color = :gray25, linewidth = 0.6)
    hidedecorations!(ax)
    # With a fixed aspect the map only fills part of the axis box; the map itself
    # occupies `ax.scene.viewport`, so size the colorbar to that rectangle. The
    # colorbar tells its width (so its column shrinks to fit) but not its height.
    map_height = map(r -> r.widths[2], ax.scene.viewport)
    Colorbar(
        fig[cb...], hm; label = units, tellwidth = true, tellheight = false,
        width = width, height = map_height
    )
    return ax, hm
end

# ---------------------------------------------------------------------------
# Figure 2: SoilModel performance (SYPD vs grid resolution, all architectures)
# ---------------------------------------------------------------------------

# Plots simulated years per wallclock day (SYPD) for `SoilModel` across grid
# resolutions, one curve per architecture. The data is the `bench202` suite (the
# `:soil_heat` configuration, i.e. soil heat conduction without hydrology) in the
# benchmark JSON store, written by
# `benchmark/manual_benchmarking.jl` (see `docs/joss/joss_benchmark.slurm` for a
# single-node run of all four architectures). Architectures absent from the store
# — or resolutions skipped/failed there — are simply not drawn.
const BENCHMARK_RESULTS_JSON = normpath(joinpath(@__DIR__, "..", "..", "benchmark", "assets", "benchmark_results.json"))
const BENCHMARK_SOIL_SUITE = "bench202" # the :soil_heat SoilModel resolution sweep

# Horizontal resolutions (degrees) marked by dashed vertical reference lines
const BENCHMARK_REFERENCE_RESOLUTIONS = (10, 5, 2, 1, 0.5)

"""
Number of land columns at the horizontal `resolution` (degrees), interpolated linearly in log-log
space between the grids of the benchmark sweep (`sweep_resolutions`, `sweep_columns`), and
extrapolated with the slope of the outermost segment outside of it.
"""
function columns_at_resolution(resolution, sweep_resolutions, sweep_columns)
    order = sortperm(sweep_resolutions)
    log_resolutions = log.(sweep_resolutions[order])
    log_columns = log.(sweep_columns[order])
    x = log(resolution)
    k = clamp(searchsortedlast(log_resolutions, x), 1, length(log_resolutions) - 1)
    slope = (log_columns[k + 1] - log_columns[k]) / (log_resolutions[k + 1] - log_resolutions[k])
    return exp(log_columns[k] + slope * (x - log_resolutions[k]))
end

# Store label => (legend name, color). The CPU and the Reactant (CPU)
# labels each share one curve; the first present wins (see the `plotted_names` guard below).
const BENCHMARK_ARCH_STYLES = (
    "cpu-x86" => ("CPU", :navy),
    "cpu-arm" => ("CPU", :navy),
    "gpu-nvidia" => ("GPU", :forestgreen),
    "reactant-cpu-x86" => ("Reactant (CPU)", :darkorange),
    "reactant-cpu-arm" => ("Reactant (CPU)", :darkorange),
    "reactant-cpu" => ("Reactant (CPU)", :darkorange),
    "reactant-gpu" => ("Reactant (GPU)", :crimson),
)

fig2 = Figure(size = (640, 460))
ax2 = Axis(
    fig2[1, 1],
    xlabel = "Number of land columns",
    ylabel = "Simulated years per wallclock day (SYPD)",
    xscale = log10, yscale = log10,
    # label every decade of SYPD (e.g. 10¹ and 10³) for easier comparison with other models
    yticks = LogTicks(-3:9),
    title = "Terrarium SoilModel performance (heat conduction only)"
)

plotted_names = String[]
max_sypd = 1.0
sweep_resolutions = Float64[]
sweep_columns = Int[]
if isfile(BENCHMARK_RESULTS_JSON)
    results = JSON3.read(read(BENCHMARK_RESULTS_JSON, String), Dict{String, Any})
    for (label, (name, color)) in BENCHMARK_ARCH_STYLES
        haskey(results, label) || continue
        name in plotted_names && continue
        overview = get(get(results[label], "overview", Dict()), BENCHMARK_SOIL_SUITE, nothing)
        overview === nothing && continue
        # JSON has no NaN literal: skipped/failed resolutions are stored as null.
        sypd = [something(x, NaN) for x in overview["sypd"]]
        ncolumns = [Int(x) for x in overview["ncolumns"]]
        resolution = [180 / (2 * Int(n)) for n in overview["nlat_half"]]
        ok = isfinite.(sypd)
        any(ok) || continue
        global max_sypd = max(max_sypd, maximum(sypd[ok]))

        lines!(ax2, ncolumns[ok], sypd[ok]; color, linewidth = 1.5, label = name)
        scatter!(ax2, ncolumns[ok], sypd[ok]; markersize = 9, color)
        # all architectures share the grids of the sweep; keep one set for the reference lines
        isempty(sweep_resolutions) && append!(sweep_resolutions, resolution)
        isempty(sweep_columns) && append!(sweep_columns, ncolumns)
        push!(plotted_names, name)
    end
end

if isempty(plotted_names)
    Label(fig2[1, 1], "No benchmark data — run benchmark/manual_benchmarking.jl", tellwidth = false)
else
    # dashed vertical lines at reference horizontal resolutions, labeled at the top of the axes
    for degrees in BENCHMARK_REFERENCE_RESOLUTIONS
        ncolumns_ref = columns_at_resolution(degrees, sweep_resolutions, sweep_columns)
        vlines!(ax2, ncolumns_ref; color = :gray50, linestyle = :dash, linewidth = 1)
        # shifted about half a centimeter (19 units at 96 units per inch) to the left of the line
        text!(ax2, ncolumns_ref, 3 * max_sypd; text = "$(degrees)°", align = (:center, :bottom), offset = (-19, 0), fontsize = 10, color = :gray30)
    end
    axislegend(ax2; position = :lb)
    # headroom above the fastest curve for the reference line labels
    ylims!(ax2, nothing, 8 * max_sypd)
end
save(joinpath(assets_dir, "figure2.png"), fig2)
@info "Saved figure2.png"

# ---------------------------------------------------------------------------
# Figure 3: land-surface climatology, 2x3 panels
#   top:    skin temperature, GPP, max annual snow depth
#   bottom: evapotranspiration, evaporative fraction, albedo
# ---------------------------------------------------------------------------


# The SpeedyWeather fields (`shf`, `shuf`, `albedo`, `sd`) carry values or a
# constant fallback over ocean, and the Terrarium output variables should be NaN
# there. On GPU, however, SpeedyWeather's bitrounding of the output turns those
# NaNs into -0.0 (its interpolation produces NaNs with all mantissa bits set,
# which the rounding carry turns into the sign bit). A skin temperature of
# exactly -0.0 °C does not occur over land, so the bitwise -0.0 (or NaN) points
# of the raw skin temperature mark the ocean, including coastal points that
# touch the ocean in the output interpolation. The resulting land mask is
# applied to every panel so that they share the same land-only footprint.
land_mask = NCDataset(joinpath(run_dir, "output.nc")) do ds
    raw_skin_temperature = ds["skin_temperature"].var[:, :, end]
    ocean = isnan.(raw_skin_temperature) .| (signbit.(raw_skin_temperature) .& (raw_skin_temperature .== 0))
    # also drop the cropped-out latitudes so that they do not affect the color ranges
    on_map = (ds["lat"][:] .>= MAP_LATITUDES[1])'
    .!ocean .& on_map
end

# GPP is stored as kg C / m² / s; rescale to g C / m² / day for readability.
gpp = time_mean(outputs[:gross_primary_production]) * 8.64e7
gpp[.!land_mask] .= NaN32

# Skin temperature (°C) is written directly from the Terrarium state, so it is
# already 2D.
skin_temperature = time_mean(outputs[:skin_temperature])
skin_temperature[.!land_mask] .= NaN32

snow_depth = annual_max_mean(outputs[:sd])
snow_depth[.!land_mask] .= NaN32

# Total evapotranspiration is the sum of Terrarium's ground evaporation, canopy
# evaporation, transpiration, and sublimation fluxes, each stored in m (liquid
# water) / s; rescale to mm / day.
evapotranspiration = (
    time_mean(outputs[:evaporation_ground]) +
        time_mean(outputs[:evaporation_canopy]) +
        time_mean(outputs[:transpiration]) +
        time_mean(outputs[:sublimation])
) * 8.64e7
evapotranspiration[.!land_mask] .= NaN32

# Evaporative fraction EF = LE / (LE + H), with LE = L_v * (humidity flux).
# Points where the time-mean total turbulent flux LE + H is not positive have no
# well-defined partitioning and are masked out; the land mask is applied too.
shf = time_mean(outputs[:shf])
shuf = time_mean(outputs[:shuf])
lhf = latent_heat_vaporization * shuf
tur = lhf + shf
tur_nz = ifelse.(tur .> 0, tur, oneunit(eltype(tur)))
ef = ifelse.(tur .> 0, lhf ./ tur_nz, NaN32)
ef[.!land_mask] .= NaN32

# Albedo is a SpeedyWeather radiation diagnostic defined over the globe (snow
# drives the high land values); mask it to land for consistency.
albedo = time_mean(outputs[:albedo])
albedo[.!land_mask] .= NaN32

# Height chosen so each row's cell ≈ map height (width / aspect) + title, minimizing the
# vertical letterboxing that the fixed map aspect leaves inside each cell.
fig3 = Figure(size = (1400, 400), padding = (5, 5, 5, 5), colgap = 4, rowgap = 8)
map_panel!(
    fig3, (1, 1), (1, 2), skin_temperature;
    title = "Mean skin temperature", units = "°C",
    colormap = :balance, colorrange = (-30, 30), width = 10
)
map_panel!(
    fig3, (1, 3), (1, 4), gpp;
    title = "Mean gross primary production", units = "g C m⁻² day⁻¹",
    colormap = :YlGn, colorrange = robust_colorrange(gpp; bounds = (0, 0.99)), width = 10
)
# Snow-free land (annual maximum below 1 mm of water) is drawn white rather than in the
# lightest color of the colormap; values above 0.5 m saturate.
map_panel!(
    fig3, (1, 5), (1, 6), snow_depth;
    title = "Max annual snow depth water equivalent", units = "m",
    colormap = :Blues, colorrange = (1.0e-3, 0.5), lowclip = :white, width = 10
)
map_panel!(
    fig3, (2, 1), (2, 2), evapotranspiration;
    title = "Mean evapotranspiration", units = "mm day⁻¹",
    colormap = :YlGnBu, colorrange = (0, robust_colorrange(evapotranspiration; bounds = (0, 0.99))[2]), width = 10
)
map_panel!(
    fig3, (2, 3), (2, 4), ef;
    title = "Evaporative fraction",
    colormap = :viridis, colorrange = (0, 1), width = 10
)
map_panel!(
    fig3, (2, 5), (2, 6), albedo;
    title = "Mean surface albedo",
    colormap = :viridis, colorrange = robust_colorrange(albedo), width = 10
)
save(joinpath(assets_dir, "figure3.png"), fig3)
@info "Saved figure3.png"
