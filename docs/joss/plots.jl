# Climatological mean maps for the coupled SpeedyWeather-Terrarium run `run_0002`.
#
# Loads `outputs/run_0002/output.nc`, computes the time mean of every output
# variable and saves one map per variable (with coastlines) on a linear
# equirectangular (plate carrée) projection to `docs/joss/assets/climate/`.
# A linear projection is required because `heatmap!` warps only the corners of
# its textured quad, so a nonlinear projection (e.g. Robinson) would skew the
# field relative to the per-vertex coastlines.
#
# Run from the repository root with
#     julia --project=examples docs/joss/plots.jl

using Rasters, NCDatasets
using CairoMakie, GeoMakie
using Statistics

output_dir = joinpath(@__DIR__, "assets", "climate")
mkpath(output_dir)

outputs = RasterStack("outputs/run_0002/output.nc", lazy = true)

# Output variables grouped by dimensionality.
# The 3D atmospheric variables are plotted at the lowest sigma layer (last index, sigma = 0.9375).
atmosphere = (:temp, :u, :v, :vor)
soil = (:st,)
surface = (
    :mslp, :sd, :cloud_top, :albedo, :shf, :shuf,
    :rain_conv, :rain_cond, :snow_cond,
    :rain_conv_rate, :rain_cond_rate, :snow_cond_rate,
)

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
    plot_climatology(name, var; layer = nothing)

Plot the climatological (time) mean of the `Raster` `var` as a map on a linear
equirectangular `GeoAxis` with coastlines and save it to `output_dir`.
For variables with a vertical dimension, `layer` selects the level to plot
(default: the last one).
"""
function plot_climatology(name, var; layer = nothing)
    # The first snapshot is the uninitialized initial condition, so skip it.
    climatology = dropdims(mean(var[ntuple(_ -> :, ndims(var) - 1)..., 2:end]; dims = Ti); dims = Ti)
    data = replace(Array(climatology), missing => NaN32)

    title = get(var.metadata, "long_name", string(name))
    if ndims(data) == 3
        k = something(layer, size(data, 3))
        data = data[:, :, k]
        title *= " (layer $k)"
    end

    lon, lat, data = to_map_order(Array(dims(var, X)), Array(dims(var, Y)), data)

    # Robust color range so that localized spikes (e.g. precipitation rates)
    # do not wash out the map.
    finite = data[isfinite.(data)]
    colorrange = quantile(finite, [0.01, 0.99])
    colorrange[1] == colorrange[2] && (colorrange = extrema(finite))

    units = get(var.metadata, "units", "")

    fig = Figure(size = (900, 480))
    ax = GeoAxis(fig[1, 1]; dest = "+proj=longlat +datum=WGS84", title = title)
    hm = heatmap!(ax, lon, lat, data; colormap = :turbo, colorrange = colorrange)
    lines!(ax, GeoMakie.coastlines(); color = :white, linewidth = 0.8)
    hidedecorations!(ax)
    Colorbar(fig[1, 2], hm; label = units)

    path = joinpath(output_dir, "$name.png")
    save(path, fig)
    @info "Saved $path"
    return path
end

for name in surface
    plot_climatology(name, outputs[name])
end
for name in atmosphere # lowest atmospheric layer
    plot_climatology(name, outputs[name]; layer = size(outputs[name], 3))
end
for name in soil # single soil layer
    plot_climatology(name, outputs[name]; layer = 1)
end
