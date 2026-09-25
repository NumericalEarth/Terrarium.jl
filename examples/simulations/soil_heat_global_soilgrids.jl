# # [Global soil heat conduction with heterogeneous soil from SoilGrids 2.0](@id soil_heat_global_soilgrids)
# This example extends the [global soil heat conduction example](@ref soil_heat_global) with a
# heterogeneous, multi-layer soil stratigraphy whose texture (sand/silt/clay fractions) is
# prescribed from the [SoilGrids 2.0](https://soilgrids.org) dataset, accessed via
# [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl).
#
# The key new concept demonstrated here are *namespaced input variables*: each soil horizon of
# the [`SoilStratigraphy`](@ref) declares its own `sand_fraction`, `silt_fraction`, `clay_fraction`,
# and `thickness` input variables inside a namespace named after the horizon. Input sources are matched
# to these variables via namespaced names such as `:horizon1 => :sand_fraction`.

using Terrarium

using CUDA
using Dates
using Rasters, NCDatasets
using Statistics

using CairoMakie, GeoMakie

import RingGrids
import DisplayAs #hide

# NumericalEarth.jl provides the `SoilGrids2` dataset together with download and
# preprocessing machinery.
using NumericalEarth.DataWrangling: MetadataSet
using NumericalEarth.SoilGrids: SoilGrids2

input_dir = "inputs" #hide
@info "Current working directory: $(pwd())" #hide

# First we check if a GPU is available and choose the architecture correspondingly.
arch = CUDA.functional() ? GPU() : CPU()
@info "Setting up simulation on $arch" #hide

# As in the [global example](@ref soil_heat_global), we load a land-sea mask at ~1° resolution
# and set up a masked [`ColumnRingGrid`](@ref) with 30 exponentially spaced soil layers.
NF = Float32
land_sea_frac = convert.(NF, dropdims(Raster(joinpath(input_dir, "era5-land_land_sea_mask_N72.nc")), dims = Ti))
land_sea_frac_field = RingGrids.FullGaussianGrid(Matrix(land_sea_frac), input_as = Matrix)
land_mask = land_sea_frac_field .> 0.5
grid = ColumnRingGrid(arch, NF, ExponentialSpacing(N = 30), land_mask.grid, land_mask)
grid_lon, grid_lat = RingGrids.get_lonlats(grid.rings) # in radians

# ## Loading soil texture from SoilGrids 2.0
# SoilGrids 2.0 provides global predictions of soil properties on a ~10 km grid for the six
# standard [GlobalSoilMap](https://www.isric.org/projects/globalsoilmap) depth intervals
# (0-5, 5-15, 15-30, 30-60, 60-100, and 100-200 cm). Sand, silt, and clay are mass fractions
# of the fine earth (< 2 mm) component reported in g/kg following the USDA particle-size
# classification (clay < 2 µm, silt 2-50 µm, sand 50-2000 µm); see the
# [GlobalSoilMap specifications (2015)](https://www.isric.org/sites/default/files/GlobalSoilMap_specifications_december_2015_2.pdf)
# for the full reference. NumericalEarth.jl downloads the data on first use (~several hundred
# MB, cached afterwards), converts the fractions to the unit interval, and returns Oceananigans
# `Field`s with the vertical axis ordered from the deepest layer (k=1, 100-200 cm) to the
# surface layer (k=6, 0-5 cm).
function Terrarium.InputSources(dataset::SoilGrids2, grid::ColumnRingGrid, horizons = (Symbol(:horizon, i) for i in 1:6); name = nameof(typeof(dataset)), verbose = true)
    soilgrids_vars = (:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)
    metadataset = MetadataSet(soilgrids_vars...; dataset)
    arch = RingGrids.architecture(grid.rings)
    soilgrids_inputs = []
    for (idx, horizon) in enumerate(horizons)
        layer_inputs = Dict()
        for var in soilgrids_vars
            verbose && @info "Loading input data for $var on $horizon"
            var_field = Field(getproperty(metadataset, var))
            ring_field = RingGrids.on_architecture(arch, RingGrids.FullClenshawField(interior(var_field)[:, (end - 1):-1:2, end - idx + 1], input_as = Matrix))
            target_field = RingGrids.Field(grid.rings)
            RingGrids.interpolate!(target_field, ring_field)
            layer_inputs[var] = InputSource(grid, Field(target_field, grid); name = horizon => var, domain = Terrarium.Ground())
        end
        # Ensure that mineral texture components with each horizon sum to unity
        Terrarium.normalize_texture!(layer_inputs[:sand_fraction].field, layer_inputs[:silt_fraction].field, layer_inputs[:clay_fraction].field)
        append!(soilgrids_inputs, values(layer_inputs))
    end
    return InputSources(name, soilgrids_inputs...)
end

soilgrids_inputs = InputSources(SoilGrids2(), grid)

fig = heatmap(RingGrids.Field(on_architecture(CPU(), soilgrids_inputs.sources[2].field), grid)[:, 1])
DisplayAs.PNG(fig) #hide

# ## Heterogeneous soil stratigraphy
# We now construct a six-layer `SoilStratigraphy` using the `SoilGridsStratigraphy` convenience constructor.
porosity = SoilPorositySURFEX(eltype(grid))
strat = SoilGridsStratigraphy(eltype(grid); porosity)
soil = SoilEnergyWaterCarbon(eltype(grid); strat)
model = SoilModel(grid; soil)

# We reuse the simple latitude-dependent climatology from the [global example](@ref soil_heat_global)
# for the initial and boundary conditions.
mean_annual_temperature(lat) = 20 - abs(40 * sin(lat))

lon_masked = grid_lon[land_mask]
lat_masked = grid_lat[land_mask]

function initial_soil_temperature(x, z)
    latᵢ = lat_masked[round(Int, x)]
    T₀ = mean_annual_temperature(latᵢ)
    T = T₀ - 0.05 * z
    return T
end

function get_temperature_bc(lon::AbstractVector, lat::AbstractVector, amplitude = 10.0)
    lon_device = on_architecture(arch, NF.(lon))
    lat_device = on_architecture(arch, NF.(lat))
    function periodic_bc(x::NF, t::NF) where {NF}
        lonₓ = lon_device[round(Int, x)]
        latₓ = lat_device[round(Int, x)]
        T₀ = mean_annual_temperature(latₓ)
        seconds_per_day = NF(24 * 3600)
        T = T₀ + NF(amplitude) * sin(2π * t / seconds_per_day - lonₓ)
        return T
    end
    return periodic_bc
end

bc = PrescribedSurfaceTemperature(:T_ub, get_temperature_bc(lon_masked, lat_masked))
inits = (temperature = initial_soil_temperature,)

# Initialize the model; the input sources are matched to the namespaced input variables of
# the prescribed soil horizons.
integrator = initialize(model, ForwardEuler(NF); inputs = soilgrids_inputs, boundary_conditions = bc, initializers = inits)

# We can verify that the SoilGrids texture has been correctly assigned to the first horizon:
sand1 = RingGrids.Field(arch, interior(integrator.state.namespaces.horizon1.sand_fraction), grid)
fig = heatmap(sand1[:, 1, 1], title = "Sand fraction of the uppermost horizon")
DisplayAs.PNG(fig) #hide

# ## Running the simulation
# We wrap the integrator in an Oceananigans `Simulation` and attach a `JLD2Writer` that saves the
# soil temperature field every six hours over a 10-day simulation.
using Oceananigans: JLD2Writer, TimeInterval, prettytime
using JLD2

output_dir = mkpath("outputs") #hide
output_file = joinpath(output_dir, "soil_heat_global_soilgrids.jld2")

simulation = Simulation(integrator; Δt = 300.0, stop_time = 10 * 24 * 3600.0)
simulation.output_writers[:temperature] = JLD2Writer(
    integrator,
    (; temperature = integrator.state.temperature);
    filename = output_file,
    schedule = TimeInterval(3600),
    overwrite_files = true,
)

@time run!(simulation)

# ## Visualizing the change in soil temperature
# We read the saved temperature back as a `FieldTimeSeries` and, at the 25 cm soil layer, compute
# the change relative to the initial condition (`Ts[1]`). A divergent blue-white-red colormap
# centered at zero isolates the texture-dependent thermal response. First, the change at the final time:
Ts = FieldTimeSeries(output_file, "temperature")
k = size(Ts, 3) - 3 # vertical index of the 25 cm soil layer

soil_temperature_change(t) = RingGrids.Field(CPU(), interior(Ts[t])[:, 1, :] .- interior(Ts[1])[:, 1, :], grid)

fig = heatmap(soil_temperature_change(length(Ts.times))[:, k], title = "Change in 25 cm soil temperature (vs. initial)", colorrange = (-5, 5), colormap = :bwr)
DisplayAs.PNG(fig) #hide

# Finally, we animate the full evolution of the change over the 10-day simulation. Makie cannot
# convert an `Observable`-wrapped `RingGrids.Field` directly, so we extract the longitude/latitude
# axes once and animate the plain `Matrix` of each frame.
ΔT₁ = soil_temperature_change(1)[:, k]
lond = RingGrids.get_lond(ΔT₁)
latd = RingGrids.get_latd(ΔT₁)

let n_t = Observable(1)
    data = @lift Matrix(soil_temperature_change($n_t)[:, k])
    plot_title = @lift "Change in 25 cm soil temperature at t = " * prettytime(Ts.times[$n_t])
    fig = Figure(size = (1000, 500))
    ax = Axis(fig[1, 1], title = plot_title, xlabel = "Longitude", ylabel = "Latitude")
    hm = heatmap!(ax, lond, latd, data, colorrange = (-5, 5), colormap = :bwr)
    Colorbar(fig[1, 2], hm, label = "ΔT (°C)")
    Makie.record(fig, joinpath("plots", "soil_temperature_change.mp4"), 1:length(Ts.times), framerate = 8) do i
        n_t[] = i
    end
end

# ![Soil temperature change animation](soil_temperature_change.mp4)
