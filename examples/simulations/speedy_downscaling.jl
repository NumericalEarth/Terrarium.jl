using Terrarium

using CUDA
using Dates
using Rasters, NCDatasets
using Statistics

using CairoMakie, GeoMakie

import RingGrids
import SpeedyWeather as Speedy

# Choose architecture based on available hardware
terrarium_arch = CUDA.functional() ? GPU() : CPU()
speedy_arch = CUDA.functional() ? Speedy.GPU() : Speedy.CPU()

"""
Naive implementation of a SpeedyWeather "wet" land model based on Terrarium.
Operates with two separate ring grids: a lower-resolution grid for Speedy and a
higher-resolution grid for Terrarium, with `RingGrids.interpolate!` used to couple them.
"""
struct TerrariumWetLand{
        NF,
        LG <: Speedy.LandGeometry,
        TM <: Terrarium.ModelIntegrator{NF},
        IU <: RingGrids.AbstractInterpolator,
        ID <: RingGrids.AbstractInterpolator,
        FT <: RingGrids.AbstractField,
        FS <: RingGrids.AbstractField,
    } <: Speedy.AbstractWetLand
    "Speedy spectral grid (low resolution)"
    spectral_grid::Speedy.SpectralGrid

    "Speedy land model geometry"
    geometry::LG

    "Initialized Terrarium model integrator (high resolution)"
    integrator::TM

    "Pre-computed interpolator: Speedy ring grid (low-res) → Terrarium ring grid (high-res)"
    interp_up::IU

    "Pre-computed interpolator: Terrarium ring grid (high-res) → Speedy ring grid (low-res)"
    interp_down::ID

    "Reusable 2D buffer Field on Terrarium ring grid (for upscaling Speedy → Terrarium)"
    buffer_terrarium::FT

    "Reusable 2D buffer Field on Speedy ring grid (for downscaling Terrarium → Speedy)"
    buffer_speedy::FS

    function TerrariumWetLand(
            integrator::Terrarium.ModelIntegrator{NF, Arch, Grid},
            speedy_ring_grid::RingGrids.AbstractGrid,
            terrarium_ring_grid::RingGrids.AbstractGrid;
            spectral_grid_kwargs...
        ) where {NF, Arch, Grid <: ColumnRingGrid}
        spectral_grid = Speedy.SpectralGrid(speedy_ring_grid; NF, spectral_grid_kwargs...)
        interp_up = RingGrids.interpolator(terrarium_ring_grid, speedy_ring_grid)
        interp_down = RingGrids.interpolator(speedy_ring_grid, terrarium_ring_grid)
        buffer_terrarium = zeros(NF, terrarium_ring_grid)
        buffer_speedy = zeros(NF, speedy_ring_grid)
        land_grid = integrator.grid
        Δz = on_architecture(CPU(), land_grid.z.Δᵃᵃᶜ)
        geometry = Speedy.LandGeometry(1, Δz[end])
        return new{NF, typeof(geometry), typeof(integrator), typeof(interp_up), typeof(interp_down), typeof(buffer_terrarium), typeof(buffer_speedy)}(
            spectral_grid, geometry, integrator, interp_up, interp_down, buffer_terrarium, buffer_speedy
        )
    end
end

Speedy.variables(land::TerrariumWetLand) = (
    Speedy.PrognosticVariable(name = :soil_temperature, dims = Speedy.Grid3D(), namespace = :land),
    Speedy.PrognosticVariable(name = :soil_moisture, dims = Speedy.Grid3D(), namespace = :land),
)

function Speedy.initialize!(
        vars::Speedy.Variables,
        land::TerrariumWetLand{NF},
        model::Speedy.PrimitiveEquation,
    ) where {NF}
    Terrarium.initialize!(land.integrator)
    state = land.integrator.state
    grid = land.integrator.model.grid
    # Downscale Terrarium initial state to Speedy resolution
    scatter_land!(land.buffer_terrarium, interior(state.temperature)[:, 1, end], grid)
    RingGrids.interpolate!(land.buffer_speedy, land.buffer_terrarium, land.interp_down)
    vars.prognostic.land.soil_temperature .= land.buffer_speedy .+ NF(273.15)
    scatter_land!(land.buffer_terrarium, interior(state.saturation_water_ice)[:, 1, end], grid)
    RingGrids.interpolate!(land.buffer_speedy, land.buffer_terrarium, land.interp_down)
    vars.prognostic.land.soil_moisture .= land.buffer_speedy
    return nothing
end

function Speedy.timestep!(
        vars::Speedy.Variables,
        land::TerrariumWetLand{NF},
        model::Speedy.PrimitiveEquation,
    ) where {NF}
    speedy_timestep!(vars, land)
    return nothing
end

## Scatter land-only Oceananigans interior data (length Nh) back to a full-resolution RingGrids Field2D.
## Sea grid points are set to zero.
function scatter_land!(out::RingGrids.AbstractField, data::AbstractArray, grid::ColumnRingGrid)
    fill!(out, zero(eltype(out)))
    out.data[grid.mask.data] .= data
    return out
end

function speedy_timestep!(
        vars::Speedy.Variables,
        land::TerrariumWetLand{NF},
    ) where {NF}
    # land constants
    consts = land.integrator.model.constants
    state = land.integrator.state
    terrarium_grid = land.integrator.model.grid
    mask = terrarium_grid.mask.data
    # Upscale Speedy fields to Terrarium resolution and update inputs
    ## For each field: interpolate to full Terrarium ring grid, then gather only the land-masked points
    RingGrids.interpolate!(land.buffer_terrarium, vars.grid.temperature[:, end], land.interp_up)
    land.buffer_terrarium.data .-= NF(273.15)  ## convert K → °C
    interior(state.inputs.air_temperature)[:, 1, 1] .= land.buffer_terrarium.data[mask]
    RingGrids.interpolate!(land.buffer_terrarium, vars.grid.pressure, land.interp_up)
    land.buffer_terrarium.data .= exp.(land.buffer_terrarium.data)  ## convert log(Pa) → Pa
    interior(state.inputs.air_pressure)[:, 1, 1] .= land.buffer_terrarium.data[mask]
    RingGrids.interpolate!(land.buffer_terrarium, vars.grid.humidity[:, end], land.interp_up)
    interior(state.inputs.specific_humidity)[:, 1, 1] .= land.buffer_terrarium.data[mask]
    RingGrids.interpolate!(land.buffer_terrarium, vars.parameterizations.rain_rate, land.interp_up)
    interior(state.inputs.rainfall)[:, 1, 1] .= land.buffer_terrarium.data[mask]
    RingGrids.interpolate!(land.buffer_terrarium, vars.parameterizations.snow_rate, land.interp_up)
    interior(state.inputs.snowfall)[:, 1, 1] .= land.buffer_terrarium.data[mask]
    RingGrids.interpolate!(land.buffer_terrarium, vars.parameterizations.surface_wind_speed, land.interp_up)
    interior(state.inputs.windspeed)[:, 1, 1] .= land.buffer_terrarium.data[mask]
    RingGrids.interpolate!(land.buffer_terrarium, vars.parameterizations.surface_shortwave_down, land.interp_up)
    interior(state.inputs.surface_shortwave_down)[:, 1, 1] .= land.buffer_terrarium.data[mask]
    RingGrids.interpolate!(land.buffer_terrarium, vars.parameterizations.surface_longwave_down, land.interp_up)
    interior(state.inputs.surface_longwave_down)[:, 1, 1] .= land.buffer_terrarium.data[mask]
    # run land forward over speedy timestep interval;
    # we use a smaller actual timestep to ensure stability
    Terrarium.run!(land.integrator, period = vars.prognostic.clock.Δt, Δt = 300.0)
    # Downscale Terrarium output fields to Speedy resolution
    scatter_land!(land.buffer_terrarium, interior(state.skin_temperature)[:, 1, 1], terrarium_grid)
    RingGrids.interpolate!(land.buffer_speedy, land.buffer_terrarium, land.interp_down)
    vars.prognostic.land.soil_temperature .= land.buffer_speedy .+ NF(273.15)
    scatter_land!(land.buffer_terrarium, interior(state.saturation_water_ice)[:, 1, end], terrarium_grid)
    RingGrids.interpolate!(land.buffer_speedy, land.buffer_terrarium, land.interp_down)
    vars.prognostic.land.soil_moisture .= land.buffer_speedy
    scatter_land!(land.buffer_terrarium, interior(state.sensible_heat_flux)[:, 1, 1], terrarium_grid)
    RingGrids.interpolate!(land.buffer_speedy, land.buffer_terrarium, land.interp_down)
    vars.prognostic.land.sensible_heat_flux .= land.buffer_speedy
    scatter_land!(land.buffer_terrarium, interior(state.latent_heat_flux)[:, 1, 1], terrarium_grid)
    RingGrids.interpolate!(land.buffer_speedy, land.buffer_terrarium, land.interp_down)
    vars.prognostic.land.surface_humidity_flux .= land.buffer_speedy ./ consts.Llg
    scatter_land!(land.buffer_terrarium, interior(state.surface_longwave_up)[:, 1, 1], terrarium_grid)
    RingGrids.interpolate!(land.buffer_speedy, land.buffer_terrarium, land.interp_down)
    vars.parameterizations.land.surface_longwave_up .= land.buffer_speedy
    scatter_land!(land.buffer_terrarium, interior(state.surface_shortwave_up)[:, 1, 1], terrarium_grid)
    RingGrids.interpolate!(land.buffer_speedy, land.buffer_terrarium, land.interp_down)
    vars.parameterizations.land.surface_shortwave_up .= land.buffer_speedy
    return nothing
end

# quick test of default Speedy PrimitiveWetModel on GPU
speedy_ring_grid = RingGrids.FullGaussianGrid(24, speedy_arch)
speedy_ring_grid_cpu = on_architecture(CPU(), speedy_ring_grid)
spectral_grid = Speedy.SpectralGrid(speedy_ring_grid)
orography = Speedy.EarthOrography(spectral_grid, smoothing = false)
primitive_wet = Speedy.PrimitiveWetModel(spectral_grid; orography)
sim = Speedy.initialize!(primitive_wet)
Speedy.run!(sim, period = Day(1))

# Higher-resolution ring grid on Speedy's architecture for use in coupling buffers/interpolators
terrarium_ring_grid = RingGrids.FullGaussianGrid(72 * 2, speedy_arch)
terrarium_ring_grid_cpu = on_architecture(CPU(), terrarium_ring_grid)

# Load SpeedyWeather land-sea mask at Terrarium resolution; land = 1, sea = 0 (fractional)
## Threshold at 0.5 to get a Bool field suitable for ColumnRingGrid masking
land_sea_mask_raw = RingGrids.get_asset(
    "data/boundary_conditions/land-sea_mask.nc",
    from_assets = true, name = "lsm",
    ArrayType = RingGrids.FullClenshawField, FileFormat = NCDataset,
)
land_sea_mask_terrarium = clamp.(RingGrids.grid_cell_average!(RingGrids.Field(terrarium_ring_grid_cpu), land_sea_mask_raw), 0, 1)
land_sea_mask_atmos = clamp.(RingGrids.grid_cell_average!(RingGrids.Field(speedy_ring_grid_cpu), land_sea_mask_raw), 0, 1)

Nz = 30
Δz_min = 0.05
grid = ColumnRingGrid(terrarium_arch, Float32, ExponentialSpacing(; N = Nz, Δz_min), terrarium_ring_grid_cpu, land_sea_mask_terrarium .> 0.0f0)
# Initial conditions
soil_initializer = SoilInitializer(eltype(grid))
soil = SoilEnergyWaterCarbon(eltype(grid), hydrology = SoilHydrology(eltype(grid)))
# Land model with "prescribed" atmosphere (from the perspective of the land model at least...)
# vegetation = PrescribedVegetationCarbon(eltype(grid))
model = LandModel(grid; initializer = soil_initializer, vegetation = nothing, soil)
initializers = (;)
integrator = initialize(model, ForwardEuler(eltype(grid)); initializers)
# check if land model works standalone (with default atmospheric state)
timestep!(integrator, 60.0) # one step
run!(integrator, period = Hour(1), Δt = 300.0) # one hour
Terrarium.initialize!(integrator) # reinitialize before setting up atmosphere

# Initialize Terrarium-Speedy land model using separate low-res (Speedy) and high-res (Terrarium) ring grids
land = TerrariumWetLand(integrator, speedy_ring_grid, terrarium_ring_grid)
# Set up coupled model
land_sea_mask = Speedy.LandSeaMask(land.spectral_grid)
surface_heat_flux = Speedy.SurfaceHeatFlux(land.spectral_grid, land = Speedy.PrescribedLandHeatFlux())
surface_humidity_flux = Speedy.SurfaceHumidityFlux(land.spectral_grid, land = Speedy.PrescribedLandHumidityFlux())
output = Speedy.NetCDFOutput(land.spectral_grid, Speedy.PrimitiveWetModel, path = "outputs/")
time_stepping = Speedy.Leapfrog(land.spectral_grid, Δt_at_T31 = Minute(15))
primitive_wet_coupled = Speedy.PrimitiveWetModel(
    land.spectral_grid;
    land,
    surface_heat_flux,
    surface_humidity_flux,
    land_sea_mask,
    time_stepping,
    output
)
# add soil temperature as output variable for Speedy simulation
Speedy.add!(primitive_wet_coupled.output, Speedy.SoilTemperatureOutput())
# initialize coupled simulation
sim_coupled = Speedy.initialize!(primitive_wet_coupled)
# run it (kept for reference) — we'll also run a stepwise loop below to build an animation
period = Month(1)
@info "Running simulation for $period"
@time Speedy.run!(sim_coupled; period)
Terrarium.checkfinite!(integrator.state.prognostic)

# --- 2×2 Animation: land and atmosphere state evolution during the coupled run ---
# Re-initialize the simulation to start from the beginning for the animation.
Speedy.initialize!(sim_coupled; steps = 240, output = false)

# Coordinate axes for terrarium (high-res) and Speedy (low-res) grids, computed once.
# We replicate what RingGridsMakieExt does internally for heatmap (no mutating variant exists)
# so that we can drive heatmap!(ax, lond, latd, obs::Observable{Matrix}) in the record loop.
lond_hi = Float32.(RingGrids.get_lond(terrarium_ring_grid_cpu))
latd_hi = Float32.(RingGrids.get_latd(terrarium_ring_grid_cpu))
lond_lo = Float32.(RingGrids.get_lond(speedy_ring_grid_cpu))
latd_lo = Float32.(RingGrids.get_latd(speedy_ring_grid_cpu))

# CPU buffer for scatter_land! → Matrix conversion during each animation frame.
buffer_anim = zeros(Float32, terrarium_ring_grid_cpu)

# Helper: scatter land-only Oceananigans field data at `layer` into `buffer_anim` and return
# as a (nlon_hi × nlat_hi) Matrix suitable for heatmap!.
# Sea grid cells are set to NaN so they render as masked (transparent) in the animation.
function terrarium_frame(oc_field, layer)
    data_cpu = Array(interior(oc_field)[:, 1, layer])
    fill!(buffer_anim, NaN32)
    buffer_anim.data[grid.mask.data] .= data_cpu
    return Matrix(buffer_anim)
end

# Helper: extract a (nlon_lo × nlat_lo) Matrix from a Speedy 2D ring-grid field (moves to CPU).
# An optional element-wise transform is applied to the flat data before reshaping.
function speedy_frame(rg_field_2d; transform = identity)
    data_cpu = transform(Array(rg_field_2d.data))
    return reshape(data_cpu, length(lond_lo), length(latd_lo))
end

# Initial matrices for all four panels.
mat_tsoil = terrarium_frame(land.integrator.state.temperature, Nz)
mat_evap = terrarium_frame(land.integrator.state.evaporation_ground, 1)
mat_tair = speedy_frame(sim_coupled.variables.grid.temperature[:, end]; transform = x -> x .- 273.15f0)
mat_prec = speedy_frame(sim_coupled.variables.parameterizations.rain_rate)

# Build the 2×2 Figure layout.
# Axes occupy columns 1 and 3; their matching Colorbars occupy columns 2 and 4.
function heatmap_axis(fig, row, col, title)
    return Axis(
        fig[row, col];
        title, aspect = 2, titlesize = 12,
        xticks = 0:60:360, yticks = -60:30:60,
        xticklabelsize = 9, yticklabelsize = 9,
        xtickformat = vs -> ["$(round(Int, v))˚E" for v in vs],
        ytickformat = vs -> ["$(round(Int, v))˚N" for v in vs],
    )
end


# Fixed colorbar ranges — standardised so colors are consistent across all frames.
# Adjust these if your simulation uses very different conditions.
crange_tsoil = (-30.0f0, 30.0f0)    # °C  — typical land surface temperature range
crange_evap = (0.0f0, 1.0f-5)     # m s⁻¹ — bare-soil evaporation
crange_tair = (-30.0f0, 30.0f0)    # °C  — tropospheric temperature at lowest model level
crange_prec = (0.0f0, 2.0f-7)     # m s⁻¹ — rain rate

Makie.with_theme(fontsize = 16) do
    fig_anim = Figure(size = (1400, 700))

    ax_tsoil = heatmap_axis(fig_anim, 1, 1, "Soil temperature (°C)")
    ax_evap = heatmap_axis(fig_anim, 1, 3, "Evaporation (m s⁻¹)")
    ax_tair = heatmap_axis(fig_anim, 2, 1, "Air temperature (°C)")
    ax_prec = heatmap_axis(fig_anim, 2, 3, "Precipitation (m s⁻¹)")

    obs_tsoil = Observable(mat_tsoil)
    obs_evap = Observable(mat_evap)
    obs_tair = Observable(mat_tair)
    obs_prec = Observable(mat_prec)

    hm_tsoil = heatmap!(ax_tsoil, lond_hi, latd_hi, obs_tsoil; colormap = :temperaturemap, colorrange = crange_tsoil)
    hm_evap = heatmap!(ax_evap, lond_hi, latd_hi, obs_evap; colormap = :viridis, colorrange = crange_evap)
    hm_tair = heatmap!(ax_tair, lond_lo, latd_lo, obs_tair; colormap = :temperaturemap, colorrange = crange_tair)
    hm_prec = heatmap!(ax_prec, lond_lo, latd_lo, obs_prec; colormap = :Blues, colorrange = crange_prec)

    Colorbar(fig_anim[1, 2], hm_tsoil; label = "°C", ticklabelsize = 8)
    Colorbar(fig_anim[1, 4], hm_evap; label = "m s⁻¹", ticklabelsize = 8)
    Colorbar(fig_anim[2, 2], hm_tair; label = "°C", ticklabelsize = 8)
    Colorbar(fig_anim[2, 4], hm_prec; label = "m s⁻¹", ticklabelsize = 8)

    clock_anim = sim_coupled.variables.prognostic.clock
    time_label = Observable(string(clock_anim.time))
    Label(fig_anim[0, :], time_label; fontsize = 14)

    mkpath("plots")
    Makie.record(fig_anim, "plots/speedy_terrarium_bare_soil_atmosphere_10days.mp4", 1:240; framerate = 12) do _
        Speedy.run!(sim_coupled, period = Hour(1))
        obs_tsoil[] = terrarium_frame(land.integrator.state.temperature, Nz)
        obs_evap[] = terrarium_frame(land.integrator.state.evaporation_ground, 1)
        obs_tair[] = speedy_frame(sim_coupled.variables.grid.temperature[:, end]; transform = x -> x .- 273.15f0)
        obs_prec[] = speedy_frame(sim_coupled.variables.parameterizations.rain_rate)
        time_label[] = string(clock_anim.time)
    end
end

## Plot a RingGrids field: converts to CPU, draws heatmap, and labels the colorbar as "name / units".
function land_heatmap(field, grid::ColumnRingGrid, label; kwargs...)
    fig = heatmap(RingGrids.Field(field, grid); kwargs...)
    filter(x -> x isa Makie.Colorbar, fig.content)[1].label = label
    return fig
end

function atmos_heatmap(field, ring_grid, label; kwargs...)
    fig = heatmap(RingGrids.Field(field, ring_grid); kwargs...)
    filter(x -> x isa Makie.Colorbar, fig.content)[1].label = label
    return fig
end

# Land variables — convert masked Oceananigans field to full CPU ring grid, then plot
Tsoil_fig = land_heatmap(RingGrids.Field(on_architecture(CPU(), integrator.state.temperature), grid)[:, end - 5], terrarium_ring_grid_cpu, "Temperature / °C", title = "Soil temperature", size = (800, 400))
save("plots/terrarium_speedy_Tsoil_hires.png", Tsoil_fig)
Tsurf_fig = land_heatmap(RingGrids.Field(on_architecture(CPU(), integrator.state.skin_temperature), grid)[:, 1], terrarium_ring_grid_cpu, "Temperature / °C", title = "Skin temperature", size = (800, 400))
save("plots/terrarium_speedy_Tsurf_hires.png", Tsurf_fig)
Hs_fig = land_heatmap(RingGrids.Field(on_architecture(CPU(), integrator.state.sensible_heat_flux), grid)[:, 1], terrarium_ring_grid_cpu, "Sensible heat flux / W m⁻²", title = "Sensible heat flux", size = (800, 400))
save("plots/terrarium_speedy_Hs_hires.png", Hs_fig)
Hl_fig = land_heatmap(RingGrids.Field(on_architecture(CPU(), integrator.state.latent_heat_flux), grid)[:, 1], terrarium_ring_grid_cpu, "Latent heat flux / W m⁻²", title = "Latent heat flux", size = (800, 400))
save("plots/terrarium_speedy_Hl_hires.png", Hl_fig)
E_fig = land_heatmap(RingGrids.Field(on_architecture(CPU(), integrator.state.evaporation_ground), grid)[:, 1], terrarium_ring_grid_cpu, "Evaporation / m s⁻¹", title = "Evaporation", size = (800, 400))
save("plots/terrarium_speedy_E_hires.png", E_fig)
# sat_fig = land_heatmap(RingGrids.Field(on_architecture(CPU(), integrator.state.saturation_water_ice), grid)[:, end], terrarium_ring_grid_cpu, "Saturation / -", title = "Surface saturation", size = (800, 400))

# Atmosphere variables (Speedy v0.19 API) — already on FullGaussianGrid(16)
Tair_fig = atmos_heatmap(sim_coupled.variables.grid.temperature[:, 8].data .- 273.15f0, speedy_ring_grid_cpu, "Temperature / °C", title = "Air temperature", size = (800, 400))
pres_fig = atmos_heatmap(exp.(sim_coupled.variables.grid.pressure.data), speedy_ring_grid_cpu, "Pressure / Pa", title = "Surface pressure", size = (800, 400))
srad_fig = atmos_heatmap(sim_coupled.variables.parameterizations.surface_shortwave_down.data, speedy_ring_grid_cpu, "Shortwave down / W m⁻²", title = "Surface shortwave down", size = (800, 400))

# Pick a point somewhere in the mid-latitudes and plot vertical profiles
T = on_architecture(CPU(), interior(integrator.state.temperature)[8000, 1, :])
sat = on_architecture(CPU(), interior(integrator.state.saturation_water_ice)[8000, 1, :])
f = on_architecture(CPU(), interior(integrator.state.liquid_water_fraction)[8000, 1, :])
zs = on_architecture(CPU(), znodes(integrator.state.temperature))

# Plot temperature, saturation, and liquid fraction vertical profiles
Makie.scatterlines(T, zs)
Makie.scatterlines(sat, zs)
Makie.scatterlines(f, zs)
