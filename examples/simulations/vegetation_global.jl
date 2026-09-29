# # [Simulating global vegetation with prescribed LAI](@id vegetation_prescribed_global)
# This example sets up a global [`VegetationModel`](@ref) driven by the [`PrescribedPhenology`](@ref)
# scheme, using the ERA5-Land leaf area index (LAI) climatology for high vegetation (`lai_hv`) as a
# annually repeating, time-varying input. The phenology factor `ϕ` is derived from the prescribed LAI
# and used by the vegetation carbon-cycle processes.

using Terrarium

using CUDA
using Dates
using Rasters, NCDatasets
using Rasters: X, Y, Ti

using CairoMakie, GeoMakie

import RingGrids

import DisplayAs #hide

# Run on GPU if available, otherwise CPU.
arch = CUDA.functional() ? GPU() : CPU()
NF = Float32
@info "Setting up simulation on $arch" #hide

# ## Grid and land mask
# As in the [global soil heat example](@ref soil_heat_global), we build a masked
# [`ColumnRingGrid`](@ref) at the native ~10km resolution of ERA5-Land, keeping only grid points with
# >50% land cover.
land_sea_frac = RingGrids.Field(arch, ERA5LandInvariants(), "lsm"; NF)
model_rings = land_sea_frac.grid
land_mask = land_sea_frac .> 0.5
grid = ColumnRingGrid(arch, NF, ExponentialSpacing(N = 10), land_mask.grid, land_mask)

# ## Prescribed LAI climatology
# The `ERA5LandLeafAreaIndex` asset holds a daily LAI climatology (1980–2010) at the native 0.1°
# resolution. Note that, due to the high resolution, it may take a few minutes to download the data.
lai_asset = ERA5LandLeafAreaIndex()
lai_asset_path = Terrarium.get_asset(lai_asset)

# We next load the asset into a `RasterStack` with `lazy = true` to avoid loading the full dataset.
# cycle = true` makes the LAI climatology repeat every year over the whole simulation.
lai_raster = RasterStack(lai_asset_path, lazy = true)
lai_highveg = convert.(NF, replace_missing(lai_raster[:lai_hv], zero(NF)))
lai_input = InputSource(grid, lai_highveg; source_grid = Terrarium.native_grid(lai_asset), name = :leaf_area_index, cycle = true, domain = Terrarium.Canopy())

# ## Vegetation model
# Here we set up a [`PrescribedVegetation`](@ref) model which uses [`PrescribedPhenology`](@ref) to take LAI as a prescribed input.
# Photosynthesis and stomatal conductance are still simulated but are not used by the standalone [`VegetationModel`](@ref).
# We provide the LAI climatology to the model via `inputs = lai_input`.
vegetation = PrescribedVegetation(NF)
model = VegetationModel(grid; vegetation)
integrator = initialize(model; inputs = lai_input)

# Now we'll define a small helper function to plot a horizontal field on the ring grid:
plot_field(field; kwargs...) = heatmap(RingGrids.Field(arch, interior(field), grid)[:, 1, 1]; kwargs...)

# Before starting the simulation, let's first take a look at the initial prescribed LAI for the month of January:
fig = plot_field(integrator.state.leaf_area_index, title = "Leaf area index", colorrange = (0, 6))
DisplayAs.PNG(fig) #hide

# ## Running the simulation
# We advance the simulation for three months with a daily timestep. By default, the [`PrescribedAtmosphere`](@ref)
# for [`VegetationModel`](@ref) uses globally constant values, so we shouldn't expect realistic daily values. However,
# we can still check that the cyclic LAI input drives the seasonal evolution of downstream variables like gross primary
# production (GPP). Note that this simulation is currently quite slow due to I/O overhead, but this will improve in the near future.
@time run!(integrator, period = Month(3), Δt = Day(1), show_progress = true)

fig_LAI = plot_field(integrator.state.leaf_area_index, title = "Leaf area index", colorrange = (0, 6))
DisplayAs.PNG(fig) #hide

fig_GPP = plot_field(integrator.state.gross_primary_production * 24 * 3600, title = "Gross Primary Production", colorrange = (0, 6))
DisplayAs.PNG(fig) #hide
