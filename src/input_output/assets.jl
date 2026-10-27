# File formats
abstract type FileFormat end
struct NetCDF <: FileFormat end

@enum AssetGrid native N72

"""$(TYPEDSIGNATURES)
File extension (including the leading dot) used by files of the given [`FileFormat`](@ref).
"""
file_extension(::NetCDF) = ".nc"

# Asset types

"""
    $TYPEDEF

Lightweight base type for Terrarium land data assets.
"""
abstract type AbstractLandAsset end

"""
    $TYPEDEF

Time-invariant spatial fields from ERA5-Land at the native 0.1° x 0.1° resolution.

| Variable | Description | Units |
|:---------|:------------|:------|
| `cvh`  | High vegetation cover | fraction (0–1) |
| `lsm`  | Land-sea mask (fraction of land) | fraction (0–1) |
| `tvl`  | Type of low vegetation | categorical index |
| `cvl`  | Low vegetation cover | fraction (0–1) |
| `z`    | Surface geopotential (orography × g) | m²/s² |
| `slt`  | Soil type | categorical index |
| `dl`   | Lake total depth | m |
| `cl`   | Lake cover | fraction (0–1) |
| `si10` | 10 metre wind speed | m/s |
| `tvh`  | Type of high vegetation | categorical index |
"""
struct ERA5LandInvariants <: AbstractLandAsset end

artifact_name(::ERA5LandInvariants) = "era5-land-invariants"
varnames(::ERA5LandInvariants) = ["cvh", "lsm", "tvl", "cvl", "z", "slt", "dl", "cl", "si10", "tvh"]
format(::ERA5LandInvariants) = NetCDF()
native_grid(::ERA5LandInvariants) = RingGrids.FullClenshawGrid(900)
data_indices(::ERA5LandInvariants) = (:, :, 5)

"""
    $TYPEDEF

Leaf area index daily climatology for 1980-2010 from ERA5-Land at the native 0.1° x 0.1° resolution.

| Variable | Description | Units |
|:---------|:------------|:------|
| `lai_lv` | Leaf area index, low vegetation | m²/m² |
| `lai_hv` | Leaf area index, high vegetation | m²/m² |
"""
struct ERA5LandLeafAreaIndex{grid} <: AbstractLandAsset end

ERA5LandLeafAreaIndex(resolution::AssetGrid = native) = ERA5LandLeafAreaIndex{resolution}()

artifact_name(::ERA5LandLeafAreaIndex{native}) = "era5-land-leaf-area-index"
native_grid(::ERA5LandLeafAreaIndex{native}) = RingGrids.FullClenshawGrid(900)

artifact_name(::ERA5LandLeafAreaIndex{N72}) = "era5-land-leaf-area-index-N72"
native_grid(::ERA5LandLeafAreaIndex{N72}) = RingGrids.FullGaussianGrid(72)

varnames(::ERA5LandLeafAreaIndex) = ["lai_lv", "lai_hv"]
format(::ERA5LandLeafAreaIndex) = NetCDF()
data_indices(::ERA5LandLeafAreaIndex) = (:, :, :)

"""
    $TYPEDEF

One year of ERA5-Land hourly meterological variables regridded to approximately 1° x 1° resolution (72 Gaussian rings).

| Variable | Description | Units |
|:---------|:------------|:------|
| `t2m`  | 2 metre air temperature | K |
| `d2m`  | 2 metre dewpoint temperature | K |
| `tp`   | Total precipitation | m |
| `sf`   | Snowfall (water equivalent) | m |
| `sp`   | Surface pressure | Pa |
| `ssrd` | Surface solar (shortwave) radiation downwards | J/m² |
| `strd` | Surface thermal (longwave) radiation downwards | J/m² |
| `u10`  | 10 metre eastward (U) wind component | m/s |
| `v10`  | 10 metre northward (V) wind component | m/s |
"""
struct ERA5LandForcings <: AbstractLandAsset end

artifact_name(::ERA5LandForcings) = "era5-land-forcings-N72"
varnames(::ERA5LandForcings) = ["t2m", "d2m", "tp", "sf", "sp", "ssrd", "strd", "u10", "v10"]
format(::ERA5LandForcings) = NetCDF()
native_grid(::ERA5LandForcings) = RingGrids.FullGaussianGrid(72)
data_indices(::ERA5LandForcings) = (:, :, :)

# get_asset

"""
    $TYPEDSIGNATURES

Download (if necessary) the given `asset` and return the path to its data file within the installed
artifact directory. The artifact is installed via [`get_artifact`](@ref) and the data file is located
by its [`file_extension`](@ref). Reading the file is left to the caller; use [`load_asset`](@ref) to
read a variable into a data `Field` wrapped on the asset's `native_grid`.
"""
function get_asset(asset::AbstractLandAsset)
    artifact_dir = get_artifact(artifact_name(asset))
    fmt = format(asset)
    path = locate_asset_file(artifact_dir, fmt)
    return path
end

"""
    $TYPEDSIGNATURES

Download (if necessary) the given `asset` via [`get_asset`](@ref) and read the variable `name`
from its data file, returning a suitable Array-type based on the asset's file format. The asset's
`indices` are applied when reading (e.g. to select a single time record), and `fill_value`
replaces missing data; it defaults to `NF(NaN)`.

The underlying read is dispatched to an I/O extension based on the asset's [`format`](@ref); load
Rasters.jl and NCDatasets.jl to enable reading NetCDF and other raster files.
"""
function load_asset(asset::AbstractLandAsset, name::String; NF = Float32, fill_value = NF(NaN))
    path = get_asset(asset)
    fmt = format(asset)
    grid = native_grid(asset)
    return load_asset(path, name, grid, fmt, NF; indices = data_indices(asset), fill_value)
end

"""
    load_asset(path, name, fmt, ::Type{NF}; indices, fill_value) where {NF}

Dispatch implemented by I/O backends which reads variable `name` from the asset file at `path`.
"""
load_asset(path, name, fmt::FileFormat, ::Type{NF}; kwargs...) where {NF} = error("no extension loaded that handles $fmt; did you call `using Rasters`?")

"""
    $TYPEDSIGNATURES

Locate the single data file of the given `format` within an installed artifact directory `dir`,
searching recursively. Errors unless exactly one matching file is found.
"""
function locate_asset_file(dir::String, format::FileFormat)
    ext = file_extension(format)
    matches = String[]
    for (root, _, files) in walkdir(dir)
        for file in files
            endswith(file, ext) && push!(matches, joinpath(root, file))
        end
    end
    length(matches) == 1 || error("expected exactly one '$ext' file in artifact directory $dir, found $(length(matches))")
    return only(matches)
end

"""
    $TYPEDSIGNATURES

Retrieve the path to the artifact with the given `name`. Throws an `AssertionError` if no artifact with `name` exists in `Artifacts.toml`.
"""
function get_artifact(name::String)
    project_root = pkgdir(Terrarium)
    # fallback to @__DIR__ if pkgdir fails
    project_root = isnothing(project_root) ? (@__DIR__) : project_root
    artifact_toml = joinpath(project_root, "Artifacts.toml")
    # Check if the artifact already exists in the TOML
    hash = Pkg.Artifacts.artifact_hash(name, artifact_toml)
    @assert !isnothing(hash) "no artifact with name $name found"
    Pkg.Artifacts.ensure_artifact_installed(name, artifact_toml)
    return Pkg.Artifacts.artifact_path(hash)
end

RingGrids.Field(asset::AbstractLandAsset, name::String; kwargs...) = RingGrids.Field(CPU(), asset, name; kwargs...)
RingGrids.Field(arch, asset::AbstractLandAsset, name::String; kwargs...) = RingGrids.Field(arch, native_grid(asset), asset, name; kwargs...)
function RingGrids.Field(arch, grid::RingGrids.AbstractGrid, asset::AbstractLandAsset, name::String; kwargs...)
    data = on_architecture(arch, Matrix(load_asset(asset, name; kwargs...)))
    field = RingGrids.Field(on_architecture(arch, grid), size(data)[3:end]...)
    field .= reshape(data, :, size(data)[3:end]...)
    return field
end
