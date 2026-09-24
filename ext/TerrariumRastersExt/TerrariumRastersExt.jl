module TerrariumRastersExt

using Terrarium
using Terrarium: XY, XYZ, timestamp
using Unitful: NoUnits

using Dates
using DocStringExtensions
using Interpolations
using Oceananigans
using Oceananigans.OutputReaders: Cyclical, Clamp
using Rasters
using Rasters: TimeDim

import Oceananigans.Architectures: on_architecture
import RingGrids

const Extrapolation = Union{Cyclical, Clamp}

function Terrarium.load_asset(path, name, grid, ::Terrarium.NetCDF, ::Type{NF}; indices = (:, :, :), fill_value = NF(NaN), lazy = true) where {NF}
    raster = convert.(NF, replace_missing(Raster(path; name, lazy), fill_value))
    data = reconcile_latitudes(view(raster, indices...), grid)
    return data
end

"""
    $TYPEDEF

Input source that reads data from a time-varying `Raster`, linearly interpolating in time between records.
When `cycle = true`, the time axis is treated as periodic (e.g. an annual climatology) and repeated over the
whole simulation. This type should generally not be constructed directly but rather via the `InputSource`
constructor interface; static (time-invariant) rasters instead yield a plain `FieldInputSource`.
"""
struct RasterInputSource{NF, name, VL, TT, IM <: AbstractVector{Int}, RS <: AbstractRaster{NF}, SG, UT, EX <: Extrapolation} <: InputSource{NF, name}
    "Variable location: spatial dimensions and the model domain the variable lives on"
    loc::VL

    "Physical units"
    units::UT

    "Index map for the grid mask"
    idxmap::IM

    "Reference time"
    reftime::TT

    "Data raster"
    raster::RS

    "Source grid for raster"
    source_grid::SG

    "Extrapolation mode; only `Cyclical` and `Clamp` are supported"
    extrapolation::EX
end

"""
    InputSource(grid::ColumnRingGrid, raster::AbstractRaster; name = raster.name, units, reftime, cycle)

Create an `InputSource` for the given `Raster` data and `grid`. A time-invariant raster is regridded once
and returned as a plain `Terrarium.FieldInputSource`; a time-varying raster is returned as a 
`RasterInputSource` that linearly interpolates in time, optionally cycling the time axis (`cycle = true`).
The `name` can either be a plain `Symbol` or a namespaced path; see [`Terrarium.varpath`](@ref).
"""
function Terrarium.InputSource(
        grid::ColumnRingGrid{NF},
        raster::AbstractRaster{NF};
        source_grid = grid.rings,
        name = raster.name,
        domain::Terrarium.Optional{Terrarium.VarDomain} = nothing,
        units = NoUnits,
        timedim = Ti,
        reftime = nothing,
        cycle = false
    ) where {NF}
    raster = reconcile_latitudes(raster, grid.rings)
    extrapolation = cycle ? Cyclical() : Clamp()
    if timedim != Ti
        td = dims(raster, timedim)
        raster = set(raster, td => Ti(td.val))
    end
    return RasterInputSource(grid, source_grid, raster, dims(raster, Ti), name, domain, units, reftime, extrapolation)
end

# Land grids delegate to the discretization of their ground domain, which carries the ring grid.
Terrarium.InputSource(grid::Terrarium.AbstractLandGrid, raster::AbstractRaster; kwargs...) = Terrarium.InputSource(ground_domain(grid), raster; kwargs...)

# Static (time-invariant) rasters reduce to a plain `FieldInputSource`, regridded only once into a `RingGrids.Field`.
function RasterInputSource(
        grid::ColumnRingGrid{NF},
        source_grid::RingGrids.AbstractGrid,
        raster::AbstractRaster{NF},
        ::Nothing,
        name, domain, units, args...
    ) where {NF}
    idxmap = findall(Array(grid.mask))
    field = similar(grid.mask, NF)
    fill!(parent(field), zero(NF))
    if grid.rings == source_grid
        @inbounds parent(field)[idxmap] .= view(raster, idxmap)
    else
        source_field = RingGrids.Field(Array(raster), source_grid)
        RingGrids.interpolate!(field, source_field)
    end
    # Construct and return a FieldInputSource from the RingGrids `field`
    return Terrarium.InputSource(grid, field; name, domain, units)
end

# Time-varying rasters retain the custom source with lazy time interpolation (and optional cycling).
function RasterInputSource(
        grid::ColumnRingGrid{NF},
        source_grid::RingGrids.AbstractGrid,
        raster::AbstractRaster{NF},
        ::TimeDim,
        name, domain, units, reftime, extrapolation
    ) where {NF}
    # get indices from grid mask
    idxmap = on_architecture(architecture(grid), findall(Array(grid.mask)))
    # infer the VarDims and subsequently the Field location from the data dimensions
    rloc = Terrarium.VarLocation(Terrarium.vardims(raster), domain)
    # infer reference time
    reftime = default_reftime(raster, reftime)
    raster = Rasters.setdims(raster, convert_time_axis(dims(raster, Ti)))
    path = Terrarium.varpath(name)
    return RasterInputSource{NF, path, typeof(rloc), typeof(reftime), typeof(idxmap), typeof(raster), typeof(source_grid), typeof(units), typeof(extrapolation)}(
        rloc, units, idxmap, reftime, raster, source_grid, extrapolation
    )
end

Terrarium.variables(source::RasterInputSource) = Terrarium.with_scope(
    Base.front(Terrarium.varpath(source)),
    Terrarium.input(Terrarium.varname(source), source.loc; units = source.units)
) |> tuple

# Infer VarDims based on the axes defined in the Raster
Terrarium.vardims(A::AbstractDimArray) = Terrarium.vardims(dims(A, X, Y, Z)...)
Terrarium.vardims(::X, ::Y) = XY()
Terrarium.vardims(::X, ::Y, ::Z) = XYZ()

# A time-varying source is initialized by writing its value at the start time.
Terrarium.initialize!(inputs, grid, clock, fields, source::RasterInputSource) =
    Terrarium.update_inputs!(inputs, grid, clock, fields, source)

function Terrarium.update_inputs!(inputs, grid, clock, fields, source::RasterInputSource)
    name = Terrarium.varname(source)
    if hasproperty(inputs, name)
        field = getproperty(inputs, name)
        update_from_raster!(field, grid, clock, source)
    end
    return nothing
end

function update_from_raster!(field, grid, clock, source::RasterInputSource{NF}) where {NF}
    raster = source.raster
    idxmap = source.idxmap
    timedim = dims(raster, Ti)
    N = length(timedim)
    t1 = timestamp(NF, source.reftime, first(timedim.val))
    tN = timestamp(NF, source.reftime, last(timedim.val))
    ti = timestamp(NF, source.reftime, clock.time)
    Δt₀ = (tN - t1) / (N - 1)
    period = tN - t1 + Δt₀ # approximate for non-equidistant spacing
    # search for indices between which t lies, converting everything to relative time in seconds;
    # note that we use clock.time again here because searchsorted applies by to the second argument
    indexes = searchsorted(timedim.val, clock.time, by = t -> t1 + mod(timestamp(NF, source.reftime, t) - t1, period))
    lower, upper = last(indexes), first(indexes)
    @inbounds if t1 < ti < tN || source.extrapolation isa Cyclical
        lower = lower < 1 ? N : lower
        upper = upper > N ? 1 : upper
        x1 = interpolate_to_grid(grid, source.source_grid, raster[Ti(lower)])[idxmap]
        x2 = interpolate_to_grid(grid, source.source_grid, raster[Ti(upper)])[idxmap]
        t1 = timestamp(NF, source.reftime, timedim[lower])
        t2 = timestamp(NF, source.reftime, timedim[upper])
        # Linear interpolation between points
        Δt = t2 > t1 ? Terrarium.convert_dt(NF, t2 - t1) : Δt₀
        ϵ = Terrarium.convert_dt(NF, ti - t1)
        x_interp = x1 + ϵ * (x2 - x1) / Δt
        return set!(field, x_interp)
    else
        # Flat extrapolation (corresponding to Clamp)
        x_end = interpolate_to_grid(grid, source.source_grid, raster[Ti(min(upper, N))])
        return set!(field, x_end[idxmap])
    end
end

# Land grids delegate to the discretization of their ground domain, which carries the ring grid.
interpolate_to_grid(grid::Terrarium.AbstractLandGrid, source_grid::RingGrids.AbstractGrid, data::Raster) = interpolate_to_grid(ground_domain(grid), source_grid, data)

function interpolate_to_grid(grid::ColumnRingGrid, source_grid::RingGrids.AbstractGrid, data::Raster)
    arch = architecture(grid)
    data_device = on_architecture(arch, reshape(data.data, :)) # NOTE: only works if underlying data is an in-memory 2D array
    field = RingGrids.Field(data_device, on_architecture(arch, source_grid))
    if grid.rings === source_grid
        return field
    else
        return RingGrids.interpolate(grid.rings, field)
    end
end

to_seconds(time::Month) = to_seconds(Day(30) * Dates.value(time))
to_seconds(time::Day) = to_seconds(Hour(24) * Dates.value(time))
to_seconds(time::Hour) = to_seconds(Second(3600) * Dates.value(time))
to_seconds(time::Second) = Dates.value(time)
to_seconds(time) = time

convert_time_axis(timedim::Ti) = timedim
function convert_time_axis(timedim::Ti{<:Rasters.Lookup{<:Integer}})
    reftime = default_reftime(timedim)
    return Ti(map(t -> typeof(reftime)(t) - reftime, timedim.val))
end

# Use specified reference time, if provided (i.e. not nothing)
default_reftime(data::AbstractRaster, reftime) = reftime
# Otherwise, take first value from time dimension
default_reftime(data::AbstractRaster, ::Nothing) = default_reftime(dims(data, Ti))
default_reftime(timedim::Ti) = first(timedim)
function default_reftime(timedim::Ti{<:Rasters.Lookup{<:Integer}})
    t_min = minimum(timedim.val)
    return if length(timedim) == 12
        Month(t_min)
    elseif length(timedim) ∈ (365, 366)
        Day(t_min)
    elseif length(timedim) == 24
        Hour(t_min)
    else
        Second(t_min)
    end
end

# Adapt rules for Rasters
on_architecture(to, raster::AbstractRaster) = rebuild(
    raster,
    data = on_architecture(to, raster.data),
    dims = map(d -> rebuild(d, val = on_architecture(to, d.val)), dims(raster))
)

# Drop latitudes at the poles for RingGrids
reconcile_latitudes(raster::AbstractRaster, ::RingGrids.AbstractGrid) = @view(raster[Y(Where(lat -> -90 < lat < 90))])
reconcile_latitudes(raster::AbstractRaster, grid) = raster

end
