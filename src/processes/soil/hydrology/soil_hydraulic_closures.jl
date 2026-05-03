# Saturation ↔ Hydraulic (pressure) head

"""
    $TYPEDEF

Represents a closure relating saturation of water/ice in soil pores to a corresponding
pressure (or hydraulic) head. Note that here "pressure head" is defined to be synonymous
with hydraulic head, i.e. including all both elevation and hydrostatic pressure contributions.
This relation is typically described by soil property-dependent *soil-water retention curve* (SWRC)
which is here defined in implementations of [`AbstractSoilHydraulics`](@ref).
"""
@kwdef struct SoilSaturationPressureClosure <: AbstractSoilWaterClosure end

variables(::SoilSaturationPressureClosure) = (
    auxiliary(:pressure_head, XYZ(), units = u"m", desc = "Total hydraulic pressure head in m water displaced at standard pressure"),
)

"""
    $TYPEDSIGNATURES

Computes `pressure_head` ``Ψ = ψm + ψz + ψh`` from the current `saturation_water_ice` state.
"""
function closure!(
        state, grid,
        closure::SoilSaturationPressureClosure,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        args...
    ) where {NF}
    # apply saturation correction
    adjust_saturation_profile!(state, grid, hydrology)
    # update water table
    compute_water_table!(state, grid, hydrology)
    # determine pressure head from saturation
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    out = (pressure_head = state.pressure_head,)
    fields = get_fields(state, hydrology, bgc; except = out)
    launch!(
        grid, XYZ, saturation_to_pressure_kernel!,
        out, fields, closure, hydrology, strat, bgc
    )
    return nothing
end

"""
    $TYPEDSIGNATURES

Computes `saturation_water_ice` from the current `pressure_head` state.
"""
function invclosure!(
        state, grid,
        closure::SoilSaturationPressureClosure,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        args...
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    out = (saturation_water_ice = state.saturation_water_ice,)
    fields = get_fields(state, hydrology, bgc; except = out)
    # determine saturation from pressure
    launch!(
        grid, XYZ, pressure_to_saturation_kernel!,
        out, fields, closure, hydrology, strat, bgc
    )
    # apply saturation correctionh
    adjust_saturation_profile!(state, grid, hydrology)
    # update water table
    compute_water_table!(state, grid, hydrology)
    return nothing
end

@propagate_inbounds function pressure_to_saturation!(
        saturation_water_ice, i, j, k, grid, fields,
        closure::SoilSaturationPressureClosure,
        hydrology::SoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    fgrid = get_field_grid(grid)
    ψ = fields.pressure_head[i, j, k] # assumed given
    # get the elevation (z-coord) of k'th layer and reference (surface)
    z = znode(i, j, k, fgrid, Center(), Center(), Center())
    # TODO: we need a more user friendly interface for this...
    z_ref = znode(i, j, fgrid.Nz + 1, fgrid, Center(), Center(), Face())
    # elevation pressure head
    ψz = z - z_ref
    # compute hydrostatic pressure head assuming impermeable lower boundary
    # TODO: relax this assumption in the future?
    z₀ = fields.water_table[i, j, 1]
    ψh = max(0, z₀ - z)
    # remove hydrostatic and elevation components
    ψm = ψ - ψh - ψz
    swrc = get_swrc(hydrology)
    por = porosity(i, j, k, grid, fields, strat, bgc)
    vol_water_ice_content = swrc(ψm; θsat = por)
    saturation_water_ice[i, j, k] = vol_water_ice_content / por
    return nothing
end

@propagate_inbounds function saturation_to_pressure!(
        pressure_head, i, j, k, grid, fields,
        closure::SoilSaturationPressureClosure,
        hydrology::SoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    fgrid = get_field_grid(grid)
    sat = fields.saturation_water_ice[i, j, k] # assumed given
    # get the elevation (z-coord) of k'th layer and reference (surface)
    z = znode(i, j, k, fgrid, Center(), Center(), Center())
    z_ref = znode(i, j, fgrid.Nz + 1, fgrid, Center(), Center(), Face())
    # get inverse of SWRC
    inv_swrc = inv(get_swrc(hydrology))
    por = porosity(i, j, k, grid, fields, strat, bgc)
    sat_res = residual_saturation(get_hydraulic_properties(hydrology))
    # compute matric pressure head
    ψm = inv_swrc(max(sat * por, sat_res); θsat = por)
    @assert isfinite(ψm) "NaN/infinite matric potential ψm = $ψm with sat=$sat por=$por resid=$sat_res"
    # compute elevation pressure head
    ψz = z - z_ref
    # compute hydrostatic pressure head assuming impermeable lower boundary
    # TODO: can we generalize this for arbitrary lower boundaries?
    z₀ = fields.water_table[i, j, 1]
    ψh = max(0, z₀ - z)
    # compute total pressure head as sum of ψh + ψm + ψz
    # note that ψh and ψz will cancel out in the saturated zone
    pressure_head[i, j, k] = ψh + ψm + ψz
    return nothing
end

# Kernels

@kernel inbounds = true function pressure_to_saturation_kernel!(
        out, grid, fields,
        closure::SoilSaturationPressureClosure,
        args...
    )
    i, j, k = @index(Global, NTuple)
    pressure_to_saturation!(out.saturation_water_ice, i, j, k, grid, fields, closure, args...)
end

@kernel inbounds = true function saturation_to_pressure_kernel!(
        out, grid, fields,
        closure::SoilSaturationPressureClosure,
        args...
    )
    i, j, k = @index(Global, NTuple)
    saturation_to_pressure!(out.pressure_head, i, j, k, grid, fields, closure, args...)
end

# Default debug hooks
@inline debughook!(::typeof(pressure_to_saturation_kernel!), out, args...) = checkfinite!(out)
@inline debughook!(::typeof(saturation_to_pressure_kernel!), out, args...) = checkfinite!(out)
