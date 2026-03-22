# Prescribed radiative fluxes

"""
    $TYPEDEF

Represents the simplest scheme for the radiative budget where outgoing shortwave and longwave
radiation are given as input variables. Net radiation is diagnosed by summing all radiative fluxes:

```math
R_{\\text{net}} = S_{\\uparrow} - S_{\\downarrow} + L_{\\uparrow} - L_{\\downarrow}
```
"""
struct PrescribedRadiativeFluxes{NF} <: AbstractRadiativeFluxes{NF} end

PrescribedRadiativeFluxes(::Type{NF}) where {NF} = PrescribedRadiativeFluxes{NF}()

## Process methods

variables(::PrescribedRadiativeFluxes) = (
    input(:surface_shortwave_up, XY(), units = u"W/m^2", desc = "Outgoing (upwelling) shortwave radiation"),
    input(:surface_longwave_up, XY(), units = u"W/m^2", desc = "Outgoing (upwelling) longwave radiation"),
    auxiliary(:surface_net_radiation, XY(), units = u"W/m^2", desc = "Net outgoing (positive up) radiation"),
)

function compute_auxiliary!(
        state, grid,
        rad::PrescribedRadiativeFluxes,
        seb::AbstractSurfaceEnergyBalance,
        atmos::AbstractAtmosphere,
        args...
    )
    out = auxiliary_fields(state, rad)
    fields = get_fields(state, rad, atmos; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, rad, atmos)
    return nothing
end

## Kernel functions

"""
    $TYPEDSIGNATURES

Compute upwelling shortwave and longwave radiation at a grid point.
"""
@propagate_inbounds function compute_surface_upwelling_radiation(i, j, grid, fields, rad::PrescribedRadiativeFluxes, args...)
    surface_shortwave_up = fields.surface_shortwave_up[i, j]
    surface_longwave_up = fields.surface_longwave_up[i, j]
    return (; surface_shortwave_up, surface_longwave_up)
end

"""
    $TYPEDSIGNATURES

Compute net radiation and store in auxiliary fields at a grid point.
"""
@propagate_inbounds function compute_radiative_fluxes!(
        out, i, j, grid, fields,
        rad::PrescribedRadiativeFluxes,
        atmos::AbstractAtmosphere,
        args...
    )
    # Compute and store net radiation
    out.surface_net_radiation[i, j, 1] = compute_surface_net_radiation(i, j, grid, fields, rad, atmos)
    return out
end

# Diagnosed radiative fluxes

"""
    $TYPEDEF

Computes outgoing shortwave and longwave radiation according to separately specified
schemes for the albedo, skin temperature, and atmospheric inputs.
"""
struct DiagnosedRadiativeFluxes{NF} <: AbstractRadiativeFluxes{NF} end

DiagnosedRadiativeFluxes(::Type{NF}) where {NF} = DiagnosedRadiativeFluxes{NF}()

"""
    compute_shortwave_up(::DiagnosedRadiativeFluxes, surface_shortwave_down, α)

Compute outgoing shortwave radiation from the incoming `surface_shortwave_down` and albedo `α`.
"""
@inline function compute_shortwave_up(::DiagnosedRadiativeFluxes, surface_shortwave_down, α)
    surface_shortwave_up = α * surface_shortwave_down
    return surface_shortwave_up
end

"""
    compute_longwave_up(::DiagnosedRadiativeFluxes, constants::PhysicalConstants, surface_longwave_down, Ts, ϵ)

Compute outgoing longwave radiation from incoming `surface_longwave_down`, surface temperature `Ts`, and emissivity `ϵ`.
"""
@inline function compute_longwave_up(::DiagnosedRadiativeFluxes, constants::PhysicalConstants, surface_longwave_down, Ts, ϵ)
    T = celsius_to_kelvin(constants, Ts)
    # outgoing LW radiation is the sum of the radiant emittance and the residual incoming radiation
    surface_longwave_up = stefan_boltzmann(constants, T, ϵ) + (1 - ϵ) * surface_longwave_down
    return surface_longwave_up
end

## Process methods

variables(::DiagnosedRadiativeFluxes) = (
    auxiliary(:surface_shortwave_up, XY(), units = u"W/m^2", desc = "Outgoing (upwelling) shortwave radiation"),
    auxiliary(:surface_longwave_up, XY(), units = u"W/m^2", desc = "Outgoing (upwelling) longwave radiation"),
    auxiliary(:surface_net_radiation, XY(), units = u"W/m^2", desc = "Net radiation budget"),
)

function compute_auxiliary!(
        state, grid,
        rad::DiagnosedRadiativeFluxes,
        seb::AbstractSurfaceEnergyBalance,
        atmos::AbstractAtmosphere,
        consts::PhysicalConstants,
        args...
    )
    (; skin_temperature, albedo) = seb
    out = auxiliary_fields(state, rad)
    fields = get_fields(state, rad, skin_temperature, albedo, atmos; except = out)
    launch!(
        grid, XY, compute_auxiliary_kernel!,
        out, fields, rad, skin_temperature, albedo, atmos, consts
    )
    return nothing
end

## Kernel functions

@propagate_inbounds function compute_surface_upwelling_radiation(
        i, j, grid, fields,
        rad::DiagnosedRadiativeFluxes,
        skinT::AbstractSkinTemperature,
        abd::AbstractAlbedo,
        atmos::AbstractAtmosphere,
        consts::PhysicalConstants
    )
    # Get inputs
    SW_down = shortwave_down(i, j, grid, fields, atmos)
    LW_down = longwave_down(i, j, grid, fields, atmos)
    Tsurf = skin_temperature(i, j, grid, fields, skinT)
    α = albedo(i, j, grid, fields, abd)
    ϵ = emissivity(i, j, grid, fields, abd)

    # Compute fluxes
    surface_shortwave_up = compute_shortwave_up(rad, SW_down, α)
    surface_longwave_up = compute_longwave_up(rad, consts, LW_down, Tsurf, ϵ)

    # Return fluxes as named tuple
    return (; surface_shortwave_up, surface_longwave_up)
end

@propagate_inbounds function compute_radiative_fluxes!(
        out, i, j, grid, fields,
        rad::DiagnosedRadiativeFluxes,
        skinT::AbstractSkinTemperature,
        abd::AbstractAlbedo,
        atmos::AbstractAtmosphere,
        consts::PhysicalConstants
    )
    # Unpack outputs
    (; surface_shortwave_up, surface_longwave_up) = out

    # Compute and store outgoing fluxes
    outgoing_fluxes = compute_surface_upwelling_radiation(i, j, grid, fields, rad, skinT, abd, atmos, consts)
    surface_shortwave_up[i, j, 1] = outgoing_fluxes.surface_shortwave_up
    surface_longwave_up[i, j, 1] = outgoing_fluxes.surface_longwave_up

    # Compute and store net radiation
    fields = merge(fields, (; surface_shortwave_up, surface_longwave_up))
    out.surface_net_radiation[i, j, 1] = compute_surface_net_radiation(i, j, grid, fields, rad, atmos)
    return out
end

@propagate_inbounds function compute_surface_net_radiation(
        i, j, grid, fields,
        rad::AbstractRadiativeFluxes,
        atmos::AbstractAtmosphere
    )
    # Get inputs
    SW_down = shortwave_down(i, j, grid, fields, atmos)
    SW_up = shortwave_up(i, j, grid, fields, rad)
    LW_down = longwave_down(i, j, grid, fields, atmos)
    LW_up = longwave_up(i, j, grid, fields, rad)

    # Compute and return net radiation
    surface_net_radiation = compute_surface_net_radiation(rad, SW_up, SW_down, LW_up, LW_down)
    return surface_net_radiation
end

# Common implementations

"""
    $TYPEDEF

Compute the net radiation budget given incoming and outgoing shortwave and longwave radiation.
"""
@inline function compute_surface_net_radiation(
        ::AbstractRadiativeFluxes,
        SW_up,
        SW_down,
        LW_up,
        LW_down
    )
    # Sum up radiation fluxes; note that by convention fluxes are positive upward
    rad_net = SW_up - SW_down + LW_up - LW_down
    return rad_net
end

## Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, rad::AbstractRadiativeFluxes, args...)
    i, j = @index(Global, NTuple)
    compute_radiative_fluxes!(out, i, j, grid, fields, rad, args...)
end
