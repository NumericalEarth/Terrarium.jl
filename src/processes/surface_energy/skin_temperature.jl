# Prescribed skin temperature

"""
    $TYPEDEF

Simple scheme for prescribed skin temperatures from input variables.

Properties:
$FIELDS
"""
@kwdef struct PrescribedSkinTemperature{NF} <: AbstractSkinTemperature{NF}
    "Assumed thermal conductivity at the surface [W m⁻¹ K⁻¹]"
    κₛ::NF = 2.0
end

PrescribedSkinTemperature(::Type{NF}; kwargs...) where {NF} = PrescribedSkinTemperature{NF}(; kwargs...)

## Top-level interface methods

variables(::PrescribedSkinTemperature) = (
    auxiliary(:ground_heat_flux, XY(), units = u"W/m^2", desc = "Ground heat flux"),
    input(:skin_temperature, XY(), units = u"°C", desc = "Longwave emission temperature of the land surface in °C"),
)

@inline compute_auxiliary!(state, grid, ::PrescribedSkinTemperature, args...) = nothing

@inline compute_tendencies!(state, grid, ::PrescribedSkinTemperature, args...) = nothing

## Kernel functions

@propagate_inbounds compute_skin_temperature(i, j, grid, fields, skinT::PrescribedSkinTemperature) = fields.skin_temperature[i, j]

# Implicit skin temperature

"""
    $TYPEDEF

Scheme for an implicit skin temperature ``T_s`` satisfying:
```math
R_{\\text{net}}(T_s) = H_s(T_s) + H_l(T_s) + G(T_s, T_g)
```
where ``R_{\\text{net}}`` is the net radiation budget, ``H_s`` is the sensible heat flux, ``H_l`` is the latent
heat flux from sublimation and evapotranspiration, ``G`` is the ground heat flux, and ``T_g`` is the ground
temperature, or temperature of the uppermost subsurface (soil or snow) layer.

Properties:
$FIELDS
"""
@kwdef struct ImplicitSkinTemperature{NF} <: AbstractSkinTemperature{NF}
    "Assumed thermal conductivity at the surface [W m⁻¹ K⁻¹]"
    κₛ::NF = 2.0
end

ImplicitSkinTemperature(::Type{NF}; kwargs...) where {NF} = ImplicitSkinTemperature{NF}(; kwargs...)

"""
    $TYPEDSIGNATURES

Compute the implicit update of the skin temperature from the given ground surface temperature
`Tg`, ground heat flux `G`, and distance `Δz`.
"""
@inline function compute_skin_temperature(skinT::ImplicitSkinTemperature, Tg, G, Δz)
    # Compute new skin temperature T₀ by setting G equal to the half-cell heat flux
    # TODO: use thermal conductivity from the soil and/or canopy?
    κₛ = skinT.κₛ
    Ts = Tg - G * Δz / (2 * κₛ)
    return Ts
end

"""
    $TYPEDSIGNATURES

Compute the ground heat flux as the residual of the net radiation `R_net` and the
sensible `H_s` and latent `H_l` heat flux.
"""
@inline function compute_ground_heat_flux(::AbstractSkinTemperature, R_net, H_s, H_l)
    # Compute ground heat flux as the residual of R_net and turbulent fluxes
    G = R_net + H_s + H_l
    return G
end

## Top-level interface methods

variables(::ImplicitSkinTemperature) = (
    prognostic(:skin_temperature, XY(), units = u"°C", desc = "Longwave emission temperature of the land surface in °C"),
    auxiliary(:ground_heat_flux, XY(), units = u"W/m^2", desc = "Ground heat flux"),
    input(:ground_temperature, XY(), units = u"°C", desc = "Temperature of the uppermost ground or soil grid cell in °C"),
)

""" $TYPEDSIGNATURES """
@inline function compute_auxiliary!(
        state, grid,
        skinT::ImplicitSkinTemperature,
        seb::AbstractSurfaceEnergyBalance,
        args...
    )
    compute_ground_heat_flux!(state, grid, skinT, seb)
    return nothing
end

"""
    $TYPEDSIGNATURES

Update `skin_temperature` according to the current state of `ground_heat_flux`.
"""
function update_skin_temperature!(state, grid, skinT::ImplicitSkinTemperature)
    out = prognostic_fields(state, skinT)
    fields = get_fields(state, skinT; except = out)
    launch!(grid, XY, compute_skin_temperature_kernel!, out, fields, skinT)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute `ground_heat_flux` as the residual ``R_\\text{net} - H_s - H_l``.
"""
function compute_ground_heat_flux!(
        state, grid,
        skinT::AbstractSkinTemperature,
        seb::AbstractSurfaceEnergyBalance
    )
    rad = get_radiative_fluxes(seb)
    tur = get_turbulent_fluxes(seb)
    out = auxiliary_fields(state, skinT)
    fields = get_fields(state, skinT, rad, tur; except = out)
    launch!(grid, XY, compute_ground_heat_flux_kernel!, out, fields, skinT, rad, tur)
    return nothing
end

## Kernel functions

"""
    $TYPEDSIGNATURES

Estimate the skin temperature from the current `ground_heat_flux` at grid cell `i, j`.
"""
@propagate_inbounds function compute_skin_temperature(
        i, j, grid, fields,
        skinT::ImplicitSkinTemperature
    )
    # Get thickness of topmost soil/ground grid cell
    field_grid = get_field_grid(grid)
    Δz₁ = Δzᵃᵃᶜ(i, j, field_grid.Nz, field_grid)
    # Get ground heat flux
    G = fields.ground_heat_flux[i, j]
    # Get ground temperature
    Tg = fields.ground_temperature[i, j]
    Ts = compute_skin_temperature(skinT, Tg, G, Δz₁)
    return Ts
end

"""
    $TYPEDSIGNATURES

Compute the ground heat flux from the surface net radiation and sensible/latent heat flux at grid cell `i, j`.
"""
@propagate_inbounds function compute_ground_heat_flux(
        i, j, grid, fields,
        skinT::AbstractSkinTemperature,
        ::AbstractRadiativeFluxes,
        ::AbstractTurbulentFluxes
    )
    # Get individual flux terms
    R_net = fields.surface_net_radiation[i, j]
    H_s = fields.sensible_heat_flux[i, j]
    H_l = fields.latent_heat_flux[i, j]
    # Compute ground heat flux
    G = compute_ground_heat_flux(skinT, R_net, H_s, H_l)
    return G
end

# Kernels

@kernel function compute_skin_temperature_kernel!(out, grid, fields, skinT::ImplicitSkinTemperature, args...)
    i, j = @index(Global, NTuple)

    # Compute and store skin temperature
    out.skin_temperature[i, j, 1] = compute_skin_temperature(i, j, grid, fields, skinT, args...)
end

@kernel function compute_ground_heat_flux_kernel!(out, grid, fields, skinT::AbstractSkinTemperature, args...)
    i, j = @index(Global, NTuple)

    # Compute and store ground heat flux
    out.ground_heat_flux[i, j, 1] = compute_ground_heat_flux(i, j, grid, fields, skinT, args...)
end
