# Diagnosed ground heat flux

"""
    $TYPEDEF

Ground heat flux `G` diagnosed from the rest of the surface energy balance. With all fluxes positive
upward (aligned with `+z`), the energy arriving at the skin from below must balance the radiative and
turbulent losses above, so the atmosphere-side *demand* is the residual `G = R_net + H_s + H_l`.

How that demand is realized depends on the skin temperature scheme: [`PrescribedSkinTemperature`](@ref)
has no conduction target of its own and stores the demand directly, whereas
[`ImplicitSkinTemperature`](@ref) stores the explicit conductive flux `2κ_g (T_g - T_s) / Δz_g`
evaluated at the current skin temperature, which coincides with the demand only at convergence.
"""
struct DiagnosedGroundHeatFlux{NF} <: AbstractGroundHeatFlux{NF} end

DiagnosedGroundHeatFlux(::Type{NF}) where {NF} = DiagnosedGroundHeatFlux{NF}()

variables(::DiagnosedGroundHeatFlux) = (
    auxiliary(:ground_heat_flux, Ground(Top()), units = u"W/m^2", desc = "Ground heat flux"),
)

# Prescribed ground heat flux

"""
    $TYPEDEF

Ground heat flux `G` supplied externally as an input variable, for example assembled by a coupler from
the atmosphere-side surface energy budget. Nothing is computed: `compute_ground_heat_flux!` is a no-op,
so whatever was written into the `ground_heat_flux` field survives to the ground (soil) top boundary
condition. The same positive-upward convention applies as for [`DiagnosedGroundHeatFlux`](@ref).

Note that with all four surface fluxes prescribed, the residual identity `G = R_net + H_s + H_l` is
*not* enforced by construction; it is the caller's responsibility to supply a consistent set.
"""
struct PrescribedGroundHeatFlux{NF} <: AbstractGroundHeatFlux{NF} end

PrescribedGroundHeatFlux(::Type{NF}) where {NF} = PrescribedGroundHeatFlux{NF}()

variables(::PrescribedGroundHeatFlux) = (
    input(:ground_heat_flux, Ground(Top()), units = u"W/m^2", desc = "Ground heat flux"),
)

## Top-level interface methods

""" $TYPEDSIGNATURES """
@inline function compute_auxiliary!(
        state, grid,
        ghf::AbstractGroundHeatFlux,
        seb::AbstractSurfaceEnergyBalance,
        args...
    )
    compute_ground_heat_flux!(state, grid, ghf, get_skin_temperature(seb), seb)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute and store `ground_heat_flux` on `grid`, dispatching on `ghf` and `skinT` to the type-specific
kernel function. For [`PrescribedGroundHeatFlux`](@ref) this is a no-op, since the field is an input.
"""
function compute_ground_heat_flux!(
        state, grid,
        ghf::AbstractGroundHeatFlux,
        skinT::AbstractSkinTemperature,
        seb::AbstractSurfaceEnergyBalance
    )
    out = auxiliary_fields(state, ghf)
    fields = get_fields(state, seb; except = out)
    launch!(grid, XY, compute_ground_heat_flux_kernel!, out, fields, ghf, skinT, seb)
    return nothing
end

"""
    $TYPEDSIGNATURES

The ground heat flux is a prescribed input variable, so there is nothing to diagnose.
"""
@inline compute_ground_heat_flux!(state, grid, ::PrescribedGroundHeatFlux, args...) = nothing

## Kernel functions

"""
    $TYPEDSIGNATURES

Compute the ground heat flux *demand* at grid cell `i, j`: the flux implied by the radiative budget and
the turbulent fluxes, `G = R_net + H_s + H_l`. This is the quantity the implicit skin temperature solve
inverts its conduction relation against.
"""
@propagate_inbounds function compute_ground_heat_flux_demand(i, j, grid, fields, ghf::DiagnosedGroundHeatFlux)
    # Get individual flux terms
    R_net = fields.surface_net_radiation[i, j, end]
    H_s = fields.sensible_heat_flux[i, j, end]
    H_l = fields.latent_heat_flux[i, j, end]
    # Compute ground heat flux
    G₀ = compute_ground_heat_flux_demand(ghf, R_net, H_s, H_l)
    return G₀
end

"""
    $TYPEDSIGNATURES

When the ground heat flux is prescribed, the demand *is* the prescribed flux; the surface energy budget
is not re-closed from the radiative and turbulent terms.
"""
@propagate_inbounds compute_ground_heat_flux_demand(i, j, grid, fields, ::PrescribedGroundHeatFlux) = fields.ground_heat_flux[i, j, end]

"""
    $TYPEDSIGNATURES

Compute the residual ground heat flux that would close the surface energy balance. With all fluxes
positive upward (aligned with `+z`), the energy arriving at the skin from below must balance
the radiative and turbulent losses above, so `G = R_net + H_s + H_l`.
"""
@inline function compute_ground_heat_flux_demand(::DiagnosedGroundHeatFlux, R_net, H_s, H_l)
    G₀ = R_net + H_s + H_l
    return G₀
end

"""
    $TYPEDSIGNATURES

For `PrescribedSkinTemperature`, set the ground heat flux directly to the demand, i.e. `G₀ = R_net + H_s + H_l`.
"""
@propagate_inbounds function compute_ground_heat_flux(
        i, j, grid, fields,
        ghf::DiagnosedGroundHeatFlux,
        ::PrescribedSkinTemperature,
        ::AbstractSurfaceEnergyBalance
    )
    return compute_ground_heat_flux_demand(i, j, grid, fields, ghf)
end

"""
    $TYPEDSIGNATURES

Compute the conductive ground heat flux from the current `skin_temperature` and `ground_temperature`.
"""
@propagate_inbounds function compute_ground_heat_flux(
        i, j, grid, fields,
        ::DiagnosedGroundHeatFlux,
        skinT::ImplicitSkinTemperature,
        ::AbstractSurfaceEnergyBalance
    )
    Tg, κg, Δzg = ground_thermal_interface(i, j, grid, fields, skinT)
    Ts = fields.skin_temperature[i, j, end]
    return 2 * κg * (Tg - Ts) / Δzg
end

"""
    $TYPEDSIGNATURES

Per-cell mutating variant used by the fused surface-energy-balance kernel: store the ground heat flux
into the auxiliary output field `out`.
"""
@propagate_inbounds function compute_ground_heat_flux!(
        out, i, j, grid, fields,
        ghf::DiagnosedGroundHeatFlux,
        skinT::AbstractSkinTemperature,
        seb::AbstractSurfaceEnergyBalance
    )
    out.ground_heat_flux[i, j, end] = compute_ground_heat_flux(i, j, grid, fields, ghf, skinT, seb)
    return nothing
end

"""
    $TYPEDSIGNATURES

The prescribed ground heat flux is an input field, so the fused surface-energy-balance kernel leaves it
untouched.
"""
@propagate_inbounds compute_ground_heat_flux!(out, i, j, grid, fields, ::PrescribedGroundHeatFlux, ::AbstractSkinTemperature, ::AbstractSurfaceEnergyBalance) = nothing

# Kernels

@kernel function compute_ground_heat_flux_kernel!(out, grid, fields, ghf::AbstractGroundHeatFlux, args...)
    i, j = @index(Global, NTuple)
    # Forward to mutating compute_ground_heat_flux!
    compute_ground_heat_flux!(out, i, j, grid, fields, ghf, args...)
end
