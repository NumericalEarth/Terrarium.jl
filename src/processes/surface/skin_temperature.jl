# Prescribed skin temperature

"""
    $TYPEDEF

Simple scheme for prescribed skin temperatures from input variables.

Properties:
$FIELDS
"""
@parameterized @kwdef struct PrescribedSkinTemperature{NF} <: AbstractSkinTemperature{NF}
    "Assumed thermal conductivity at the surface"
    @param κₛ::NF = 1.0 (units = u"W/m/K", bounds = Positive)
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

# The skin temperature is prescribed (an input field), so there is no implicit solve: the fused
# surface-energy-balance kernel just evaluates the fluxes from the prescribed skin temperature.
@propagate_inbounds solve_skin_temperature!(out, i, j, grid, fields, ::PrescribedSkinTemperature, seb, snow, seb_args...) = nothing

# Implicit skin temperature

"""
    $TYPEDEF

Scheme for an implicit skin temperature ``T_s`` satisfying:
```math
R_{\\text{net}}(T_s) + H_s(T_s) + H_l(T_s) - (1 - f_{\\text{snow}})\\, G(T_s, T_g) - f_{\\text{snow}}\\, S(T_s, T_{\\text{snow}}) = 0
```
where ``R_{\\text{net}}`` is the net radiation budget, ``H_s`` is the sensible heat flux, ``H_l`` is the latent
heat flux from sublimation and evapotranspiration, ``G`` is the conductive flux from the skin into the
snow-free ground (``T_g`` its temperature), ``S`` is the conductive flux from the skin into the top of the
snowpack over the snow-covered fraction (``T_{\\text{snow}}`` its temperature), and ``f_{\\text{snow}}``
is the snow-covered area fraction (``f_{\\text{snow}} = 0`` and ``S`` absent without snow). ``G`` and
``S`` are each computed from their own *unblended* conduction target (see
[`ground_thermal_interface`](@ref) and [`snow_thermal_interface`](@ref)).

Properties:
$FIELDS
"""
@parameterized struct ImplicitSkinTemperature{NF, Solver} <: AbstractSkinTemperature{NF}
    "Assumed thermal conductivity at the surface"
    @param κₛ::NF (units = u"W/m/K", bounds = Positive)

    "Numerical solver for the implicit skin temperature"
    solver::Solver
end

ImplicitSkinTemperature(::Type{NF}; κₛ::NF = NF(1.0), solver = default_skin_temperature_solver(NF)) where {NF} = ImplicitSkinTemperature{NF, typeof(solver)}(κₛ, solver)

"""
    $TYPEDSIGNATURES

Construct the default solver for the implicit skin temperature: a Newton root-finder
([`RootSolver`](@ref) backed by RootSolvers.jl) with a small iteration budget.
"""
function default_skin_temperature_solver(::Type{NF}) where {NF}
    # Default to a Newton root-finder (via RootSolvers.jl) with a small iteration budget
    return RootSolver(NF; max_iterations = 10)
end

"""
    $TYPEDSIGNATURES

Return the conduction target `(Tg, κ, Δz)` for the snow-free ground: the uppermost ground layer's
`ground_temperature`, the assumed surface conductivity `κₛ`, and the top ground cell thickness. This is
always the *unblended* ground-only target, used both when there is no snow and (weighted by
`1 - f_snow`) for the bare-ground share of the skin-temperature solve when there is.
"""
@propagate_inbounds function ground_thermal_interface(i, j, grid, fields, skinT::ImplicitSkinTemperature)
    field_grid = get_field_grid(grid)
    Δz₁ = Δzᵃᵃᶜ(i, j, field_grid.Nz, field_grid)
    Tg = fields.ground_temperature[i, j]
    return (Tg, skinT.κₛ, Δz₁)
end

"""
    $TYPEDSIGNATURES

Return the conduction target `(Tsnow, κsnow, dsnow)` for the snow-covered fraction: the snow's own
(bulk) temperature, its thermal conductivity recovered from the density scheme, and its depth floored at
[`min_snow_conduction_thickness`](@ref). Without snow (`snow === nothing`), returns a placeholders with
`κsnow = 0` such that the snow conductive heat flux reduces to zero.
"""
@propagate_inbounds function snow_thermal_interface(i, j, grid, fields, snow::AbstractSnow, constants::PhysicalConstants)
    ρ_snow = compute_snow_density(i, j, grid, fields, snow.density)
    κ_snow = compute_thermal_conductivity(snow, constants.material, ρ_snow)
    T_snow = snow_temperature(i, j, grid, fields, snow)
    d_snow = max(snow_depth(i, j, grid, fields, snow), min_snow_conduction_thickness(i, j, grid, fields, snow))
    return (T_snow, κ_snow, d_snow)
end
# Fallback for case where snow == nothing
@propagate_inbounds snow_thermal_interface(i, j, grid, fields, ::Nothing, constants::PhysicalConstants) = (zero(eltype(grid)), zero(eltype(grid)), one(eltype(grid)))

"""
    $TYPEDSIGNATURES

Compute the residual ground heat flux that would close the surface energy balance. With all fluxes
positive upward (aligned with `+z`), the energy arriving at the skin from below must balance
the radiative and turbulent losses above, so `G = R_net + H_s + H_l`.
"""
@inline function compute_ground_heat_flux_demand(::AbstractSkinTemperature, R_net, H_s, H_l)
    G₀ = R_net + H_s + H_l
    return G₀
end

## Top-level interface methods

variables(::ImplicitSkinTemperature) = (
    prognostic(:skin_temperature, XY(), units = u"°C", desc = "Longwave emission temperature of the land surface in °C"),
    auxiliary(:ground_heat_flux, XY(), units = u"W/m^2", desc = "Ground heat flux"),
    input(:ground_temperature, XY(), units = u"°C", desc = "Temperature of the uppermost ground or soil grid cell in °C"),
)

"""
    $TYPEDSIGNATURES

Seed the prognostic `skin_temperature` with the current `ground_temperature` so the implicit
nonlinear solve starts from a physically sensible guess close to the root.
"""
function initialize!(state, grid, ::ImplicitSkinTemperature, args...)
    set!(state.skin_temperature, state.ground_temperature)
    return nothing
end

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

Compute and store `ground_heat_flux` on `grid`, dispatching on `skinT` to the type-specific kernel
function: the atmosphere-side demand ``R_\\text{net} + H_s + H_l`` for [`PrescribedSkinTemperature`](@ref)
(which has no separate conduction target), or the explicit conductive flux ``2\\kappa_g(T_g - T_s)/\\Delta z_g``
for [`ImplicitSkinTemperature`](@ref) (all fluxes positive upward).
"""
function compute_ground_heat_flux!(
        state, grid,
        skinT::AbstractSkinTemperature,
        seb::AbstractSurfaceEnergyBalance
    )
    out = auxiliary_fields(state, skinT)
    fields = get_fields(state, seb; except = out)
    launch!(grid, XY, compute_ground_heat_flux_kernel!, out, fields, skinT, seb)
    return nothing
end

## Kernel functions

"""
    $TYPEDSIGNATURES

Compute the ground heat flux *demand* from the surface net radiation and sensible/latent heat flux at grid cell
`i, j`: i.e. the flux implied by the radiative budget and turbulent fluxes, `G = R_net + H_s + H_l`.
"""
@propagate_inbounds function compute_ground_heat_flux_demand(
        i, j, grid, fields,
        skinT::AbstractSkinTemperature,
        ::AbstractSurfaceEnergyBalance
    )
    # Get individual flux terms
    R_net = fields.surface_net_radiation[i, j]
    H_s = fields.sensible_heat_flux[i, j]
    H_l = fields.latent_heat_flux[i, j]
    # Compute ground heat flux
    G₀ = compute_ground_heat_flux_demand(skinT, R_net, H_s, H_l)
    return G₀
end


"""
    $TYPEDSIGNATURES

For `PrescribedSkinTemperature`, set the ground heat flux directly to the demand, i.e. `G₀ = R_net + H_s + H_l`.
"""
@propagate_inbounds function compute_ground_heat_flux(
        i, j, grid, fields,
        skinT::PrescribedSkinTemperature,
        seb::AbstractSurfaceEnergyBalance
    )
    return compute_ground_heat_flux_demand(i, j, grid, fields, skinT, seb)
end

"""
    $TYPEDSIGNATURES

Compute the conductive ground heat flux from the current `skin_temperature` and `ground_temperature`.
"""
@propagate_inbounds function compute_ground_heat_flux(
        i, j, grid, fields,
        skinT::ImplicitSkinTemperature,
        ::AbstractSurfaceEnergyBalance
    )
    Tg, κg, Δzg = ground_thermal_interface(i, j, grid, fields, skinT)
    Ts = fields.skin_temperature[i, j]
    return 2 * κg * (Tg - Ts) / Δzg
end

"""
    $TYPEDSIGNATURES

Per-cell mutating variant used by the fused surface-energy-balance kernel: store the ground heat flux
into the auxiliary output field `out`.
"""
@propagate_inbounds function compute_ground_heat_flux!(out, i, j, grid, fields, skinT::AbstractSkinTemperature, seb::AbstractSurfaceEnergyBalance)
    out.ground_heat_flux[i, j, 1] = compute_ground_heat_flux(i, j, grid, fields, skinT, seb)
    return nothing
end

"""
    $TYPEDSIGNATURES

Invert the (linear) ground-only conduction relation for the implicit skin temperature `Ts` given the
atmosphere-side demanded flux `G` (`= R_net + H_s + H_l`): `Ts = Tg − G/(2κg/Δzg)`. This is the no-snow
special case (`f_snow = 0`) of the snow-aware method below; it is a separate method (rather than a default
`snow = nothing`) purely so it can skip the unused `snow_thermal_interface`/`snow_cover_fraction` calls.
"""
@inline function compute_skin_temperature(i, j, grid, fields, skinT::ImplicitSkinTemperature{NF}, args...) where {NF}
    # Get inputs
    R_net = fields.surface_net_radiation[i, j, 1]
    H_s = fields.sensible_heat_flux[i, j, 1]
    H_l = fields.latent_heat_flux[i, j, 1]
    G₀ = compute_ground_heat_flux_demand(skinT, R_net, H_s, H_l)
    Tg, κg, Δzg = ground_thermal_interface(i, j, grid, fields, skinT)
    Ts = Tg - G₀ * Δzg / (2 * κg)
    return Ts
end

"""
    $TYPEDSIGNATURES

Invert the (linear) area-weighted conduction relation for the implicit skin temperature `Ts` given the
atmosphere-side demanded flux `G` (`= R_net + H_s + H_l`), by equating `G` to the area-weighted sum of the
*unblended* ground and snow-top conductive fluxes, `(1 − f_snow)·2κg(Tg − Ts)/Δzg + f_snow·2κsnow(Tsnow − Ts)/dsnow`.
"""
@inline function compute_skin_temperature(
        i, j, grid, fields,
        skinT::ImplicitSkinTemperature{NF},
        constants::PhysicalConstants,
        snow::AbstractSnow
    ) where {NF}
    # Get inputs
    R_net = fields.surface_net_radiation[i, j, 1]
    H_s = fields.sensible_heat_flux[i, j, 1]
    H_l = fields.latent_heat_flux[i, j, 1]
    G₀ = compute_ground_heat_flux_demand(skinT, R_net, H_s, H_l)
    Tg, κg, Δzg = ground_thermal_interface(i, j, grid, fields, skinT)
    Tsnow, κsnow, dsnow = snow_thermal_interface(i, j, grid, fields, snow, constants)
    f_snow = snow_cover_fraction(i, j, grid, fields, snow)
    # Solve for skin temperature
    Dg = 2 * κg / Δzg
    Ds = 2 * κsnow / dsnow
    A = (NF(1) - f_snow) * Dg + f_snow * Ds
    B = (NF(1) - f_snow) * Dg * Tg + f_snow * Ds * Tsnow
    Ts = (B - G₀) / A
    return Ts
end

"""
    $TYPEDSIGNATURES

Surface-energy-balance residual at grid cell `i, j`, in temperature space: `Ts_prev − Ts_implicit`, where
`Ts_implicit` is the exact conduction-side inverse (see [`compute_skin_temperature`](@ref)) of the
atmosphere-side demanded flux `G_demand = R_net(Ts_prev) + H(Ts_prev) + LE(Ts_prev)`.
"""
@propagate_inbounds function compute_skin_temperature_residual!(
        out, i, j, grid, fields,
        skinT::ImplicitSkinTemperature,
        seb::AbstractSurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        snow::Optional{AbstractSnow} = nothing
    )
    # Compute all fluxes at the current Ts (this includes the explicit ground-conduction flux, stored
    # into `ground_heat_flux`); `snow` partitions the latent flux by snow-covered fraction
    compute_surface_energy_fluxes!(out, i, j, grid, fields, seb, constants, atmos, hydrology, snow)
    Ts_implicit = compute_skin_temperature(i, j, grid, fields, skinT, constants, snow)
    Ts_prev = out.skin_temperature[i, j, 1]
    return Ts_prev - Ts_implicit
end

"""
    $TYPEDSIGNATURES

Run a full nonlinear solve to determine the `skin_temperature` at grid cell `i, j` that solves the surface energy balance.
"""
@propagate_inbounds function solve_skin_temperature!(
        out, i, j, grid, fields,
        skinT::ImplicitSkinTemperature,
        seb::AbstractSurfaceEnergyBalance,
        args...
    )
    objective = ObjectiveFunction(compute_skin_temperature_residual!, :skin_temperature)
    Ts = solve!(out, (i, j), grid, fields, objective, skinT.solver, skinT, seb, args...)
    return Ts
end

# Kernels

@kernel function compute_ground_heat_flux_kernel!(out, grid, fields, skinT::AbstractSkinTemperature, args...)
    i, j = @index(Global, NTuple)
    # Forward to mutating compute_ground_heat_flux!
    compute_ground_heat_flux!(out, i, j, grid, fields, skinT, args...)
end
