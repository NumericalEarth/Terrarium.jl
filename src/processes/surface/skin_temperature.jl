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
    input(:skin_temperature, Surface(XY()), units = u"°C", desc = "Longwave emission temperature of the land surface in °C"),
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
    ground_grid = ground_domain(grid)
    Δz₁ = Δzᵃᵃᶜ(i, j, ground_grid.Nz, ground_grid)
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

## Top-level interface methods

variables(::ImplicitSkinTemperature) = (
    prognostic(:skin_temperature, Surface(XY()), units = u"°C", desc = "Longwave emission temperature of the land surface in °C"),
    input(:ground_temperature, Ground(Top(z = Center())), units = u"°C", desc = "Temperature of the uppermost ground or soil grid cell in °C"),
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

## Kernel functions

"""
    $TYPEDSIGNATURES

Invert the (linear) ground-only conduction relation for the implicit skin temperature `Ts` given the
atmosphere-side demanded flux `G` (see [`compute_ground_heat_flux_demand`](@ref)): `Ts = Tg − G/(2κg/Δzg)`.
This is the no-snow special case (`f_snow = 0`) of the snow-aware method below; it is a separate method
(rather than a default `snow = nothing`) purely so it can skip the unused
`snow_thermal_interface`/`snow_cover_fraction` calls.
"""
@inline function compute_skin_temperature(
        i, j, grid, fields,
        skinT::ImplicitSkinTemperature{NF},
        ghf::AbstractGroundHeatFlux,
        args...
    ) where {NF}
    G₀ = compute_ground_heat_flux_demand(i, j, grid, fields, ghf)
    Tg, κg, Δzg = ground_thermal_interface(i, j, grid, fields, skinT)
    Ts = Tg - G₀ * Δzg / (2 * κg)
    return Ts
end

"""
    $TYPEDSIGNATURES

Invert the (linear) area-weighted conduction relation for the implicit skin temperature `Ts` given the
atmosphere-side demanded flux `G` (see [`compute_ground_heat_flux_demand`](@ref)), by equating `G` to the
area-weighted sum of the *unblended* ground and snow-top conductive fluxes,
`(1 − f_snow)·2κg(Tg − Ts)/Δzg + f_snow·2κsnow(Tsnow − Ts)/dsnow`.
"""
@inline function compute_skin_temperature(
        i, j, grid, fields,
        skinT::ImplicitSkinTemperature{NF},
        ghf::AbstractGroundHeatFlux,
        constants::PhysicalConstants,
        snow::AbstractSnow
    ) where {NF}
    G₀ = compute_ground_heat_flux_demand(i, j, grid, fields, ghf)
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
atmosphere-side demanded flux supplied by the ground heat flux sub-process (see
[`compute_ground_heat_flux_demand`](@ref)), which for [`DiagnosedGroundHeatFlux`](@ref) is
`G_demand = R_net(Ts_prev) + H(Ts_prev) + LE(Ts_prev)`.
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
    # Compute all fluxes at the current Ts (with a diagnosed ground heat flux this includes the
    # explicit ground-conduction flux, stored into `ground_heat_flux`); `snow` partitions the latent
    # flux by snow-covered fraction
    compute_surface_energy_fluxes!(out, i, j, grid, fields, seb, constants, atmos, hydrology, snow)
    Ts_implicit = compute_skin_temperature(i, j, grid, fields, skinT, get_ground_heat_flux(seb), constants, snow)
    Ts_prev = out.skin_temperature[i, j, end]
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
