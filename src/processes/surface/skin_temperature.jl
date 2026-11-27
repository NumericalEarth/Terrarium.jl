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

Scheme for an implicit skin temperature `Tₛ` satisfying the surface energy balance

    R_net(Tₛ) + H(Tₛ) + LE(Tₛ) = G(Tₛ)

where `R_net` is the net radiation budget, `H` is the sensible heat flux, `LE` is the latent heat flux,
and `G` is the conductive heat flux from the skin into the medium below it (all positive upward). On the
snow-free surface (superscript `sf`), `G = Gˢᶠ = 2κₛ(T_g − Tₛ)/Δz_g` conducts into the uppermost ground
layer of temperature `T_g` and thickness `Δz_g` (see [`ground_thermal_interface`](@ref)), and `LE` is the
evaporation and transpiration latent flux. On the snow-covered surface (superscript `sn`),
`G = Gˢⁿ = 2κ_snow(T_snow − Tₛ)/d_snow` conducts through half the bulk snow depth `d_snow` into the
snowpack of temperature `T_snow` (see [`snow_thermal_interface`](@ref)), and `LE` is the sublimation
latent flux.

The skin temperature is found as the root of `Tₛ − Tₛ*(Tₛ)`, where `Tₛ*` inverts the (linear)
conduction relation for the demanded flux `G₀ = R_net + H + LE` (see [`compute_skin_temperature`](@ref)).

!!! note
    With a snow component, the solve currently uses a single grid-cell skin temperature, for which the
    snow-free and snow-covered conduction relations are combined by area weighting with the snow-covered
    fraction `f_snow`: `(1 − f_snow)·Gˢᶠ(Tₛ) + f_snow·Gˢⁿ(Tₛ) = G₀`. This is a transitional state; the
    snow-free and snow-covered tiles are to be solved separately.

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
the conduction target of the snow-free surface, used both when there is no snow and for the snow-free
tile when there is.
"""
@propagate_inbounds function ground_thermal_interface(i, j, grid, fields, skinT::ImplicitSkinTemperature)
    ground_grid = ground_domain(grid)
    Δz₁ = Δzᵃᵃᶜ(i, j, ground_grid.Nz, ground_grid)
    Tg = fields.ground_temperature[i, j]
    return (Tg, skinT.κₛ, Δz₁)
end

"""
    $TYPEDSIGNATURES

Return the conduction target `(Tsnow, κsnow, dsnow)` for the snow-covered surface: the snow's own
(bulk) temperature, its thermal conductivity recovered from the density scheme, and its depth floored at
[`min_snow_conduction_thickness`](@ref). Without snow (`snow === nothing`), returns placeholders with
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
    auxiliary(:skin_temperature, Surface(XY()), units = u"°C", desc = "Longwave emission temperature of the land surface in °C"),
    input(:ground_temperature, Ground(Top(z = Center())), units = u"°C", desc = "Temperature of the uppermost ground or soil grid cell in °C"),
)

"""
    $TYPEDSIGNATURES

Seed the auxiliary `skin_temperature` with the current `ground_temperature` so the implicit
nonlinear solve starts from a physically sensible guess close to the root. The field persists
across time steps, so each subsequent solve starts from the previous solution.
"""
function initialize!(state, grid, ::ImplicitSkinTemperature, args...)
    set!(state.skin_temperature, state.ground_temperature)
    return nothing
end

## Kernel functions

"""
    $TYPEDSIGNATURES

Invert the (linear) ground conduction relation of the snow-free tile for the implicit skin temperature
`Tₛˢᶠ` given the atmosphere-side demanded flux `G₀` (see [`compute_ground_heat_flux_demand`](@ref)),

    Tₛˢᶠ = T_g − G₀·Δz_g/(2κₛ)

where `T_g`, `κₛ`, and `Δz_g` are the ground conduction target (see [`ground_thermal_interface`](@ref)).
This is the method for the snow-free tile ([`SnowFree`](@ref)), which is also the whole grid cell in
configurations without snow; `constants` and `snow` are not used.
"""
@propagate_inbounds function compute_skin_temperature(
        i, j, grid, fields,
        ::SnowFree,
        skinT::ImplicitSkinTemperature{NF},
        seb::AbstractSurfaceEnergyBalance,
        constants::PhysicalConstants,
        snow::Optional{AbstractSnow} = nothing
    ) where {NF}
    G₀ = compute_ground_heat_flux_demand(i, j, grid, fields, get_ground_heat_flux(seb), seb)
    Tg, κg, Δzg = ground_thermal_interface(i, j, grid, fields, skinT)
    Ts = Tg - G₀ * Δzg / (2 * κg)
    return Ts
end

"""
    $TYPEDSIGNATURES

Invert the (linear) snow conduction relation of the snow-covered tile for the implicit skin temperature
`Tₛˢⁿ` given the atmosphere-side demanded flux `G₀` (see [`compute_ground_heat_flux_demand`](@ref)),

    Tₛˢⁿ = T_snow − G₀·d_snow/(2κ_snow)

where `T_snow`, `κ_snow`, and `d_snow` are the snow conduction target (see
[`snow_thermal_interface`](@ref)). This is the method for the snow-covered tile ([`SnowCovered`](@ref)); it does not depend on the
snow-covered area fraction. It requires an actual snow component, since the `nothing` placeholder of
[`snow_thermal_interface`](@ref) has zero conductivity.
"""
@propagate_inbounds function compute_skin_temperature(
        i, j, grid, fields,
        ::SnowCovered,
        skinT::ImplicitSkinTemperature{NF},
        seb::AbstractSurfaceEnergyBalance,
        constants::PhysicalConstants,
        snow::AbstractSnow
    ) where {NF}
    G₀ = compute_ground_heat_flux_demand(i, j, grid, fields, get_ground_heat_flux(seb), seb)
    Tsnow, κsnow, dsnow = snow_thermal_interface(i, j, grid, fields, snow, constants)
    Ts = Tsnow - G₀ * dsnow / (2 * κsnow)
    return Ts
end

"""
    $TYPEDSIGNATURES

Grid-cell skin temperature that satisfies the area-weighted conduction relation
`(1 − f_snow)·Gˢᶠ(Tₛ) + f_snow·Gˢⁿ(Tₛ) = G₀` for the demanded flux `G₀`. Since both conductive fluxes are
linear in `Tₛ`, this is the conductance-weighted mean of the snow-free and snow-covered tile inverses
`Tₛˢᶠ` and `Tₛˢⁿ` (the methods of [`compute_skin_temperature`](@ref) for [`SnowFree`](@ref) and
[`SnowCovered`](@ref)),

    Tₛ = ((1 − f_snow)·D_g·Tₛˢᶠ + f_snow·D_snow·Tₛˢⁿ) / ((1 − f_snow)·D_g + f_snow·D_snow)

with the conductances `D_g = 2κₛ/Δz_g` and `D_snow = 2κ_snow/d_snow`.

With snow, this is a transitional kernel function used while the surface energy balance is solved for a
single grid-cell skin temperature; it is to be replaced by separate snow-free and snow-covered tile
solves.
"""
@propagate_inbounds function compute_skin_temperature(
        i, j, grid, fields,
        skinT::ImplicitSkinTemperature{NF},
        seb::AbstractSurfaceEnergyBalance,
        constants::PhysicalConstants,
        snow::AbstractSnow
    ) where {NF}
    Ts_sf = compute_skin_temperature(i, j, grid, fields, SnowFree(), skinT, seb, constants, snow)
    Ts_sn = compute_skin_temperature(i, j, grid, fields, SnowCovered(), skinT, seb, constants, snow)
    _, κg, Δzg = ground_thermal_interface(i, j, grid, fields, skinT)
    _, κsnow, dsnow = snow_thermal_interface(i, j, grid, fields, snow, constants)
    f_snow = snow_cover_fraction(i, j, grid, fields, snow)
    Dg = 2 * κg / Δzg
    Ds = 2 * κsnow / dsnow
    w_sf = (one(NF) - f_snow) * Dg
    w_sn = f_snow * Ds
    Ts = (w_sf * Ts_sf + w_sn * Ts_sn) / (w_sf + w_sn)
    return Ts
end

"""
    $TYPEDSIGNATURES

Grid-cell skin temperature without snow, i.e. the inverse of the snow-free tile ([`SnowFree`](@ref)).
"""
@propagate_inbounds compute_skin_temperature(
    i, j, grid, fields,
    skinT::ImplicitSkinTemperature,
    seb::AbstractSurfaceEnergyBalance,
    constants::PhysicalConstants,
    ::Nothing = nothing
) = compute_skin_temperature(i, j, grid, fields, SnowFree(), skinT, seb, constants, nothing)

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
    # Transitional (Phase 3 of the snow tiling plan replaces this with separate tile solves): with snow,
    # the single grid-cell skin temperature inverts the area-weighted blend of the snow-free and
    # snow-covered conduction relations
    Ts_implicit = compute_skin_temperature(i, j, grid, fields, skinT, seb, constants, snow)
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
