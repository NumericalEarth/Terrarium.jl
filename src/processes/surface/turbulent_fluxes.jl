# Prescribed turbulent fluxes

"""
    $TYPEDEF

Represents the simplest case where the turbulent (sensible and latent) heat fluxes are prescribed
via input variables.
"""
struct PrescribedTurbulentFluxes{NF} <: AbstractTurbulentFluxes{NF} end

PrescribedTurbulentFluxes(::Type{NF}) where {NF} = PrescribedTurbulentFluxes{NF}()

variables(::PrescribedTurbulentFluxes) = (
    input(:sensible_heat_flux, Surface(XY()), units = u"W/m^2", desc = "Sensible heat flux at the surface [W m⁻²]"),
    input(:latent_heat_flux, Surface(XY()), units = u"W/m^2", desc = "Latent heat flux at the surface [W m⁻²]"),
)

# The turbulent fluxes are prescribed input variables, so there is nothing to diagnose.
@inline compute_auxiliary!(state, grid, ::PrescribedTurbulentFluxes, args...) = nothing

# Diagnosed turbulent fluxes

"""
    $TYPEDEF

Represents the standard case where the turbulent (sensible and latent) heat fluxes are diagnosed
from atmosphere and soil conditions.
"""
struct DiagnosedTurbulentFluxes{NF} <: AbstractTurbulentFluxes{NF} end

DiagnosedTurbulentFluxes(::Type{NF}) where {NF} = DiagnosedTurbulentFluxes{NF}()

"""
    $TYPEDSIGNATURES

Compute the sensible heat flux [W/m²] as a function of the bulk aerodynamic temperature gradient `Q_T` [K m/s]
and the density `ρₐ` [kg/m³] and specific heat capacity `cₐ` [J/kg K] of air.
"""
function compute_sensible_heat_flux(::DiagnosedTurbulentFluxes, Q_T, ρₐ, cₐ)
    Hₛ = cₐ * ρₐ * Q_T
    return Hₛ
end

"""
    $TYPEDSIGNATURES

Compute the latent heat flux as a function of the humidity flux `Q_h` [m/s], the density `ρₐ` [kg/m³] of air,
and the specific latent heat of vaporization or sublimation `L` [J/kg].
"""
function compute_latent_heat_flux(::DiagnosedTurbulentFluxes, Q_h, ρₐ, L)
    Hₗ = L * ρₐ * Q_h
    return Hₗ
end

"""
    $TYPEDSIGNATURES
    
Computes the difference in vapor pressure between a saturated surface at temperature `T` [°C]
and the atmosphere, defined by its specific humidity `q_air` [kg/kg] and pressure `p` [Pa]. 
Relies on [Thermodynamics.jl](https://github.com/CliMA/Thermodynamics.jl) via
[`partial_pressure_vapor`](@extref Thermodynamics.partial_pressure_vapor) and
[`saturation_vapor_pressure`](@ref saturation_vapor_pressure).
"""
function vapor_pressure_difference(c::ThermodynamicConstants, p, q_air, T)
    e_air = Thermodynamics.partial_pressure_vapor(c, p, q_air)
    e_sat_s = saturation_vapor_pressure(c, T)
    Δe = e_sat_s - e_air
    return Δe
end

"""
    $TYPEDSIGNATURES

Computes the difference in specific humidity between a saturated surface at temperature `T` [°C]
and the atmosphere, defined by its specific humidity `q_air` [kg/kg] and pressure `p` [Pa].
"""
function specific_humidity_difference(c::ThermodynamicConstants{NF}, p, q_air, T) where {NF}
    # Bound T from below at absolute zero to prevent nonphysical thermodynamics
    T_C = max(T, NF(-273))
    T_K = celsius_to_kelvin(c, T_C)
    # TODO: should use surface air density for better accuracy
    ρₐ = Thermodynamics.air_density(c, T_K, p, q_air)
    q_sat = saturation_specific_humidity_vapor(c, T_C, ρₐ)
    Δq = q_sat - q_air
    return Δq
end

## Top-level interface methods

variables(::DiagnosedTurbulentFluxes) = (
    auxiliary(:sensible_heat_flux, Surface(XY()), units = u"W/m^2", desc = "Sensible heat flux at the surface [W m⁻²]"),
    auxiliary(:latent_heat_flux, Surface(XY()), units = u"W/m^2", desc = "Latent heat flux at the surface [W m⁻²]"),
)

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        tur::DiagnosedTurbulentFluxes,
        seb::AbstractSurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    )
    skinT = seb.skin_temperature
    out = auxiliary_fields(state, tur)
    fields = get_fields(state, tur, skinT, atmos, args...; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, tur, skinT, constants, atmos, args...)
    return nothing
end

## Kernel functions

"""
    $TYPEDSIGNATURES

Computes the specific humidity difference [kg/kg] between a saturated surface at temperature `T` [°C] and the current atmospheric fields.
"""
@propagate_inbounds function compute_specific_humidity_difference(i, j, grid, fields, atmos::AbstractAtmosphere, c::PhysicalConstants, T)
    q_air = specific_humidity(i, j, grid, fields, atmos)
    pres = air_pressure(i, j, grid, fields, atmos)
    Δq = specific_humidity_difference(c.thermodynamics, pres, q_air, T)
    return Δq
end

"""
    $TYPEDSIGNATURES

Computes the vapor pressure difference [Pa] between a saturated surface at temperature `T` [°C] and the current atmospheric fields.
"""
@propagate_inbounds function compute_vapor_pressure_difference(i, j, grid, fields, atmos::AbstractAtmosphere, c::PhysicalConstants, T)
    q_air = specific_humidity(i, j, grid, fields, atmos)
    pres = air_pressure(i, j, grid, fields, atmos)
    Δe = vapor_pressure_difference(c.thermodynamics, pres, q_air, T)
    return Δe
end

"""
    $TYPEDSIGNATURES

Compute the sensible heat flux at `i, j` based on the current skin temperature and atmospheric conditions.
"""
@inline function compute_sensible_heat_flux(
        i, j, grid, fields,
        tur::DiagnosedTurbulentFluxes,
        skinT::AbstractSkinTemperature,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    rₐ = aerodynamic_resistance(i, j, grid, fields, atmos) # aerodynamic resistance
    Tₛ = skin_temperature(i, j, grid, fields, skinT) # skin temperature
    Tₐ = air_temperature(i, j, grid, fields, atmos) # air temperature
    pₐ = air_pressure(i, j, grid, fields, atmos)
    qₐ = specific_humidity(i, j, grid, fields, atmos)
    cₐ = specific_heat_capacity_moist_air(constants.thermodynamics, qₐ) # specific heat capacity of moist air
    # TODO: density should be evaluated at surface temperature for better accuracy
    ρₐ = Thermodynamics.air_density(constants.thermodynamics, celsius_to_kelvin(constants.thermodynamics, Tₐ), pₐ, qₐ)
    Q_T = (Tₛ - Tₐ) / rₐ  # bulk aerodynamic temperature-gradient
    # Calculate sensible heat flux (positive upwards)
    Hₛ = compute_sensible_heat_flux(tur, Q_T, ρₐ, cₐ)
    return Hₛ
end

"""
    $TYPEDSIGNATURES

Compute the latent heat flux at `i, j` based on the current skin temperature and atmospheric conditions.
When an evapotranspiration scheme is provided, uses the ET-aware humidity flux; otherwise uses bare ground evaporation.

With a snow component, the flux is partitioned by snow-covered area fraction `f_snow`: the snow-free
fraction `(1 − f_snow)` evaporates from the ground/canopy (latent heat of vaporization), while the
snow-covered fraction sublimates from the snowpack (latent heat of sublimation, see
[`compute_snow_sublimation_flux`](@ref)). Without snow it reduces to the bare ground/canopy latent flux.
"""
@inline function compute_latent_heat_flux(
        i, j, grid, fields,
        tur::DiagnosedTurbulentFluxes{NF},
        skinT::AbstractSkinTemperature,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        surface_hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        snow::Optional{AbstractSnow} = nothing,
    ) where {NF}
    Q_h = if isnothing(surface_hydrology)
        # Direct calculation of evaporative flux without ET coupling
        Tₛ = skin_temperature(i, j, grid, fields, skinT)
        rₐ = aerodynamic_resistance(i, j, grid, fields, atmos) # aerodynamic resistance
        Δq = compute_specific_humidity_difference(i, j, grid, fields, atmos, constants, Tₛ)
        Δq / rₐ  # simplified humidity flux w/o separate ET
    else
        # Compute humidity flux using the given ET scheme
        evtr = get_evapotranspiration(surface_hydrology)
        compute_surface_humidity_flux(i, j, grid, fields, evtr, constants, atmos)
    end

    # Get atmospheric variables and snow cover fraction
    # TODO: density should be evaluated at surface temperature for better accuracy
    ρₐ = air_density(i, j, grid, fields, atmos, constants)
    f = snow_cover_fraction(i, j, grid, fields, snow)
    # Retrieve constants
    L_lv = constants.thermodynamics.latent_heat_vaporization
    L_sg = constants.thermodynamics.latent_heat_sublimation
    ρ_w = constants.material.density_water
    # Compute latent heat flux for bare ground and snow sublimation flux
    Hₗ_ground = compute_latent_heat_flux(tur, Q_h, ρₐ, L_lv)
    E_subl = compute_snow_sublimation_flux(i, j, grid, fields, snow, atmos, constants, skinT)
    Hₗ_snow = ρ_w * L_sg * E_subl
    # Compute area-weighted total latent heat flux (bare ground + snow sublimation)
    Hₗ = (NF(1) - f) * Hₗ_ground + f * Hₗ_snow
    return Hₗ
end

# Kernels

@kernel function compute_auxiliary_kernel!(out, grid, fields, tur::DiagnosedTurbulentFluxes, args...)
    i, j = @index(Global, NTuple)
    # compute sensible heat flux
    out.sensible_heat_flux[i, j, end] = compute_sensible_heat_flux(i, j, grid, fields, tur, args...)
    # compute latent heat flux - pass all args to allow dispatch on evtr presence
    out.latent_heat_flux[i, j, end] = compute_latent_heat_flux(i, j, grid, fields, tur, args...)
end

# Per-process mutating variant used by the fused surface-energy-balance kernel.

"""
    $TYPEDSIGNATURES

Compute the turbulent (sensible and latent) heat fluxes from the current skin temperature and store
them into the auxiliary output fields `out`.
"""
@propagate_inbounds function compute_turbulent_fluxes!(out, i, j, grid, fields, tur::DiagnosedTurbulentFluxes, skinT, constants, atmos, hydrology, snow)
    out.sensible_heat_flux[i, j, end] = compute_sensible_heat_flux(i, j, grid, fields, tur, skinT, constants, atmos)
    out.latent_heat_flux[i, j, end] = compute_latent_heat_flux(i, j, grid, fields, tur, skinT, constants, atmos, hydrology, snow)
    return nothing
end

"""
    $TYPEDSIGNATURES

Prescribed turbulent fluxes are input fields supplied by an external coupler (already present in
`fields`), so there is nothing to compute or store.
"""
@propagate_inbounds compute_turbulent_fluxes!(out, i, j, grid, fields, ::PrescribedTurbulentFluxes, args...) = nothing
