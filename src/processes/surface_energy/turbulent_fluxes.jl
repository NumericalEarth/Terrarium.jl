# Prescribed turbulent fluxes

"""
    $TYPEDEF

Represents the simplest case where the turbulent (sensible and latent) heat fluxes are prescribed
via input variables.
"""
struct PrescribedTurbulentFluxes{NF} <: AbstractTurbulentFluxes{NF} end

PrescribedTurbulentFluxes(::Type{NF}) where {NF} = PrescribedTurbulentFluxes{NF}()

variables(::PrescribedTurbulentFluxes) = (
    input(:sensible_heat_flux, XY(), units = u"W/m^2", desc = "Sensible heat flux at the surface [W m⁻²]"),
    input(:latent_heat_flux, XY(), units = u"W/m^2", desc = "Latent heat flux at the surface [W m⁻²]"),
)

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

Compute the sensible heat flux as a function of the bulk aerodynamic temperature gradient `Q_T` [K m/s]
and the density `ρₐ` [kg/m³] and specific heat capacity `cₐ` [J/kg K] of air.
"""
function compute_sensible_heat_flux(::DiagnosedTurbulentFluxes, Q_T, ρₐ, cₐ)
    Hₛ = cₐ * ρₐ * Q_T
    return Hₛ
end

"""
    $TYPEDSIGNATURES

Compute the latent heat flux as a function of the humidity flux `Q_h` [m/s], the density `ρₐ` [kg/m³] of air,
and the specific latent heat of fusion `Lsl` [J/kg].
"""
function compute_latent_heat_flux(::DiagnosedTurbulentFluxes, Q_h, ρₐ, Lsl)
    Hₗ = Lsl * ρₐ * Q_h
    return Hₗ
end

"""
    $TYPEDSIGNATURES
    
Computes the difference in vapor pressure between a saturated surface at temperature `T`
and the atmosphere, defined by its specific humidity `q_air`.
"""
function vapor_pressure_difference(c::PhysicalConstants, p, q_air, T)
    e_air = partial_pressure_vapor(c, p, q_air)
    e_sat_s = saturation_vapor_pressure(c, T)
    Δe = e_sat_s - e_air
    return Δe
end

"""
    $TYPEDSIGNATURES

Computes the difference in specific humidity between a saturated surface at temperature `T`
and the atmosphere, defined by its specific humidity `q_air`.
"""
function specific_humidity_difference(c::PhysicalConstants, p, q_air, T)
    T_K = celsius_to_kelvin(c, T)
    # TODO: should use surface air density for better accuracy
    ρₐ = air_density(c, T_K, p, q_air)
    q_sat = q_vap_saturation(c, T, ρₐ)
    Δq = q_sat - q_air
    return Δq
end

## Top-level interface methods

variables(::DiagnosedTurbulentFluxes) = (
    auxiliary(:sensible_heat_flux, XY(), units = u"W/m^2", desc = "Sensible heat flux at the surface [W m⁻²]"),
    auxiliary(:latent_heat_flux, XY(), units = u"W/m^2", desc = "Latent heat flux at the surface [W m⁻²]"),
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
    fields = get_fields(state, tur, skinT, atmos; except = out)
    launch!(
        grid, XY, compute_auxiliary_kernel!, out, fields,
        tur, skinT, constants, atmos
    )
    return nothing
end

## Kernel functions

"""
    $TYPEDSIGNATURES

Computes the specific humidity difference between a saturated surface at temperature `T` and the current atmospheric fields.
"""
@propagate_inbounds function compute_specific_humidity_difference(i, j, grid, fields, atmos::AbstractAtmosphere, c::PhysicalConstants, T)
    q_air = specific_humidity(i, j, grid, fields, atmos)
    pres = air_pressure(i, j, grid, fields, atmos)
    Δq = specific_humidity_difference(c, pres, q_air, T)
    return Δq
end

"""
    $TYPEDSIGNATURES

Computes the vapor pressure difference between a saturated surface at temperature `T` and the current atmospheric fields.
"""
@propagate_inbounds function compute_vapor_pressure_difference(i, j, grid, fields, atmos::AbstractAtmosphere, c::PhysicalConstants, T)
    q_air = specific_humidity(i, j, grid, fields, atmos)
    pres = air_pressure(i, j, grid, fields, atmos)
    Δe = vapor_pressure_difference(c, pres, q_air, T)
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
    let cₐ = cp_d(constants), # specific heat capacity of air
            rₐ = aerodynamic_resistance(i, j, grid, fields, atmos), # aerodynamic resistance
            Tₛ = skin_temperature(i, j, grid, fields, skinT), # skin temperature
            Tₐ = air_temperature(i, j, grid, fields, atmos), # air temperature
            pres = air_pressure(i, j, grid, fields, atmos),
            q_air = specific_humidity(i, j, grid, fields, atmos),
            # TODO: density should be evaluated at surface temperature for better accuracy
            ρₐ = air_density(constants, celsius_to_kelvin(constants, Tₐ), pres, q_air),
            Q_T = (Tₛ - Tₐ) / rₐ  # bulk aerodynamic temperature-gradient
        # Calculate sensible heat flux (positive upwards)
        Hₛ = compute_sensible_heat_flux(tur, Q_T, ρₐ, cₐ)
        return Hₛ
    end
end

"""
    $TYPEDSIGNATURES

Compute the bare ground latent heat flux at `i, j` based on the current skin temperature
and atmospheric conditions. This imlementation assumes that evaporation is the only contributor
to the latent heat flux.
"""
@inline function compute_latent_heat_flux(
        i, j, grid, fields,
        tur::DiagnosedTurbulentFluxes,
        skinT::AbstractSkinTemperature,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    let L = constants.Llg, # specific latent heat of vaporization of water
            Tₛ = skin_temperature(i, j, grid, fields, skinT),
            Tₐ = air_temperature(i, j, grid, fields, atmos), # air temperature
            pres = air_pressure(i, j, grid, fields, atmos),
            q_air = specific_humidity(i, j, grid, fields, atmos),
            # TODO: density should be evaluated at surface temperature for better accuracy
            ρₐ = air_density(constants, celsius_to_kelvin(constants, Tₐ), pres, q_air),
            rₐ = aerodynamic_resistance(i, j, grid, fields, atmos), # aerodynamic resistance
            Δq = compute_specific_humidity_difference(i, j, grid, fields, atmos, constants, Tₛ),
            Q_h = Δq / rₐ  # humidity flux
        # Calculate latent heat flux (positive upwards)
        Hₗ = compute_latent_heat_flux(tur, Q_h, ρₐ, L)
        return Hₗ
    end
end

"""
    $TYPEDSIGNATURES

Compute the latent heat flux at `i, j` based on the given evapotranspiration scheme.
This implementation derives the latent heat flux from the [`surface_humidity_flux`](@ref)
defined by `evtr` which is assumed to be already computed.
"""
@inline function compute_latent_heat_flux(
        i, j, grid, fields,
        tur::DiagnosedTurbulentFluxes,
        evtr::AbstractEvapotranspiration,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    let L = constants.Llg, # specific latent heat of vaporization of water
            Tₐ = air_temperature(i, j, grid, fields, atmos), # air temperature
            pres = air_pressure(i, j, grid, fields, atmos),
            q_air = specific_humidity(i, j, grid, fields, atmos),
            # TODO: density should be evaluated at surface temperature for better accuracy
            ρₐ = air_density(constants, celsius_to_kelvin(constants, Tₐ), pres, q_air),
            Q_h = surface_humidity_flux(i, j, grid, fields, evtr)   # humidity flux
        # Calculate latent heat flux (positive upwards)
        Hₗ = compute_latent_heat_flux(tur, Q_h, ρₐ, L)
        return Hₗ
    end
end

# Kernels

@kernel function compute_auxiliary_kernel!(
        out, grid, fields,
        tur::DiagnosedTurbulentFluxes,
        skinT::AbstractSkinTemperature,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    i, j = @index(Global, NTuple)
    # compute sensible heat flux
    out.sensible_heat_flux[i, j, 1] = compute_sensible_heat_flux(i, j, grid, fields, tur, skinT, constants, atmos)
    # compute latent heat flux
    out.latent_heat_flux[i, j, 1] = compute_latent_heat_flux(i, j, grid, fields, tur, skinT, constants, atmos)
end

# TODO: Can these dispatches be standardized to reduce redundancy?
@kernel function compute_auxiliary_kernel!(
        out, grid, fields,
        tur::DiagnosedTurbulentFluxes,
        skinT::AbstractSkinTemperature,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        evtr::AbstractEvapotranspiration
    )
    i, j = @index(Global, NTuple)
    # compute sensible heat flux
    out.sensible_heat_flux[i, j, 1] = compute_sensible_heat_flux(i, j, grid, fields, tur, skinT, constants, atmos)
    # compute latent heat flux
    out.latent_heat_flux[i, j, 1] = compute_latent_heat_flux(i, j, grid, fields, tur, evtr, constants, atmos)
end
