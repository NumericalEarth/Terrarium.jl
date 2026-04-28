"""
    $TYPEDEF

Standard implementation of the surface energy balance (SEB) that computes the radiative,
turbulent, and ground energy fluxes at the surface. The SEB is also responsible for defining
and solving the so-called *skin temperature* (effective emission temperature of the land surface)
as well as the albedo.
"""
struct SurfaceEnergyBalance{
        NF,
        SkinTemperature <: AbstractSkinTemperature{NF},
        TurbulentFluxes <: AbstractTurbulentFluxes{NF},
        RadiativeFluxes <: AbstractRadiativeFluxes{NF},
        Albedo <: AbstractAlbedo{NF},
    } <: AbstractSurfaceEnergyBalance{NF}
    "Scheme for determining skin temperature and ground heat flux"
    skin_temperature::SkinTemperature

    "Scheme for determining the net radiation budget"
    radiative_fluxes::RadiativeFluxes

    "Scheme for computing turbulent (sensible and latent) heat fluxes"
    turbulent_fluxes::TurbulentFluxes

    "Scheme for parameterizing surface albedo"
    albedo::Albedo
end

function SurfaceEnergyBalance(
        ::Type{NF};
        radiative_fluxes::AbstractRadiativeFluxes = DiagnosedRadiativeFluxes(NF),
        turbulent_fluxes::AbstractTurbulentFluxes = DiagnosedTurbulentFluxes(NF),
        skin_temperature::AbstractSkinTemperature = ImplicitSkinTemperature(NF),
        albedo::AbstractAlbedo = ConstantAlbedo(NF)
    ) where {NF}
    return SurfaceEnergyBalance(skin_temperature, radiative_fluxes, turbulent_fluxes, albedo)
end

variables(seb::SurfaceEnergyBalance) = tuplejoin(
    variables(seb.albedo),
    variables(seb.skin_temperature),
    variables(seb.radiative_fluxes),
    variables(seb.turbulent_fluxes)
)

""" $TYPEDSIGNATURES """
@inline function compute_auxiliary!(
        state, grid,
        seb::SurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        args...
    )
    compute_surface_energy_fluxes!(state, grid, seb, constants, atmos, hydrology, args...)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the surface energy fluxes on `grid` based on the current atmospheric state.
"""
function compute_surface_energy_fluxes!(
        state, grid,
        seb::SurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        args...
    )
    evtr = isnothing(hydrology) ? nothing : get_evapotranspiration(hydrology)
    # Construct outputs as auxiliaries + skin temperature (which is prognostic)
    out = (skin_temperature = state.skin_temperature, auxiliary_fields(state, seb)...)
    fields = get_fields(state, seb, atmos, evtr)
    launch!(grid, XY, compute_surface_energy_fluxes_kernel!, out, fields, seb, constants, atmos, evtr, args...)
    return nothing
end

# Kernels (fused)

@kernel inbounds = true function compute_surface_energy_fluxes_kernel!(
        out, grid, fields,
        seb::SurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    )
    i, j = @index(Global, NTuple)

    # Compute fluxes based on current skin temperature
    compute_surface_energy_fluxes!(out, i, j, grid, fields, seb, constants, atmos, args...)
end

@kernel inbounds = true function compute_surface_energy_fluxes_kernel!(
        out, grid, fields,
        seb::SurfaceEnergyBalance{NF, <:ImplicitSkinTemperature},
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    ) where {NF}
    i, j = @index(Global, NTuple)

    # Compute fluxes based on current skin temperature
    compute_surface_energy_fluxes!(out, i, j, grid, fields, seb, constants, atmos, args...)
    # Update skin temperature
    out.skin_temperature[i, j, 1] = compute_skin_temperature(i, j, grid, fields, seb.skin_temperature)
    # Recompute fluxes from updated skin temperature
    compute_surface_energy_fluxes!(out, i, j, grid, fields, seb, constants, atmos, args...)
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Fused kernel function that computes the radiative and turbulent fluxes, as well as the ground heat flux.
"""
@propagate_inbounds function compute_surface_energy_fluxes!(
        out, i, j, grid, fields,
        seb::SurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        evtr::Optional{AbstractEvapotranspiration} = nothing,
        args...
    )
    # Compute radiative fluxes
    radiative_fluxes = compute_surface_upwelling_radiation(i, j, grid, fields, seb.radiative_fluxes, seb.skin_temperature, seb.albedo, constants, atmos)
    out.surface_shortwave_up[i, j, 1] = radiative_fluxes.surface_shortwave_up
    out.surface_longwave_up[i, j, 1] = radiative_fluxes.surface_longwave_up
    out.surface_net_radiation[i, j, 1] = compute_surface_net_radiation(i, j, grid, fields, seb.radiative_fluxes, atmos)
    # Compute turbulent fluxes
    out.sensible_heat_flux[i, j, 1] = compute_sensible_heat_flux(i, j, grid, fields, seb.turbulent_fluxes, seb.skin_temperature, constants, atmos)
    if isnothing(evtr)
        # Bare ground evaporation, no coupling with ET
        out.latent_heat_flux[i, j, 1] = compute_latent_heat_flux(i, j, grid, fields, seb.turbulent_fluxes, seb.skin_temperature, constants, atmos)
    else
        # Coupling with surface hydrology ET scheme
        out.latent_heat_flux[i, j, 1] = compute_latent_heat_flux(i, j, grid, fields, seb.turbulent_fluxes, evtr, constants, atmos)
    end
    # Compute ground heat flux
    out.ground_heat_flux[i, j, 1] = compute_ground_heat_flux(i, j, grid, fields, seb.skin_temperature, seb.radiative_fluxes, seb.turbulent_fluxes)
    return nothing
end
