"""
    $TYPEDSIGNATURES

Return a `KernelFunctionOperation` that computes the skin temperature residual over all grid cells.
"""
diagnose_skin_temperature_residual(state, model::LandModel) = diagnose_skin_temperature_residual(state, model, model.surface_energy_balance, model.snow)
function diagnose_skin_temperature_residual(
        state,
        model::AbstractModel,
        seb::SurfaceEnergyBalance,
        snow::Optional{AbstractSnow} = nothing
    )
    return kernel_operation2D(state, model.grid, seb.skin_temperature, model.constants, snow) do i, j, grid, fields, skinT::ImplicitSkinTemperature, args...
        Ts_implicit = compute_skin_temperature(i, j, grid, fields, skinT, args...)
        Ts_prev = state.skin_temperature[i, j, 1]
        return Ts_prev - Ts_implicit
    end
end

"""
    $TYPEDSIGNATURES

Return a `KernelFunctionOperation` that computes the kinematic surface humidity flux [m/s] over all grid cells.
"""
diagnose_surface_humidity_flux(state, model::LandModel) =
    diagnose_surface_humidity_flux(state, model, get_evapotranspiration(model.surface_hydrology), model.constants, model.atmosphere)

function diagnose_surface_humidity_flux(
        state,
        model::AbstractModel,
        evapotranspiration::AbstractEvapotranspiration,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    return kernel_operation2D(compute_surface_humidity_flux, state, model.grid, evapotranspiration, constants, atmos)
end

"""
    $TYPEDSIGNATURES

Return a `KernelFunctionOperation` that computes the evaporation flux [m/s liquid water] over all grid cells.
The kinematic humidity flux is rescaled by the air-to-water density ratio and weighted by the snow-free area fraction.
"""
diagnose_evaporation_flux(state, model::LandModel) =
    diagnose_evaporation_flux(state, model, get_evapotranspiration(model.surface_hydrology), model.constants, model.atmosphere, model.snow)

function diagnose_evaporation_flux(
        state,
        model::AbstractModel,
        evapotranspiration::AbstractEvapotranspiration,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        snow::Optional{AbstractSnow} = nothing
    )
    function surface_evapotranspiration_flux(
            i, j, grid, fields,
            evapotranspiration::AbstractEvapotranspiration,
            constants::PhysicalConstants,
            atmos::AbstractAtmosphere,
            snow::Optional{AbstractSnow}
        )
        Qh = compute_surface_humidity_flux(i, j, grid, fields, evapotranspiration, constants, atmos)
        ρ_a = air_density(i, j, grid, fields, atmos, constants)
        ρ_w = constants.material.density_water
        f_snow = snow_cover_fraction(i, j, grid, fields, snow)
        return (1 - f_snow) * Qh * ρ_a / ρ_w
    end
    return kernel_operation2D(surface_evapotranspiration_flux, state, model.grid, evapotranspiration, constants, atmos, snow)
end
