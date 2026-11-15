"""
    $TYPEDSIGNATURES

Return a `KernelFunctionOperation` that computes the total water stored in each ground column (m of water
equivalent): the liquid water and ice in the soil pore space, `∫ ϕ S dz` with porosity `ϕ` and saturation `S`,
plus the `surface_excess_water` pool of the surface runoff scheme, if there is one. This is the quantity whose
change over time a column water balance compares against the boundary fluxes (rain, evaporation, and runoff).

Like the other diagnostics, the operation is constructed anew on each call and is intended for diagnostics,
output, and tests; it can be indexed directly, e.g. `diagnose_total_ground_water(state, model)[i, j, 1]`.
"""
diagnose_total_ground_water(state, model::SoilModel) = diagnose_total_ground_water(state, model, model.soil)
diagnose_total_ground_water(state, model::LandModel) = diagnose_total_ground_water(state, model, model.soil, get_surface_runoff(model.surface_hydrology))

function diagnose_total_ground_water(
        state,
        model::AbstractModel,
        soil::AbstractSoil,
        runoff::Optional{AbstractSurfaceRunoff} = nothing
    )
    hydrology = get_hydrology(soil)
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    return kernel_operation2D(compute_total_ground_water, state, model.grid, hydrology, strat, bgc, runoff)
end
