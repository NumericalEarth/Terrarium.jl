"""
    $TYPEDSIGNATURES

Launch [`compute_snow_interface_fluxes!`](@ref) to diagnose the snow↔surface/soil coupling fluxes from the
snow state and the surface energy balance outputs: the blended soil-top heat flux (`soil_heat_flux`), the
grid-cell-mean snow-base flux (`snow_basal_heat_flux`), the grid-cell-mean snow-top conductive flux
(`snow_surface_heat_flux`; both drive the snowpack's own energy tendency), and the snow
surface sublimation rate (`sublimation`, the snow-fraction bulk-aerodynamic vapor flux at the converged
skin temperature). Must run after the surface energy balance (which sets `ground_heat_flux` and the skin
temperature). No-op when there is no snowpack (`snow === nothing`).
"""
compute_snow_interface_fluxes!(state, grid, ::Nothing, args...) = nothing
function compute_snow_interface_fluxes!(
        state, grid,
        snow::SingleLayerSnow,
        seb::AbstractSurfaceEnergyBalance,
        soil::AbstractSoil,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    # atmosphere fields (air temperature/pressure/humidity, windspeed) are needed for the sublimation flux
    out = (
        sublimation = state.sublimation,
        soil_heat_flux = state.soil_heat_flux,
        snow_basal_heat_flux = state.snow_basal_heat_flux,
        snow_surface_heat_flux = state.snow_surface_heat_flux,
    )
    fields = get_fields(state, snow, seb, soil, atmos)
    launch!(grid, XY, compute_snow_interface_fluxes_kernel!, out, fields, snow, seb, soil, constants, atmos)
    return nothing
end

"""
    $TYPEDSIGNATURES

Snow→soil basal conductive heat flux `Q_base` [W/m²] at grid cell `i, j`, positive upward (soil → snow),
as the series-resistance conduction between the soil's top half-cell and the snow layer:
`Q_base = (T_soil − T_snow) / (Δz_soil/(2κ_soil) + d_snow/(2κ_snow))`. Both the local soil top-layer
thermal conductivity `κ_soil` and the snow conductivity `κ_snow` are recovered from their respective
composition/density schemes rather than stored. The snow-side conduction thickness is floored at
[`min_snow_conduction_thickness`](@ref) as `d_snow → 0`, matching the floor used elsewhere in the snow
thermodynamics; without the soil-side term this reduces to the previous snow-resistance-only closure,
which is only valid once the soil resistance is genuinely negligible next to the (floored) snow's.
"""
@propagate_inbounds function compute_snow_basal_heat_flux(
        i, j, grid, fields,
        snow::SingleLayerSnow,
        soil::AbstractSoil,
        constants::PhysicalConstants
    )
    strat = get_stratigraphy(soil)
    hydrology = get_hydrology(soil)
    bgc = get_biogeochemistry(soil)
    energy = get_energy_balance(soil)
    ground_grid = ground_domain(grid)
    k = ground_grid.Nz
    composition = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    κ_soil = compute_thermal_conductivity(energy.thermal_properties, composition)
    Δz_soil = Δzᵃᵃᶜ(i, j, k, ground_grid)

    ρ_snow = compute_snow_density(i, j, grid, fields, snow.density)
    κ_snow = compute_thermal_conductivity(snow, constants.material, ρ_snow)
    d_snow = max(fields.snow_depth[i, j], min_snow_conduction_thickness(i, j, grid, fields, snow))

    T_soil = fields.ground_temperature[i, j]
    T_snow = fields.snow_temperature[i, j]
    R_soil = Δz_soil / (2 * κ_soil)
    R_snow = d_snow / (2 * κ_snow)
    return (T_soil - T_snow) / (R_soil + R_snow)
end

"""
    $TYPEDSIGNATURES

Snow-top conductive heat flux [W/m²] at grid cell `i, j` (positive upward, i.e. energy leaving the snow
top into the skin), `S = 2κ_snow(T_snow − T_skin)/max(d_snow, d_min)` — the same unblended snow
conduction target the skin-temperature solve closes against (see
[`Terrarium.snow_thermal_interface`](@ref)). This is a per-unit-snow-area flux; the snowpack energy
tendency ([`compute_snow_energy_tendency`](@ref)'s `Q_top`) uses the grid-cell mean `f_snow·S`, which is
applied at the call site in [`compute_snow_interface_fluxes!`](@ref).
"""
@propagate_inbounds function compute_snow_surface_heat_flux(
        i, j, grid, fields,
        snow::SingleLayerSnow,
        constants::PhysicalConstants
    )
    T_snow, κ_snow, d_snow = snow_thermal_interface(i, j, grid, fields, snow, constants)
    Ts = fields.skin_temperature[i, j]
    return 2 * κ_snow * (T_snow - Ts) / d_snow
end

"""
    $TYPEDSIGNATURES

Diagnose the snow↔surface/soil coupling fluxes at grid cell `i, j`, run after the surface energy balance:
the blended soil-top heat flux `f·Q_base + (1 − f)·G` (with `G` the bare-ground `ground_heat_flux`; used only as the soil's top boundary condition), the grid-cell-mean snow-base
flux `f·Q_base` (see [`compute_snow_basal_heat_flux`](@ref)), the grid-cell-mean snow-top conductive flux
`f·S` (see [`compute_snow_surface_heat_flux`](@ref)), and the snow surface sublimation rate (see
[`compute_snow_sublimation_flux`](@ref), the same snow-fraction vapor flux the surface energy balance uses
for the latent-flux partition). All snowpack fluxes are grid-cell means scaled by the snow cover fraction `f`
(as is the sublimation) since the snow state is a grid-cell mean. The column (soil + snow) then loses
`(1 − f)·G + f·S`, the flux delivered to the skin.
"""
@propagate_inbounds function compute_snow_interface_fluxes!(
        out, i, j, grid, fields,
        snow::SingleLayerSnow,
        seb::AbstractSurfaceEnergyBalance,
        soil::AbstractSoil,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    f = snow_cover_fraction(i, j, grid, fields, snow)
    E_subl = compute_snow_sublimation_flux(i, j, grid, fields, snow, atmos, constants, seb.skin_temperature)
    out.sublimation[i, j, end] = f * E_subl
    G = fields.ground_heat_flux[i, j, end]
    Q_base = compute_snow_basal_heat_flux(i, j, grid, fields, snow, soil, constants)
    S = compute_snow_surface_heat_flux(i, j, grid, fields, snow, constants)
    out.soil_heat_flux[i, j, end] = f * Q_base + (one(f) - f) * G
    out.snow_basal_heat_flux[i, j, end] = f * Q_base
    out.snow_surface_heat_flux[i, j, end] = f * S
    return nothing
end

@kernel inbounds = true function compute_snow_interface_fluxes_kernel!(out, grid, fields, snow::SingleLayerSnow, args...)
    i, j = @index(Global, NTuple)
    compute_snow_interface_fluxes!(out, i, j, grid, fields, snow, args...)
end
