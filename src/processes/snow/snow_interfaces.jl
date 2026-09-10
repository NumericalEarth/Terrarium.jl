"""
    $TYPEDSIGNATURES

Launch [`compute_snow_interface_fluxes!`](@ref) to diagnose the snow↔surface/soil coupling fluxes from the
snow state and the surface energy balance outputs: the blended soil-top heat flux (`soil_heat_flux`), the
snow-top conductive flux (`surface_heat_flux`, drives the snowpack's own energy tendency), and the snow
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
        basal_heat_flux = state.basal_heat_flux,
        surface_heat_flux = state.surface_heat_flux,
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
    field_grid = get_field_grid(grid)
    k = field_grid.Nz
    composition = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    κ_soil = compute_thermal_conductivity(energy.thermal_properties, composition)
    Δz_soil = Δzᵃᵃᶜ(i, j, k, field_grid)

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

Blended soil-top heat flux [W/m²] at grid cell `i, j`: the snow-cover-fraction-weighted combination of
the snow→soil basal conductive flux `Q_base` (see [`compute_snow_basal_heat_flux`](@ref)) and the
*explicit* bare-ground conductive flux `G` (`ground_heat_flux`, already the unblended per-bare-ground-area
quantity — see the `ImplicitSkinTemperature`-specific `compute_ground_heat_flux` in `skin_temperature.jl`),
`f_snow·Q_base + (1 − f_snow)·G`.
"""
@propagate_inbounds function compute_snow_soil_heat_flux(
        i, j, grid, fields,
        snow::SingleLayerSnow,
        constants::PhysicalConstants,
        soil::AbstractSoil
    )
    G = fields.ground_heat_flux[i, j]
    f = fields.snow_cover_fraction[i, j]
    Q_base = compute_snow_basal_heat_flux(i, j, grid, fields, snow, soil, constants)
    return f * Q_base + (one(f) - f) * G
end

"""
    $TYPEDSIGNATURES

Snow-top conductive heat flux [W/m²] at grid cell `i, j` (positive upward, i.e. energy leaving the snow
top into the skin), `S = 2κ_snow(T_snow − T_skin)/max(d_snow, d_min)` — the same unblended snow
conduction target the skin-temperature solve closes against (see
[`Terrarium.snow_thermal_interface`](@ref)). This is the per-unit-snow-area flux that drives the
snowpack's own top-of-pack energy tendency ([`compute_snow_energy_tendency`](@ref)'s `Q_top`); it replaces
a previous unweighted alias to the whole-grid-cell `ground_heat_flux`, which forced the entire snowpack
with a flux sized for the whole cell regardless of how small the snow-covered fraction actually was.
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
the blended soil-top heat flux (see [`compute_snow_soil_heat_flux`](@ref)), the snow-top conductive flux
(see [`compute_snow_surface_heat_flux`](@ref)), and the snow surface sublimation rate (see
[`compute_snow_sublimation_flux`](@ref), the same snow-fraction vapor flux the surface energy balance uses
for the latent-flux partition).
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
    out.sublimation[i, j, 1] = f * E_subl
    out.basal_heat_flux[i, j, 1] = compute_snow_soil_heat_flux(i, j, grid, fields, snow, constants, soil)
    out.surface_heat_flux[i, j, 1] = compute_snow_surface_heat_flux(i, j, grid, fields, snow, constants)
    return nothing
end

@kernel inbounds = true function compute_snow_interface_fluxes_kernel!(out, grid, fields, snow::SingleLayerSnow, args...)
    i, j = @index(Global, NTuple)
    compute_snow_interface_fluxes!(out, i, j, grid, fields, snow, args...)
end
