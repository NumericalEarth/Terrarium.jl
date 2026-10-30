# Snow mass balance, meltwater outflow, and the depth-integrated energy/mass tendencies.

"""
    $TYPEDSIGNATURES

Snow-surface sublimation rate [m/s SWE] at grid cell `i, j`. The snow surface is treated as saturated:
a bulk-aerodynamic vapor flux `Δq/rₐ` evaluated at the skin temperature, with `Δq` taken over ice for
a sub-freezing surface (the saturation humidity already dispatches over ice for `T ≤ 0` — see [`saturation_specific_humidity_vapor`](@ref)).
The water-vapor mass flux `ρₐ·Δq/rₐ` is converted to a snow-water-equivalent rate via `ρ_w`. Zero without
snow (i.e. when `snow === nothing`).
"""
@propagate_inbounds compute_snow_sublimation_flux(i, j, grid, fields, ::Nothing, atmos, constants, skinT) = zero(eltype(grid))
@propagate_inbounds function compute_snow_sublimation_flux(
        i, j, grid, fields,
        snow::AbstractSnow,
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants,
        skinT::AbstractSkinTemperature
    )
    Tₛ = skin_temperature(i, j, grid, fields, skinT)
    rₐ = aerodynamic_resistance(i, j, grid, fields, atmos)
    Δq = compute_specific_humidity_difference(i, j, grid, fields, atmos, constants, Tₛ) # over ice for Tₛ ≤ 0
    ρ_a = air_density(i, j, grid, fields, atmos, constants)
    ρ_w = constants.material.density_water
    # saturated vapor mass flux converted to a snow-water-equivalent rate, area-weighted by the snow-covered
    # fraction `f` to give the grid-cell-mean sublimation (W_snow and Ū_snow are grid-cell means)
    E_subl = ρ_a * (Δq / rₐ) / ρ_w
    return E_subl
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Snow water equivalent (SWE) tendency (m/s) at grid cell `i, j`:
```
dW_snow/dt = S + R_snow − M − E_subl
```
where `S` is snowfall, `R_snow = f_snow · rainfall` the rain intercepted by the snow-covered fraction,
`M` the Darcy meltwater outflow (see [`snow_meltwater_flux`](@ref)), and `E_subl` the sublimation rate.
`M` is omitted under implicit integration of `snow_water_equivalent`, where the implicit step drains the
meltwater.
"""
@propagate_inbounds function compute_snow_water_tendency(
        i, j, grid, fields,
        snow::SingleLayerSnow{NF},
        atmos::AbstractAtmosphere,
        water_timestepping::Timestepping = Explicit()
    ) where {NF}
    f_snow = snow_cover_fraction(i, j, grid, fields, snow)
    # TODO: account for canopy interception
    S = snowfall(i, j, grid, fields, atmos)
    M = explicit_meltwater_flux(i, j, grid, fields, snow, water_timestepping)
    R = rainfall(i, j, grid, fields, atmos)
    R_snow = f_snow * R
    E_subl = fields.sublimation[i, j]
    W = fields.snow_water_equivalent[i, j]
    @assert_kernel W > zero(NF) || E_subl ≈ zero(NF)
    @assert_kernel W > zero(NF) || M ≈ zero(NF)
    dWdt = S + R_snow - M - E_subl
    return dWdt
end

# Meltwater outflow in the explicit tendency: drained by the implicit step under implicit integration.
@propagate_inbounds explicit_meltwater_flux(i, j, grid, fields, snow::AbstractSnow, ::Explicit) = snow_meltwater_flux(i, j, grid, fields, snow)
@inline explicit_meltwater_flux(i, j, grid, fields, snow::AbstractSnow{NF}, ::Implicit) where {NF} = zero(NF)
