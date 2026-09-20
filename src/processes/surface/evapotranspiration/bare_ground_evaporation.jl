"""
    BareGroundEvaporation{NF, GR} <: AbstractEvapotranspiration

Evaporation scheme for bare ground that calculates the humidity flux as

```math
E = \\beta \\frac{\\Delta q}{r_a}
```
where `Δq` is the specific humidity difference, `rₐ` is aerodynamic resistance,
and `β` is an evaporation limiting factor.
"""
struct BareGroundEvaporation{NF, GR <: AbstractGroundEvaporationResistanceFactor} <: AbstractEvapotranspiration{NF}
    "Parameterization for ground resistance to evaporation/sublimation"
    ground_resistance::GR
end

BareGroundEvaporation(
    ::Type{NF};
    ground_resistance::GR = SoilMoistureResistanceFactor(NF)
) where {NF, GR} = BareGroundEvaporation{NF, GR}(ground_resistance)

@propagate_inbounds ground_evapotranspiration_flux(i, j, grid, fields, evaporation::BareGroundEvaporation, args...) = fields.evaporation_ground[i, j]

""" $TYPEDSIGNATURES """
@inline ground_evaporation_conductance(::BareGroundEvaporation, β, rₐ) = β / rₐ

"""
    $TYPEDSIGNATURES

Evaluate the bare-ground, kinematic surface humidity flux [m/s] from the current skin temperature and
evaporation conductance in `fields`.
"""
@propagate_inbounds function compute_surface_humidity_flux(
        i, j, grid, fields,
        evaporation::BareGroundEvaporation,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    )
    Qh_gnd = compute_surface_humidity_fluxes(i, j, grid, fields, evaporation, constants, atmos)
    return Qh_gnd
end

# Top-level interface methods

variables(::BareGroundEvaporation) = (
    # Skin-driven vapor conductance β/rₐ (independent of skin temperature; held fixed during the SEB solve)
    auxiliary(:ground_evaporation_conductance, XY(), units = u"m/s", desc = "Ground evaporation vapor conductance"),
    auxiliary(:evaporation_ground, XY(), units = u"m/s", desc = "Ground evaporation flux in meters liquid water height"),
    input(:skin_temperature, XY(), units = u"°C", desc = "Skin temperature of the surface"),
)

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        evaporation::BareGroundEvaporation,
        ::NoCanopyInterception,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::Optional{AbstractSoil} = nothing,
        snow::Optional{AbstractSnow} = nothing,
    )
    out = auxiliary_fields(state, evaporation)
    # merge the snow cover fraction so the ground evaporation can be scaled by the snow-free fraction
    fields = merge(get_fields(state, evaporation, atmos, soil; except = out), get_fields(state, snow))
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, evaporation, constants, atmos, soil, snow)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the skin-driven ground evaporation vapor conductance β/rₐ at grid cell `i, j` for the
given bare-ground `evaporation` scheme. This conductance is independent of the skin temperature.
"""
@propagate_inbounds function compute_evapotranspiration_conductances(
        i, j, grid, fields,
        evaporation::BareGroundEvaporation,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::Optional{AbstractSoil} = nothing
    )
    rₐ = aerodynamic_resistance(i, j, grid, fields, atmos) # aerodynamic resistance
    β = ground_evaporation_resistance_factor(i, j, grid, fields, evaporation.ground_resistance, soil)
    g_gnd = ground_evaporation_conductance(evaporation, β, rₐ)
    return g_gnd
end

"""
    $TYPEDSIGNATURES

Compute and store the skin-driven ground evaporation vapor conductance on `grid` for the given
bare-ground `evaporation` scheme.
"""
@propagate_inbounds function compute_evapotranspiration_conductances!(
        out, i, j, grid, fields,
        evaporation::BareGroundEvaporation,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::Optional{AbstractSoil} = nothing
    )
    g_gnd = compute_evapotranspiration_conductances(i, j, grid, fields, evaporation, constants, atmos, soil)
    out.ground_evaporation_conductance[i, j, 1] = g_gnd
    return out
end

@propagate_inbounds function compute_surface_humidity_fluxes(
        i, j, grid, fields,
        evaporation::BareGroundEvaporation,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    )
    Ts = fields.skin_temperature[i, j]
    g = fields.ground_evaporation_conductance[i, j]
    Δq = compute_specific_humidity_difference(i, j, grid, fields, atmos, constants, Ts)
    Qh_gnd = humidity_flux(evaporation, Δq, g)
    return Qh_gnd
end

@propagate_inbounds function compute_evapotranspiration_fluxes!(
        out, i, j, grid, fields,
        evaporation::BareGroundEvaporation{NF},
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        snow::Optional{AbstractSnow} = nothing,
    ) where {NF}
    # Compute raw evaporation flux
    Qh_gnd = compute_surface_humidity_fluxes(i, j, grid, fields, evaporation, constants, atmos)
    # Ground evaporation occurs only over the snow-free fraction (1 − f_snow); the snow-covered fraction
    # sublimates instead (see `compute_snow_sublimation_flux`). No scaling without snow (f_snow = 0).
    f_snow = snow_cover_fraction(i, j, grid, fields, snow)
    # Get air/water densities
    ρ_a = air_density(i, j, grid, fields, atmos, constants)
    ρ_w = constants.material.density_water
    # Scale Qh_gnd by snow-free fraction and and scale by air-water density ratio to get evaporative flux E_gnd
    E_gnd = (NF(1) - f_snow) * Qh_gnd * ρ_a / ρ_w
    out.evaporation_ground[i, j, 1] = E_gnd
    return out
end
# Kernels


@kernel inbounds = true function compute_auxiliary_kernel!(
        out, grid, fields,
        evapotranspiration::BareGroundEvaporation,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::Optional{AbstractSoil} = nothing,
        snow::Optional{AbstractSnow} = nothing,
    )
    i, j = @index(Global, NTuple)

    # First compute conductances
    compute_evapotranspiration_conductances!(out, i, j, grid, fields, evapotranspiration, constants, atmos, soil)
    # TODO: Annoyingly, we need to explicitly add these to `fields`; need a better solution to this problem
    conductances = (ground_evaporation_conductance = out.ground_evaporation_conductance,)
    fields = merge(fields, conductances)
    # Compute ET fluxes from stored conductances; `snow` scales ground evaporation by the snow-free fraction
    compute_evapotranspiration_fluxes!(out, i, j, grid, fields, evapotranspiration, constants, atmos, snow)
end
