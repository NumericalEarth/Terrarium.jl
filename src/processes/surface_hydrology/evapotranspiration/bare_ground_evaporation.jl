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

@propagate_inbounds surface_humidity_flux(i, j, grid, fields, evaporation::BareGroundEvaporation, args...) = fields.evaporation_ground[i, j]

# Top-level interface methods

variables(::BareGroundEvaporation) = (
    auxiliary(:evaporation_ground, XY(), units = u"m/s", desc = "Ground evaporation contribution to surface humidity flux"),
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
        args...
    )
    out = auxiliary_fields(state, evaporation)
    fields = get_fields(state, evaporation, atmos, soil; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, evaporation, constants, atmos, soil)
    return nothing
end

# Kernel functions

@propagate_inbounds function compute_evapotranspiration!(
        out, i, j, grid, fields,
        evaporation::BareGroundEvaporation,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::Optional{AbstractSoil} = nothing
    )
    Ts = fields.skin_temperature[i, j]
    rₐ = aerodynamic_resistance(i, j, grid, fields, atmos) # aerodynamic resistance
    β = ground_evaporation_resistance_factor(i, j, grid, fields, evaporation.ground_resistance, soil)
    Δq = compute_specific_humidity_difference(i, j, grid, fields, atmos, constants, Ts)
    # Calculate water evaporation flux in m/s (positive upwards)
    E_g = β * Δq / rₐ
    return out.evaporation_ground[i, j, 1] = E_g
end
