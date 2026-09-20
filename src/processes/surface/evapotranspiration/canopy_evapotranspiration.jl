"""
    $TYPEDEF

Canopy evapotranspiration scheme from PALADYN ([willeitPALADYNV10Comprehensive2016; Eq. (5)](@cite)) 
that includes a canopy evaporation term based on the saturation fraction of canopy water defined by the
canopy hydrology scheme.

```math
E_{\\text{ground}} = \\beta \\frac{\\Delta q}{r_a + r_e}
```
```math
E_{\\text{can}} = f_{\\text{can}} \\frac{\\Delta q}{r_a}
```
```math
T_{\\text{can}} = \\frac{\\Delta q}{r_a + r_s}
```

Properties:
$FIELDS

# References
* [willeitPALADYNV10Comprehensive2016](@cite) Willeit and Ganopolski, Geoscientific Model Development (2016)
"""
@parameterized @kwdef struct PALADYNCanopyEvapotranspiration{NF, GR <: AbstractGroundEvaporationResistanceFactor} <: AbstractEvapotranspiration{NF}
    "Drag coefficient for the transfer of heat and water between the ground and canopy"
    @param C_can::NF (bounds = Positive,)

    "Parameterization for ground resistance to evaporation/sublimation"
    @component ground_resistance::GR
end

function PALADYNCanopyEvapotranspiration(
        ::Type{NF};
        C_can = NF(0.006),
        ground_resistance = ConstantEvaporationResistanceFactor(typeof(C_can))
    ) where {NF}
    return PALADYNCanopyEvapotranspiration{NF, typeof(ground_resistance)}(C_can, ground_resistance)
end

"""
    $TYPEDSIGNATURES

Compute the ground evaporation conductance from the given resistance factor β and aerodynamic resistances.
"""
@inline function ground_evaporation_conductance(::PALADYNCanopyEvapotranspiration, β, rₐ, rₐ_can)
    g_gnd = β / (rₐ + rₐ_can)
    return g_gnd
end

"""
    $TYPEDSIGNATURES

Compute the transpiration vapor conductance [m/s] from aerodynamic resistance `rₐ` and stomatal
conductance `g_stm`. The transpiration flux is this conductance times the humidity gradient.
"""
@inline function transpiration_conductance(::PALADYNCanopyEvapotranspiration{NF}, rₐ, g_stm) where {NF}
    # From the general formula for conductances in series:
    # g_total = g_1 * g_2 / (g_1 + g_2)
    # substituting g_1 = 1 / r_1
    g_trp = g_stm / (1 + g_stm * rₐ)
    return g_trp
end

"""
    $TYPEDSIGNATURES

Compute the canopy evaporation vapor conductance [m/s] from the current canopy saturation fraction `f_can`
and aerodynamic resistance `rₐ`.
"""
@inline function canopy_evaporation_conductance(::PALADYNCanopyEvapotranspiration, f_can, rₐ)
    g_can = f_can / rₐ
    return g_can
end

# Top-level interface methods

variables(::PALADYNCanopyEvapotranspiration{NF}) where {NF} = (
    # Skin-driven vapor conductances (independent of skin temperature; held fixed during the SEB solve)
    auxiliary(:ground_evaporation_conductance, XY(); units = u"m/s", desc = "Ground evaporation vapor conductance"),
    auxiliary(:canopy_evaporation_conductance, XY(); units = u"m/s", desc = "Canopy evaporation vapor conductance"),
    auxiliary(:transpiration_conductance, XY(); units = u"m/s", desc = "Transpiration vapor conductance"),
    # Partitioned humidity fluxes (skin-driven terms refreshed from the converged skin temperature by the finalize pass)
    auxiliary(:evaporation_canopy, XY(); desc = "Canopy evaporation flux in meters liquid water height", units = u"m/s"),
    auxiliary(:evaporation_ground, XY(), units = u"m/s", desc = "Ground evaporation flux in meters liquid water height"),
    auxiliary(:transpiration, XY(), units = u"m/s", desc = "Transpiration evaporation flux in meters liquid water height"),
    input(:skin_temperature, XY(); units = u"°C", desc = "Skin temperature"),
    input(:ground_temperature, XY(); default = NF(1), units = u"°C", desc = "Ground surface temperature"),
)

@propagate_inbounds function ground_evapotranspiration_flux(i, j, grid, fields, ::PALADYNCanopyEvapotranspiration)
    E_gnd = fields.evaporation_ground[i, j]
    E_trp = fields.transpiration[i, j]
    ET = E_gnd + E_trp
    return ET
end

"""
    $TYPEDSIGNATURES

Compute the kinematic surface humidity flux [m/s] from the current skin temperature and conductances in `fields`.
"""
@propagate_inbounds function compute_surface_humidity_flux(
        i, j, grid, fields,
        evtr::PALADYNCanopyEvapotranspiration,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        args...
    )
    Qh_gnd, Qh_trp, Qh_can = compute_surface_humidity_fluxes(i, j, grid, fields, evtr, constants, atmos, args...)
    Qh = Qh_gnd + Qh_trp + Qh_can
    return Qh
end

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        evapotranspiration::PALADYNCanopyEvapotranspiration,
        interception::AbstractCanopyInterception,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::AbstractSoil,
        vegetation::AbstractVegetation,
        args...
    )
    out = auxiliary_fields(state, evapotranspiration)
    fields = get_fields(state, evapotranspiration, interception, atmos, soil, vegetation; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, evapotranspiration, interception, constants, atmos, soil, vegetation)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the skin-driven vapor conductances (`ground_evaporation_conductance`,
`canopy_evaporation_conductance`, and `transpiration_conductance`) at grid cell `i, j` for the
given scheme `evapotranspiration` and process dependencies. These conductances are independent
of the skin temperature.
"""
@propagate_inbounds function compute_evapotranspiration_conductances(
        i, j, grid, fields,
        evapotranspiration::PALADYNCanopyEvapotranspiration,
        interception::AbstractCanopyInterception,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::AbstractSoil,
        vegetation::AbstractVegetation,
        args...
    )
    # Get inputs
    g_stm = fields.canopy_water_conductance[i, j] # stomatal conductance (assumed to be defined by vegetation)

    # Compute aerodynamic resistances and resistance factors
    rₐ = aerodynamic_resistance(i, j, grid, fields, atmos) # aerodynamic resistance
    rₐ_can = aerodynamic_resistance(i, j, grid, fields, atmos, evapotranspiration, vegetation) # aerodynamic resistance between ground and canopy
    f_can = saturation_canopy_water(i, j, grid, fields, interception)
    β = ground_evaporation_resistance_factor(i, j, grid, fields, evapotranspiration.ground_resistance, soil)

    # Compute skin-driven vapor conductances (independent of skin temperature)
    g_gnd = ground_evaporation_conductance(evapotranspiration, β, rₐ, rₐ_can)
    g_can = canopy_evaporation_conductance(evapotranspiration, f_can, rₐ)
    g_trp = transpiration_conductance(evapotranspiration, rₐ, g_stm)
    return g_gnd, g_trp, g_can
end

"""
    $TYPEDSIGNATURES

Compute and store the skin-driven vapor conductances on `grid` for the given scheme
`evapotranspiration` and process dependencies.
"""
@propagate_inbounds function compute_evapotranspiration_conductances!(
        out, i, j, grid, fields,
        evapotranspiration::PALADYNCanopyEvapotranspiration,
        interception::AbstractCanopyInterception,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::AbstractSoil,
        vegetation::AbstractVegetation,
        args...
    )
    # Compute conductances
    g_gnd, g_trp, g_can = compute_evapotranspiration_conductances(i, j, grid, fields, evapotranspiration, interception, constants, atmos, soil, vegetation, args...)

    # Store skin-driven vapor conductances in corresponding output Fields
    out.ground_evaporation_conductance[i, j, 1] = g_gnd
    out.canopy_evaporation_conductance[i, j, 1] = g_can
    out.transpiration_conductance[i, j, 1] = g_trp
    return out
end

"""
    $TYPEDEF

Compute unscaled `transpiration`, `evaporation_ground`, and `evaporation_canopy` fluxes on `grid`.
Following the implementation of PALADYN, `Qh_can` is clamped to be strictly positive (no canopy dew formation).
"""
@propagate_inbounds function compute_surface_humidity_fluxes(
        i, j, grid, fields,
        evapotranspiration::PALADYNCanopyEvapotranspiration{NF},
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere
    ) where {NF}
    # Get inputs
    Ts = fields.skin_temperature[i, j] # skin temperature (top of canopy)
    Tg = fields.ground_temperature[i, j] # ground temperature (top snow/soil layer)
    g_gnd = fields.ground_evaporation_conductance[i, j] # ground evaporation conductance
    g_can = fields.canopy_evaporation_conductance[i, j] # canopy evaporation conductance
    g_trp = fields.transpiration_conductance[i, j] # canopy transpiration conductance

    # Compute specific humidity differences
    Δqs = compute_specific_humidity_difference(i, j, grid, fields, atmos, constants, Ts) # humidity difference between canopy and atmosphere
    Δqg = compute_specific_humidity_difference(i, j, grid, fields, atmos, constants, Tg) # humidity difference between ground and canopy

    # Compute the kinematic surface humidity fluxes for each component; negative values of Δq result in deposition
    Qh_gnd = humidity_flux(evapotranspiration, Δqg, g_gnd)
    Qh_can = humidity_flux(evapotranspiration, Δqs, g_can)
    # We clamp transpiration to be nonnegative since presently foliar/stomatal uptake is not represented
    Qh_trp = max(humidity_flux(evapotranspiration, Δqs, g_trp), zero(NF))
    return Qh_gnd, Qh_trp, Qh_can
end

"""
    $TYPEDEF

Compute `transpiration`, `evaporation_ground`, and `evaporation_canopy` fluxes on `grid`
for the given scheme `evapotranspiration` and process dependencies.
"""
@propagate_inbounds function compute_evapotranspiration_fluxes!(
        out, i, j, grid, fields,
        evapotranspiration::PALADYNCanopyEvapotranspiration{NF},
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        snow::Optional{AbstractSnow} = nothing,
        args...
    ) where {NF}
    # Compute humidity fluxes
    Qh_gnd, Qh_trp, Qh_can = compute_surface_humidity_fluxes(i, j, grid, fields, evapotranspiration, constants, atmos)

    # Get air/water densities
    ρ_a = air_density(i, j, grid, fields, atmos, constants)
    ρ_w = constants.material.density_water
    r = ρ_a / ρ_w # ratio of air to water density [-]

    # Rescale by snow-covered fraction (if applicable) and convert to liquid water flux
    f_snow = snow_cover_fraction(i, j, grid, fields, snow)
    f_bare = NF(1) - f_snow
    out.evaporation_ground[i, j, 1] = f_bare * Qh_gnd * r
    out.transpiration[i, j, 1] = f_bare * Qh_trp * r
    out.evaporation_canopy[i, j, 1] = f_bare * Qh_can * r
    return out
end

"""
    $TYPEDSIGNATURES

Compute the aerodynamic resistance between the ground and canopy as a function of LAI and SAI.
"""
@inline function aerodynamic_resistance(
        i, j, grid, fields,
        atmos::AbstractAtmosphere,
        evapotranspiration::PALADYNCanopyEvapotranspiration{NF},
        ::AbstractVegetation # included just to make explicit the dependence on vegetation fields
    ) where {NF}
    @inbounds let
        LAI = max(fields.leaf_area_index[i, j], zero(NF))
        SAI = max(fields.stem_area_index[i, j], zero(NF))
        Vₐ = windspeed(i, j, grid, fields, atmos)
        C = evapotranspiration.C_can  # drag coefficient for the canopy
        rₐ_can = (1 - exp(-LAI - SAI)) / (C * Vₐ)
        return rₐ_can
    end
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(
        out, grid, fields,
        evapotranspiration::PALADYNCanopyEvapotranspiration,
        interception::AbstractCanopyInterception,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        soil::AbstractSoil,
        vegetation::AbstractVegetation,
        snow::Optional{AbstractSnow} = nothing,
        args...
    )
    i, j = @index(Global, NTuple)
    # First compute conductances
    compute_evapotranspiration_conductances!(out, i, j, grid, fields, evapotranspiration, interception, constants, atmos, soil, vegetation, args...)
    # TODO: Annoyingly, we need to explicitly add these to `fields`; need a better solution to this problem
    conductances = (
        ground_evaporation_conductance = out.ground_evaporation_conductance,
        transpiration_conductance = out.transpiration_conductance,
        canopy_evaporation_conductance = out.canopy_evaporation_conductance,
    )
    fields = merge(fields, conductances)
    # Compute ET fluxes from stored conductances
    compute_evapotranspiration_fluxes!(out, i, j, grid, fields, evapotranspiration, constants, atmos, snow)
end
