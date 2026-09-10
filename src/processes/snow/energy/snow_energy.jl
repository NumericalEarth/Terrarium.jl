"""
    $TYPEDSIGNATURES

Advected heat flux [W/m²] carried into the snowpack by precipitation, relative to liquid water at 0 °C
(the `U = 0` reference of the `FreeWater` enthalpy closure). Fresh snow `P_s` arrives as ice, which sits
`L_sl` below the liquid reference, plus sensible heat for `T_air < 0`; rain-on-snow `R_on_snow` arrives as
liquid carrying only its sensible heat for `T_air > 0`. The latent heat released when rain refreezes in a
cold pack is captured implicitly by the enthalpy closure, so it is *not* added here (adding `L_sl` would
double-count relative to the liquid-water reference).
"""
@inline function compute_snow_precip_heat_flux(::AbstractSnow, constants::PhysicalConstants, P_s::NF, R_on_snow::NF, T_air::NF) where {NF}
    ρ_w = constants.material.density_water
    L_sl = constants.thermodynamics.latent_heat_fusion
    cp_i = constants.thermodynamics.specific_heat_capacity_ice
    cp_w = constants.thermodynamics.specific_heat_capacity_liquid_water
    T_ref = constants.thermodynamics.temperature_reference
    Q_snow = ρ_w * P_s * (cp_i * min(T_air, T_ref) - L_sl)
    Q_rain = ρ_w * R_on_snow * cp_w * max(T_air, T_ref)
    Q_prcp = Q_snow + Q_rain
    return Q_prcp
end

"""
    $TYPEDSIGNATURES

Bulk volumetric heat capacity `C_snow` [J/m³/K] of the snowpack treated as an ice–liquid–air mixture,
given the bulk snow density `ρ_snow` [kg/m³] and the liquid water fraction `liq` ∈ [0,1] of the water
substance. The water-substance mass per unit snow volume is `ρ_snow`, of which `(1 − liq)` is ice and
`liq` is liquid; the remaining void space is dry air. The corresponding constituent volume fractions are
```math
\\begin{aligned}
θ_{ice} &= ρ_{snow}·(1 − liq)/ρ_i,\\
θ_{liq} &= ρ_{snow}·liq/ρ_w,\\
θ_{air} &= 1 − θ_{ice} − θ_{liq}
\\end{aligned}
```
and the heat capacity is the volume-weighted sum over the ice and liquid constituents,
`C_snow = cp_i·ρ_i·θ_ice + cp_w·ρ_w·θ_liq`. The air's own sensible-heat storage (≈0.1%) is neglected; note
the air is still reflected in the bulk density `ρ_snow < ρ_ice`.
"""
@propagate_inbounds function compute_snow_volumetric_heat_capacity(
        ::AbstractSnow{NF},
        constants::PhysicalConstants,
        ρ_snow::NF,
        liq::NF
    ) where {NF}
    cp_i = constants.thermodynamics.specific_heat_capacity_ice
    cp_w = constants.thermodynamics.specific_heat_capacity_liquid_water
    ρ_w = constants.material.density_water
    ρ_i = constants.material.density_ice
    # ice and liquid volume fractions per unit snow volume; the remaining void is air, whose
    # sensible-heat storage (≈0.1%) is neglected
    θ_ice = ρ_snow * (one(NF) - liq) / ρ_i
    θ_liq = ρ_snow * liq / ρ_w
    C_snow = cp_i * ρ_i * θ_ice + cp_w * ρ_w * θ_liq
    return C_snow
end

"""
    $TYPEDSIGNATURES

Depth-integrated snow energy tendency [W/m²] at grid cell `i, j` (all fluxes positive upward):
```
dŪ_snow/dt = Q_base − Q_top + Q_precip + Q_subl
```
where `Q_top`/`Q_base` are the surface/basal heat fluxes, `Q_precip` the advected precipitation heat
(see [`compute_snow_precip_heat_flux`](@ref)), and `Q_subl` an advective correction for sublimation.

The sublimation correction `Q_subl = ρ_w·L_sl·E_subl` is required because the latent heat flux
carries the full sublimation enthalpy `ρ_w·L_sg·E_subl`, whereas the mass leaving the snowpack departs as ice,
whose specific enthalpy relative to the liquid-water reference is `−L_sl`. Adding back `ρ_w·L_sl·E_subl`
leaves the snowpack with a net loss of `ρ_w·(L_sg − L_sl)·E_subl = ρ_w·L_lg·E_subl`, the vaporization enthalpy
carried by the departing vapor.

Note that no explicit *meltwater* energy term appears because meltwater drains as liquid water at 0 °C,
which is the zero-enthalpy reference (`U = 0`) of the `FreeWater` closure, so it carries no enthalpy
out of the snowpack.

The conductive/sublimation terms `Q_base - Q_top + Q_subl` are gated to zero when the pack is empty at
the *start* of the step (`W == 0`), since they are otherwise evaluated against a degenerate zero-depth
layer (`Q_top`/`Q_base` conduction targets floored at `min_conduction_thickness`, giving an arbitrarily
large flux relative to a genuinely-thin pack's tiny heat capacity). `Q_prcp` is *not* gated: it is a pure
source term from the precipitation rate alone, well-defined regardless of the existing pack depth. Gating
it too — as a previous implementation did, multiplying the whole `dUdt` including `Q_prcp` by `W > 0` —
zeroed the energy input on the very first step any snow accumulates onto bare ground (`W` is read before
that step's mass increment), leaving the freshly-fallen, physically cold snow with zero recorded
enthalpy. Under the `FreeWater` closure (`U = 0` ≡ liquid water at 0 °C) that reads as instantaneously
fully melted, triggering spurious immediate Darcy drainage regardless of how cold the air actually was.
"""
@propagate_inbounds function compute_snow_energy_tendency(
        i, j, grid, fields,
        snow::SingleLayerSnow,
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants
    )
    ρ_w = constants.material.density_water
    L_sl = constants.thermodynamics.latent_heat_fusion
    W = fields.snow_water_equivalent[i, j]
    Q_top = fields.surface_heat_flux[i, j]   # positive upward: energy leaving the snow top
    Q_base = fields.basal_heat_flux[i, j]    # positive upward: energy entering the snow base from soil
    E_subl = fields.sublimation[i, j]        # positive upward: snow mass loss to water vapor
    f_snow = snow_cover_fraction(i, j, grid, fields, snow)
    P_s = snowfall(i, j, grid, fields, atmos)
    R = rainfall(i, j, grid, fields, atmos)
    T_air = air_temperature(i, j, grid, fields, atmos)
    R_snow = f_snow * R
    Q_prcp = compute_snow_precip_heat_flux(snow, constants, P_s, R_snow, T_air)
    Q_subl = ρ_w * L_sl * E_subl # correction to account for enthalpy of lost ice
    dUdt = (Q_base - Q_top + Q_subl) * (W > 0) + Q_prcp
    return dUdt
end

# Helper functions

"""
    $TYPEDSIGNATURES

Volumetric snow internal energy `U_snow = Ū_snow/max(d_snow, d_min)` [J/m³] from the depth-integrated
energy `Ū_snow` [J/m²] and the snow depth `d_snow` [m]. The depth is floored at the minimum thermal
thickness `d_min` (see [`min_snow_conduction_thickness`](@ref)) rather than a machine-`eps` offset. This
bounds the snow temperature recovered downstream: with only an `eps` offset, any residual `Ū_snow` over
a vanishing `d_snow` gives a huge `U_snow` and hence a snow temperature far below physical bounds, which
then corrupts the (cover-fraction-blended) skin temperature and basal heat flux. Flooring at `d_min`
gives a thin snowpack a bounded effective heat capacity `C_snow·d_min`, so it stores negligible energy
and is thermally transient — the ground heat flux passes essentially unmediated to the soil
(`f_snow → 0` in the blend) while the pack still stores mass and modifies the surface albedo.
"""
@inline compute_snow_volumetric_energy(Ū_snow::NF, d_snow::NF, d_min::NF) where {NF} = Ū_snow / max(d_snow, d_min)
