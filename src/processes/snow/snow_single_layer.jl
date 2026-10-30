"""
    $TYPEDEF

Simple single-layer snow scheme (loosely based on the Utah Energy Balance model,
[tarbotonSpatiallyDistributedEnergy1994](@cite)). The snowpack is represented as a single lumped layer
with a bulk density `ρ_snow` supplied by a snow-density scheme (constant by default), from which the thermal
properties follow. The prognostic state is the depth-integrated (column) internal energy `snow_energy`
`Ū_snow` [J/m²] and the `snow_water_equivalent` `W_snow` [m]; snow depth, cover fraction, and thermal conductivity
are diagnosed from these and the bulk density.

Properties:
$FIELDS

# References
* [tarbotonSpatiallyDistributedEnergy1994](@cite) Tarboton, Chowdhury and Jackson (1994)
"""
@parameterized struct SingleLayerSnow{NF, Cover, Density, Conductivity, Hydraulics, Albedo, Closure} <: AbstractSnow{NF}
    "Snow areal coverage parameterization"
    @component cover::Cover

    "Bulk snow density parameterization"
    @component density::Density

    "Snow thermal conductivity parameterization"
    @component thermal_conductivity::Conductivity

    "Snow hydraulic properties"
    @component hydraulic_properties::Hydraulics

    "Snow albedo parameterization"
    @component albedo::Albedo

    "Snow energy-temperature closure"
    @component closure::Closure
end

function SingleLayerSnow(
        ::Type{NF};
        cover = FractionalSnowCover(NF),
        density = ConstantSnowDensity(NF),
        thermal_conductivity = PowerLawSnowThermalConductivity(NF),
        hydraulic_properties = ConstantSnowHydraulics(NF),
        albedo = ConstantSnowAlbedo(NF),
        closure = SnowEnergyTemperatureClosure(NF),
    ) where {NF}
    # No field is directly typed `NF` (the minimum conduction thickness is now derived from the grid
    # rather than stored, see `min_snow_conduction_thickness`), so `NF` is supplied explicitly here and
    # carried only via the `AbstractSnow{NF}` supertype bound, mirroring `LandModel`/`SnowModel`.
    return SingleLayerSnow{NF, typeof(cover), typeof(density), typeof(thermal_conductivity), typeof(hydraulic_properties), typeof(albedo), typeof(closure)}(
        cover, density, thermal_conductivity, hydraulic_properties, albedo, closure
    )
end

"""
    $TYPEDSIGNATURES

Minimum snow thickness (m) used to regularize the snow thermodynamics (conductive fluxes and the
volumetric energy/temperature closure) as the pack thins toward zero: half the thickness of the
uppermost soil grid cell, so the floor scales with the grid resolution rather than a fixed constant.
"""
@propagate_inbounds function min_snow_conduction_thickness(i, j, grid, fields, ::SingleLayerSnow)
    ground_grid = ground_domain(grid)
    return Δzᵃᵃᶜ(i, j, ground_grid.Nz, ground_grid) / 2
end

# Top-level interface

"""
    $TYPEDSIGNATURES

Snow layer depth `d_snow = W_snow·ρ_w/ρ_snow` [m], converting the water-equivalent depth `W_snow` [m] to the physical
snow depth using the water density `ρ_w` and the bulk snow density `ρ_snow`.
"""
@inline compute_snow_depth(::AbstractSnow, W_snow::NF, ρ_snow::NF, ρ_w::NF) where {NF} = max(W_snow, zero(NF)) * ρ_w / ρ_snow

"""
    $TYPEDSIGNATURES

Bulk snow density `ρ_snow` [kg/m³] of the snowpack, delegating to the process's density scheme.
"""
@inline snow_density(snow::SingleLayerSnow) = snow_density(snow.density)

"""
    $TYPEDSIGNATURES

Bulk snow thermal conductivity `κ_snow` [W/m/K], delegating to the process's thermal conductivity scheme
with the bulk density `ρ_snow`.
"""
@inline compute_thermal_conductivity(snow::SingleLayerSnow, constants::MaterialConstants, ρ_snow) =
    compute_thermal_conductivity(snow.thermal_conductivity, constants, ρ_snow)

"""$TYPEDSIGNATURES"""
@inline get_closure(snow::SingleLayerSnow) = snow.closure

# Process methods

variables(snow::SingleLayerSnow) = (
    prognostic(:snow_energy, Snow(XY()); closure = get_closure(snow), units = u"J/m^2", desc = "Depth-integrated (column) internal energy of the snowpack relative to water at 0°C"),
    prognostic(:snow_water_equivalent, Snow(XY()); units = u"m", desc = "Snow water equivalent (ice + retained liquid)"),
    auxiliary(:snow_depth, Snow(XY()); units = u"m", desc = "Snow layer depth"),
    auxiliary(:snow_cover_fraction, Snow(XY()); bounds = UnitInterval, desc = "Sub-grid snow-covered area fraction"),
    auxiliary(:meltwater_flux, Snow(XY()); units = u"m/s", desc = "Meltwater outflow from the snowpack base (SWE)"),
    input(:surface_heat_flux, Snow(Top()); units = u"W/m^2", desc = "Net heat flux at the snow surface (positive upward)"),
    input(:basal_heat_flux, Snow(Bottom()); units = u"W/m^2", desc = "Conductive heat flux at the snow base (positive upward, soil → snow)"),
    input(:sublimation, Snow(Top()); units = u"m/s", desc = "Sublimation/evaporation rate from the snow surface (SWE)"),
)

"""
    $TYPEDSIGNATURES

Diagnose the snow geometric properties (`snow_depth`, `snow_cover_fraction`) from the current snow water
equivalent and bulk density, and the `meltwater_flux` from the liquid fraction if `snow_water_equivalent` is
integrated explicitly by `timestepper`. Under implicit integration, `meltwater_flux` holds the drainage of
the last implicit step instead (see [`implicit_update!`](@ref)).
"""
function compute_auxiliary!(
        state, grid,
        snow::SingleLayerSnow,
        constants::PhysicalConstants,
        timestepper::Optional{AbstractTimeStepper} = nothing
    )
    water_timestepping = timestepping(timestepper, state, Val(:snow_water_equivalent))
    out = auxiliary_fields(state, snow)
    fields = get_fields(state, snow; except = out)
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, snow, constants, water_timestepping)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the snow water-equivalent and depth-integrated energy tendencies (see [`compute_snow_tendencies!`](@ref)),
driven by the atmospheric precipitation inputs and the prescribed surface/basal heat fluxes and
sublimation. The meltwater outflow is omitted if `snow_water_equivalent` is integrated implicitly by
`timestepper`.
"""
function compute_tendencies!(
        state, grid,
        snow::SingleLayerSnow,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        timestepper::Optional{AbstractTimeStepper} = nothing
    )
    water_timestepping = timestepping(timestepper, state, Val(:snow_water_equivalent))
    tendencies = tendency_fields(state, snow)
    fields = get_fields(state, snow, atmos)
    launch!(grid, XY, compute_tendencies_kernel!, tendencies, fields, snow, atmos, constants, water_timestepping)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Accumulate the snow water-equivalent and depth-integrated energy tendencies at grid cell `i, j` from the
mass and energy balances (see [`compute_snow_water_tendency`](@ref) and [`compute_snow_energy_tendency`](@ref)).
"""
@propagate_inbounds function compute_snow_tendencies!(
        tendencies, i, j, grid, fields,
        snow::SingleLayerSnow,
        atmos::AbstractAtmosphere,
        constants::PhysicalConstants,
        water_timestepping::Timestepping = Explicit()
    )
    tendencies.snow_water_equivalent[i, j, end] += compute_snow_water_tendency(i, j, grid, fields, snow, atmos, water_timestepping)
    tendencies.snow_energy[i, j, end] += compute_snow_energy_tendency(i, j, grid, fields, snow, atmos, constants)
    return nothing
end
"""
    $TYPEDSIGNATURES

Compute the snow depth and cover fraction at grid cell `i, j`, and the meltwater outflow if
`snow_water_equivalent` is integrated explicitly (see [`compute_meltwater_flux!`](@ref)).
"""
@propagate_inbounds function compute_snow_properties!(
        out, i, j, grid, fields,
        snow::SingleLayerSnow,
        constants::PhysicalConstants,
        water_timestepping::Timestepping = Explicit()
    )
    W_snow = fields.snow_water_equivalent[i, j]
    ρ_w = constants.material.density_water
    ρ_snow = compute_snow_density(i, j, grid, fields, snow.density)
    out.snow_depth[i, j, end] = compute_snow_depth(snow, W_snow, ρ_snow, ρ_w)
    out.snow_cover_fraction[i, j, end] = compute_snow_cover_fraction(snow.cover, W_snow)
    compute_meltwater_flux!(out, i, j, grid, fields, snow, water_timestepping)
    return nothing
end

"""
    $TYPEDSIGNATURES

Store the Darcy-type meltwater outflow `M_r` (m/s SWE) at grid cell `i, j` in `meltwater_flux`, evaluated
at the diagnosed liquid water fraction (see [`compute_meltwater_outflow`](@ref)). A no-op under implicit
integration of `snow_water_equivalent`, where the implicit step sets it.
"""
@propagate_inbounds function compute_meltwater_flux!(out, i, j, grid, fields, snow::SingleLayerSnow, ::Explicit)
    θ_liq = liquid_water_fraction(i, j, grid, fields, snow)
    out.meltwater_flux[i, j, end] = compute_meltwater_outflow(snow.hydraulic_properties, θ_liq)
    return nothing
end

@inline compute_meltwater_flux!(out, i, j, grid, fields, snow::SingleLayerSnow, ::Implicit) = nothing

"""
    $TYPEDSIGNATURES

Snow meltwater outflow `M_r` (m/s SWE) at grid cell `i, j`, the `meltwater_flux` auxiliary: the outflow at
the current liquid fraction under explicit integration (see [`compute_meltwater_flux!`](@ref)), and the
drainage of the last implicit step otherwise.
"""
@propagate_inbounds snow_meltwater_flux(i, j, grid, fields, snow::SingleLayerSnow) = fields.meltwater_flux[i, j]

## Field-level accessors

@propagate_inbounds snow_water_equivalent(i, j, grid, fields, ::SingleLayerSnow) = fields.snow_water_equivalent[i, j]

@propagate_inbounds snow_energy(i, j, grid, fields, ::SingleLayerSnow) = fields.snow_energy[i, j]

@propagate_inbounds snow_depth(i, j, grid, fields, ::SingleLayerSnow) = fields.snow_depth[i, j]

@propagate_inbounds snow_cover_fraction(i, j, grid, fields, ::SingleLayerSnow) = fields.snow_cover_fraction[i, j]

@propagate_inbounds snow_temperature(i, j, grid, fields, ::SingleLayerSnow) = fields.snow_temperature[i, j]

@propagate_inbounds liquid_water_fraction(i, j, grid, fields, ::SingleLayerSnow) = fields.snow_liquid_fraction[i, j]

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, snow::AbstractSnow, args...)
    i, j = @index(Global, NTuple)
    compute_snow_properties!(out, i, j, grid, fields, snow, args...)
end

@kernel inbounds = true function compute_tendencies_kernel!(tendencies, grid, fields, snow::SingleLayerSnow, args...)
    i, j = @index(Global, NTuple)
    compute_snow_tendencies!(tendencies, i, j, grid, fields, snow, args...)
end
