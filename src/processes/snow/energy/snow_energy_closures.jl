# Snow energy closure

"""
    $TYPEDEF

Energy–temperature closure for snow volumes. For [`SingleLayerSnow`](@ref), the depth-averaged snow
temperature `T_snow` (°C) and liquid water fraction `θ_liq` are recovered from the depth-integrated internal energy
`Ū_snow` (J/m²) using the medium-agnostic `FreeWater` enthalpy relations, treating the bulk snowpack as an ice-water-air
mixture. The internal energy is defined as:
```math
U(T) = T_{\text{snow}} \\times C(T) - \\rho_{snow} L_{sl} (1 - F(T))
```
with `C(T)` the temperature-dependent volumetric heat capacity of the snowpack (J/m³/K), 
`ρ_snow L_sl = ρ_w L_sl θ` the volumetric latent heat of fusion (J/m³), and `F(T) = θ_liq/θ` 
the fraction of the total (liquid water + ice) volumetric water content that is liquid. 

"""
struct SnowEnergyTemperatureClosure{NF} <: AbstractEnergyClosure end

SnowEnergyTemperatureClosure(::Type{NF}) where {NF} = SnowEnergyTemperatureClosure{NF}()

variables(::SnowEnergyTemperatureClosure) = (
    auxiliary(:snow_temperature, XY(), units = u"°C", desc = "Depth-averaged snow temperature in °C (≤ 0)"),
    auxiliary(:snow_liquid_fraction, XY(), bounds = UnitInterval, desc = "Liquid (unfrozen) fraction of the snow water substance"),
)

# Process-level closure entry points (dispatch on the snow process, mirroring the soil interface).
@inline closure!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants) =
    closure!(state, grid, get_closure(snow), snow, constants)

@inline invclosure!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants) =
    invclosure!(state, grid, get_closure(snow), snow, constants)

"""
    $TYPEDSIGNATURES

Initialize the snow internal energy by evaluating the inverse closure (temperature → energy). Assumes
`snow_temperature` and `snow_water_equivalent` have already been initialized.
"""
function initialize!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants, args...)
    invclosure!(state, grid, get_closure(snow), snow, constants)
    return nothing
end

"""
    $TYPEDSIGNATURES

Forward closure: recover `snow_temperature` and `snow_liquid_fraction` from the prognostic `snow_energy`.
"""
function closure!(
        state, grid,
        closure::SnowEnergyTemperatureClosure,
        snow::SingleLayerSnow,
        constants::PhysicalConstants,
        args...
    )
    kernel_args = (closure, snow, constants)
    out = closure_fields(state, snow)
    fields = get_fields(state, kernel_args...)
    launch!(grid, XY, snow_energy_to_temperature_kernel!, out, fields, kernel_args...)
    return nothing
end

# Helper functions

"""
    $TYPEDSIGNATURES

Inverse closure: compute the prognostic `snow_energy` from a prescribed `snow_temperature` (used for
initialization). `snow_water_equivalent` must be initialized first, since the snow depth `d_snow` enters
the depth-integrated energy.
"""
function invclosure!(
        state, grid,
        closure::SnowEnergyTemperatureClosure,
        snow::SingleLayerSnow,
        constants::PhysicalConstants,
        args...
    )
    kernel_args = (closure, snow, constants)
    # manually add prognostic snow_energy to output fields
    out = merge(closure_fields(state, snow), (snow_energy = state.snow_energy,))
    fields = get_fields(state, kernel_args...)
    launch!(grid, XY, snow_temperature_to_energy_kernel!, out, fields, kernel_args...)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Recover the snow temperature and liquid water fraction from the depth-integrated energy at grid cell `i, j`.
"""
@propagate_inbounds function energy_to_temperature!(
        out, i, j, grid, fields,
        ::SnowEnergyTemperatureClosure{NF},
        snow::SingleLayerSnow,
        constants::PhysicalConstants
    ) where {NF}
    W_snow = fields.snow_water_equivalent[i, j] # assumed given (prognostic)
    Ū_snow = ifelse(W_snow > 0, fields.snow_energy[i, j], zero(NF)) # assumed given (prognostic)
    L_sl = constants.thermodynamics.latent_heat_fusion
    ρ_w = constants.material.density_water
    ρ_snow = compute_snow_density(i, j, grid, fields, snow.density)
    d_snow = compute_snow_depth(snow, W_snow, ρ_snow, ρ_w)
    # Volumetric latent heat of fusion
    ρLθ = ρ_snow * L_sl # ρ_snow L_sl = ρ_w θ L_sl by definition
    # Compute liquid water fraction from depth-integrated energy and latent heat of fusion
    liq = liquid_water_fraction(FreeWater(), Ū_snow, d_snow * ρLθ)
    # liq will be 1 for W_snow = 0, so gate by W_snow > 0
    out.snow_liquid_fraction[i, j, 1] = liq * (W_snow > 0)
    # Snow temperature cannot exceed 0°C, so clip the free-water temperature at zero. The energy above
    # the fully-melted (0°C, all-liquid) reference, i.e. the positive part `U_snow > 0`, is not stored; it
    # is derived on demand where needed to determine snow melt.
    C_snow = compute_snow_volumetric_heat_capacity(snow, constants, ρ_snow, liq)
    U_snow = compute_snow_volumetric_energy(Ū_snow, d_snow, min_snow_conduction_thickness(i, j, grid, fields, snow))
    T = energy_to_temperature(FreeWater(), U_snow, ρLθ, C_snow)
    out.snow_temperature[i, j, 1] = min(T, zero(T))
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the depth-integrated snow energy from a prescribed temperature at grid cell `i, j`.
"""
@propagate_inbounds function temperature_to_energy!(
        out, i, j, grid, fields,
        ::SnowEnergyTemperatureClosure,
        snow::SingleLayerSnow,
        constants::PhysicalConstants
    )
    T = min(fields.snow_temperature[i, j], zero(eltype(grid))) # assumed given (initialization)
    W_snow = fields.snow_water_equivalent[i, j] # assumed given (prognostic)
    L_sl = constants.thermodynamics.latent_heat_fusion
    ρ_w = constants.material.density_water
    ρ_snow = compute_snow_density(i, j, grid, fields, snow.density)
    d_snow = compute_snow_depth(snow, W_snow, ρ_snow, ρ_w)
    ρLθ = ρ_snow * L_sl # ρ_snow L_sl = ρ_w θ L_sl by definition
    liq = zero(eltype(grid)) # always initialize snow fully frozen
    out.snow_liquid_fraction[i, j, 1] = liq
    C_snow = compute_snow_volumetric_heat_capacity(snow, constants, ρ_snow, liq)
    U_snow = T * C_snow - ρLθ * (one(liq) - liq)
    out.snow_energy[i, j, 1] = U_snow * d_snow # integrate by d_snow
    return nothing
end

# Kernels
#
# These wrap the shared `energy_to_temperature!` / `temperature_to_energy!` kernel functions (dispatched
# on the snow closure type) in 2D (XY) kernels. They use snow-specific names because KernelAbstractions
# `@kernel` does not support multiple methods of one kernel — the soil energy closures define 3D (XYZ)
# kernels under the base names.

@kernel inbounds = true function snow_energy_to_temperature_kernel!(out, grid, fields, closure::SnowEnergyTemperatureClosure, args...)
    i, j = @index(Global, NTuple)
    energy_to_temperature!(out, i, j, grid, fields, closure, args...)
end

@kernel inbounds = true function snow_temperature_to_energy_kernel!(out, grid, fields, closure::SnowEnergyTemperatureClosure, args...)
    i, j = @index(Global, NTuple)
    temperature_to_energy!(out, i, j, grid, fields, closure, args...)
end
