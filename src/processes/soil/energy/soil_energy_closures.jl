"""
    $TYPEDEF

Defines the constitutive relationship between the the internal energy and temperature of a
soil volume, i.e.
```math
U(T) = T\\times C(T) - \\rho_w L_{sl} \\theta (1 - F(T))
```
where T is temperature (°C), C(T) is the temperature-dependent heat capacity (J/m³/K), 
ρ_w L_{sl} θ is the volumetric latent heat of fusion (J/m³),  and F(T) = θ_w/θ is the constitutive
relation between T and the unfrozen fraction of pore water with θ the sum of the volumetric fractions of water and ice. 
Note that, under this formulation, zero energy corresponds to 0°C with no ice, i.e. all pore water fully thawed.

The closure relation is defined as being a mapping from the conserved quantity (energy) to the continuous
quantity (temperature), i.e. the inverse of U(T).
"""
struct SoilEnergyTemperatureClosure <: AbstractEnergyClosure end

"""
Defines `temperature` as the closure variable for `SoilEnergyTemperatureClosure`.
"""
variables(::SoilEnergyTemperatureClosure) = (
    auxiliary(:temperature, Ground(XYZ()), units = u"°C", desc = "Temperature of the soil volume in °C"),
    auxiliary(:liquid_water_fraction, Ground(XYZ()), bounds = UnitInterval, desc = "Fraction of unfrozen water in the pore space"),
)

function closure!(
        state, grid,
        closure::SoilEnergyTemperatureClosure,
        energy::SoilThermodynamics,
        ground::AbstractSoil,
        constants::PhysicalConstants,
        args...
    )
    (; hydrology, strat, biogeochem) = ground
    fc = freezecurve(energy.thermal_properties, hydrology)
    kernel_args = (closure, fc, energy, hydrology, strat, biogeochem, constants)
    # get closure fields (outputs)
    out = closure_fields(state, energy)
    # collect state/input fields
    fields = get_fields(state, kernel_args...; except = out)
    launch!(grid, XYZ, energy_to_temperature_kernel!, out, fields, kernel_args...)
    return nothing
end

function invclosure!(
        state, grid,
        closure::SoilEnergyTemperatureClosure,
        energy::SoilThermodynamics,
        ground::AbstractSoil,
        constants::PhysicalConstants,
        args...
    )
    (; hydrology, strat, biogeochem) = ground
    fc = freezecurve(energy.thermal_properties, hydrology)
    kernel_args = (closure, fc, energy, hydrology, strat, biogeochem, constants)
    # here we mannually collect the output fields since one is the prognostic variable
    out = (internal_energy = state.internal_energy, liquid_water_fraction = state.liquid_water_fraction)
    fields = get_fields(state, kernel_args...; except = out)
    launch!(grid, XYZ, temperature_to_energy_kernel!, out, fields, kernel_args...)
    return nothing
end

@propagate_inbounds function temperature_to_energy!(
        out, i, j, k, grid, fields,
        ::SoilEnergyTemperatureClosure,
        ::FreeWater,
        energy::SoilThermodynamics{NF, OP, SoilEnergyTemperatureClosure},
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants
    ) where {NF, OP}
    T = fields.temperature[i, j, k] # assumed given
    ρw = constants.material.density_water
    Lsl = constants.thermodynamics.latent_heat_fusion
    ρL = ρw * Lsl
    por = porosity(i, j, k, grid, fields, strat, bgc)
    sat = saturation_water_ice(i, j, k, grid, fields, hydrology)
    # calculate unfrozen water content from temperature
    # N.B. For the free water freeze curve, the mapping from temperature to unfrozen water content
    # within the phase change region is indeterminate since it is assumed that T = 0. As such, we
    # have to assume here that the liquid water fraction is zero if T < 0 and one otherwise. This method
    # should therefore only be used for initialization and should **not** be involved in the calculation
    # of tendencies.
    liq = out.liquid_water_fraction[i, j, k] = ifelse(
        T >= zero(T),
        one(sat),
        zero(sat),
    )
    # add liquid water fraction to fields
    fields = merge(fields, (; liquid_water_fraction = out.liquid_water_fraction))
    solid = soil_matrix(i, j, k, grid, fields, strat, bgc)
    soil = SoilComposition(por, sat, liq, solid)
    C = compute_heat_capacity(energy.thermal_properties, soil)
    # compute energy from temperature, heat capacity, and ice fraction
    U = out.internal_energy[i, j, k] = T * C - ρL * sat * por * (1 - liq)
    return U
end

@propagate_inbounds function energy_to_temperature!(
        out, i, j, k, grid, fields,
        ::SoilEnergyTemperatureClosure,
        fc::FreeWater,
        energy::SoilThermodynamics,
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants
    )

    U = fields.internal_energy[i, j, k] # assumed given
    ρw = constants.material.density_water
    Lsl = constants.thermodynamics.latent_heat_fusion
    por = porosity(i, j, k, grid, fields, strat, bgc)
    sat = saturation_water_ice(i, j, k, grid, fields, hydrology)
    ρLθ = ρw * Lsl * sat * por
    # calculate unfrozen water content
    liq = out.liquid_water_fraction[i, j, k] = liquid_water_fraction(fc, U, ρLθ)
    # add liquid water fraction to fields
    fields = merge(fields, (; liquid_water_fraction = out.liquid_water_fraction))
    # calculate soil volumetric fractions
    solid = soil_matrix(i, j, k, grid, fields, strat, bgc)
    soil = SoilComposition(por, sat, liq, solid)
    C = compute_heat_capacity(energy.thermal_properties, soil)
    # calculate temperature from internal energy and liquid water fraction
    T = out.temperature[i, j, k] = energy_to_temperature(fc, U, ρLθ, C)
    return T
end

# Kernels

@kernel inbounds = true function temperature_to_energy_kernel!(out, grid, fields, args...)
    i, j, k = @index(Global, NTuple)
    temperature_to_energy!(out, i, j, k, grid, fields, args...)
end

@kernel inbounds = true function energy_to_temperature_kernel!(out, grid, fields, args...)
    i, j, k = @index(Global, NTuple)
    energy_to_temperature!(out, i, j, k, grid, fields, args...)
end
