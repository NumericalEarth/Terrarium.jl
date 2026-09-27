"""
    $TYPEDEF

Base type for coupled snow processes. Implementations should typically couple an energy balance and a
mass (water equivalent) balance for a snowpack overlying the ground surface. Snow is modeled as
its own coupled process (a sibling of soil), reusing the medium-agnostic `FreeWater` enthalpy relations
for the energy↔temperature closure (the same closure used for the soil; see [Soil energy balance](@ref)).
"""
abstract type AbstractSnow{NF} <: AbstractCoupledProcesses{NF} end

"""
    snow_water_equivalent(i, j, grid, fields, ::AbstractSnow)

Retrieve the snow water equivalent (SWE) `W` (m) — the total water substance (ice + retained liquid).
"""
function snow_water_equivalent end

"""
    snow_energy(i, j, grid, fields, ::AbstractSnow)

Retrieve the depth-integrated (column) snow internal energy `Ū_snow` [J/m²] relative to ice at 0°C.
"""
function snow_energy end

"""
    snow_depth(i, j, grid, fields, ::AbstractSnow)

Compute or retrieve the snow layer depth `d_snow` [m].
"""
function snow_depth end

"""
    snow_cover_fraction(i, j, grid, fields, ::AbstractSnow)

Compute or retrieve the sub-grid snow-covered area fraction `f_snow` ∈ [0,1].
"""
function snow_cover_fraction end

"""
    snow_thermal_conductivity(i, j, grid, fields, ::AbstractSnow)

Compute or retrieve the bulk snow thermal conductivity `κ_snow` [W/m/K].
"""
function snow_thermal_conductivity end

"""
    $TYPEDEF

Base type for snow density schemes.
"""
abstract type AbstractSnowDensity{NF} end

number_format(::Type{<:AbstractSnowDensity{NF}}) where {NF} = NF

"""
    snow_density(i, j, grid, fields, ::AbstractSnowDensity)

Compute or retrieve the bulk snow density `ρ_snow` [kg/m³].
"""
function compute_snow_density end

"""
    $TYPEDEF

Base type for snow albedo parameterizations.
"""
abstract type AbstractSnowAlbedo{NF} end

number_format(::Type{<:AbstractSnowAlbedo{NF}}) where {NF} = NF

# Defaults for snow = nothing

# No snow component present (`snow === nothing`): treat as zero snow cover, so surface processes that
# take an optional snow argument fall back to their snow-free behavior.
@propagate_inbounds snow_cover_fraction(i, j, grid, fields, ::Nothing) = zero(eltype(grid))

@propagate_inbounds snow_water_equivalent(i, j, grid, fields, ::Nothing) = zero(eltype(grid))

@propagate_inbounds snow_energy(i, j, grid, fields, ::Nothing) = zero(eltype(grid))

@propagate_inbounds snow_temperature(i, j, grid, fields, ::Nothing) = zero(eltype(grid))

@propagate_inbounds snow_depth(i, j, grid, fields, ::Nothing) = zero(eltype(grid))

@propagate_inbounds snow_thermal_conductivity(i, j, grid, fields, ::Nothing) = zero(eltype(grid))

@propagate_inbounds snow_meltwater_flux(i, j, grid, fields, ::Nothing) = zero(eltype(grid))
