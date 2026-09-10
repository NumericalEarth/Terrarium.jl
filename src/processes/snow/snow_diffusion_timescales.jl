#####
##### Snow heat diffusion timescale
#####

# Fallback: no snow ⇒ no additional diffusive restriction.
cell_diffusion_timescale(state, grid, ::Nothing, constants::PhysicalConstants) = convert(eltype(grid), Inf)

"""
    $TYPEDSIGNATURES

Return the minimum thermal diffusion timescale ``τ = \\max(d_{\\text{snow}}, d_{\\text{min}})² \\,
C_{\\text{snow}} / κ_{\\text{snow}}`` over the grid for the snow-top conduction operator, mirroring
the [`cell_diffusion_timescale`](@ref) pattern used for soil. Uses the same floored conduction thickness
([`min_snow_conduction_thickness`](@ref)) that enters the actual conductive-flux calculations
([`compute_snow_surface_heat_flux`](@ref), [`compute_snow_basal_heat_flux`](@ref)), so the reported
timescale reflects the stability limit of the discretization as implemented — the floor bounds it below
for a vanishing pack, while a genuinely thick pack still reports its own (longer, less restrictive)
timescale rather than being pinned to the floor.
"""
function cell_diffusion_timescale(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants)
    field_grid = get_field_grid(grid)
    fields = get_fields(state, snow)
    τ = KernelFunctionOperation{Center, Center, Nothing}(
        compute_snow_diffusion_timescale, field_grid, fields, snow, constants
    )
    return minimum(τ)
end

"""
    $TYPEDSIGNATURES

Kernel function returning the snow thermal diffusion timescale ``\\max(d_{\\text{snow}}, d_{\\text{min}})²
\\, C_{\\text{snow}} / κ_{\\text{snow}}`` at grid cell `i, j`, with the bulk thermal conductivity and heat
capacity computed from the local snow density and liquid fraction.
"""
@inline @propagate_inbounds function compute_snow_diffusion_timescale(
        i, j, grid, fields,
        snow::SingleLayerSnow,
        constants::PhysicalConstants
    )
    ρ_snow = compute_snow_density(i, j, grid, fields, snow.density)
    κ_snow = compute_thermal_conductivity(snow, constants.material, ρ_snow)
    liq = fields.snow_liquid_fraction[i, j]
    C_snow = compute_snow_volumetric_heat_capacity(snow, constants, ρ_snow, liq)
    d_snow = max(snow_depth(i, j, grid, fields, snow), min_snow_conduction_thickness(i, j, grid, fields, snow))
    # thermal diffusivity α = κ / C ⟹ τ = d_snow² / α = d_snow² C / κ
    return d_snow^2 * C_snow / κ_snow
end
