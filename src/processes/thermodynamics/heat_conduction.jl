# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the internal energy tendency `∂U∂t` as the divergence of the diffusive and advective heat
fluxes,

    ∂U∂t = -∂z(q_diff + q_adv),

where `q_diff` is the [`diffusive_heat_flux`](@ref) and `q_adv` the [`advective_heat_flux`](@ref)
(zero unless the soil water flows).
"""
@propagate_inbounds function compute_energy_tendency(
        i, j, k, grid, fields,
        energy::AbstractThermodynamics,
        args...
    )
    # Operators require the underlying Oceananigans grid
    ground_grid = ground_domain(grid)

    # Divergence of heat fluxes
    ∂U∂t = -∂zᵃᵃᶜ(i, j, k, ground_grid, diffusive_heat_flux, fields, energy, args...) +
        compute_advective_energy_tendency(i, j, k, grid, fields, energy, args...)
    return ∂U∂t
end

"""
    $TYPEDSIGNATURES

Compute the part `-∂z(q_adv)` of the internal energy tendency at `i, j, k` that is due to the
divergence of the [`advective_heat_flux`](@ref) `q_adv`, through all faces of the cell. When the
heat fluxes are integrated implicitly, only the part due to the boundary faces is treated explicitly
([`compute_boundary_advective_energy_tendency`](@ref)).
"""
@propagate_inbounds function compute_advective_energy_tendency(
        i, j, k, grid, fields,
        energy::AbstractThermodynamics,
        args...
    )
    ground_grid = ground_domain(grid)
    return -∂zᵃᵃᶜ(i, j, k, ground_grid, advective_heat_flux, fields, energy, args...)
end

"""
    $TYPEDSIGNATURES

Advective heat flux (W/m², positive upward) at the cell face `i, j, k`. The default is zero, i.e. no
heat is transported by a moving medium; see the method for soil water flow following the Richards
equation.
"""
@inline advective_heat_flux(i, j, k, grid, fields, energy::AbstractThermodynamics, args...) = zero(grid)

"""
    $TYPEDSIGNATURES

Internal energy tendency `∂U∂t` at `i, j, k` from the [`diffusive_heat_flux`](@ref) `q` through the
boundary faces of the column only (bottom of `k = 1`, top of `k = Nz`):

    ∂U∂t = -(δ_{k,Nz} q_{k+½} - δ_{k,1} q_{k-½}) / Δz_k.

Zero for flux and no-flux conditions, whose halos have zero gradient (flux conditions are added
separately); nonzero, e.g., for a gradient condition on `temperature`.
"""
@propagate_inbounds function compute_boundary_energy_tendency(
        i, j, k, grid, fields,
        energy::AbstractThermodynamics,
        args...
    )
    ground_grid = ground_domain(grid)
    Nz = size(ground_grid, 3)
    # Diffusive heat fluxes through the lower (k) and upper (k + 1) faces of cell k
    q_lower = diffusive_heat_flux(i, j, k, ground_grid, fields, energy, args...)
    q_upper = diffusive_heat_flux(i, j, k + 1, ground_grid, fields, energy, args...)
    # Keep only the fluxes through the boundary faces of the column
    q_bottom = ifelse(k == 1, q_lower, zero(q_lower))
    q_top = ifelse(k == Nz, q_upper, zero(q_upper))
    ∂U∂t = -(q_top - q_bottom) / Δzᵃᵃᶜ(i, j, k, ground_grid)
    return ∂U∂t
end

"""
    $TYPEDSIGNATURES

Compute the part of the internal energy tendency `∂U∂t` at `i, j, k` due to the advective heat flux
through the boundary faces of the column that remains explicit when the interior fluxes are
integrated implicitly (see [`Picard`](@ref)). The default is zero, i.e. no heat is transported by a
moving medium; see the method for soil water flow following the Richards equation.
"""
@inline compute_boundary_advective_energy_tendency(i, j, k, grid, fields, energy::AbstractThermodynamics, args...) = zero(grid)

# Diffusive heat flux term passed to ∂z operator
"""
    $TYPEDSIGNATURES

Compute the diffusive (Fourier) heat flux `q = -κ ∂T/∂z` at the cell face `i, j, k`, interpolating the
medium-specific thermal conductivity to the face.
"""
@propagate_inbounds function diffusive_heat_flux(
        i, j, k, grid, fields,
        energy::AbstractThermodynamics,
        args...
    )
    # Get temperature field
    T = fields.temperature
    # Compute and interpolate conductivity to grid cell faces
    κ = ℑzᵃᵃᶠ(i, j, k, grid, compute_thermal_conductivity, fields, energy, args...)
    # Fourier's law: q = -κ ∂T/∂z
    q = -κ * ∂zᵃᵃᶠ(i, j, k, grid, T)
    return q
end

""" $TYPEDSIGNATURES """
@propagate_inbounds function compute_thermal_conductivity(
        i, j, k, grid, fields,
        energy::AbstractThermodynamics,
        args...
    )
    return compute_thermal_conductivity(energy.thermal_properties)
end
