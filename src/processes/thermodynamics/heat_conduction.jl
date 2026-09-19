"""
    $TYPEDEF

Represents an explicit formulation of the two-phase heat conduction operator in 1D:

```math
\\frac{\\partial U(T,\\phi)}{\\partial t} = \\boldsymbol{\\nabla} \\cdot \\left[ \\kappa(T) \\boldsymbol{\\nabla}_x T(x,t) \\right]
```
where \$T\$ is temperature [K], \$U\$ is internal energy [J m⁻³], and \$\\kappa\$ is the thermal conductivity [W m K⁻¹].
"""
@kwdef struct ExplicitTwoPhaseHeatConduction <: AbstractHeatOperator end

# Kernel functions

"""
    $TYPEDSIGNATURES

Compute the internal energy tendency `∂U∂t` as the divergence of the diffusive heat flux for the
explicit two-phase heat conduction operator.
"""
@propagate_inbounds function compute_energy_tendency(
        i, j, k, grid, fields,
        energy::AbstractThermodynamics,
        args...
    )
    # Operators require the underlying Oceananigans grid
    ground_grid = ground_domain(grid)

    # Divergence of heat fluxes
    ∂U∂t = -∂zᵃᵃᶜ(i, j, k, ground_grid, diffusive_heat_flux, fields, energy, args...)
    return ∂U∂t
end

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
