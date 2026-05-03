# Forcing interface for soil hydrology

"""
    forcing(i, j, k, grid, clock, fields, evapotranspiration::AbstractEvapotranspiration, ::AbstractSoilHydrology)

Compute and return the evapotranspiration forcing for soil moisture at the given indices `i, j, k`.
The ET forcing is just the `surface_humidity_flux` rescaled by the thickness of layer `k`.
"""
@inline function forcing(i, j, k, grid, clock, fields, evapotranspiration::AbstractEvapotranspiration, ::AbstractSoilHydrology)
    let Δz = Δzᵃᵃᶜ(i, j, k, grid)
        Qh = surface_humidity_flux(i, j, grid, fields, evapotranspiration)
        ∂θ∂t = -Qh / Δz # rescale by layer thickness to get water content flux
        return ∂θ∂t * (k == grid.Nz)
    end
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, evapotranspiration::AbstractEvapotranspiration, args...)
    i, j = @index(Global, NTuple)
    compute_evapotranspiration!(out, i, j, grid, fields, evapotranspiration, args...)
end
