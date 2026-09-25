using Terrarium
using Test

using Enzyme
using Statistics

"""
Build a snow-free, vegetation-free `LandModel` whose surface energy balance is fully prescribed, so
that `ground_heat_flux` is an input field forming a clean boundary-condition seam into the soil.
"""
function build_prescribed_seb_model(arch, ::Type{NF}) where {NF}
    grid = ColumnGrid(arch, NF, ExponentialSpacing(Δz_max = 1.0, N = 10))
    seb = PrescribedSurfaceEnergyBalance(eltype(grid))
    return LandModel(grid; surface_energy_balance = seb, vegetation = nothing, snow = nothing)
end

function mean_soil_temperature_step!(integrator, Δt)
    timestep!(integrator, Δt)
    return mean(interior(integrator.state.temperature))
end

@testset "Prescribed SEB: adjoint w.r.t. ground heat flux" begin
    model = build_prescribed_seb_model(CPU(), Float64)
    initializers = (
        temperature = (x, z) -> 5.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> 0.5,
    )
    integrator = initialize(model; initializers)
    set!(integrator.state.ground_heat_flux, 25.0)
    dintegrator = make_zero(integrator)
    Δt = 60.0
    Enzyme.autodiff(
        set_runtime_activity(Reverse),
        mean_soil_temperature_step!,
        Active,
        Duplicated(integrator, dintegrator),
        Const(Δt)
    )
    dG = Array(interior(dintegrator.state.ground_heat_flux))
    @test all(isfinite.(dG))
    # A positive ground heat flux is directed upward and so cools the column: the sensitivity of the
    # mean soil temperature to the prescribed flux must be strictly negative.
    @test all(dG .< 0)
    @test all(isfinite.(interior(dintegrator.state.internal_energy)))
end
