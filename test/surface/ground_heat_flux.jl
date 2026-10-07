using Terrarium
using Test

using Oceananigans.BoundaryConditions: BoundaryCondition, Flux

"""
Build a snow-free, vegetation-free `LandModel` whose surface energy balance uses the given
`ground_heat_flux` sub-process, and return `(land, integrator)` initialized to a mild soil profile.
"""
function ground_heat_flux_land_model(
        ground_heat_flux;
        skin_temperature = PrescribedSkinTemperature(Float64),
        turbulent_fluxes = PrescribedTurbulentFluxes(Float64)
    )
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(Δz_max = 1.0, N = 20))
    seb = SurfaceEnergyBalance(eltype(grid); skin_temperature, turbulent_fluxes, ground_heat_flux)
    land = LandModel(grid; surface_energy_balance = seb, vegetation = nothing, snow = nothing)
    initializers = (
        temperature = (x, z) -> 5.0 - 0.02 * z,
        saturation_water_ice = (x, z) -> 0.5,
    )
    return land, initialize(land; initializers)
end

@testset "Ground heat flux: variable classification" begin
    NF = Float64
    diagnosed_vars = Terrarium.variables(SurfaceEnergyBalance(NF))
    prescribed_vars = Terrarium.variables(PrescribedSurfaceEnergyBalance(NF))
    # The field must be declared exactly once, by the ground heat flux sub-process alone.
    @test count(v -> Terrarium.varname(v) === :ground_heat_flux, diagnosed_vars) == 1
    @test count(v -> Terrarium.varname(v) === :ground_heat_flux, prescribed_vars) == 1
    # ... as an auxiliary when diagnosed and as an input when prescribed.
    diagnosed = only(filter(v -> Terrarium.varname(v) === :ground_heat_flux, collect(diagnosed_vars)))
    prescribed = only(filter(v -> Terrarium.varname(v) === :ground_heat_flux, collect(prescribed_vars)))
    @test diagnosed isa Terrarium.AuxiliaryVariable
    @test prescribed isa Terrarium.InputVariable
    # The skin temperature schemes no longer declare it themselves.
    @test !any(v -> Terrarium.varname(v) === :ground_heat_flux, Terrarium.variables(PrescribedSkinTemperature(NF)))
    @test !any(v -> Terrarium.varname(v) === :ground_heat_flux, Terrarium.variables(ImplicitSkinTemperature(NF)))
end

@testset "Ground heat flux: PrescribedSurfaceEnergyBalance alias" begin
    NF = Float64
    seb = PrescribedSurfaceEnergyBalance(NF)
    # Guards the type parameter ordering of `SurfaceEnergyBalance`, which is the most likely thing
    # to get silently wrong when the alias and the struct drift apart.
    @test seb isa PrescribedSurfaceEnergyBalance
    @test seb isa PrescribedSurfaceEnergyBalance{NF}
    @test Terrarium.get_skin_temperature(seb) isa PrescribedSkinTemperature
    @test Terrarium.get_radiative_fluxes(seb) isa PrescribedRadiativeFluxes
    @test Terrarium.get_turbulent_fluxes(seb) isa PrescribedTurbulentFluxes
    @test Terrarium.get_ground_heat_flux(seb) isa PrescribedGroundHeatFlux
    # The default SEB diagnoses everything and must *not* match the alias.
    @test !(SurfaceEnergyBalance(NF) isa PrescribedSurfaceEnergyBalance)
    @test Terrarium.get_ground_heat_flux(SurfaceEnergyBalance(NF)) isa DiagnosedGroundHeatFlux
    # The albedo is left free by the alias.
    @test PrescribedSurfaceEnergyBalance(NF; albedo = DiagnosticAlbedo(NF)) isa PrescribedSurfaceEnergyBalance
end

@testset "Ground heat flux: prescribed flux is not overwritten" begin
    land, integrator = ground_heat_flux_land_model(PrescribedGroundHeatFlux(Float64))
    state = integrator.state
    @test hasproperty(state.inputs, :ground_heat_flux)
    @test !hasproperty(state.auxiliary, :ground_heat_flux)

    # Deliberately inconsistent with the residual closure: G ≠ R_net + H_s + H_l. A diagnosed
    # ground heat flux would overwrite it; a prescribed one must not.
    G = 25.0
    set!(state.ground_heat_flux, G)
    set!(state.skin_temperature, 3.0)
    set!(state.sensible_heat_flux, 111.0)
    set!(state.latent_heat_flux, 222.0)
    Terrarium.closure!(state, land)
    compute_auxiliary!(state, land)
    compute_boundary_conditions!(state, land)
    @test all(interior(state.ground_heat_flux) .== G)

    # Without snow the soil-top energy BC *is* the prescribed field, so the coupler's value routes
    # straight to the soil column's Neumann boundary condition.
    energy_top_bc = state.internal_energy.boundary_conditions.top
    @test isa(energy_top_bc, BoundaryCondition{<:Flux})
    @test energy_top_bc.condition === state.ground_heat_flux

    timestep!(integrator, 60.0)
    @test all(interior(state.ground_heat_flux) .== G)
    @test all(isfinite.(interior(state.internal_energy)))
end

@testset "Ground heat flux: prescribed flux conserves energy" begin
    land, integrator = ground_heat_flux_land_model(PrescribedGroundHeatFlux(Float64))
    state = integrator.state
    ground_grid = Terrarium.ground_domain(Terrarium.get_grid(land))
    Δz = [Terrarium.Δzᵃᵃᶜ(1, 1, k, ground_grid) for k in 1:ground_grid.Nz]
    column_energy(state) = sum(Array(interior(state.internal_energy))[1, 1, :] .* Δz)

    G = 25.0
    set!(state.ground_heat_flux, G)
    U₀ = column_energy(state)
    Δt = 60.0
    N = 20
    for _ in 1:N
        timestep!(integrator, Δt)
    end
    # All fluxes are positive upward, so a positive `G` removes energy from the ground column at
    # exactly the prescribed rate.
    @test column_energy(state) - U₀ ≈ -G * Δt * N rtol = 1.0e-6
end

@testset "Ground heat flux: diagnosed flux closes the balance" begin
    land, integrator = ground_heat_flux_land_model(DiagnosedGroundHeatFlux(Float64))
    state = integrator.state
    @test hasproperty(state.auxiliary, :ground_heat_flux)
    @test !hasproperty(state.inputs, :ground_heat_flux)

    set!(state.skin_temperature, 10.0)
    set!(state.sensible_heat_flux, 20.0)
    set!(state.latent_heat_flux, 30.0)
    Terrarium.closure!(state, land)
    compute_auxiliary!(state, land)
    compute_boundary_conditions!(state, land)

    # With a prescribed skin temperature, the diagnosed ground heat flux is the residual that
    # closes the surface energy balance, G = R_net + H_s + H_l.
    R_net = Array(interior(state.surface_net_radiation))
    @test all(isapprox.(Array(interior(state.ground_heat_flux)), R_net .+ 20 .+ 30; rtol = 1.0e-6))
end
