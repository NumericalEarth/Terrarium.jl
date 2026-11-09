# Stefan problem benchmark of the coupled soil energy and water time stepping
module SoilStefanProblemTests

using Terrarium
using Test

using SpecialFunctions: erf, erfc
using Terrarium: Δzᵃᵃᶜ, get_hydrology, get_stratigraphy, get_biogeochemistry, get_energy_balance, soil_composition,
    compute_heat_capacity, compute_thermal_conductivity, cell_diffusion_timescale

# Stefan problem benchmark: freezing of a soil column from the surface, compared with the classical two-phase
# Neumann solution (e.g. Carslaw and Jaeger, 1959, §11.2) for a semi-infinite medium, initially thawed at the
# temperature Tᵢ > 0, whose surface is held at Tₛ < 0 from t = 0. The front lies at the depth X(t) = 2λ √(αₛ t),
# where λ solves the Stefan condition
#
#   (Tₘ - Tₛ) e^{-λ²} / erf(λ) - (κₗ / κₛ) ν (Tᵢ - Tₘ) e^{-λ²ν²} / erfc(λν) = λ √π L / Cₛ,    ν = √(αₛ / αₗ),
#
# with melting point Tₘ = 0 °C, thermal conductivities κ, heat capacities C, and diffusivities α = κ / C of the frozen
# (s) and thawed (l) soil, and the volumetric latent heat L = ρ_w L_sl θ. The column starts from the Neumann profile
# at t₀, when the front lies at X₀, so that the surface condition becomes the flux of the Neumann solution,
# κₛ (Tₘ - Tₛ) / (erf(λ) √(π αₛ t)), which is supported by both explicit and implicit time stepping.
#
# The soil uses a Richards-equation hydrology, so that `IMEX` solves the water before the energy, with the
# hydraulic conductivity lagged at the liquid water fraction of the previous step. In the saturated column the water
# does not flow and the Neumann solution applies; in the unsaturated column the water drains slowly under gravity,
# which changes the latent heat along the column, so `IMEX` is compared with the explicit reference only.
#
# The Neumann solution agrees with that of CryoGrid.jl (`stefan_analytic.jl`, after Hu et al., 1996, Eq. 13), with the
# roles of the phases exchanged since CryoGrid's benchmark thaws a frozen column. As there, the front is required to
# lie within 1 cm of the analytical one. Unlike CryoGrid's benchmark (a step to a prescribed surface temperature at
# t = 0, over five years with daily steps), the column starts from the Neumann profile at t₀ under the Neumann surface
# flux, since value conditions on `temperature` are not supported under implicit time stepping and the explicit
# reference must remain affordable; over the nine days of this benchmark, daily steps are too coarse.

module StefanProblemTestUtils

    using Terrarium
    using SpecialFunctions: erf, erfc
    using Terrarium: Δzᵃᵃᶜ, get_hydrology, get_stratigraphy, get_biogeochemistry, get_energy_balance, soil_composition,
        compute_heat_capacity, compute_thermal_conductivity

    const POROSITY = 0.4

    function richards_soil(NF; saturated_conductivity)
        hydraulic_properties = ConstantSoilHydraulics(
            NF; swrc = VanGenuchten(α = 2.0, n = 2.0), saturated_conductivity, unsat_hydraulic_cond = UnsatKVanGenuchten(NF)
        )
        hydrology = SoilHydrology(NF, RichardsEq(); hydraulic_properties)
        strat = HomogeneousSoilStratigraphy(NF; porosity = ConstantSoilPorosity(NF; mineral_porosity = POROSITY))
        biogeochem = ConstantSoilCarbonDensity(NF; ρ_soc = 0.0)
        return SoilEnergyWaterCarbon(NF; hydrology, strat, biogeochem)
    end

    # thermal conductivity and heat capacity of the frozen (top) and thawed (bottom) cells of a column
    function phase_properties(integrator)
        (; state, model) = integrator
        soil = model.soil
        energy, hydrology, strat, bgc = get_energy_balance(soil), get_hydrology(soil), get_stratigraphy(soil), get_biogeochemistry(soil)
        fields = get_fields(state, energy, hydrology, strat, bgc)
        grid = get_grid(model)
        function properties(k)
            composition = soil_composition(1, 1, k, grid, fields, strat, hydrology, bgc)
            return (C = compute_heat_capacity(energy.thermal_properties, composition), κ = compute_thermal_conductivity(energy.thermal_properties, composition))
        end
        return properties(ground_domain(grid).Nz), properties(1)
    end

    # root λ of the Stefan condition, by bisection
    function neumann_λ(; κₛ, κₗ, Cₛ, Cₗ, L, Tₛ, Tᵢ)
        ν = sqrt((κₛ / Cₛ) / (κₗ / Cₗ))
        residual(λ) = -Tₛ * exp(-λ^2) / erf(λ) - (κₗ / κₛ) * ν * Tᵢ * exp(-λ^2 * ν^2) / erfc(λ * ν) - λ * sqrt(π) * L / Cₛ
        a, b = 1.0e-6, 5.0
        for _ in 1:100
            m = (a + b) / 2
            residual(m) > 0 ? (a = m) : (b = m)
        end
        return (a + b) / 2
    end

    # Neumann temperature (°C) at the depth x (m) and time t (s)
    function neumann_temperature(x, t; λ, αₛ, αₗ, Tₛ, Tᵢ)
        X = 2λ * sqrt(αₛ * t)
        ν = sqrt(αₛ / αₗ)
        frozen = Tₛ - Tₛ * erf(x / (2 * sqrt(αₛ * t))) / erf(λ)
        thawed = Tᵢ - Tᵢ * erfc(x / (2 * sqrt(αₗ * t))) / erfc(λ * ν)
        return x < X ? frozen : thawed
    end

    # Freeze a column of `N` cells of thickness `Δz` from the Neumann profile with the front at `X₀` until the front of
    # the Neumann solution reaches `X₁`, with `timestepper` and the time step `Δt` (rounded to divide the duration;
    # `nothing` for half the explicit diffusive limit of the heat conduction). Returns the numerical front depth, i.e.
    # the frozen thickness ∑ (1 - F) Δz, the temperature profile, and the Neumann front and profile.
    function freeze_column(
            timestepper, Δt; saturation, saturated_conductivity, Tₛ = -5.0, Tᵢ = 2.0, X₀ = 0.1, X₁ = 0.5, N = 200, Δz = 0.02
        )
        NF = Float64
        grid = ColumnGrid(CPU(), NF, UniformSpacing(; Δz, N))
        soil = richards_soil(NF; saturated_conductivity)
        # bulk properties of the frozen and thawed soil
        probe = initialize(SoilModel(grid; soil); initializers = (temperature = (x, z) -> z > -1 ? -1.0 : 1.0, saturation_water_ice = saturation))
        frozen, thawed = phase_properties(probe)
        constants = probe.model.constants
        L = constants.material.density_water * constants.thermodynamics.latent_heat_fusion * POROSITY * saturation
        λ = neumann_λ(; κₛ = frozen.κ, κₗ = thawed.κ, Cₛ = frozen.C, Cₗ = thawed.C, L, Tₛ, Tᵢ)
        αₛ, αₗ = frozen.κ / frozen.C, thawed.κ / thawed.C
        t₀ = (X₀ / (2λ))^2 / αₛ
        t₁ = (X₁ / (2λ))^2 / αₛ
        # heat flux (W/m², positive upward) of the Neumann solution through the surface at the model time t
        flux_amplitude = frozen.κ * (-Tₛ) / (erf(λ) * sqrt(π * αₛ))
        surface_heat_flux(x, t) = flux_amplitude / sqrt(t + t₀)
        initial_temperature(x, z) = neumann_temperature(-z, t₀; λ, αₛ, αₗ, Tₛ, Tᵢ)
        model = SoilModel(grid; soil, timestepper)
        integrator = initialize(
            model; boundary_conditions = SoilHeatFlux(surface_heat_flux), initializers = (temperature = initial_temperature, saturation_water_ice = saturation)
        )
        duration = t₁ - t₀
        if isnothing(Δt)
            Δt = Terrarium.cell_diffusion_timescale(integrator.state, grid, get_energy_balance(soil), soil) / 2
        end
        nsteps = ceil(Int, duration / Δt)
        for _ in 1:nsteps
            timestep!(integrator, duration / nsteps)
        end
        Δzs = [Δzᵃᵃᶜ(1, 1, k, ground_domain(grid)) for k in 1:N]
        X = sum((1 .- interior(integrator.state.liquid_water_fraction)[1, 1, :]) .* Δzs)
        T = copy(interior(integrator.state.temperature)[1, 1, :])
        z = collect(znodes(ground_domain(grid), Center()))
        T_neumann = [neumann_temperature(-zₖ, t₁; λ, αₛ, αₗ, Tₛ, Tᵢ) for zₖ in z]
        return (; X, T, X_neumann = 2λ * sqrt(αₛ * t₁), T_neumann, λ, Δt = duration / nsteps, ΔT = Tᵢ - Tₛ)
    end

end

using .StefanProblemTestUtils: freeze_column

@testset "Stefan problem: saturated column without water flow (Neumann solution)" begin
    NF = Float64
    Δz = 0.02
    kwargs = (saturation = 1.0, saturated_conductivity = 1.0e-5, Δz)
    reference = freeze_column(ForwardEuler(NF), nothing; kwargs...)
    reference_error = abs(reference.X - reference.X_neumann)
    @info "Stefan problem (saturated): explicit reference" reference.λ reference.Δt reference.X reference.X_neumann reference_error maximum(abs, reference.T .- reference.T_neumann)
    # the explicit reference resolves the front within a fraction of a cell
    @test reference_error < Δz / 4
    @test maximum(abs, reference.T .- reference.T_neumann) < 0.02 * reference.ΔT
    for Δt in (3600.0, 10800.0)
        result = freeze_column(IMEX(ForwardEuler(NF), BackwardEuler(NF)), Δt; kwargs...)
        front_error = abs(result.X - result.X_neumann)
        profile_error = maximum(abs, result.T .- result.T_neumann)
        @info "Stefan problem (saturated): IMEX" Δt result.X front_error profile_error
        # the front of IMEX lies within half a cell of the error of the explicit reference, and within 1 cm of
        # the Neumann front, the tolerance of CryoGrid's benchmark
        @test front_error < reference_error + Δz / 2
        @test front_error < 0.01
        @test profile_error < 0.04 * result.ΔT
    end
end

@testset "Stefan problem: unsaturated column with slow drainage (explicit reference)" begin
    NF = Float64
    Δz = 0.02
    kwargs = (saturation = 0.8, saturated_conductivity = 1.0e-7, Δz)
    reference = freeze_column(ForwardEuler(NF), nothing; kwargs...)
    @info "Stefan problem (unsaturated): explicit reference" reference.Δt reference.X reference.X_neumann
    for Δt in (3600.0, 10800.0)
        result = freeze_column(IMEX(ForwardEuler(NF), BackwardEuler(NF)), Δt; kwargs...)
        front_difference = abs(result.X - reference.X)
        profile_difference = maximum(abs, result.T .- reference.T)
        @info "Stefan problem (unsaturated): IMEX" Δt result.X front_difference profile_difference
        @test front_difference < Δz / 2
        @test profile_difference < 0.04 * result.ΔT
    end
end

end # module SoilStefanProblemTests
