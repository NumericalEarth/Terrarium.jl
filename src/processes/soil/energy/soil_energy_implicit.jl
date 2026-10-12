# Linearly implicit soil heat conduction: implicit operator and process hooks of `SoilThermodynamics`.
#
# Backward Euler step of ∂U/∂t = ∂z(κ ∂z T) - ∂z(C_w q T_up) + G_exp, with conduction and upwind advection by
# the water flux q solved in temperature and the energy updated conservatively. With the `Picard` solver, for
# iterations m = 0, …, M-1, starting from U⁰ = Uⁿ (plus any heat moved by the closure of the water step) and
# T⁰ ← closure(U⁰):
#
#   1. `linearize!`: Tᵐ ← closure(Uᵐ), lagged κᵐ (faces), regularized apparent heat capacity C_appᵐ (centers),
#      advective face velocity C_w q (faces), and rhs = C_appᵐ Tᵐ - Uᵐ + U⁺,    U⁺ = U⁰ + Δt G_exp;
#   2. `solve!`: C_appᵐ Tᵐ⁺¹ - Δt ∂z(κᵐ ∂z Tᵐ⁺¹) + Δt ∂z(C_w q T_upᵐ⁺¹) = rhs;
#   3. `implicit_update!`: Uᵐ⁺¹ = U⁺ + Δt [∂z(κᵐ ∂z Tᵐ⁺¹) - ∂z(C_w q T_upᵐ⁺¹)]    (interior faces only);
#   4. `implicit_closure!`: Tᵐ⁺¹, liquid water fraction ← closure(Uᵐ⁺¹).
#
# The solver calls the hooks through the `SoilModel`/`LandModel` forwarding methods and the
# `SoilEnergyWaterCarbon` methods in `soil_coupled.jl`.

"""
    $TYPEDSIGNATURES

Return the implicit operator of the soil internal energy: a [`TridiagonalOperator`](@ref) on
`ground_domain(grid)` for heat conduction in temperature, with a face velocity for upwind advection of
heat by the water flux if the `hydrology` lets the water flow (`SoilHydrology` with `RichardsEq`).
Validates on the host that the freeze curve is `FreeWater` (the only one with an implemented regularized
apparent heat capacity) and that `temperature` has no value boundary conditions (not yet supported).
Flux boundary conditions on `internal_energy` and gradient boundary conditions on `temperature` are
supported; the latter prescribe the diffusive flux `-κ ∂T/∂z` through the boundary face, which enters the
explicit tendency via [`compute_boundary_energy_tendency`](@ref).
"""
function implicit_operator(energy::SoilThermodynamics, hydrology::AbstractSoilHydrology, state, grid)
    energy.thermal_properties.freezecurve isa FreeWater || throw(
        ArgumentError(
            "implicit soil heat conduction currently supports only the `FreeWater` freeze curve, got $(nameof(typeof(energy.thermal_properties.freezecurve)))"
        )
    )
    bcs = state.temperature.boundary_conditions
    for (side, bc) in ((:top, bcs.top), (:bottom, bcs.bottom))
        bc isa BoundaryCondition{<:Value} && throw(
            ArgumentError(
                "value boundary conditions on `temperature` (found at the $side boundary) are not yet supported when `internal_energy` is integrated implicitly; use a flux boundary condition on `internal_energy` (e.g. `SoilHeatFlux`), a gradient boundary condition on `temperature`, or an explicit timestepper"
            )
        )
    end
    return TridiagonalOperator(ground_domain(grid); advection = has_water_flow(hydrology))
end

# Whether the soil water flows, so that it advects heat.
has_water_flow(::AbstractSoilHydrology) = false
has_water_flow(::SoilHydrology{NF, RichardsEq}) where {NF} = true

"""
    $TYPEDSIGNATURES

Linearize the implicit heat operator of the soil internal energy at the current iterate (linearly
implicit backward Euler in temperature, conservative update of the internal energy). On entry,
`explicit_energy` holds `U⁺ = U⁰ + Δt G_exp`, where `G_exp` contains the flux boundary conditions (e.g.
[`SoilHeatFlux`](@ref) and [`GeothermalHeatFlux`](@ref)), the diffusive fluxes implied by gradient
boundary conditions on `temperature`, the advective heat fluxes through the boundary faces, and the
heat carried by water crossing the boundaries ([`compute_boundary_advected_heat!`](@ref)), but not the
fluxes through the interior faces. This hook

1. re-evaluates the energy closure, so that temperature and liquid water fraction are consistent with
   the internal energy and the current saturation (changed by the water step since `tⁿ`),
2. evaluates the face conductivity `κ` ([`compute_implicit_conductivity!`](@ref)), the regularized
   apparent heat capacity `C_app` ([`compute_implicit_capacity!`](@ref)), and, with water flow, the
   advective face velocity `C_w q` from the `water_flux` of the step
   ([`compute_implicit_advective_velocity!`](@ref)), storing them in the `face_conductivity`,
   `diagonal_coefficient`, and `face_velocity` of `operator`,
3. and assembles the right-hand side `C_app Tᵐ - Uᵐ + U⁺` of
   `C_app Tᵐ⁺¹ - Δt ∂z(κ ∂z Tᵐ⁺¹) + Δt ∂z(C_w q T_upᵐ⁺¹) = C_app Tᵐ - Uᵐ + U⁺`
   ([`compute_implicit_energy_rhs`](@ref)).

[`implicit_update!`](@ref) then updates the internal energy conservatively: for any number of
iterations, the change of column-integrated energy is exactly `Δt` times the net boundary flux in
`G_exp`.
"""
function linearize!(
        operator::TridiagonalOperator,
        rhs,
        state,
        grid,
        energy::SoilThermodynamics,
        soil::AbstractSoil,
        constants::PhysicalConstants,
        explicit_energy,
        Δt
    )
    hydrology = get_hydrology(soil)
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    fc = freezecurve(energy.thermal_properties, hydrology)
    # temperature and liquid water fraction at the current iterate
    closure!(state, grid, energy.closure, energy, soil, constants)
    (; diagonal_coefficient, face_conductivity, face_velocity) = operator
    coefficients = (; diagonal_coefficient, face_conductivity, face_velocity)
    # internal energy, temperature, liquid water fraction, water flux, soil composition
    fields = get_fields(state, energy, hydrology, strat, bgc)
    # lagged coefficients at the latest iterate
    launch!(
        grid, XYZ, compute_implicit_heat_coefficients_kernel!,
        coefficients, fields, energy, fc, hydrology, strat, bgc, constants
    )
    # right-hand side C_app Tᵐ - Uᵐ + U⁺
    launch!(grid, XYZ, compute_implicit_energy_rhs_kernel!, rhs, fields, diagonal_coefficient, explicit_energy)
    return nothing
end

"""
    $TYPEDSIGNATURES

Conservative update of the soil internal energy,
`Uᵐ⁺¹ = U⁺ + Δt [∂z(κ ∂z Tᵐ⁺¹) - ∂z(C_w q T_upᵐ⁺¹)]`, from the implicit diffusive and
advective fluxes through the interior faces ([`compute_implicit_energy_update`](@ref)), with
`U⁺ = explicit_energy` and the lagged face conductivity `κ` and face velocity `C_w q` of `operator`.
"""
function implicit_update!(
        state, grid,
        energy::SoilThermodynamics,
        soil::AbstractSoil,
        operator::TridiagonalOperator,
        explicit_energy, Δt
    )
    launch!(
        grid, XYZ, compute_implicit_energy_update_kernel!,
        state.internal_energy, operator.face_conductivity, operator.face_velocity, state.temperature, explicit_energy, Δt
    )
    return nothing
end

"""
    $TYPEDSIGNATURES

Re-diagnose the temperature and liquid water fraction from the internal energy with the energy closure
[`closure!`](@ref) after every iteration. `runoff` and `iteration` are not used.
"""
function implicit_closure!(
        state, grid,
        energy::SoilThermodynamics,
        soil::AbstractSoil,
        constants::PhysicalConstants,
        runoff, iteration::Iteration
    )
    closure!(state, grid, energy.closure, energy, soil, constants)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Store the lagged thermal conductivity at face `i, j, k` in `face_conductivity`, interpolated as in the
explicit [`diffusive_heat_flux`](@ref).
"""
@propagate_inbounds function compute_implicit_conductivity!(
        face_conductivity, i, j, k, grid, fields,
        energy::SoilThermodynamics,
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    ground_grid = ground_domain(grid)
    face_conductivity[i, j, k] = ℑzᵃᵃᶠ(i, j, k, ground_grid, compute_thermal_conductivity, fields, energy, hydrology, strat, bgc)
    return nothing
end

"""
    $TYPEDSIGNATURES

Store the regularized apparent heat capacity `C_app` (J/m³/K) at cell `i, j, k` in
`diagonal_coefficient`, for the `FreeWater` freeze curve,

    C_app = C + C_pc    if 0 < F < 1 (phase change),
    C_app = C           otherwise,

where `C` is the bulk heat capacity, `F` the liquid water fraction of the current iterate, and
`C_pc = 1.5 × 10⁸` J/m³/K a fixed regularization of the latent heat term (a Dirac delta at 0 °C for
`FreeWater`). `C_pc` only damps the temperature change of phase-change cells between iterates and does
not enter the conservative energy update. It is not applied to fully frozen or thawed cells, where it
would slow the lagged temperature of cells that keep cooling (warming) and overestimate their
conductive fluxes.
"""
@propagate_inbounds function compute_implicit_capacity!(
        diagonal_coefficient, i, j, k, grid, fields,
        energy::SoilThermodynamics,
        ::FreeWater,
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants
    )
    soil = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    C = compute_heat_capacity(energy.thermal_properties, soil)
    # fixed apparent capacity (J/m³/K) of the regularized latent heat term in phase-change cells
    C_pc = oftype(C, 15 * 10^7)
    liq = soil.liquid
    phase_change = (liq > zero(liq)) & (liq < one(liq))
    diagonal_coefficient[i, j, k] = C + ifelse(phase_change, C_pc, zero(C))
    return nothing
end

"""
    compute_implicit_advective_velocity!(face_velocity, i, j, k, grid, fields, energy, hydrology)

Store the advective face velocity `C_w q` (W/m²/K) of the implicit heat operator at face `i, j, k` in
`face_velocity`, and at the top boundary face for `k = Nz`, with `q` the water flux (m/s) of the step
through the face ([`compute_advective_water_flux`](@ref)) and `C_w` the volumetric heat capacity of
liquid water, as in the explicit [`advective_heat_flux`](@ref). Through the boundary faces, the
operator uses only the outflow (see [`TridiagonalOperator`](@ref)). A no-op without advection
(`face_velocity = nothing`).
"""
@propagate_inbounds function compute_implicit_advective_velocity!(
        face_velocity, i, j, k, grid, fields,
        energy::SoilThermodynamics,
        hydrology::SoilHydrology{NF, RichardsEq}
    ) where {NF}
    Nz = size(ground_domain(grid), 3)
    Cw = energy.thermal_properties.heat_capacities.water
    face_velocity[i, j, k] = Cw * compute_advective_water_flux(i, j, k, grid, fields)
    # the top face Nz + 1 is written by cell Nz; other cells rewrite their lower face
    k_top = ifelse(k == Nz, Nz + 1, k)
    face_velocity[i, j, k_top] = Cw * compute_advective_water_flux(i, j, k_top, grid, fields)
    return nothing
end

@inline compute_implicit_advective_velocity!(::Nothing, i, j, k, grid, fields, energy::SoilThermodynamics, hydrology::AbstractSoilHydrology) = nothing

"""
    $TYPEDSIGNATURES

Store the lagged apparent heat capacity ([`compute_implicit_capacity!`](@ref)), face conductivity
([`compute_implicit_conductivity!`](@ref)), and, with water flow, advective face velocity
([`compute_implicit_advective_velocity!`](@ref)) at `i, j, k` in `out.diagonal_coefficient`,
`out.face_conductivity`, and `out.face_velocity`.
"""
@propagate_inbounds function compute_implicit_heat_coefficients!(
        out, i, j, k, grid, fields,
        energy::SoilThermodynamics,
        fc::FreeWater,
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants
    )
    compute_implicit_capacity!(out.diagonal_coefficient, i, j, k, grid, fields, energy, fc, hydrology, strat, bgc, constants)
    compute_implicit_conductivity!(out.face_conductivity, i, j, k, grid, fields, energy, hydrology, strat, bgc)
    compute_implicit_advective_velocity!(out.face_velocity, i, j, k, grid, fields, energy, hydrology)
    return nothing
end

"""
    $TYPEDSIGNATURES

Right-hand side `C_app Tᵐ - Uᵐ + U⁺` of the linearized system at Picard iterate `m`, with
`C_app = diagonal_coefficient` and `U⁺ = explicit_energy`.
"""
@propagate_inbounds function compute_implicit_energy_rhs(i, j, k, grid, fields, diagonal_coefficient, explicit_energy)
    ∂U∂T = diagonal_coefficient[i, j, k]
    T = fields.temperature[i, j, k]
    U = fields.internal_energy[i, j, k]
    U⁺ = explicit_energy[i, j, k]
    rhs = ∂U∂T * T - U + U⁺
    return rhs
end

"""
    $TYPEDSIGNATURES

Conservative energy update `U = U⁺ + Δt (Q_diff + Q_adv)` at `i, j, k` from the implicit heating rates
(W/m³) of conduction, `Q_diff = ∂z(κ ∂z T)` through the interior faces
([`compute_implicit_diffusion_tendency`](@ref)), and of advection, `Q_adv = -∂z(C_w q T_up)` through
the interior faces and the boundary outflow ([`compute_implicit_advection_tendency`](@ref)), with
`U⁺ = explicit_energy`, `κ = face_conductivity`, and `C_w q = face_velocity` (`nothing` without water
flow).
"""
@propagate_inbounds function compute_implicit_energy_update(i, j, k, grid, face_conductivity, face_velocity, temperature, explicit_energy, Δt)
    ground_grid = ground_domain(grid)
    Q_diff = compute_implicit_diffusion_tendency(i, j, k, ground_grid, face_conductivity, temperature)
    Q_adv = compute_implicit_advection_tendency(i, j, k, ground_grid, face_velocity, temperature)
    U⁺ = explicit_energy[i, j, k]
    U = U⁺ + Δt * (Q_diff + Q_adv)
    return U
end

# Kernels

@kernel inbounds = true function compute_implicit_heat_coefficients_kernel!(out, grid, fields, args...)
    i, j, k = @index(Global, NTuple)
    compute_implicit_heat_coefficients!(out, i, j, k, grid, fields, args...)
end

@kernel inbounds = true function compute_implicit_energy_rhs_kernel!(rhs, grid, fields, diagonal_coefficient, explicit_energy)
    i, j, k = @index(Global, NTuple)
    rhs[i, j, k] = compute_implicit_energy_rhs(i, j, k, grid, fields, diagonal_coefficient, explicit_energy)
end

@kernel inbounds = true function compute_implicit_energy_update_kernel!(internal_energy, grid, face_conductivity, face_velocity, temperature, explicit_energy, Δt)
    i, j, k = @index(Global, NTuple)
    internal_energy[i, j, k] = compute_implicit_energy_update(i, j, k, grid, face_conductivity, face_velocity, temperature, explicit_energy, Δt)
end
