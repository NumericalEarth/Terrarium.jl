# Implicit snow meltwater drainage: implicit operator and process hooks of `SingleLayerSnow` for
# `snow_water_equivalent`.
#
# Drainage removes liquid water at the zero-enthalpy reference (liquid water at 0 °C), so it changes neither the
# snow energy Ū nor the ice content W_i = (1 − θ_liq) W = −Ū / (ρ_w L_f), clamped to [0, W]. With Ū⁺ and
# W⁺ = Wⁿ + Δt G_exp from the explicit stage (snowfall, rain, and sublimation), the backward Euler step of the
# drainage M_r(θ_liq, W) is a scalar equation per column, written in the liquid fraction θ via W = W_i / (1 − θ):
#
#     G(θ) = W_i / (1 − θ) + Δt M_r(θ, W_i / (1 − θ)) − W⁺ = 0.
#
# Each `Picard` iteration takes one Newton step: linearizing G about the current iterate θᵐ,
# G(θ) ≈ G(θᵐ) + G′(θᵐ) (θ − θᵐ), and setting this to zero gives the diagonal linear system
#
#     G′(θᵐ) θᵐ⁺¹ = G′(θᵐ) θᵐ − G(θᵐ).
#
# The outflow is the minimum of the Darcy rate and the storage limit (see `compute_meltwater_outflow`), so G is
# increasing in θ and convex on each branch. Within a branch, a Newton step from any θ ≥ θ* (the root) lands in
# [θ*, θ], so the iterates approach the root from above; a step across the switch between the branches may
# undershoot, so the update bounds θ below by min(L_c, θ⁺), at or below the root. Each iteration calls the hooks:
#
#   1. `linearize!`: the diagonal G′(θᵐ) and right-hand side G′(θᵐ) θᵐ − G(θᵐ) of the `DiagonalOperator`, at
#      the liquid fraction θᵐ of the current iterate Wᵐ, bounded to [0, θ⁺];
#   2. `solve!`: θᵐ⁺¹ in `snow_liquid_fraction`, the target of the implicit operator (see `implicit_variable`);
#   3. `implicit_update!`: Wᵐ⁺¹ = W_i / (1 − θᵐ⁺¹) with min(L_c, θ⁺) ≤ θᵐ⁺¹ ≤ θ⁺, since the iteration starts from Wⁿ, which
#      may lie below the root, and `meltwater_flux` ← (W⁺ − Wᵐ⁺¹) / Δt, the drainage passed on to the surface
#      hydrology in the next step;
#   4. `implicit_closure!`: the snow closure at the new iterate.
#
# A fully liquid pack (W_i = 0, θ = 1) drains at min(K_sat, (1 − L_c) W / τ_m), so its backward Euler step has
# the exact solution max(W⁺ − Δt K_sat, W⁺ / (1 + Δt (1 − L_c) / τ_m)), independent of the iterate.

"""
    implicit_operator(snow::AbstractSnow, state, grid, ::Val{name})

Return the implicit operator of the snow prognostic variable `name`: a [`DiagonalOperator`](@ref) for the
meltwater drainage of `snow_water_equivalent` (see [`linearize!`](@ref)). The fallback throws an
`ArgumentError`.
"""
function implicit_operator(snow::AbstractSnow, state, grid, ::Val{name}) where {name}
    throw(
        ArgumentError(
            "no implicit operator is defined for variable `$name` of $(nameof(typeof(snow))); route the variable explicitly"
        )
    )
end

implicit_operator(snow::SingleLayerSnow, state, grid, ::Val{:snow_water_equivalent}) = DiagonalOperator(similar(state.snow_water_equivalent))

# The meltwater drainage is linearized in the liquid fraction.
implicit_variable(::AbstractVariable{:snow_water_equivalent}) = :snow_liquid_fraction

"""
    $TYPEDSIGNATURES

Fill the diagonal and right-hand side of the Newton step for the liquid fraction of the meltwater drainage
([`compute_implicit_meltwater_coefficients!`](@ref)), given the explicit stage `W⁺ = explicit_water`.
"""
function linearize!(
        operator::DiagonalOperator, rhs, state, grid,
        snow::SingleLayerSnow,
        constants::PhysicalConstants,
        explicit_water, Δt,
        ::Iteration{:snow_water_equivalent}
    )
    out = (; diagonal = operator.diagonal, rhs)
    fields = get_fields(state, snow)
    launch!(grid, XY, compute_implicit_meltwater_coefficients_kernel!, out, fields, explicit_water, snow, constants, Δt)
    return nothing
end

"""
    $TYPEDSIGNATURES

Update `snow_water_equivalent` from the solved liquid fraction and store the drainage of the step in
`meltwater_flux` ([`compute_implicit_snow_water_update!`](@ref)).
"""
function implicit_update!(
        state, grid,
        snow::SingleLayerSnow,
        constants::PhysicalConstants,
        operator::DiagonalOperator,
        explicit_water, Δt,
        ::Iteration{:snow_water_equivalent}
    )
    out = (; state.snow_water_equivalent, state.meltwater_flux)
    fields = get_fields(state, snow)
    launch!(grid, XY, compute_implicit_snow_water_update_kernel!, out, fields, explicit_water, snow, constants, Δt)
    return nothing
end

"""
    $TYPEDSIGNATURES

Re-evaluate the snow closure (temperature and liquid fraction) at the new iterate of `snow_water_equivalent`.
"""
function implicit_closure!(state, grid, snow::SingleLayerSnow, constants::PhysicalConstants, ::Iteration{:snow_water_equivalent})
    closure!(state, grid, snow, constants)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Ice content `W_i = −Ū⁺ / (ρ_w L_f)` (m SWE) of the snowpack at grid cell `i, j`, clamped to `[0, W⁺]`, from the
explicit stage `W⁺ = explicit_water` and the snow energy `Ū⁺`; unchanged by the meltwater drainage.
"""
@propagate_inbounds function compute_snow_ice_content(i, j, grid, fields, explicit_water, constants::PhysicalConstants)
    ρ_w = constants.material.density_water
    L_f = constants.thermodynamics.latent_heat_fusion
    W⁺ = explicit_water[i, j]
    Ū⁺ = fields.snow_energy[i, j]
    return clamp(-Ū⁺ / (ρ_w * L_f), zero(W⁺), max(W⁺, zero(W⁺)))
end

"""
    $TYPEDSIGNATURES

Diagonal `G′(θᵐ)` and right-hand side `G′(θᵐ) θᵐ − G(θᵐ)` at grid cell `i, j` of the Newton step for the liquid
fraction, with `G(θ) = W_i / (1 − θ) + Δt M_r(θ, W_i / (1 − θ)) − W⁺`, the storage-limited outflow `M_r`
([`compute_meltwater_outflow`](@ref)), the ice content `W_i`
([`compute_snow_ice_content`](@ref)), and `W⁺ = explicit_water`. The linearization point is the liquid fraction
`1 − W_i / Wᵐ` of the current iterate, bounded to `[0, θ⁺]`. The coefficients are the identity for a pack
without ice or snow, which [`compute_implicit_snow_water_update!`](@ref) treats separately.
"""
@propagate_inbounds function compute_implicit_meltwater_coefficients!(
        out, i, j, grid, fields, explicit_water,
        snow::SingleLayerSnow{NF},
        constants::PhysicalConstants,
        Δt
    ) where {NF}
    hydraulics = snow.hydraulic_properties
    W⁺ = explicit_water[i, j]
    Wᵐ = fields.snow_water_equivalent[i, j]
    W_i = compute_snow_ice_content(i, j, grid, fields, explicit_water, constants)
    has_ice = W_i > zero(NF)
    # liquid fractions of the explicit stage and of the current iterate
    θ⁺ = one(NF) - W_i / ifelse(has_ice, W⁺, one(NF))
    θᵐ = one(NF) - W_i / ifelse(Wᵐ > zero(NF), Wᵐ, W⁺)
    # θ < 1 for a pack with ice; a pack without ice (θ⁺ = 1), whose coefficients are unused, is evaluated at θ = 0
    # so that no branch divides by zero
    θ = ifelse(has_ice, clamp(θᵐ, zero(NF), θ⁺), zero(NF))
    ice_fraction = one(NF) - θ
    W = W_i / ice_fraction
    G = W + Δt * compute_meltwater_outflow(hydraulics, θ, W) - W⁺
    ∂G∂θ = W_i / ice_fraction^2 + Δt * compute_meltwater_outflow_derivative(hydraulics, θ, W_i)
    out.diagonal[i, j, end] = ifelse(has_ice, ∂G∂θ, one(NF))
    out.rhs[i, j, end] = ifelse(has_ice, ∂G∂θ * θ - G, θ⁺)
    return nothing
end

"""
    $TYPEDSIGNATURES

Snow water equivalent `W = W_i / (1 − θ)` at grid cell `i, j` from the solved liquid fraction
`θ = snow_liquid_fraction`, bounded to `[min(L_c, θ⁺), θ⁺]` with the capillary retention `L_c`, so that
`W ≤ W⁺ = explicit_water` and a Newton step that undershoots the root at the switch between the branches of the
outflow drains neither ice nor capillary water, and the drainage `meltwater_flux = (W⁺ − W) / Δt`. A pack
without ice (`W_i = 0`) takes the exact solution `W = max(W⁺ − Δt K_sat, W⁺ / (1 + Δt (1 − L_c) / τ_m))` of its
step, and a step without snow (`W⁺ ≤ 0`) leaves `W = W⁺`.
"""
@propagate_inbounds function compute_implicit_snow_water_update!(
        out, i, j, grid, fields, explicit_water,
        snow::SingleLayerSnow{NF},
        constants::PhysicalConstants,
        Δt
    ) where {NF}
    hydraulics = snow.hydraulic_properties
    W⁺ = explicit_water[i, j]
    W_i = compute_snow_ice_content(i, j, grid, fields, explicit_water, constants)
    has_ice = W_i > zero(NF)
    θ⁺ = one(NF) - W_i / ifelse(has_ice, W⁺, one(NF))
    θ_min = min(hydraulics.capillary_retention, θ⁺)
    θ = clamp(fields.snow_liquid_fraction[i, j], θ_min, θ⁺)
    W_newton = W_i / ifelse(has_ice, one(NF) - θ, one(NF))
    # fully liquid pack: drainage min(K_sat, (1 − L_c) W / τ_m), solved exactly
    K_sat = hydraulics.saturated_conductivity
    storage_rate = (one(NF) - hydraulics.capillary_retention) / hydraulics.drainage_timescale
    W_liquid = max(W⁺ - Δt * K_sat, W⁺ / (one(NF) + Δt * storage_rate))
    W = ifelse(W⁺ > zero(NF), ifelse(has_ice, W_newton, W_liquid), W⁺)
    out.snow_water_equivalent[i, j, end] = W
    out.meltwater_flux[i, j, end] = (W⁺ - W) / Δt
    return nothing
end

# Kernels

@kernel inbounds = true function compute_implicit_meltwater_coefficients_kernel!(out, grid, fields, explicit_water, snow::SingleLayerSnow, args...)
    i, j = @index(Global, NTuple)
    compute_implicit_meltwater_coefficients!(out, i, j, grid, fields, explicit_water, snow, args...)
end

@kernel inbounds = true function compute_implicit_snow_water_update_kernel!(out, grid, fields, explicit_water, snow::SingleLayerSnow, args...)
    i, j = @index(Global, NTuple)
    compute_implicit_snow_water_update!(out, i, j, grid, fields, explicit_water, snow, args...)
end
