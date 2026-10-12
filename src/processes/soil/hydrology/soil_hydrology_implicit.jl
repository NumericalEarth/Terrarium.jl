# Implicit Richards flow: implicit operator and process hooks of `SoilHydrology{NF, RichardsEq}`, after the
# mass-conservative modified Picard iteration of Celia et al. (1990).
#
# With porosity φ, saturation s, volumetric water content θ = φ s, and total pressure head
# Ψ = ψₘ(θ) + ψ_z + ψ_h (matric, elevation, and hydrostatic head), the backward Euler step reads
#
#     φ (sⁿ⁺¹ - sⁿ) / Δt = ∂z(K ∂z Ψⁿ⁺¹) + φ G_exp,
#
# where G_exp holds the flux boundary conditions on saturation, the Darcy fluxes through the boundary faces,
# and the sources (evapotranspiration, forcing). With ψ_z and ψ_h lagged, ∂θ/∂Ψ = C, the specific moisture
# capacity. Linearizing θ about iterate m with lagged conductivity Kᵐ gives, with s⁺ = sⁿ + Δt G_exp, the
# tridiagonal system for the total pressure head
#
#     Cᵐ Ψᵐ⁺¹ - Δt ∂z(Kᵐ ∂z Ψᵐ⁺¹) = Cᵐ Ψᵐ - φ (sᵐ - s⁺).
#
# Each `Picard` iteration m = 0, …, M-1, starting from s⁰ = sⁿ, calls the hooks:
#
#   1. `linearize!`: hydraulic conductivity, lagged Kᵐ (faces) and Cᵐ (centers, bounded below by
#      `minimum_moisture_capacity` so that frozen saturated cells stay regular), and the right-hand side;
#   2. `solve!`: the system above for Ψᵐ⁺¹;
#   3. `implicit_update!`: sᵐ⁺¹ = s⁺ + Δt ∂z(Kᵐ ∂z Ψᵐ⁺¹) / φ through the interior faces, which conserves water
#      exactly, and `water_flux` ← -Kᵐ ∂z Ψᵐ⁺¹, the flux with which the heat step advects heat;
#   4. `implicit_closure!`: redistribution of over- and undersaturation, water table, Ψᵐ⁺¹, and hydraulic
#      conductivity. Since every iteration restarts from s⁺, only the final one routes surface excess water
#      to the runoff pool and moves the heat of redistributed water.
#
# The solver calls the hooks through the `SoilModel`/`LandModel` forwarding methods and the
# `SoilEnergyWaterCarbon` methods in `soil_coupled.jl`.

# Process level: `SoilHydrology{NF, RichardsEq}`

"""
    $TYPEDSIGNATURES

Fallback for hydrology processes without an implicit operator; throws an `ArgumentError`.
"""
function implicit_operator(hydrology::AbstractSoilHydrology, state, grid)
    throw(
        ArgumentError(
            "implicit soil water flow requires a `SoilHydrology` with `RichardsEq` vertical flow, got $(nameof(typeof(hydrology)))"
        )
    )
end

"""
    $TYPEDSIGNATURES

Allocate the [`TridiagonalOperator`](@ref) of `saturation_water_ice` on `ground_domain(grid)`. Throws
an `ArgumentError` unless the closure is the [`SoilSaturationPressureClosure`](@ref) and `pressure_head`
has no value boundary conditions.
"""
function implicit_operator(hydrology::SoilHydrology{NF, RichardsEq}, state, grid) where {NF}
    get_closure(hydrology) isa SoilSaturationPressureClosure || throw(
        ArgumentError(
            "implicit soil water flow requires the `SoilSaturationPressureClosure`, got $(nameof(typeof(get_closure(hydrology))))"
        )
    )
    bcs = state.pressure_head.boundary_conditions
    for (side, bc) in ((:top, bcs.top), (:bottom, bcs.bottom))
        bc isa BoundaryCondition{<:Value} && throw(
            ArgumentError(
                "value boundary conditions on `pressure_head` (found at the $side boundary) are not yet supported when `saturation_water_ice` is integrated implicitly; use a flux boundary condition on `saturation_water_ice` (e.g. `InfiltrationFlux`), a gradient boundary condition on `pressure_head`, or an explicit timestepper"
            )
        )
    end
    return TridiagonalOperator(ground_domain(grid))
end

"""
    $TYPEDSIGNATURES

Update the hydraulic conductivity ([`compute_hydraulics!`](@ref)), fill the lagged capacity and face
conductivity of `operator` ([`compute_implicit_water_flow_coefficients!`](@ref)), and assemble the
right-hand side `C Ψᵐ - φ (sᵐ - s⁺)` in `rhs` ([`compute_implicit_water_flow_rhs`](@ref)), with
`s⁺ = explicit_saturation`. [CeliaEtAl1990](@cite) recommend at least two iterations.

# References

* [CeliaEtAl1990](@cite) Celia et al., Water Resources Research 26, 1483–1496 (1990)
"""
function linearize!(
        operator::TridiagonalOperator, rhs, state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        constants::PhysicalConstants,
        explicit_saturation, Δt
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    (; diagonal_coefficient, face_conductivity) = operator
    coefficients = (; diagonal_coefficient, face_conductivity)
    compute_hydraulics!(state, grid, hydrology, soil)
    fields = get_fields(state, hydrology, strat, bgc)
    launch!(grid, XYZ, compute_implicit_water_flow_coefficients_kernel!, coefficients, fields, hydrology, strat, bgc)
    launch!(grid, XYZ, compute_implicit_water_flow_rhs_kernel!, rhs, fields, diagonal_coefficient, explicit_saturation, strat, bgc)
    return nothing
end

"""
    $TYPEDSIGNATURES

Update `saturation_water_ice` conservatively from the solved pressure head
([`compute_implicit_saturation_update`](@ref)) and store the interior Darcy fluxes in `water_flux`
([`compute_implicit_water_flux`](@ref)).
"""
function implicit_update!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        operator::TridiagonalOperator,
        explicit_saturation, Δt
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    fields = get_fields(state, hydrology, strat, bgc)
    launch!(
        grid, XYZ, compute_implicit_saturation_update_kernel!,
        state.saturation_water_ice, fields, operator.face_conductivity, explicit_saturation, strat, bgc, Δt
    )
    launch!(grid, XYZ, compute_implicit_water_flux_kernel!, state.water_flux, operator.face_conductivity, state.pressure_head)
    return nothing
end

"""
    $TYPEDSIGNATURES

Redistribute over- and undersaturation ([`adjust_saturation_profile!`](@ref)), then update the water
table, pressure head, and hydraulic conductivity. Only in the final `iteration` is the surface excess
routed to `runoff` and the heat of the redistributed water moved.
"""
function implicit_closure!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        constants::PhysicalConstants,
        runoff::Optional{AbstractSurfaceRunoff},
        iteration::Iteration
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    # route the surface excess and move heat only once per step, in the final iteration
    runoff = isfinal(iteration) ? runoff : nothing
    energy = isfinal(iteration) ? get_energy_balance(soil) : nothing
    adjust_saturation_profile!(state, grid, hydrology, runoff, strat, bgc, energy)
    compute_water_table!(state, grid, hydrology)
    compute_pressure_head!(state, grid, get_closure(hydrology), hydrology, soil)
    compute_hydraulics!(state, grid, hydrology, soil)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Store the hydraulic conductivity (m/s) of the current iterate at face `i, j, k` in `face_conductivity`,
as in the explicit [`darcy_flux`](@ref) ([`darcy_conductivity`](@ref)).
"""
@propagate_inbounds function compute_implicit_conductivity!(
        face_conductivity, i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, RichardsEq}
    ) where {NF}
    ground_grid = ground_domain(grid)
    face_conductivity[i, j, k] = darcy_conductivity(i, j, k, ground_grid, fields.pressure_head, fields.hydraulic_conductivity)
    return nothing
end

"""
    $TYPEDSIGNATURES

Store the specific moisture capacity `C = ∂θ/∂ψ` (1/m) of the current iterate at cell `i, j, k` in
`diagonal_coefficient` ([`compute_specific_moisture_capacity`](@ref)), bounded below by
[`minimum_moisture_capacity`](@ref).
"""
@propagate_inbounds function compute_implicit_capacity!(
        diagonal_coefficient, i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    C = compute_specific_moisture_capacity(i, j, k, grid, fields, hydrology, strat, bgc)
    C_min = minimum_moisture_capacity(get_hydraulic_properties(hydrology))
    diagonal_coefficient[i, j, k] = max(C, C_min)
    return nothing
end

"""
    $TYPEDSIGNATURES

Store the capacity ([`compute_implicit_capacity!`](@ref)) and face conductivity
([`compute_implicit_conductivity!`](@ref)) at `i, j, k` in `out`.
"""
@propagate_inbounds function compute_implicit_water_flow_coefficients!(
        out, i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    compute_implicit_capacity!(out.diagonal_coefficient, i, j, k, grid, fields, hydrology, strat, bgc)
    compute_implicit_conductivity!(out.face_conductivity, i, j, k, grid, fields, hydrology)
    return nothing
end

"""
    $TYPEDSIGNATURES

Right-hand side `C Ψᵐ - φ (sᵐ - s⁺)` at `i, j, k`, with `C = diagonal_coefficient` and
`s⁺ = explicit_saturation`.
"""
@propagate_inbounds function compute_implicit_water_flow_rhs(
        i, j, k, grid, fields, diagonal_coefficient, explicit_saturation,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    ∂θ∂Ψ = diagonal_coefficient[i, j, k]
    Ψ = fields.pressure_head[i, j, k]
    s = fields.saturation_water_ice[i, j, k]
    s⁺ = explicit_saturation[i, j, k]
    φ = porosity(i, j, k, grid, fields, strat, bgc)
    return ∂θ∂Ψ * Ψ - φ * (s - s⁺)
end

"""
    $TYPEDSIGNATURES

Saturation `s = s⁺ + Δt Q_diff / φ` at `i, j, k`, with `Q_diff = ∂z(K ∂z Ψ)` (1/s) through the interior
faces ([`compute_implicit_diffusion_tendency`](@ref)), `s⁺ = explicit_saturation`, and
`K = face_conductivity`.
"""
@propagate_inbounds function compute_implicit_saturation_update(
        i, j, k, grid, fields, face_conductivity, explicit_saturation,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        Δt
    )
    ground_grid = ground_domain(grid)
    Q_diff = compute_implicit_diffusion_tendency(i, j, k, ground_grid, face_conductivity, fields.pressure_head)
    φ = porosity(i, j, k, grid, fields, strat, bgc)
    s⁺ = explicit_saturation[i, j, k]
    s = s⁺ + Δt * Q_diff / φ
    return s
end

"""
    $TYPEDSIGNATURES

Darcy flux `q = -K ∂z Ψ` (m/s, positive upward) at interior face `i, j, k`, with `K = face_conductivity`;
at the bottom face, the current `water_flux` (from `tⁿ`) is kept.
"""
@propagate_inbounds function compute_implicit_water_flux(i, j, k, grid, water_flux, face_conductivity, pressure_head)
    ground_grid = ground_domain(grid)
    q = -face_conductivity[i, j, k] * ∂zᵃᵃᶠ(i, j, k, ground_grid, pressure_head)
    return ifelse(k == 1, water_flux[i, j, k], q)
end

# Kernels

# Faces k = 1, …, Nz; the top boundary face Nz + 1 is not written.
@kernel inbounds = true function compute_implicit_water_flux_kernel!(water_flux, grid, face_conductivity, pressure_head)
    i, j, k = @index(Global, NTuple)
    water_flux[i, j, k] = compute_implicit_water_flux(i, j, k, grid, water_flux, face_conductivity, pressure_head)
end

@kernel inbounds = true function compute_implicit_water_flow_coefficients_kernel!(out, grid, fields, args...)
    i, j, k = @index(Global, NTuple)
    compute_implicit_water_flow_coefficients!(out, i, j, k, grid, fields, args...)
end

@kernel inbounds = true function compute_implicit_water_flow_rhs_kernel!(rhs, grid, fields, diagonal_coefficient, explicit_saturation, strat, bgc)
    i, j, k = @index(Global, NTuple)
    rhs[i, j, k] = compute_implicit_water_flow_rhs(i, j, k, grid, fields, diagonal_coefficient, explicit_saturation, strat, bgc)
end

@kernel inbounds = true function compute_implicit_saturation_update_kernel!(saturation_water_ice, grid, fields, face_conductivity, explicit_saturation, strat, bgc, Δt)
    i, j, k = @index(Global, NTuple)
    saturation_water_ice[i, j, k] = compute_implicit_saturation_update(i, j, k, grid, fields, face_conductivity, explicit_saturation, strat, bgc, Δt)
end
