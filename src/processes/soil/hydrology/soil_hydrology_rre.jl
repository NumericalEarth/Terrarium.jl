"""
    RichardsEq{PS} <: AbstractVerticalFlow

[`SoilHydrology`](@ref) flow operator implementing the mixed saturation-pressure form
of the Richardson-Richards equation.

State variables defined by the Richards' formulation of `SoilHydrology`:

- `saturation_water_ice`: saturation level of water and ice in the pore space.
- `hydraulic_conductivity`: hydraulic conductivity at cell centers (m/s).
- `water_table``: elevation of the water table (m).
- `liquid_water_fraction`: fraction of unfrozen liquid water in the pore space (dimensionless).
- `water_flux`: Darcy flux (m/s, positive upward) through the cell faces that moved the water in the
  step (see [`compute_water_flux`](@ref)).

Excess water that reaches the soil surface is routed to the `surface_excess_water` pool defined by
[`AbstractSurfaceRunoff`](@ref) process (see [`adjust_saturation_profile!`](@ref)), if passed as a dependency.
For standalone soil hydrology with no surface runoff, surface excess water is discarded.

See also [`SoilSaturationPressureClosure`](@ref) and [`AbstractSoilHydraulics`](@ref) for details regarding the
closure relating saturation and pressure head.
"""
@kwdef struct RichardsEq <: AbstractVerticalFlow end

variables(hydrology::SoilHydrology{NF, RichardsEq}) where {NF} = (
    prognostic(:saturation_water_ice, Ground(XYZ()); closure = get_closure(hydrology), bounds = UnitInterval, desc = "Saturation level of water and ice in the pore space"),
    auxiliary(:hydraulic_conductivity, Ground(XYZ(z = Face())), units = u"m/s", desc = "Hydraulic conductivity of soil volumes in m/s"),
    auxiliary(:water_table, Ground(XY()), units = u"m", desc = "Elevation of the water table in meters"),
    auxiliary(:water_flux, Ground(XYZ(z = Face())), units = u"m/s", desc = "Darcy flux through the cell faces in m/s, positive upward"),
    input(:liquid_water_fraction, Ground(XYZ()), default = NF(1), bounds = UnitInterval, desc = "Fraction of unfrozen water in the pore space"),
)

default_timestepping(::SoilHydrology{NF, RichardsEq}) where {NF} = Implicit()

# Top-level interface methods

""" $TYPEDSIGNATURES """
function initialize!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        constants::PhysicalConstants,
        args...
    ) where {NF}
    # Assume saturation is given as initial condition and apply closure!
    # TODO: Though rare, there may be cases where we want to set an initial
    # condition for pressure instead of saturation. This would need to be
    # refactored into another initialization method.
    closure!(state, grid, hydrology.closure, hydrology, soil)
    compute_auxiliary!(state, grid, hydrology, soil, constants)
    return nothing
end

""" $TYPEDSIGNATURES """
function compute_auxiliary!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        args...
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    out = auxiliary_fields(state, hydrology)
    fields = get_fields(state, hydrology, strat, bgc; except = out)
    launch!(grid, XYZ, compute_hydraulics_kernel!, out, fields, hydrology, strat, bgc)
    return nothing
end

"""
    $TYPEDSIGNATURES

Fill the `pressure_head` halo and apply the top/bottom flux boundary conditions for
`saturation_water_ice`. The `fields` passed to those boundary conditions are built explicitly here
(rather than using `state.inputs`, which only carries declared `input` variables) so that a
discrete-form condition like [`saturation_infiltration_bc`](@ref) can read auxiliary fields
(`infiltration`) and per-horizon namespaces (via `strat`/`bgc`) that only exist once the full
`state` does. `infiltration` is only merged in when present, since standalone (non-`LandModel`)
configurations may not declare it at all.
"""
function compute_boundary_conditions!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    fill_halo_regions!(state.pressure_head, state)
    infiltration_field = hasproperty(state, :infiltration) ? (; infiltration = state.infiltration) : (;)
    fields = merge(get_fields(state, hydrology, strat, bgc), infiltration_field)
    compute_z_bcs!(state.tendencies.saturation_water_ice, state.saturation_water_ice, architecture(grid), state.clock, fields)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the saturation tendency of the Richards-equation `hydrology`, with the evapotranspiration sink
of `evtr` (if given) and the user-defined `vwc_forcing` as source terms. `timestepping` is the resolved
[`Timestepping`](@ref) class of `saturation_water_ice` (see
[`timestepping(state, ::Val)`](@ref timestepping)). Under [`Explicit`](@ref) the full divergence of the
Darcy flux is added. Under [`Implicit`](@ref) the fluxes through the interior faces are omitted
(integrated by the implicit sub-stepper, see [`Picard`](@ref)), leaving the fluxes through the two
boundary faces (see [`compute_boundary_darcy_flux_divergence`](@ref)). The flux boundary conditions on
`saturation_water_ice` (e.g. [`InfiltrationFlux`](@ref)) from `compute_boundary_conditions!` and the
source terms are part of the tendency in both cases.

The Darcy fluxes at `tⁿ` through all faces are stored in `water_flux` ([`compute_water_flux`](@ref)),
with which the soil energy advects heat. Under [`Implicit`](@ref) time stepping, the implicit step
overwrites the interior fluxes with those of its conservative update (see [`implicit_update!`](@ref)).
"""
function compute_tendencies!(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration} = nothing,
        timestepping::Timestepping = Explicit()
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    tendencies = tendency_fields(state, hydrology)
    fields = get_fields(state, hydrology, strat, bgc, evtr)
    clock = state.clock
    launch!(
        grid, XYZ, compute_tendencies_kernel!,
        tendencies, clock, fields, hydrology, strat, bgc, constants, evtr, timestepping
    )
    # Darcy fluxes at tⁿ through all faces (pressure head halos are filled)
    launch!(grid, XYZ, compute_water_flux_kernel!, state.water_flux, state.pressure_head, state.hydraulic_conductivity)
    return nothing
end

# Kernel functions

"""
    $SIGNATURES

Compute the volumetric water content (VWC) tendency at grid cell `i, j k` according to the
Richardson-Richards equation. Note that the VWC tendency is not scaled by the porosity and
is thus not the same as the saturation tendency. Under [`Explicit`](@ref) time stepping (the default)
the full divergence of the Darcy flux is included; under [`Implicit`](@ref) only the boundary-face
fluxes are ([`compute_boundary_darcy_flux_divergence`](@ref)), the interior ones being integrated by the
implicit sub-stepper.
"""
@propagate_inbounds function compute_volumetric_water_content_tendency(
        i, j, k, grid, clock, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        constants::PhysicalConstants,
        evapotranspiration::Optional{AbstractEvapotranspiration},
        timestepping::Timestepping = Explicit()
    ) where {NF}
    # Compute divergence of water fluxes
    # ∂θ∂t = ∇⋅K(θ)∇Ψ + S, where Ψ = ψₘ + ψₕ + ψz, and S is a forcing term for sources and sinks such as ET losses
    ∂θ∂t = (
        - compute_darcy_flux_divergence(i, j, k, grid, fields, timestepping)                     # Darcy flux
            + forcing(i, j, k, grid, clock, fields, evapotranspiration, hydrology, constants)  # ET source/sink
            + forcing(i, j, k, grid, clock, fields, hydrology.vwc_forcing, hydrology)          # generic user-defined forcing
    )
    return ∂θ∂t
end

"""
    $SIGNATURES

Divergence `∂q/∂z` (1/s) of the [`darcy_flux`](@ref) `q` at cell `i, j, k`: over all faces under
[`Explicit`](@ref) time stepping, and over the boundary faces of the column only (see
[`compute_boundary_darcy_flux_divergence`](@ref)) under [`Implicit`](@ref).
"""
@propagate_inbounds function compute_darcy_flux_divergence(i, j, k, grid, fields, ::Explicit)
    ground_grid = ground_domain(grid)
    return ∂zᵃᵃᶜ(i, j, k, ground_grid, darcy_flux, fields.pressure_head, fields.hydraulic_conductivity)
end

@propagate_inbounds compute_darcy_flux_divergence(i, j, k, grid, fields, ::Implicit) =
    compute_boundary_darcy_flux_divergence(i, j, k, grid, fields)

"""
    $SIGNATURES

Part of the Darcy flux divergence at `i, j, k` due to the two boundary faces of the column (bottom face
of cell `k = 1`, top face of cell `k = Nz`), omitting all interior faces:

    (δ_{k,Nz} q_{k+½} - δ_{k,1} q_{k-½}) / Δz_k,

with `q` the [`darcy_flux`](@ref). This is the explicit part when the interior fluxes are integrated
implicitly (see [`Picard`](@ref)). The boundary fluxes are computed from the `pressure_head` halo: they
vanish for the default (no-flux) conditions and carry the prescribed flux for a gradient condition on
`pressure_head`. Flux boundary conditions on `saturation_water_ice` (e.g. [`InfiltrationFlux`](@ref))
are added separately by `compute_boundary_conditions!`.
"""
@propagate_inbounds function compute_boundary_darcy_flux_divergence(i, j, k, grid, fields)
    ground_grid = ground_domain(grid)
    Nz = size(ground_grid, 3)
    ψ = fields.pressure_head
    K = fields.hydraulic_conductivity
    # Darcy fluxes through the lower (k) and upper (k + 1) faces of cell k
    q_lower = darcy_flux(i, j, k, ground_grid, ψ, K)
    q_upper = darcy_flux(i, j, k + 1, ground_grid, ψ, K)
    # keep only the boundary faces
    q_bottom = ifelse(k == 1, q_lower, zero(q_lower))
    q_top = ifelse(k == Nz, q_upper, zero(q_upper))
    return (q_top - q_bottom) / Δzᵃᵃᶜ(i, j, k, ground_grid)
end

"""
    $SIGNATURES

Kernel function for computing the Darcy flux `q = -K ∂ψ/∂z` (m/s, positive upward) over layer faces from
the total pressure head `ψ` and the face hydraulic conductivity `K` (see [`darcy_conductivity`](@ref)).
"""
@propagate_inbounds function darcy_flux(i, j, k, grid, ψ, K)
    # Darcy's law: q = -K ∂ψ/∂z
    # TODO: also account for effect of temperature on matric potential
    ∇ψ = ∂zᵃᵃᶠ(i, j, k, grid, ψ)
    Kₖ = darcy_conductivity(i, j, k, grid, ψ, K)
    ## Note that elevation and hydrostatic pressure are assumed to be accounted
    ## for in the computation of ψ, so we don't need any additional terms.
    q = -Kₖ * ∇ψ
    return q
end

"""
    $SIGNATURES

Darcy flux `q = -K ∂Ψ/∂z` (m/s, positive upward) through face `i, j, k` ([`darcy_flux`](@ref)), stored
in `water_flux` by `compute_tendencies!` at `tⁿ`. At the boundary faces it is the flux implied by the
boundary conditions on `pressure_head` (zero for no-flux, see
[`compute_boundary_darcy_flux_divergence`](@ref)).
"""
@propagate_inbounds compute_water_flux(i, j, k, grid, ψ, K) = darcy_flux(i, j, k, ground_domain(grid), ψ, K)

"""
    $SIGNATURES

Hydraulic conductivity (m/s) used by the [`darcy_flux`](@ref) at face `k`: the minimum of `K` (defined
on faces) at face `k` and at the adjacent face in the direction of the gradient of the total pressure
head `ψ` (face `k - 1` for `∂ψ/∂z < 0`, else `k + 1`).
"""
@propagate_inbounds function darcy_conductivity(i, j, k, grid, ψ, K)
    ∇ψ = ∂zᵃᵃᶠ(i, j, k, grid, ψ)
    ## Take minimum of hydraulic conductivities in the direction of flow
    Kₖ = (∇ψ < 0) * min(K[i, j, k - 1], K[i, j, k]) +
        (∇ψ >= 0) * min(K[i, j, k], K[i, j, k + 1])
    return Kₖ
end

"""
    $SIGNATURES

Matric potential `ψₘ = Ψ - ψ_h - ψ_z` (m) at cell `i, j, k`, reconstructed from the total
`pressure_head` `Ψ` by removing the hydrostatic head `ψ_h = max(0, z₀ - z)` below the water table `z₀`
and the elevation head `ψ_z = z - z_ref` relative to the soil surface `z_ref`, as in
[`SoilSaturationPressureClosure`](@ref).
"""
@propagate_inbounds function compute_matric_potential(i, j, k, grid, fields)
    ground_grid = ground_domain(grid)
    Ψ = fields.pressure_head[i, j, k]
    z = znode(i, j, k, ground_grid, Center(), Center(), Center())
    z_ref = znode(i, j, ground_grid.Nz + 1, ground_grid, Center(), Center(), Face())
    ψz = z - z_ref
    z₀ = fields.water_table[i, j, end]
    ψh = max(zero(z), z₀ - z)
    return Ψ - ψh - ψz
end

"""
    $SIGNATURES

Specific moisture capacity `∂θ/∂ψ` (1/m) at cell `i, j, k`: the derivative of the soil water retention
curve ([`specific_moisture_capacity`](@ref)) at the matric potential ([`compute_matric_potential`](@ref))
reconstructed from the total `pressure_head`. It vanishes in saturated cells.
"""
@propagate_inbounds function compute_specific_moisture_capacity(
        i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    ψm = compute_matric_potential(i, j, k, grid, fields)
    swrc = get_swrc(hydrology)
    por = porosity(i, j, k, grid, fields, strat, bgc)
    return convert(NF, specific_moisture_capacity(swrc, ψm, por))
end

"""
    specific_moisture_capacity(swrc::SWRC, ψ, θsat)

Specific moisture capacity `∂θ/∂ψ` (1/m) of the soil water retention curve `swrc` at the matric
potential `ψ` (m) for the saturated water content `θsat`, from the derivative provided by FreezeCurves.jl;
zero for saturated conditions (`ψ` at or above the [`air_entry_potential`](@ref)). The derivative is
always evaluated at an unsaturated potential, so that the saturated branch of the curve, where the
derivatives of FreezeCurves.jl involve powers of zero, never enters the result or its adjoint.

```jldoctest
julia> swrc = VanGenuchten(α = 2.0, n = 2.0);

julia> Terrarium.specific_moisture_capacity(swrc, 0.0, 0.4)
0.0

julia> Terrarium.specific_moisture_capacity(swrc, -0.5, 0.4) ≈ swrc(Terrarium.FreezeCurves.derivative, -0.5; θsat = 0.4)
true
```
"""
@inline function specific_moisture_capacity(swrc::SWRC, ψ, θsat)
    # TODO: fix upstream in FreezeCurves.jl, whose SWRC derivatives evaluate both branches of `ifelse`
    # (powers of zero at saturation, NaN adjoints for n < 2) and return an `Int` zero (`VanGenuchten`) or
    # `θsat` (`BrooksCorey`) in the saturated branch; then call `swrc(FreezeCurves.derivative, ψ; θsat)`
    # directly and remove the guard and `air_entry_potential`.
    ψ_ae = air_entry_potential(swrc, ψ)
    unsaturated = ψ < ψ_ae
    # evaluate the derivative at an unsaturated potential (1 m below air entry where saturated)
    ψ_unsat = ifelse(unsaturated, ψ, ψ_ae - one(ψ))
    ∂θ∂ψ = convert(typeof(ψ), swrc(FreezeCurves.derivative, ψ_unsat; θsat))
    return ifelse(unsaturated, ∂θ∂ψ, zero(∂θ∂ψ))
end

"""
    air_entry_potential(swrc::SWRC, ψ)

Matric potential (m) above which the soil water retention curve `swrc` is saturated, in the number type
of `ψ`: zero by default and for `VanGenuchten`, and `-ψₛ` for `BrooksCorey`.
"""
@inline air_entry_potential(::SWRC, ψ) = zero(ψ)
@inline air_entry_potential(swrc::BrooksCorey, ψ) = -convert(typeof(ψ), swrc.ψₛ)

"""
    $SIGNATURES

Compute the hydraulic conductivity at the center of the grid cell `i, j, k`.
"""
@propagate_inbounds function hydraulic_conductivity(
        i, j, k, grid, fields,
        hydrology::SoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    soil = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    return hydraulic_conductivity(hydrology.hydraulic_properties, soil)
end

"""
    $TYPEDSIGNATURES

Computes the unsaturated hydraulic conductivity for `RichardsEq` configurations of `SoilHydrology`.
"""
@propagate_inbounds function compute_hydraulics!(
        out, i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    # Get underlying grid
    ground_grid = ground_domain(grid)
    # compute hydraulic conductivity
    Nz = ground_grid.Nz
    k_below = ifelse(k >= Nz, Nz, max(k - 1, one(k)))
    k_above = min(k, Nz)
    K_face = min(
        hydraulic_conductivity(i, j, k_below, grid, fields, hydrology, strat, bgc),
        hydraulic_conductivity(i, j, k_above, grid, fields, hydrology, strat, bgc)
    )
    @inbounds out.hydraulic_conductivity[i, j, k] = K_face
    k_upper = ifelse(k >= Nz, Nz + one(Nz), k)
    @inbounds out.hydraulic_conductivity[i, j, k_upper] = K_face
    return nothing
end

# Kernels

# Each cell writes its lower face; the top cell also writes the top face Nz + 1.
@kernel inbounds = true function compute_water_flux_kernel!(water_flux, grid, ψ, K)
    i, j, k = @index(Global, NTuple)
    Nz = size(ground_domain(grid), 3)
    water_flux[i, j, k] = compute_water_flux(i, j, k, grid, ψ, K)
    k_top = ifelse(k == Nz, Nz + 1, k)
    water_flux[i, j, k_top] = compute_water_flux(i, j, k_top, grid, ψ, K)
end

@kernel inbounds = true function compute_tendencies_kernel!(
        tend, grid, clock, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration},
        timestepping::Timestepping
    ) where {NF}
    i, j, k = @index(Global, NTuple)
    compute_saturation_tendency!(tend.saturation_water_ice, i, j, k, grid, clock, fields, hydrology, strat, bgc, constants, evtr, timestepping)
end
