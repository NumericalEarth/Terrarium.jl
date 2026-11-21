"""
    $TYPEDEF

Base type for schemes that determine the hydraulic conductivity at the faces between two soil cells, through which
the Darcy flux of the Richards equation flows (see [`compute_face_hydraulic_conductivity`](@ref)).
"""
abstract type AbstractFaceConductivity end

"""
    $TYPEDEF

Face conductivity at the mean liquid water content over the mean porosity of the two adjacent cells, i.e. a
porosity-weighted average of their liquid water saturation, as in the interface conductivity of CLM5
[lawrenceTechnicalDescriptionVersion2018; Eq. (7.47)](@cite). It lets water drain from a wet cell into a drier one at
a rate set by both cells, while it remains negligible between two dry cells.

# References

* [lawrenceTechnicalDescriptionVersion2018](@cite) Lawrence et al., Technical Description of Version 5.0 of the Community Land Model (CLM), NCAR (2018)
"""
struct WeightedAverageConductivity <: AbstractFaceConductivity end

"""
    $TYPEDEF

Upstream-weighted face conductivity: the conductivity of the cell from which the water flows, determined by the sign
of the gradient of the total hydraulic head across the face. Upstream weighting is the standard choice for robust
infiltration into dry soil, since a wetting front advances at the conductivity of the wet cell behind it
[forsythRobustNumericalMethods1995](@cite).

# References

* [forsythRobustNumericalMethods1995](@cite) Forsyth et al., Advances in Water Resources (1995)
"""
struct UpwindConductivity <: AbstractFaceConductivity end

"""
    $TYPEDEF

[`SoilHydrology`](@ref) flow operator implementing the mixed saturation-pressure form
of the Richardson-Richards equation.

The hydraulic conductivity at the cell faces is determined by the `face_conductivity` scheme
([`AbstractFaceConductivity`](@ref)), either [`WeightedAverageConductivity`](@ref) (default) or
[`UpwindConductivity`](@ref).

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
@kwdef struct RichardsEq{FC <: AbstractFaceConductivity} <: AbstractVerticalFlow
    "Scheme for the hydraulic conductivity at the cell faces"
    face_conductivity::FC = WeightedAverageConductivity()
end

variables(hydrology::SoilHydrology{NF, <:RichardsEq}) where {NF} = (
    prognostic(:saturation_water_ice, Ground(XYZ()); closure = get_closure(hydrology), bounds = UnitInterval, desc = "Saturation level of water and ice in the pore space"),
    auxiliary(:hydraulic_conductivity, Ground(XYZ(z = Face())), units = u"m/s", desc = "Hydraulic conductivity of soil volumes in m/s"),
    auxiliary(:water_table, Ground(XY()), units = u"m", desc = "Elevation of the water table in meters"),
    auxiliary(:water_flux, Ground(XYZ(z = Face())), units = u"m/s", desc = "Darcy flux through the cell faces in m/s, positive upward"),
    input(:liquid_water_fraction, Ground(XYZ()), default = NF(1), bounds = UnitInterval, desc = "Fraction of unfrozen water in the pore space"),
)

default_timestepping(::SoilHydrology{NF, <:RichardsEq}) where {NF} = Implicit()

# Top-level interface methods

""" $TYPEDSIGNATURES """
function initialize!(
        state, grid,
        hydrology::SoilHydrology{NF, <:RichardsEq},
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
        hydrology::SoilHydrology{NF, <:RichardsEq},
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
        hydrology::SoilHydrology{NF, <:RichardsEq},
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
(integrated by the implicit sub-stepper, see [`BackwardEuler`](@ref)), leaving the fluxes through the two
boundary faces (see [`compute_boundary_darcy_flux_divergence`](@ref)). The flux boundary conditions on
`saturation_water_ice` (e.g. [`InfiltrationFlux`](@ref)) from `compute_boundary_conditions!` and the
source terms are part of the tendency in both cases.

The Darcy fluxes at `tⁿ` through all faces are stored in `water_flux` ([`compute_water_flux`](@ref)),
with which the soil energy advects heat. Under [`Implicit`](@ref) time stepping, the implicit step
overwrites the interior fluxes with those of its conservative update (see [`implicit_update!`](@ref)).
"""
function compute_tendencies!(
        state, grid,
        hydrology::SoilHydrology{NF, <:RichardsEq},
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
is thus not the same as the saturation tendency. Like the energy tendency, it is the explicitly
integrated part of the tendency, and `timestepping` is the class of `saturation_water_ice`: under
[`Explicit`](@ref) (the default) the full divergence of the Darcy flux is included; under
[`Implicit`](@ref) only the boundary-face fluxes are ([`compute_boundary_darcy_flux_divergence`](@ref)),
the interior ones being integrated by the implicit sub-stepper.
"""
@propagate_inbounds function compute_volumetric_water_content_tendency(
        i, j, k, grid, clock, fields,
        hydrology::SoilHydrology{NF, <:RichardsEq},
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

Divergence `∂j_w/∂z` (1/s) of the [`darcy_flux`](@ref) `j_w` at cell `i, j, k`: over all faces under
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

    (δ_{k,Nz} j_{w,k+½} - δ_{k,1} j_{w,k-½}) / Δz_k,

with `j_w` the [`darcy_flux`](@ref). This is the explicit part when the interior fluxes are integrated
implicitly (see [`BackwardEuler`](@ref)). The boundary fluxes are computed from the `pressure_head` halo: they
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
    Kₖ = darcy_conductivity(i, j, k, grid, K)
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

Hydraulic conductivity (m/s) used by the [`darcy_flux`](@ref) at face `k`, i.e. the face conductivity `K`
computed by [`compute_face_hydraulic_conductivity`](@ref).
"""
@propagate_inbounds darcy_conductivity(i, j, k, grid, K) = K[i, j, k]

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
reconstructed from the total `pressure_head`. It vanishes in saturated cells, and at the
[`minimum_matric_potential`](@ref), below which the closure holds the matric potential constant.
"""
@propagate_inbounds function compute_specific_moisture_capacity(
        i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, <:RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    ψm = compute_matric_potential(i, j, k, grid, fields)
    ψm_min = minimum_matric_potential(get_hydraulic_properties(hydrology))
    swrc = get_swrc(hydrology)
    por = porosity(i, j, k, grid, fields, strat, bgc)
    dθdψ = convert(NF, specific_moisture_capacity(swrc, ψm, por))
    return ifelse(ψm > ψm_min, dθdψ, zero(dθdψ))
end

"""
    specific_moisture_capacity(swrc::SWRC, ψ, θsat)

Specific moisture capacity `∂θ/∂ψ` (1/m) of the soil water retention curve `swrc` at the matric
potential `ψ` (m) for the saturated water content `θsat`, from the derivative provided by FreezeCurves.jl,
in the number type of `ψ`; zero for saturated conditions.

```jldoctest
julia> swrc = VanGenuchten(α = 2.0, n = 2.0);

julia> Terrarium.specific_moisture_capacity(swrc, 0.0, 0.4)
0.0

julia> Terrarium.specific_moisture_capacity(swrc, -0.5, 0.4) ≈ swrc(Terrarium.FreezeCurves.derivative, -0.5; θsat = 0.4)
true
```
"""
@inline specific_moisture_capacity(swrc::SWRC, ψ, θsat) = convert(typeof(ψ), swrc(FreezeCurves.derivative, ψ; θsat))

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

Compute the infiltration capacity (m/s) of the soil surface in grid column `i, j` as

    I_max = K_sat I_ice(f)

where `K_sat` is the saturated hydraulic conductivity and `I_ice` is the ice impedance factor of the unsaturated
conductivity formulation (see [`Terrarium.ice_impedance_factor`](@ref)) for the unfrozen water fraction `f` of
the topmost soil layer, as in CLM5 [lawrenceTechnicalDescriptionVersion2018; Eq. (7.34)](@cite).

The infiltration capacity of a soil is at least its saturated conductivity: the matric suction gradient at a
wetting front raises it above `K_sat` in dry soil, and it decays toward `K_sat` as the soil wets
[greenStudiesSoilPhysics1911, philipTheoryInfiltration1957](@cite).

# References

* [lawrenceTechnicalDescriptionVersion2018](@cite) Lawrence et al., Technical Description of Version 5.0 of the Community Land Model (CLM), NCAR (2018)
* [greenStudiesSoilPhysics1911](@cite) Green and Ampt, The Journal of Agricultural Science (1911)
* [philipTheoryInfiltration1957](@cite) Philip, Soil Science (1957)
"""
@propagate_inbounds function compute_infiltration_capacity(
        i, j, grid, fields,
        hydrology::SoilHydrology{NF, <:RichardsEq}
    ) where {NF}
    props = get_hydraulic_properties(hydrology)
    f = liquid_water_fraction(i, j, grid.Nz, grid, fields, hydrology)
    K_sat = saturated_hydraulic_conductivity(props)
    I_ice = ice_impedance_factor(props.unsat_hydraulic_cond, f)
    return K_sat * I_ice
end

# evaluated inline wherever it is needed, so that it needs neither storage nor an update in `compute_auxiliary!`
@propagate_inbounds infiltration_capacity(i, j, grid, fields, hydrology::SoilHydrology{NF, <:RichardsEq}) where {NF} =
    compute_infiltration_capacity(i, j, ground_domain(grid), fields, hydrology)

"""
    $TYPEDSIGNATURES

Return the [`SoilComposition`](@ref) at the face between the soil volumes `soil_below` and `soil_above`, with the
mean porosity, the mean water and ice content, and the mean liquid water content of the two volumes, and the solid
matrix of `soil_above`. Its liquid water saturation `θw / θsat` is thus the mean liquid water content over the mean
porosity, as in the interface conductivity of CLM5 [lawrenceTechnicalDescriptionVersion2018; Eq. (7.47)](@cite).

# References

* [lawrenceTechnicalDescriptionVersion2018](@cite) Lawrence et al., Technical Description of Version 5.0 of the Community Land Model (CLM), NCAR (2018)
"""
@inline function face_soil_composition(soil_below::SoilComposition{NF}, soil_above::SoilComposition{NF}) where {NF}
    # porosity ϕ, water and ice content θwi, and liquid water content θw below (₁) and above (₂) the face
    ϕ₁ = porosity(soil_below)
    ϕ₂ = porosity(soil_above)
    θwi₁ = water_ice(soil_below)
    θwi₂ = water_ice(soil_above)
    θw₁ = water(soil_below)
    θw₂ = water(soil_above)
    # means at the face
    ϕ = (ϕ₁ + ϕ₂) / 2
    θwi = (θwi₁ + θwi₂) / 2
    θw = (θw₁ + θw₂) / 2
    # saturation ξ and liquid fraction f at the face; the guard covers a column without any water or ice,
    # where the liquid fraction is undefined
    ξ = θwi / ϕ
    f = θw / max(θwi, eps(NF))
    return SoilComposition(ϕ, ξ, f, soil_above.solid)
end

"""
    $TYPEDSIGNATURES

Compute the hydraulic conductivity at face `k` of the soil column `i, j`, i.e. between the cells `k - 1` and `k`,
according to the `face_conductivity` scheme of the Richards equation (see [`AbstractFaceConductivity`](@ref)).
At the bottom (`k = 1`) and top (`k = Nz + 1`) boundary faces, it is the conductivity of the adjacent cell.
"""
@propagate_inbounds compute_face_hydraulic_conductivity(i, j, k, grid, fields, hydrology::SoilHydrology{NF, <:RichardsEq}, args...) where {NF} =
    compute_face_hydraulic_conductivity(i, j, k, grid, fields, hydrology.vertical_flow.face_conductivity, hydrology, args...)

"""
    $TYPEDSIGNATURES

Compute the hydraulic conductivity at face `k` as the conductivity of the mean composition of the two adjacent cells
(see [`face_soil_composition`](@ref)), following the interface conductivity of CLM5
[lawrenceTechnicalDescriptionVersion2018; Eq. (7.47)](@cite).

# References

* [lawrenceTechnicalDescriptionVersion2018](@cite) Lawrence et al., Technical Description of Version 5.0 of the Community Land Model (CLM), NCAR (2018)
"""
@propagate_inbounds function compute_face_hydraulic_conductivity(
        i, j, k, grid, fields,
        ::WeightedAverageConductivity,
        hydrology::SoilHydrology{NF, <:RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    Nz = ground_domain(grid).Nz
    k_below = max(k - 1, one(k))
    k_above = min(k, Nz)
    soil_below = soil_composition(i, j, k_below, grid, fields, strat, hydrology, bgc)
    soil_above = soil_composition(i, j, k_above, grid, fields, strat, hydrology, bgc)
    soil_face = face_soil_composition(soil_below, soil_above)
    return hydraulic_conductivity(hydrology.hydraulic_properties, soil_face)
end

"""
    $TYPEDSIGNATURES

Compute the hydraulic conductivity at face `k` as the conductivity of the upstream cell: the water flows from the
higher to the lower total hydraulic head `ψ` (`pressure_head`), so the upstream cell is `k` (above the face) if the
head increases upward across the face, and `k - 1` otherwise [forsythRobustNumericalMethods1995](@cite).

# References

* [forsythRobustNumericalMethods1995](@cite) Forsyth et al., Advances in Water Resources (1995)
"""
@propagate_inbounds function compute_face_hydraulic_conductivity(
        i, j, k, grid, fields,
        ::UpwindConductivity,
        hydrology::SoilHydrology{NF, <:RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    ground_grid = ground_domain(grid)
    Nz = ground_grid.Nz
    ∇ψ = ∂zᵃᵃᶠ(i, j, k, ground_grid, fields.pressure_head)
    # select the index of the upstream cell, clamped to the column at the boundary faces
    k_upstream = clamp(ifelse(∇ψ > 0, k, k - 1), one(k), Nz)
    return hydraulic_conductivity(i, j, k_upstream, grid, fields, hydrology, strat, bgc)
end

"""
    $TYPEDSIGNATURES

Computes the hydraulic conductivity at the cell faces for `RichardsEq` configurations of `SoilHydrology` (see
[`compute_face_hydraulic_conductivity`](@ref)). Each cell writes its lower face; the top cell also writes the top
face `Nz + 1`.
"""
@propagate_inbounds function compute_hydraulics!(
        out, i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, <:RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    Nz = ground_domain(grid).Nz
    @inbounds out.hydraulic_conductivity[i, j, k] = compute_face_hydraulic_conductivity(i, j, k, grid, fields, hydrology, strat, bgc)
    k_upper = ifelse(k >= Nz, Nz + one(Nz), k)
    @inbounds out.hydraulic_conductivity[i, j, k_upper] = compute_face_hydraulic_conductivity(i, j, k_upper, grid, fields, hydrology, strat, bgc)
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
        hydrology::SoilHydrology{NF, <:RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry,
        constants::PhysicalConstants,
        evtr::Optional{AbstractEvapotranspiration},
        timestepping::Timestepping
    ) where {NF}
    i, j, k = @index(Global, NTuple)
    compute_saturation_tendency!(tend.saturation_water_ice, i, j, k, grid, clock, fields, hydrology, strat, bgc, constants, evtr, timestepping)
end
