#####
##### Soil heat diffusion timescale
#####

# Fallback
cell_diffusion_timescale(state, grid, ::AbstractSoilThermodynamics{NF}, soil, args...) where {NF} = convert(NF, Inf)

"""
    $TYPEDSIGNATURES

Return the minimum thermal diffusion timescale ``τ = Δz² C / κ`` over the soil column for the
`SoilThermodynamics` energy process, where `κ` is the bulk thermal conductivity (W m⁻¹ K⁻¹) and `C`
the bulk volumetric heat capacity (J m⁻³ K⁻¹). This is the diffusive stability limit of the explicit
heat conduction operator, whose effective temperature diffusivity is ``α = κ / C``.

The sensible heat capacity is used (not the apparent capacity, which includes the latent heat of
phase change); since the apparent capacity is never smaller, the reported timescale is conservative
near the freezing front.
"""
function cell_diffusion_timescale(
        state, grid,
        energy::SoilThermodynamics,
        soil::AbstractSoil,
        args...
    )
    strat = get_stratigraphy(soil)
    hydrology = get_hydrology(soil)
    bgc = get_biogeochemistry(soil)
    ground_grid = ground_domain(grid)
    fields = get_fields(state, energy, hydrology, strat, bgc)
    τ = KernelFunctionOperation{Center, Center, Center}(
        compute_thermal_diffusion_timescale, ground_grid, fields, energy, hydrology, strat, bgc
    )
    return minimum(τ)
end

"""
    $TYPEDSIGNATURES

Kernel function returning the thermal diffusion timescale ``Δz² C / κ`` at cell `i, j, k`, with the
bulk thermal conductivity `κ` and heat capacity `C` computed from the local soil composition.
"""
@inline @propagate_inbounds function compute_thermal_diffusion_timescale(
        i, j, k, grid, fields,
        energy::SoilThermodynamics,
        hydrology::AbstractSoilHydrology,
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    )
    soil = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    κ = compute_thermal_conductivity(energy.thermal_properties, soil)
    C = compute_heat_capacity(energy.thermal_properties, soil)
    Δz = Δzᵃᵃᶜ(i, j, k, grid)
    # thermal diffusivity α = κ / C ⟹ τ = Δz² / α = Δz² C / κ
    return Δz^2 * C / κ
end

"""
    $TYPEDSIGNATURES

Return the minimum diffusive timescale over the coupled soil processes as the minimum of the
individual energy (heat conduction) and hydrology (Richards) timescales. If the model's `timestepper`
integrates `internal_energy` (`saturation_water_ice`) implicitly (see [`AbstractIMEX`](@ref)), the
energy (hydrology) timescale is `Inf`.
"""
function cell_diffusion_timescale(
        state, grid,
        soil::SoilEnergyWaterCarbon,
        constants::PhysicalConstants,
        timestepper::Optional{AbstractTimeStepper} = nothing
    )
    energy_timestepping = timestepping(timestepper, state, Val(:internal_energy))
    hydrology_timestepping = timestepping(timestepper, state, Val(:saturation_water_ice))
    τ_energy = energy_diffusion_timescale(state, grid, soil, constants, energy_timestepping)
    τ_water = hydrology_diffusion_timescale(state, grid, soil, constants, hydrology_timestepping)
    return min(τ_energy, τ_water)
end

# Explicit heat conduction is diffusively limited; implicit is not.
energy_diffusion_timescale(state, grid, soil::AbstractSoil, constants, ::Explicit) =
    cell_diffusion_timescale(state, grid, get_energy_balance(soil), soil, constants)
energy_diffusion_timescale(state, grid, soil::AbstractSoil, constants, ::Implicit) = convert(eltype(grid), Inf)

# Explicit Richards flow is diffusively limited; implicit is not.
hydrology_diffusion_timescale(state, grid, soil::AbstractSoil, constants, ::Explicit) =
    cell_diffusion_timescale(state, grid, get_hydrology(soil), soil, constants)
hydrology_diffusion_timescale(state, grid, soil::AbstractSoil, constants, ::Implicit) = convert(eltype(grid), Inf)

#####
##### Soil hydrology (Richards) timescale
#####

# Fallback
cell_diffusion_timescale(state, grid, ::AbstractSoilHydrology{NF}, args...) where {NF} = NF(Inf)

"""
    $TYPEDSIGNATURES

Return the minimum hydraulic diffusion timescale ``τ = Δz² (∂θ/∂ψ) / K`` over the soil column for a
Richards-equation `SoilHydrology`, where `K` is the (variably saturated) hydraulic conductivity
(m s⁻¹) and `∂θ/∂ψ` the specific moisture capacity (m⁻¹). This is the diffusive stability limit of
the explicit Richards operator, whose effective moisture diffusivity is ``D = K ∂ψ/∂θ = K / (∂θ/∂ψ)``.

As the soil saturates, ``∂θ/∂ψ → 0`` and the timescale tends to zero: explicit Richards is stiff
near saturation, which is precisely the regime motivating implicit timestepping. The
`TimeStepWizard`'s `min_change` bound keeps the step from collapsing to zero in a single adjustment.
"""
function cell_diffusion_timescale(
        state, grid,
        hydrology::SoilHydrology{NF, RichardsEq},
        soil::AbstractSoil,
        args...
    ) where {NF}
    strat = get_stratigraphy(soil)
    bgc = get_biogeochemistry(soil)
    ground_grid = ground_domain(grid)
    fields = get_fields(state, hydrology, strat, bgc)
    τ = KernelFunctionOperation{Center, Center, Center}(
        compute_hydraulic_diffusion_timescale, ground_grid, fields, hydrology, strat, bgc
    )
    return minimum(τ)
end

"""
    $TYPEDSIGNATURES

Kernel function returning the hydraulic diffusion timescale ``Δz² (∂θ/∂ψ) / K`` at cell `i, j, k`.
The hydraulic conductivity `K` is evaluated at the cell centre from the local soil composition; the
specific moisture capacity `∂θ/∂ψ` is the analytic derivative of the soil-water retention curve
(from FreezeCurves) evaluated at the matric potential `ψₘ`, reconstructed from the stored total
`pressure_head` by removing its hydrostatic and elevation components. Cells with zero hydraulic
conductivity (fully dry or frozen) impose no restriction and return `Inf`.
"""
@inline @propagate_inbounds function compute_hydraulic_diffusion_timescale(
        i, j, k, grid, fields,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    soil = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    # hydraulic conductivity at the cell centre
    K = hydraulic_conductivity(get_hydraulic_properties(hydrology), soil)
    # specific moisture capacity ∂θ/∂ψ (m⁻¹) from the analytic SWRC derivative at the matric potential
    ψm = compute_matric_potential(i, j, k, grid, fields)
    swrc = get_swrc(hydrology)
    por = porosity(i, j, k, grid, fields, strat, bgc)
    ∂θ∂ψ = convert(NF, swrc(FreezeCurves.derivative, ψm; θsat = por))
    # moisture diffusivity D = K / (∂θ/∂ψ) ⟹ τ = Δz² / D = Δz² (∂θ/∂ψ) / K
    Δz = Δzᵃᵃᶜ(i, j, k, grid)
    return ifelse(K > zero(K), Δz^2 * abs(∂θ∂ψ) / K, convert(NF, Inf))
end

#####
##### Soil energy advection timescale
#####

"""
    $TYPEDSIGNATURES

Return the minimum advective timescale of the soil energy equation over the soil column: the advective
CFL limit of the explicit first-order upwind advection of sensible heat by the soil water flux (see
[`advective_heat_flux`](@ref)). It is `Inf` without water flow (e.g. `NoFlow` hydrology) and when the
model's `timestepper` integrates `internal_energy` implicitly (the implicit operator is unconditionally
stable).
"""
function cell_advection_timescale(state, grid, soil::SoilEnergyWaterCarbon, timestepper::Optional{AbstractTimeStepper} = nothing)
    energy_timestepping = timestepping(timestepper, state, Val(:internal_energy))
    return energy_advection_timescale(state, grid, soil, energy_timestepping)
end

# Explicit heat advection is CFL limited; implicit is not.
energy_advection_timescale(state, grid, soil::AbstractSoil, ::Explicit) =
    cell_advection_timescale(state, grid, get_energy_balance(soil), get_hydrology(soil), get_stratigraphy(soil), get_biogeochemistry(soil))
energy_advection_timescale(state, grid, soil::AbstractSoil, ::Implicit) = convert(eltype(grid), Inf)

cell_advection_timescale(state, grid, energy::AbstractSoilThermodynamics, hydrology::AbstractSoilHydrology, args...) = convert(eltype(grid), Inf)

"""
    $TYPEDSIGNATURES

Return the minimum over all cells of the advective timescale ``τ = C Δz / (C_w |q|)``
([`compute_heat_advection_timescale`](@ref)) for a Richards-equation hydrology, with `q` the Darcy flux
through the faces of the cell.
"""
function cell_advection_timescale(
        state, grid,
        energy::SoilThermodynamics,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    ground_grid = ground_domain(grid)
    fields = get_fields(state, energy, hydrology, strat, bgc)
    τ = KernelFunctionOperation{Center, Center, Center}(
        compute_heat_advection_timescale, ground_grid, fields, energy, hydrology, strat, bgc
    )
    return minimum(τ)
end

"""
    $TYPEDSIGNATURES

Kernel function returning the advective timescale ``τ = C Δz / (C_w |q|)`` at cell `i, j, k`, with `C`
the bulk volumetric heat capacity of the cell, `C_w` that of liquid water, and `|q|` the larger magnitude
of the [`darcy_flux`](@ref) through the two faces of the cell. This is the stability limit of the
explicit upwind advection, which changes the cell temperature at the rate `C_w q / C` (m/s). Cells
without flow return `Inf`.
"""
@inline @propagate_inbounds function compute_heat_advection_timescale(
        i, j, k, grid, fields,
        energy::SoilThermodynamics,
        hydrology::SoilHydrology{NF, RichardsEq},
        strat::AbstractStratigraphy,
        bgc::AbstractSoilBiogeochemistry
    ) where {NF}
    soil = soil_composition(i, j, k, grid, fields, strat, hydrology, bgc)
    C = compute_heat_capacity(energy.thermal_properties, soil)
    Cw = energy.thermal_properties.heat_capacities.water
    ψ = fields.pressure_head
    K = fields.hydraulic_conductivity
    q = max(abs(darcy_flux(i, j, k, grid, ψ, K)), abs(darcy_flux(i, j, k + 1, grid, ψ, K)))
    Δz = Δzᵃᵃᶜ(i, j, k, grid)
    return ifelse(q > zero(q), C * Δz / (Cw * q), convert(NF, Inf))
end
