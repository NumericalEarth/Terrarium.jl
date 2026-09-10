"""
    $TYPEDEF

Fully-coupled land model integrating atmosphere, surface energy balance, surface hydrology,
vegetation, and soil processes.

Properties:
$(TYPEDFIELDS)
"""
@parameterized @kwdef struct LandModel{
        NF,
        GridType <: AbstractLandGrid{NF},
        Vegetation <: Optional{AbstractVegetation{NF}},
        Soil <: AbstractSoil{NF},
        Snow <: Optional{AbstractSnow{NF}},
        SEB <: AbstractSurfaceEnergyBalance,
        Hydrology <: AbstractSurfaceHydrology,
        Atmosphere <: AbstractAtmosphere,
        Initializer <: AbstractInitializer,
        Timestepper <: AbstractTimeStepper{NF},
    } <: AbstractLandModel{NF, GridType}
    "Spatial discretization"
    grid::GridType

    "Vegetation processes"
    @component vegetation::Vegetation = VegetationCarbonCycle(eltype(grid))

    "Soil processes"
    @component soil::Soil = default_soil(grid, vegetation)

    "Snow processes"
    @component snow::Snow = SingleLayerSnow(eltype(grid))

    "Surface energy balance"
    @component surface_energy_balance::SEB = default_surface_energy_balance(grid, vegetation, soil)

    "Surface hydrology scheme"
    @component surface_hydrology::Hydrology = default_surface_hydrology(grid, vegetation, soil)

    "Near-surface atmospheric conditions"
    @component atmosphere::Atmosphere = PrescribedAtmosphere(eltype(grid))

    "Physical constants"
    @component constants::PhysicalConstants{NF} = PhysicalConstants(eltype(grid))

    "State variable initializer"
    @component initializer::Initializer = DefaultInitializer(eltype(grid))

    "Time stepper: a single `AbstractTimeStepper` (e.g. `ForwardEuler`, `Heun`) or an `IMEX`"
    @component timestepper::Timestepper = default_timestepper(eltype(grid))
end

function StateVariables(
        model::LandModel{NF};
        clock = Clock(time = zero(NF)),
        fields = (;),
        boundary_conditions = (;),
        input_variables = ()
    ) where {NF}
    grid = get_grid(model)
    interface_vars = interface_variables(model)
    # Snow adds a blended `soil_heat_flux` coupling auxiliary to the state when present
    vars = Variables(variables(model)..., interface_vars..., input_variables...)
    # Initialize BC fields for coupling
    ground_heat_flux = initialize(vars.ground_heat_flux, grid, clock, fields, boundary_conditions)
    infiltration = initialize(vars.infiltration, grid, clock, fields, boundary_conditions)
    # Initialize soil heat flux
    soil_heat_flux = if isnothing(model.snow)
        ground_heat_flux
    else
        initialize(vars.soil_heat_flux, grid, clock, fields, boundary_conditions)
    end
    soil_heat_flux_bc = SoilHeatFlux(soil_heat_flux)
    # Note that saturation_water_ice is dimensionless (VWC/porosity), so the physical infiltration
    # flux (m/s of water depth) must be normalized by the top-layer porosity before it can be used
    # as a flux boundary condition on saturation_water_ice. This is a discrete-form boundary
    # condition (see `saturation_infiltration_bc`) evaluated fresh against the live state at every
    # step, rather than a value/operation captured now — the full state (including the per-horizon
    # field namespace `porosity_top` needs) does not exist yet at this point in construction.
    infiltration_bc = InfiltrationFlux(get_stratigraphy(model.soil), get_biogeochemistry(model.soil))
    bcs = merge_boundary_conditions(boundary_conditions, soil_heat_flux_bc, infiltration_bc)
    # The snow-top conductive flux (drives the snowpack's own energy tendency) is a genuinely distinct
    # quantity from the bare-ground `ground_heat_flux` once snow is present (see `skin_temperature.jl`'s
    # per-fraction `G`/`S` split), so it gets its own field rather than aliasing `ground_heat_flux`.
    snow_surface_heat_flux = if isnothing(model.snow)
        ground_heat_flux
    else
        initialize(vars.snow_surface_heat_flux, grid, clock, fields, boundary_conditions)
    end
    # The snow energy tendency reads its boundary heat fluxes from the `surface_heat_flux`/`basal_heat_flux`
    # input fields; alias them to the snow-top conductive flux (`snow_surface_heat_flux`) and the
    # blended soil-top flux (`soil_heat_flux`) so no additional state fields are allocated (no-op without snow).
    snow_flux_aliases = isnothing(model.snow) ? (;) : (; surface_heat_flux = snow_surface_heat_flux, basal_heat_flux = soil_heat_flux)
    # Merge user-defined fields with BC fields
    fields = merge((; ground_heat_flux, infiltration, soil_heat_flux, snow_surface_heat_flux), snow_flux_aliases, fields)
    return StateVariables(vars, grid; clock, timestepper = get_timestepper(model), model, boundary_conditions = bcs, fields)
end

interface_variables(::LandModel) = (
    auxiliary(:soil_heat_flux, XY(); units = u"W/m^2", desc = "Blended heat flux into the soil top (snow base + bare ground)"),
    auxiliary(:snow_surface_heat_flux, XY(); units = u"W/m^2", desc = "Conductive heat flux from the skin into the top of the snowpack (positive upward); drives the snowpack's own energy tendency"),
)

function initialize!(state, model::LandModel)
    initialize!(state, model, model.initializer)
    grid = get_grid(model)
    initialize!(state, grid, model.surface_hydrology)
    # TODO: change when refactoring model/process types
    initialize!(state, grid, model.vegetation, model.constants, model.atmosphere)
    initialize!(state, grid, model.soil, model.constants)
    # Initialize the snow (invclosure: T, W -> E) after the soil; no-op without snow (`nothing`)
    initialize!(state, grid, model.snow, model.constants)
    # Initialize the SEB after the soil so that ground_temperature is available
    initialize!(state, grid, model.surface_energy_balance)
    return nothing
end

function compute_auxiliary!(state, model::LandModel)
    grid = get_grid(model)
    compute_auxiliary!(state, grid, model.atmosphere)
    compute_auxiliary!(state, grid, model.soil, model.constants)
    compute_auxiliary!(state, grid, model.snow, model.constants)
    compute_auxiliary!(state, grid, model.vegetation, model.constants, model.atmosphere, model.soil)
    compute_auxiliary!(state, grid, model.surface_hydrology, model.constants, model.atmosphere, model.soil, model.vegetation, model.snow)
    compute_auxiliary!(state, grid, model.surface_energy_balance, model.vegetation, model.snow)
    return nothing
end

function compute_boundary_conditions!(state, model::LandModel)
    grid = get_grid(model)
    # Compute surface energy fluxes by solving the SEB
    solve_surface_energy_balance!(state, grid, model.surface_energy_balance, model.constants, model.atmosphere, model.surface_hydrology, model.snow)
    # Diagnose the snow↔soil coupling fluxes (soil-top heat blend + sublimation) after the SEB; no-op without snow
    compute_snow_interface_fluxes!(state, grid, model.snow, model.surface_energy_balance, model.soil, model.constants, model.atmosphere)
    # Now that the soil/snow atmosphere and snow-ground fluxes are updated, we can compute the soil BCs
    compute_boundary_conditions!(state, grid, model.soil)
    return nothing
end

function compute_tendencies!(state, model::LandModel)
    grid = get_grid(model)
    compute_tendencies!(state, grid, model.surface_hydrology)
    compute_tendencies!(state, grid, model.soil, model.constants)
    # Snow tendencies after surface hydrology; no-op without snow (`nothing`)
    compute_tendencies!(state, grid, model.snow, model.constants, model.atmosphere)
    compute_tendencies!(state, grid, model.vegetation, model.constants, model.atmosphere)
    return nothing
end

function closure!(state, model::LandModel)
    grid = get_grid(model)
    closure!(state, grid, model.soil, model.constants, model.surface_hydrology)
    closure!(state, grid, model.snow, model.constants)
    return nothing
end

function invclosure!(state, model::LandModel)
    grid = get_grid(model)
    invclosure!(state, grid, model.soil, model.constants, model.surface_hydrology)
    invclosure!(state, grid, model.snow, model.constants)
    return nothing
end

# Time stepping

function Terrarium.timestep!(state::StateVariables, model::LandModel, timestepper::Terrarium.AbstractTimeStepper, Δt)
    if !isnothing(model.snow)
        timestep!(state, model, model.snow, timestepper, Δt)
    end
    return nothing
end


# Default soil types
default_soil(grid::AbstractLandGrid, ::Nothing) = SoilEnergyWaterCarbon(eltype(grid))
default_soil(grid::AbstractLandGrid, ::AbstractVegetation) = SoilEnergyWaterCarbon(eltype(grid), hydrology = SoilHydrology(eltype(grid), RichardsEq()))

# Default SEB
default_surface_energy_balance(grid::AbstractLandGrid, vegetation, soil) = SurfaceEnergyBalance(eltype(grid))

# Default surface hydrology
default_surface_hydrology(grid::AbstractLandGrid, ::AbstractVegetation, soil) = SurfaceHydrology(eltype(grid))
function default_surface_hydrology(grid::AbstractLandGrid, ::Nothing, soil)
    return SurfaceHydrology(
        eltype(grid),
        evapotranspiration = BareGroundEvaporation(eltype(grid)),
        canopy_interception = NoCanopyInterception(eltype(grid))
    )
end
