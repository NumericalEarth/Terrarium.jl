"""
    $TYPEDEF

Model for surface hydrology processes.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct SurfaceHydrologyModel{
        NF,
        GridType <: AbstractGrid{NF},
        Atmosphere <: AbstractAtmosphere,
        CanopyHydrology <: AbstractCanopyInterception,
        CanopyET <: AbstractEvapotranspiration,
        SurfaceRunoff <: AbstractSurfaceRunoff,
        Initializer <: AbstractInitializer,
        Timestepper <: AbstractTimeStepper{NF},
    } <: AbstractSurfaceHydrologyModel{NF, GridType}
    "Spatial grid type"
    grid::GridType

    "Atmospheric input configuration"
    atmosphere::Atmosphere = PrescribedAtmosphere(eltype(grid))

    "Canopy hydrology scheme"
    canopy_interception::CanopyHydrology = PALADYNCanopyInterception(eltype(grid))

    "Canopy evapotranspiration scheme"
    evapotranspiration::CanopyET = PALADYNCanopyEvapotranspiration(eltype(grid))

    "Surface runoff scheme"
    surface_runoff::SurfaceRunoff = DirectSurfaceRunoff(eltype(grid))

    "Physical constants"
    constants::PhysicalConstants{NF} = PhysicalConstants(eltype(grid))

    "State variable initializer"
    initializer::Initializer = DefaultInitializer(eltype(grid))

    "Time stepper: a single `AbstractTimeStepper` (e.g. `ForwardEuler`, `Heun`) or an `IMEX`"
    timestepper::Timestepper = default_timestepper(eltype(grid))
end

# Model interface methods

function compute_auxiliary!(state, model::SurfaceHydrologyModel)
    compute_auxiliary!(state, model, model.atmosphere)
    compute_auxiliary!(state, model, model.canopy_interception)
    compute_auxiliary!(state, model, model.evapotranspiration)
    compute_auxiliary!(state, model, model.surface_runoff)
    return nothing
end

function compute_tendencies!(state, model::SurfaceHydrologyModel)
    compute_tendencies!(state, model, model.atmosphere)
    compute_tendencies!(state, model, model.canopy_interception)
    compute_tendencies!(state, model, model.evapotranspiration)
    compute_tendencies!(state, model, model.surface_runoff)
    return nothing
end
