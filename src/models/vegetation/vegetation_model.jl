"""
    $TYPEDEF

Model for natural (unmanaged) vegetation processes for a single plant functional type (PFT).
Multiple PFTs can be later handled with a `TiledVegetationModel` type that composes multiple
`VegetationModel`s with different parameters for each PFT.

Properties:
$TYPEDFIELDS
"""
@parameterized @kwdef struct VegetationModel{
        NF,
        Vegetation <: AbstractVegetation{NF},
        Atmosphere <: AbstractAtmosphere{NF},
        GridType <: AbstractGrid{NF},
        Initializer <: AbstractInitializer,
        Timestepper <: AbstractTimeStepper{NF},
    } <: AbstractVegetationModel{NF, GridType}
    "Spatial grid type"
    grid::GridType

    "Atmospheric input configuration"
    @component atmosphere::Atmosphere = PrescribedAtmosphere(eltype(grid))

    "Vegetation processes"
    @component vegetation::Vegetation = VegetationCarbonCycle(eltype(grid))

    "Physical constants"
    @component constants::PhysicalConstants{NF} = PhysicalConstants(eltype(grid))

    "State variable initializer"
    @component initializer::Initializer = DefaultInitializer(eltype(grid))

    "Time stepper: a single `AbstractTimeStepper` (e.g. `ForwardEuler`, `Heun`) or an `IMEX`"
    @component timestepper::Timestepper = default_timestepper(eltype(grid))
end

function initialize!(state, model::VegetationModel)
    initialize!(state, model.grid, model.vegetation)
    return nothing
end

function compute_auxiliary!(state, model::VegetationModel)
    # Unpack vegetation model
    (; grid, atmosphere, vegetation, constants) = model
    # Compute auxiliary variables for coupled vegetation processes
    compute_auxiliary!(state, grid, vegetation, constants, atmosphere)
    return nothing
end

function compute_tendencies!(state, model::VegetationModel)
    # Unpack vegetation model
    (; grid, atmosphere, vegetation, constants) = model
    # Compute tendencies for coupled vegetation processes
    compute_tendencies!(state, grid, vegetation, constants, atmosphere)
    return nothing
end
