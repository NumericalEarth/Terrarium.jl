module Terrarium

using DocStringExtensions

using Adapt: Adapt, adapt, @adapt_structure

using Base: @propagate_inbounds

using ConstructionBase: ConstructionBase, getproperties, setproperties

using DataStructures: OrderedDict

using Dates: Dates, TimeType, Period, Year, Month, Day, Hour, Minute, Second, Millisecond

using Flatten: flatten, flattenable, reconstruct

using KernelAbstractions: @kernel, @index

# Oceananigans numerics
using Oceananigans.AbstractOperations: Average, Integral, ConditionalOperation, KernelFunctionOperation
using Oceananigans.Architectures: Architectures, AbstractArchitecture, CPU, GPU, ReactantState, architecture, on_architecture, array_type
using Oceananigans.Fields: Field, FunctionField, AbstractField, Center, Face, set!, compute!, interior, indices, location
using Oceananigans.Forcings: Forcing, ContinuousForcing, DiscreteForcing
using Oceananigans.Grids: AbstractGrid, RectilinearGrid, CallableDiscretization, ExponentialDiscretization,
    Periodic, Flat, Bounded, halo_size, isrectilinear, nodes, topology, xnodes, ynodes, znodes, znode, zspacings,
    ξnode, ηnode, rnode
using Oceananigans.Operators: ∂zᵃᵃᶜ, ∂zᵃᵃᶠ, ℑzᵃᵃᶠ, Δzᵃᵃᶜ
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Simulations: Simulation, run!, timestepper, TimeStepWizard, conjure_time_step_wizard!, Callback, add_callback!
using Oceananigans.TimeSteppers: Clock, update_state!, time_step!, tick!, reset!
using Oceananigans.Units: Time, seconds, hours, days
using Oceananigans.Utils: launch!, IterationInterval, TimeInterval

# Boundary conditions
using Oceananigans.BoundaryConditions: BoundaryConditions, BoundaryCondition, DefaultBoundaryCondition, FieldBoundaryConditions,
    ValueBoundaryCondition, FluxBoundaryCondition, GradientBoundaryCondition, NoFluxBoundaryCondition,
    ContinuousBoundaryFunction, DiscreteBoundaryFunction,
    AbstractBoundaryConditionClassification, Value, Flux, Gradient, # BC type classifications
    fill_halo_regions!, regularize_field_boundary_conditions, getbc, compute_z_bcs!

# Progress meter
using ProgressMeter: @showprogress

# Freeze curves for soil energy balance
using FreezeCurves: FreezeCurves, FreezeCurve, SFCC, SWRC, FreeWater, VanGenuchten, BrooksCorey

# Units (for testing and UI)
# Unit dimensions for length (𝐋), mass (𝐌), and time (𝐓)
using Unitful: 𝐋, 𝐌, 𝐓
using Unitful: Units, Quantity, AbstractQuantity, NoUnits
using Unitful: @u_str, uconvert, ustrip, upreferred

# Parameter handling (imported from SpeedyWeatherInternals for now)
using SpeedyWeatherInternals.ParameterEditing: ParameterEditing, ParameterTable, ComponentVector,
    Positive, Nonnegative, Unbounded, parameters, @parameterized

# Explicit imports
import DomainSets
import Downloads
import Interpolations
import ModelParameters
import Oceananigans
import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.Diagnostics: cell_diffusion_timescale
import Pkg
import ProgressMeter
import RingGrids
import RootSolvers
import Thermodynamics

"""
Alias for `DomainSets.UnitInterval()`
"""
const UnitInterval = DomainSets.UnitInterval()

"""
Alias for numeric `Quantity` with type `NF` and units `U`
"""
const LengthQuantity{NF, U} = Quantity{NF, 𝐋, U} where {NF, U <: Units}

"""
Alias for Oceananigans `AbstractBoundaryConditionClassification`
"""
const BCType = AbstractBoundaryConditionClassification

"""
Alias for Oceananigans location types, i.e. `Center` or `Face`.
"""
const CenterOrFace = Union{Center, Face}

# Re-export selected types and methods from Oceananigans
export Simulation, Clock, Field, FieldTimeSeries, KernelFunctionOperation, Center, Face
export CPU, GPU, ReactantState, architecture, on_architecture
export Value, Flux, Gradient, ValueBoundaryCondition, GradientBoundaryCondition, FluxBoundaryCondition, NoFluxBoundaryCondition
export run!, time_step!, set!, reset!, compute!, interior, znodes, zspacings, location
export TimeStepWizard, conjure_time_step_wizard!, Callback, add_callback!, IterationInterval, TimeInterval

# Re-export selected types from FreezeCurves
export SFCC, SWRC, FreeWater, VanGenuchten, BrooksCorey

# Re-export common Dates types
export Year, Month, Day, Hour, Minute, Second

# Re-export unit types
export @u_str, uconvert, ustrip

# Re-export adapt
export adapt

const DEBUG = Ref(false)

function __init__()
    DEBUG[] = haskey(ENV, "TERRARIUM_DEBUG") && ENV["TERRARIUM_DEBUG"] == "true"
    if debug_mode()
        @warn "Debug mode enabled; debug hooks will be active and performance may be degraded."
    end
    return nothing
end

# internal utility types and methods
export @assert_kernel
include("utils/utils.jl")

include("domains.jl")

export XY, XYZ
include("abstract_variables.jl")

# grids
export UniformSpacing, ExponentialSpacing, num_layers
include("grids/vertical_discretization.jl")

export ColumnGrid, ColumnRingGrid, LandGrid
export create_land_grid, ground_domain, snow_domain, canopy_domain
include("grids/grids.jl")

export ERA5LandForcings, ERA5LandInvariants, ERA5LandLeafAreaIndex
include("input_output/assets.jl")

export InputSource, InputSources, FieldInputSource, FieldTimeSeriesInputSource
export update_inputs!, varpath, varpath, VarPath
include("input_output/input_sources.jl")

# process/model interface
export get_constants, get_grid, get_initializer, variables, processes,
    compute_auxiliary!, compute_boundary_conditions!, compute_tendencies!
include("abstract_model.jl")

# state variables
export StateVariables, get_fields
include("state_variables.jl")

# default initializers
export DefaultInitializer
include("initializers.jl")

# boundary condition helper functions
export FieldBC, FieldBCs, boundary_conditions
include("boundary_conditions.jl")

# forcing helper functions
export Forcings
include("forcings.jl")

# timestepping
export timestep!, default_dt, is_adaptive, get_timestepper, timestepping, Timestepping, Explicit, Implicit
include("timesteppers/abstract_timestepper.jl")

# abstract model types
include("models/abstract_types.jl")

# numerical solvers
include("solvers/solvers.jl")

# physical processes
include("processes/processes.jl")

# concrete model implementations
include("models/models.jl")

# model integrator/simulation types and methods
export ModelIntegrator, initialize, current_time, iteration, run_timesteps!
include("timesteppers/model_integrator.jl")

# adaptive-timestepping diagnostics (cell_diffusion_timescale for the TimeStepWizard)
include("timesteppers/cell_diffusion_timescale.jl")

# Concrete timestepper implementations
export ForwardEuler
include("timesteppers/forward_euler.jl")

export Heun
include("timesteppers/heun.jl")

export IMEX, AbstractIMEX
include("timesteppers/imex.jl")

# debugging utilities
include("diagnostics/debugging.jl")
include("diagnostics/progress.jl")
include("diagnostics/surface_fluxes.jl")

end # module Terrarium
