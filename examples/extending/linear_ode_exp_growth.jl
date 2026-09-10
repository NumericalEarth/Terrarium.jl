# # Getting started with a simple exponential growth model
#
# In this example, we will set up an embarrassingly simple example to demonstrate Terrarium's model interface. Our model will have 1-dimensional exponential dynamics with a constant offset
#
# ```math
# \frac{du}{dt} = \alpha u + c + F(t)
# ```
#
# for an arbitrary prognostic variable ``u``. For the sake of this demonstration we will treat the offset ``c`` as an auxiliary/diagnostic variable even though it is constant in time. ``F(t)`` is an external forcing that we apply.

using Terrarium

#
# We begin by defining our model `struct` that subtypes [`Terrarium.AbstractModel`](@ref):
#
# A "model" in Terrarium is a subtype of `Terrarium.AbstractModel` and is a `struct` type consisting of
#  * `grid` which defines the discretization of the spatial domain
#  * `initializer` which is responsible for initializing state variables
#  * further fields that define processes, dynamics and submodels
#
# When we follow the advised naming notations of `grid` and `initializer` we inherit default methods from `Terrarium.AbstractModel` such as [`get_grid`](@ref) and [`get_initializer`](@ref). For more complex models we might need to implement custom overrides of `initialize!(state, ::Model, ::Initializer)` to initialize model states.
#
# ## What is a "grid"?
#
# The `grid` defines the spatial discretization. Our grids are based on those of [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) (and [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl)/[RingGrids.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl/tree/main/RingGrids)) in order to take advantage of their capabilities for device-agnostic parallelization.
#
# As mentioned in the documentation on [grids](@ref Grids), Terrarium currently provides two grid types:
#
# * [`ColumnGrid`](@ref) is a set of laterally independent vertical columns with dimensions ``(x, y, z)`` where ``x`` is the column dimension, ``y=1`` is constant, and ``z`` is the vertical axis,
# * [`ColumnRingGrid`](@ref) represents a global (spherical) grid of independent, vertical columns where the spatial discretization in the horizontal direction is defined by a [`RingGrids.AbstractGrid`](@extref).
#
# In both cases we need to specify the vertical discretization via an [`UniformSpacing`](@ref), [`ExponentialSpacing`](@ref) or [`PrescribedSpacing`](@ref).
#
# ## Initializer and Boundary Conditions
#
# For our basic example here the default initializer (which does nothing) will suffice, and we won't have to define a custom one.
#
# Boundary conditions are specified by passing Oceananigans [`BoundaryCondition`](@extref Oceananigans.BoundaryConditions.BoundaryCondition) types to `initialize`. In the case of a linear ODE, however, no boundary conditions are required.
#
# ## What's our `grid`?
#
# For our current example, we are defining a simple linear ODE without any spatial dynamics, so we can get away with just a single column with one vertical layer. We can define it like so:

grid = ColumnGrid(CPU(), Float64, UniformSpacing(N = 1))

# ## Defining the model
#
# We start by defining a `struct` for our model that inherits from `AbstractModel` and consists of four properties: the spatial `grid`, an `initializer`, a single [`AbstractProcess`](@ref Terrarium.AbstractProcess) defining the dynamics, which we will also implement below and the timestepper used to step the model forward in time.

@kwdef struct LinearDynamics{NF} <: Terrarium.AbstractProcess{NF}
    "Exponential growth rate"
    alpha::NF = 0.01
    "Constant offset"
    c::NF = 0.1
end
#
@kwdef struct ExpModel{NF, Grid <: Terrarium.AbstractLandGrid{NF}, Dyn, Init, TS <: Terrarium.AbstractTimeStepper} <: Terrarium.AbstractModel{NF, Grid}
    "Spatial grid on which state variables are discretized"
    grid::Grid
    "Linear dynamics process"
    dynamics::Dyn = LinearDynamics()
    "Model initializer"
    initializer::Init = DefaultInitializer(eltype(grid))
    "Time stepper (e.g. `ForwardEuler`, `Heun`, or `IMEX`)"
    timestepper::TS = ForwardEuler(eltype(grid))
end

# ## Defining the model behavior
#
# Now, we want to define our intended model behavior. For this, we need to define the following methods:
#
# * `variables(::Model)` returns a tuple of variable metadata declaring the state variables. As defined in the documentation on [state variables](@ref "State variables"), variables must be one of three types: `prognostic`, `auxiliary` (sometimes referred to as "diagnostic"), or `input`. Prognostic variables fully characterize the state of the system at any given timestep and are updated according to their tendencies (i.e. ``u`` in our example). Tendencies are automatically allocated for each prognostic variable declared by the model. In this example we will treat the offset ``c`` as an auxiliary variable, though we could also just include it as a constant in the tendency computations.
# * `compute_auxiliary!(state, ::Model)` computes the values of all auxiliary variables (if necessary) assuming that the prognostic variables of the system in state are available for the current timestep.
# * `compute_tendencies!(state, ::Model)` computes the tendencies based on the current values of the prognostic and auxiliary variables stored in state.
#
# So, let's define those:

Terrarium.variables(::ExpModel) = (
    Terrarium.prognostic(:u, XY(), desc = "Exponential growth variable"),
    Terrarium.auxiliary(:c, XY(), desc = "Constant offset for growth"),
    Terrarium.input(:F, XY(), default = 0.0, desc = "External forcing"),
)

# Here, we defined our three variables with their names as a `Symbol` and whether they are 2D variables ([`XY`](@ref)) on the spatial grid or 3D variables ([`XYZ`](@ref)) that also vary along the vertical z-axis. Here we are considering only a simple scalar model so we choose 2D (`XY`), bearing in mind that all points in the X and Y dimensions of `ColumnGrid` are independent of each other.
#
# We also need to define `compute_auxiliary!` and `compute_tendencies!` as discussed above. We will use here a pattern which is commonly employed within Terrarium: we unpack the grid and process from the model and forward the method calls to more specialized ones defined for the `LinearDynamics` process. The `compute_auxiliary!` and `compute_tendencies!` of `AbstractProcess`es follow the signatures `(state, grid, processes...)`, as you see here:

function Terrarium.compute_auxiliary!(state, model::ExpModel)
    compute_auxiliary!(state, model.grid, model.dynamics)
    return nothing
end
#
function Terrarium.compute_tendencies!(state, model::ExpModel)
    compute_tendencies!(state, model.grid, model.dynamics)
    return nothing
end

# Note that, when implementing models within the Terrarium module itself, the `Terrarium.` qualifier in the definition is not needed.
#
# ## Implementing the dynamics
#
# Next, we define the functions that compute the actual dynamics. In order to do this, we need to know a little about how the variables we just defined are handled in our [`StateVariables`](@ref Terrarium.StateVariables). The `StateVariables` hold all prognostic and auxiliary variables, their tendencies and closures and additional inputs and forcings in seperate `NamedTuples`. Note that Terrarium also defines shortcuts such that, e.g. in our example, both `state.prognostic.u` and `state.u` would work.
#
# With that in mind, let's define the methods:

function Terrarium.compute_auxiliary!(state, grid, dynamics::LinearDynamics)
    return state.auxiliary.c .= dynamics.c
end
#
function Terrarium.compute_tendencies!(state, grid, dynamics::LinearDynamics)
    return let u = state.prognostic.u,
            ∂u∂t = state.tendencies.u,
            α = dynamics.alpha,
            c = state.auxiliary.c,
            F = state.inputs.F
        ∂u∂t .= α .* u .+ c .+ F
    end
end

# These example compute functions are really the simplest possible, for more complex operations, we would need to define them via `KernelAbstractions` kernels. We will not go into further details on that in this notebook, as it is treated in [another example](@ref snow_ddm_example).
#
# However, now we have everything our model needs and we can finally use it!
#
# ## Running our model
#
# First, we will define our initial conditions.
#
# User-specified `Field` initializers passed to `initialize` can be provided in any form supported by `Oceananigans.set!` (see [the corresponding Oceananigans documentation](https://clima.github.io/OceananigansDocumentation/stable/fields#Setting-Fields)), including constants, arrays, and functions of the form `(x,z) -> val`:

initializers = (u = 1.0,)

# Then, we define our forcing. For that, our time-dependent forcing is loaded in from a `Oceananigans.FieldTimeSeries`. If you want to load the forcing from e.g. a netCDF file you can use the `RasterInputSource` that is based on [Rasters.jl](https://github.com/rafaqz/Rasters.jl). In the concrete case, we'll just generate a random forcing:

using Random

Random.seed!(1234) # set random seed

t_F = 0:1:300; #seconds
F = FieldTimeSeries(grid, XY(), t_F);
F.data .= randn(size(F));
input = InputSource(grid, F, name = :F)

# Here we constructed a 2D (`XY()`) time series on our `grid` at times `t_F` with random normal distributed data and defined our `InputSource` for our model based on it.

# Then, we construct our model from the chosen `grid`. The timestepper is configured on the model itself;
# here we choose the second-order [`Heun`](@ref) method with a timestep of 1 second via the `timestepper` keyword.

model = ExpModel(grid; timestepper = Heun(Δt = 1.0))

# We now can initialize our model, i.e. we run all pre-computation, and initialize a numerical integrator for our model by passing it
# to `initialize` along with our input/forcing data.

integrator = initialize(model; inputs = input, initializers)

# We can advance our model by one step via the `timestep!` method:

timestep!(integrator)

integrator.state.u

# But wait there's more! What if we want to actually save the results?
#
# The `integrator` data structure implements the Oceananigans model interface, so we can also use it to set up a [`Simulation`](@extref Oceananigans.Simulations.Simulation) (for more details, see [here](@ref "Setting up a `Simulation`")):

sim = Simulation(integrator; stop_time = 300.0, Δt = 1.0)

# We can then add an output writer to the simulation and finally `run!` it!

using Oceananigans: JLD2Writer, TimeInterval
using Oceananigans.Units: seconds
using JLD2

Terrarium.initialize!(integrator) #Re-initialize to start from t₀ = 0 seconds again
output_dir = mkpath(tempname())
output_file = joinpath(output_dir, "simulation.jld2")
sim.output_writers[:snapshots] = JLD2Writer(
    integrator,
    (u = integrator.state.u,);
    filename = output_file,
    overwrite_existing = true,
    including = [:grid], # include the grid with the output
    schedule = TimeInterval(10seconds)
)
run!(sim)
@assert isfile(output_file) "Output file does not exist!"
println("Simulation data saved to $(output_file)")

# Then load the output data and plot the results:

using CairoMakie
import DisplayAs

fts = FieldTimeSeries(output_file, "u")

DisplayAs.PNG(plot(1:length(fts), [fts[i][1, 1, 1] for i in 1:length(fts)]))
