module TerrariumReactantExt

# Reactant support for Terrarium.
#
# Design: a Terrarium model whose grid lives on `ReactantState` allocates its state directly on
# the device grid and is initialized on the device (but slow in uncompiled mode). Only `timestep!`/`run!`
# are traced and compiled by Reactant.
# Kernel launches inside the compiled step trace fine (this requires `CUDA` to be loaded
# alongside `Reactant`, even on CPU).

using Terrarium
using Reactant
using Reactant: TracedRNumber
using Oceananigans

using Oceananigans.Architectures: ReactantState, CPU, architecture, on_architecture

using Terrarium: Terrarium, AbstractGrid, ColumnRingGrid, AbstractModel,
    ModelIntegrator, ground_domain, get_grid, get_timestepper

const RARCH = ReactantState

@inline Terrarium.uses_reactant(::Terrarium.ReactantMarker) = true

# Grids and models that live on the device
const ReactantGrid{NF, TX, TY, TZ, ST} = AbstractGrid{NF, TX, TY, TZ, <:RARCH, ST}
const ReactantModel{NF} = AbstractModel{NF, <:ReactantGrid}

# Inside the compiled stepping loop `clock.time` is a `TracedRNumber`, and it reaches host-level input
# code through `timestamp`/`convert_dt` (e.g. `FieldTimeSeriesInputSource.update_inputs!`). The generic
# `convert(NF, Δt)` cannot produce a concrete `NF` from a traced value, so emit a traced conversion instead.
Terrarium.convert_dt(::Type{NF}, Δt::TracedRNumber) where {NF <: Number} = TracedRNumber{NF}(Δt)

include("grids.jl")
include("transfer.jl")
include("integrator.jl")

end # module
