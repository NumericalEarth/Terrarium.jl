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
using Oceananigans

using Oceananigans.Architectures: ReactantState, CPU, architecture, on_architecture

using Terrarium: Terrarium, AbstractLandGrid, ColumnGrid, ColumnRingGrid, AbstractModel,
    ModelIntegrator, get_field_grid, get_grid, get_timestepper

const RARCH = ReactantState

@inline Terrarium.uses_reactant(::Terrarium.ReactantMarker) = true

# Land grids that live on the device.
const ReactantLandGrid{NF, TX, TY, TZ} = AbstractLandGrid{NF, TX, TY, TZ, <:RARCH}
const ReactantModel{NF} = AbstractModel{NF, <:ReactantLandGrid{NF}}

include("grids.jl")
include("transfer.jl")
include("integrator.jl")

end # module
