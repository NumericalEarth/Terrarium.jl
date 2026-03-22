# Terrarium.jl

Terrarium is a new and upcoming framework for hybrid physics- and data-driven land and terrestrial ecosystem modeling across spatial and temporal scales. We envision Terrarium to be part of a new generation of Earth system component models that combine modularity, interactivity, GPU-compability and auto-differentiability (AD) for seamless integration of process-based and data-driven model components in both global and regional scale simulations.

Terrarium is being developed alongside [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) and [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) as the land component of a new, user-friendly, and fully GPU/AD-compatible Earth System Model in the Julia programming language.

!!! warning "🚧🚧 Under construction 🚧🚧"
    This is an early prototype of our model and is not production-ready. Expect things to change rapidly and break often. If you share our vision for a new paradigm of Earth system modeling and would like to get involved in the project, don’t hesitate to reach out by creating an issue on GitHub issues or sending us an email. We are always happy to welcome new collaborators!

## Installation

Terrarium is still in an early prototype stage and is not yet registered as a package in the Julia General registry.

However, you can still install the package from the repository via the package manager (type `]` in your REPL):

```
pkg> add https://github.com/TUM-PIK-ESM/Terrarium.jl
```

If you would like to start hacking on the code directly, we recommend first cloning the repository:

```
git clone https://github.com/TUM-PIK-ESM/Terrarium.jl
```

and then initializing the project environment with

```
cd Terrarium.jl/
julia --project=. -e "import Pkg; Pkg.instantiate()"
```

To run the examples, make sure to set the project directory to the appropriate directory, e.g:

```
julia --project=examples/scripts examples/scripts/soil_heat_global.jl
```

You can also directly `activate` the example project environment from your REPL by first entering the package manager with `]` and then running the command `activate examples/scripts` followed by `instantiate`.

## Quick start

A natural first step with `Terrarium` is to set up and run your very first `SoilModel`. This represents a standalone model of transient heat, water, and carbon transport over a particular choice of `grid`. We start by chosing a `ColumnGrid` which represents one or more laterally independent vertical columns:

```@example
using Terrarium

# Set up a SoilModel on a ColumnGrid with 10 vertical soil layers that will run on the CPU with 32-bit precision
num_columns = 1
arch = CPU()
grid = ColumnGrid(arch, Float32, ExponentialSpacing(N=10), num_columns)
model = SoilModel(grid)
# Prescribe a constant surface temperature of 1°C
bcs = PrescribedSurfaceTemperature(:T_ub, 1.0)
integrator = initialize(model, ForwardEuler(eltype(grid)), boundary_conditions = bcs)
# Run the simulation forward for 10 model days
@time run!(integrator, period = Day(10))
```

That's it! You already succesfully ran a (very simple) simulation with Terrarium!

Note that setting `num_columns = 1` here corresponds to a point simulation for a single vertical column. However, we can easily scale this up by set `num_columns` to any positive integer (up to the memory limit of your system, of course).

We can also easily adapt this code to run a *global* simulation over a suitable spatial grid. For this, we'll need to have [`RingGrids`](https://github.com/SpeedyWeather/RingGrids.jl) installed (or import it directly from Terrarium with `using Terrarium.RingGrids`). Optionally, if a GPU is available and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is installed in the current project or global Julia environment, we can accelerate the global simulation by simply changing `CPU` to `GPU`:

```@example
using Terrarium
using RingGrids: FullGaussianGrid
using CUDA # needs to be seaprately installed

rings = FullGaussianGrid(8) # Gaussian grid with 16 lattitudinal rings (512 points, ~9.0˚ lat/lon)
arch = CUDA.functional() ? GPU() : CPU() # run on the GPU (if available)
grid = ColumnRingGrid(arch, Float32, ExponentialSpacing(N=10), rings) # create grid
model = SoilModel(grid)
# Prescribe a constant surface temperature of 1°C
bcs = PrescribedSurfaceTemperature(:T_ub, 1.0)
integrator = initialize(model, ForwardEuler(eltype(grid)), boundary_conditions = bcs)
# Run the simulation forward for 10 model days
@time run!(integrator, period = Day(10))
```
and voila! We have just run a GPU-accelerated, global-scale simulation of soil thermal dynamics with minimal additional effort. While more realistic simulations are of course more involved, this simple example demonstrates the core of what we aim to accomplish with Terrarium; a fast, user-friendly, and highly adaptable land model that can easily be configured to run on local, regional, and global scales.

## Table of contents

```@contents
Pages = [
    "introduction/numerical_core.md",
    "introduction/mathematical_formulation.md",
    "api_reference.md",
]
Depth = 3
```
