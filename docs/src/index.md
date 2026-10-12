# Terrarium.jl

Terrarium is a framework for hybrid physics- and data-driven land and terrestrial ecosystem modeling across spatial and temporal scales. We envision Terrarium to be part of a new generation of Earth system component models that combine modularity, interactivity, GPU-compatibility and auto-differentiability (AD) for seamless integration of process-based and data-driven model components in both global and regional scale simulations.

Terrarium is being developed alongside [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) and [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) as the land component of a new, user-friendly, and fully GPU/AD-compatible Earth System Model in the Julia programming language.

## Installation

You can install and use Terrarium as a Julia package by typing `]` in your REPL and running:
```
pkg> add Terrarium
```
or alternatively running
```julia
import Pkg
Pkg.add("Terrarium")
```
followed by
```julia
using Terrarium
```

!!! compat "Julia 1.10 or later is required"
    Currently, Terrarium.jl is tested from Julia 1.10 (LTS) onwards.


If you would like to not only use Terrarium but also actively develop it (or fix bugs 🐛), you can also install it as a [development package](https://pkgdocs.julialang.org/v1/toml-files/#Developed-package):
```
pkg> dev Terrarium
```
though it is worth noting that this will clone the repository into your Julia home directory by default. You can also fork/clone the repository yourself and start hacking!
```
git clone git@github.com:NumericalEarth/Terrarium.jl
```

You can then initialize the project environment by setting the repository as your working directory and running

```
julia --project=. -e "import Pkg; Pkg.instantiate()"
```

To run the example scripts, you will need to set the project directory to the `examples/` directory,

```
julia --project=examples -e "import Pkg; Pkg.instantiate()"
julia --project=examples examples/simulations/soil_heat_global.jl
```

You can also directly `activate` the example project environment from your REPL by first entering the package manager with `]` and then running the command `activate examples` followed by `instantiate`.

## Quick start

A natural first step with `Terrarium` is to set up and run your very first `SoilModel`. This represents a standalone model of transient heat, water, and carbon transport over a particular choice of `grid`. We start by choosing a `ColumnGrid` which represents one or more laterally independent vertical columns:

```@example
using Terrarium

# Set up a SoilModel on a ColumnGrid with 10 vertical soil layers that will run on the CPU with 32-bit precision
num_columns = 1
arch = CPU()
grid = ColumnGrid(arch, Float32, ExponentialSpacing(N=10), num_columns)
# Prescribed surface temperatures currently require explicit time stepping (the default is IMEX)
model = SoilModel(grid; timestepper = ForwardEuler(Float32))
# Prescribe a constant surface temperature of 1°C
bcs = PrescribedSurfaceTemperature(:T_ub, 1.0)
integrator = initialize(model, boundary_conditions = bcs)
# Run the simulation forward for 10 model days
@time run!(integrator, period = Day(10))
```

That's it! You already successfully ran a (very simple) simulation with Terrarium!

Note that setting `num_columns = 1` here corresponds to a point simulation for a single vertical column. However, we can easily scale this up by set `num_columns` to any positive integer (up to the memory limit of your system, of course).

We can also easily adapt this code to run a *global* simulation over a suitable spatial grid. For this, we'll need to have [`RingGrids`](https://github.com/SpeedyWeather/SpeedyWeather.jl/tree/main/RingGrids) installed (or import it directly from Terrarium with `using Terrarium.RingGrids`). Optionally, if a GPU is available and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is installed in the current project or global Julia environment, we can accelerate the global simulation by simply changing `CPU` to `GPU`:

```@example
using Terrarium
using RingGrids: FullGaussianGrid
using CUDA # needs to be separately installed

rings = FullGaussianGrid(8) # Gaussian grid with 16 latitudinal rings, 8 per hemisphere (512 points, ~9.0˚ lat/lon)
arch = CUDA.functional() ? GPU() : CPU() # run on the GPU (if available)
grid = ColumnRingGrid(arch, Float32, ExponentialSpacing(N=10), rings) # create grid
model = SoilModel(grid; timestepper = ForwardEuler(Float32))
# Prescribe a constant surface temperature of 1°C
bcs = PrescribedSurfaceTemperature(:T_ub, 1.0)
integrator = initialize(model, boundary_conditions = bcs)
# Run the simulation forward for 10 model days
@time run!(integrator, period = Day(10))
```
and voila! We have just run a GPU-accelerated, global-scale simulation of soil thermal dynamics with minimal additional effort. While more realistic simulations are of course more involved, this simple example demonstrates the core of what we aim to accomplish with Terrarium; a fast, user-friendly, and highly adaptable land model that can easily be configured to run on local, regional, and global scales.

## Table of contents

### Introduction
```@contents
Pages = [
    "introduction/basic_concepts.md",
    "introduction/numerical_core.md",
    "introduction/mathematical_formulation.md",
]
Depth = 2
```

### Running Terrarium
```@contents
Pages = [
    "running/configuring.md",
    "running/initialization.md",
    "running/time_stepping.md",
    "running/input_sources.md",
]
Depth = 2
```

### Extending Terrarium
```@contents
Pages = [
    "extending/core_interfaces.md",
    "extending/state_variables.md",
    "extending/implementing_processes.md",
    "extending/coupling_processes.md",
    "examples/extending/linear_ode_exp_growth.md",
    "examples/extending/simple_snow_ddm.md",
    "examples/extending/linear_heat_conduction.md",
]
Depth = 2
```

### Models
```@contents
Pages = [
    "models/land_model.md",
    "models/soil_model.md",
    "models/vegetation_model.md",
]
Depth = 2
```

### Processes

```@contents
Pages = [
    "processes/atmosphere/atmosphere.md",
    "processes/soil/soil.md",
    "processes/surface_energy/surface_energy_balance.md",
    "processes/surface_hydrology/surface_hydrology.md",
    "processes/vegetation/vegetation.md",
    "processes/utils/physical_constants.md",
    "processes/utils/physics_utils.md",
]
```

### Other
```@contents
Pages = [
    "contributing.md",
    "api_index.md",
    "references.md",
]
Depth = 2
```
