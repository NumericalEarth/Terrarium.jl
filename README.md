# Terrarium.jl
<strong> 🌲🌡💧 Fast, differentiable, and GPU-aware land modeling across scales with [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) </strong>
---

<a href="https://numericalearth.github.io/Terrarium.jl/dev">
<img alt="Development documentation" src="https://img.shields.io/badge/documentation-latest_release-blue.svg">
</a>
<a href="https://github.com/NumericalEarth/Terrarium.jl/actions/workflows/CI.yml"><img src="https://github.com/NumericalEarth/Terrarium.jl/actions/workflows/CI.yml/badge.svg" alt="Build status: CI" /></a>
<a href="https://www.repostatus.org/#active"><img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active – The project has reached a stable, usable state and is being actively developed." /></a>
<a href="https://eupl.eu/1.2/en"> <img alt="EUPLv1.2 license" src="https://img.shields.io/badge/License-EUPLv1.2-blue.svg?style=flat-square"> </a>
<a href="https://github.com/fredrikekre/Runic.jl">
<img alt="code style: runic" src="https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black.svg?style=flat-square">
</a>

[Terrarium.jl](https://numericalearth.github.io/Terrarium.jl/dev) is a framework for hybrid physics- and data-driven land modeling across spatial and temporal scales. We envision Terrarium to be part of a new generation of Earth system component models that combine modularity, interactivity, GPU-compatibility and auto-differentiability (AD) for seamless integration of process-based and data-driven model components in both global and regional scale simulations.

Terrarium is being developed alongside [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) and [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) as the land component of a new, user-friendly, and fully GPU/AD-compatible Earth System Model in the Julia programming language.

> [!WARNING]
> 🚧🚧 Under Construction! 🚧🚧
>
> Terrarium.jl is still in an early stage of development and is not production-ready. Expect things to change rapidly and break often. If, however, you share our vision for a new paradigm of Earth system modeling and would like to get involved in the project, please don’t hesitate to reach out by creating an issue on GitHub issues or sending us an email. We are always happy to welcome new collaborators!

## Our goals
We want a land surface (and subsurface) model that is
- **Fast** enough to run global-scale simulations on a laptop at coarse (100 km) resolutions
- **Flexible** enough to scale-up to high resolution simulations in HPC environments
- **Fully GPU-compatible** with the ability to easily switch between running on CPU-based and GPU-based architectures
- **Fully auto-differentiable** with Enzyme.jl to enable systematic parameter estimation and hybrid modeling with neural differential equations
- **Modular** and **extensible** to allow for rapid prototyping of model components with varying levels of complexity
- **Interactive** and **user-friendly** to make land surface modeling fun and accessible for a larger audience of researchers and practitioners, as well as students and educators
- **Open-source** and **community-driven** to foster interdisciplinary collaboration and development

It is important to emphasize, however, what Terrarium is not:
- Terrarium is **not a comprehensive land model**. While we are always open to suggestions and contributions to add new processes, we do not aim to build a state-of-the-art terrestrial ecosystem model rivaling that of, e.g. the [CTSM](https://github.com/ESCOMP/CTSM). Terrarium will always favor simplicity, efficiency, and interoperability over process complexity.
- Terrarium is **not “just another model”**. We do not intend for users to simply download our model products and cite our papers. We want users to directly interact with our model, ideally running their own simulations and writing their own code.
- Terrarium is **not a monolithic model**. Modularity and extensibility are core to our vision. Terrarium provides a library of models, process implementations, and numerical tools which users can use to build their own simulations. We will provide guidance and a set of well-tested and stable model configurations, but we encourage users to experiment and push the limits of what those models can do.

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

Once you have the package installed, head over to the [Quick start](https://numericalearth.github.io/Terrarium.jl/dev/#Quick-start) section of the documentation to get started.

If you would like to not only use Terrarium but also actively develop it (or fix bugs 🐛), you can also install it as a [development package](https://pkgdocs.julialang.org/v1/toml-files/#Developed-package):
```
pkg> dev Terrarium
```
However, note that this will clone the repository into your local `~/.julia/dev` directory. You can alternatively directly fork/clone the repository yourself and start hacking!
```
git clone git@github.com:NumericalEarth/Terrarium.jl
```

The project environment can be initialized by setting the repository as your working directory and running

```
julia --project=. -e "import Pkg; Pkg.instantiate()"
```

To run the example scripts, you will need to also activate and set up the `examples` project environment,

```
julia --project=examples -e "import Pkg; Pkg.instantiate()"
```

which declares some additional dependencies that are needed for some of the example scripts. Note that this may take a few minutes to finish precompiling. Once finished, you can execute the example scripts within this environment:

```
julia --project=examples examples/simulations/soil_heat_global.jl
```

You can also `activate` the example project environment from your REPL by first entering the package manager with `]` and then running the command `activate examples`. This will let you interactively execute the examples from the REPL or your preferred IDE.

## Why Oceananigans?
It might initially seem strange that a land model would be built on top of a framework for ocean modeling. There are, however, some key advantages in doing so:

1. Firstly, like ocean models, land models are commonly implemented using finite difference and/or finite volume method (FDM/FVM) to approximate spatial gradients in mass and energy conservation laws. Oceananigans provides state-of-the-art tools for FVM simulation in Julia, with a focus on geophysical applications, which aligns well with our goals. Like most land models, Terrarium will initially focus on 1D column modeling; however, using Oceananigans affords us the possibility of very feasibly expanding to 2D and 3D simulations in the future!
2. Secondly, the numerical operators provided by Oceananigans are built to be both auto-differentiable and GPU-compatible out-of-the-box, which means that Terrarium can inherit these capabilities almost “for free”.
3. Finally, and perhaps most importantly, we believe in the vision pioneered by the [Climate Modeling Alliance](https://clima.caltech.edu/) and the [NumericalEarth](https://github.com/NumericalEarth/) projects for the development of a new generation of Earth System Models that are open, accessible, interactive, and capable of learning from data in a multitude of ways that go beyond traditional data assimilation.

## Contributing

An open source project is only as strong as its community of contributors. We're always happy to accept contributions, no matter how big or small!

Terrarium.jl is still in an early stage of development, so this is a golden opportunity for you to get your ideas in on the ground floor. If you have some ideas or code you would like to contribute, please don't hesitate to create an issue and get involved!

## Copyright and license

See the [NOTICE](./NOTICE) file for copyright information.

Terrarium.jl is free and open source licensed under the [European Union Public License v1.2](https://eupl.eu/1.2/en).

What does that mean for you? You are 100% free to
- Copy, modify, and redistribute the code
- Use the software as a package in your own project (regardless of license or copyright status)
- Use the software for both commercial and non-commercial purposes

However, if you make changes or modification to the code, excluding those made purely for the purpose of interoperability, **you are required to re-distribute the modified software under the EUPL v1.2 or a compatible license**. This is vital to ensure the long-term survival of the project and to foster an open, supportive, and diverse community.

## Acknowledgements

Terrarium.jl is a community-oriented project lead by the [FutureLab on Artificial Intelligence](https://www.pik-potsdam.de/en/institute/departments/complexity-science/research/artificial-intelligence) at the Potsdam Institute for Climate Impact Research (PIK) and Earth System Modeling group at the Technical University of Munich (TUM). We acknowledge funding support from
- Horizon Europe grant agreement number 101184070 (Past to Future)
- Horizon Europe grant agreement number 101137601 (ClimTip)
- Volkswagen Foundation
- Munich Data Science Institute (MDSI) at Technical University of Munich (TUM) via the Linde/MDSI Doctoral Fellowship program

<img width="409" height="91" alt="image" src="https://github.com/user-attachments/assets/d75524fc-2cd4-4951-91c6-d13f0b62229f" />

**Disclaimer**:
*Views and opinions expressed are however those of the author(s)
only and do not necessarily reflect those of the European Union or
European Research Executive Agency (REA). Neither the
European Union nor the granting authority can be held responsible
for them.*
