---
title: 'Terrarium.jl: Flexible, GPU-accelerated, and fully auto-differentiable land modeling in Julia'
tags:
  - Julia
  - land
  - land surface
  - climate
  - numerical
  - differentiable
  - GPU-accelerated
authors:
  - name: Brian Groenke
    orcid: 0000-0003-2570-9342
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Maximilian Gelbrecht
    orcid: 0000-0002-0729-6671
    equal-contrib: true
    affiliation: "1, 2"
  - name: Maha Badri
    affiliation: "1, 2"
  - name: Olivier Bonte
    orcid: 0000-0003-1806-7572
    affiliation: 3
  - name: Mohammed Ayoub Chettouh
    affiliation: 4
    orcid: 0009-0005-7158-8547
  - name: Yunan Lin
    orcid: 0009-0008-5464-7447
    affiliation: "1, 2"
  - name: Kieran Hunt
    orcid: 0000-0003-1480-3755
    affiliation: 5
  - name: Pier Luigi Vidale
    orcid: 0000-0002-1800-8460
    affiliation: 5
  - name: Sebastian Bathiany
    orcid: 0000-0001-9904-1619
    affiliation: "1, 2"
  - name: Niklas Boers
    orcid: 0000-0002-1239-9034
    affiliation: "1, 2"
affiliations:
 - name: Potsdam Institute for Climate Impact Research, Germany
   index: 1
   ror: "03e8s1d88"
 - name: Technical University of Munich, Germany
   index: 2
   ror: "02kkvpp62"
 - name: Ghent University, Belgium
   index: 3
   ror: "00cv9y106"
 - name: Max Planck Institute for Biogeochemistry, Germany
   ror: "051yxp643"
   index: 4
 - name: National Centre for Atmospheric Science, Department of Meteorology, University of Reading, United Kingdom
   index: 5
   ror: "05v62cm79"
date: 25 November 2026
bibliography: paper.bib
---

# Summary

Land surface and terrestrial ecosystem models simulate the exchange of energy, water, and carbon between soil, snow, vegetation, and the atmosphere.
They are crucial components of global-scale *Earth System Models* (ESMs) which are widely used to study the past, present, and future of Earth's climate [@flatoEarthSystemModels2011].
Terrarium.jl is a software package in the Julia programming language [@bezansonJuliaFreshApproach2017] that aims to enable fast, flexible, and user-friendly simulations of terrestrial ecosystems and land-atmosphere interactions across scales, either as a standalone model or coupled within a larger Earth system model.

The numerical implementation is based on Oceananigans.jl [@ramadhanOceananigansjlFastFriendly2020; @wagnerHighlevelHighresolutionOcean2025], which provides numerical primitives for highly efficient finite volume simulation of fluid dynamics on orthogonal grids.
Terrarium.jl builds upon these primitives to enable GPU-accelerated and differentiable simulations of the relevant governing equations for nonlinear fluid transport in porous media (i.e. soil and rock) as well as point-scale processes like the vegetation carbon cycle, surface hydrology, and the exchange of water and energy between the land surface and the atmosphere.

Terrarium.jl is being developed as part of the NumericalEarth project, which is an international community of researchers and software developers dedicated to building a new generation of multi-scale, multi-architecture, and automatically differentiable Earth system models with a focus on user-friendliness and accessibility.

# Statement of need

Most existing ESM land components, including those involved in the international Coupled Model Intercomparison Project (CMIP), are implemented in statically compiled languages such as Fortran (and less commonly, C/C++).
Although these languages are advantageous in terms of computational efficiency, they incur a significant software engineering burden and limit the accessibility of the model code for many students and researchers who lack extensive numerical programming experience.
Furthermore, reliance on Fortran/C++ inhibits the adoption of state-of-the-art tools for high-level GPU and differentiable programming, which have served as the technological foundation for recent developments in machine learning and data-driven modeling [@gelbrechtDifferentiableProgrammingEarth2023].
There is thus a general need for a new generation of ESM components built using modern numerical computing software written in higher level, dynamic languages such as Julia or Python, and that offer robust and easy-to-use tooling for GPU acceleration and automatic differentiation (AD).
This would enable systematic, gradient-based parameter estimation, uncertainty quantification, and tighter integration of physics-based and data-driven modeling workflows.

Another key limitation of many existing land models is their tendency to mix process formulations and numerics in an *ad hoc* and inconsistent manner [@clarkNumericalImplementationLand2021].
This makes it difficult for these models to incorporate state-of-the-art numerical methods since this means also needing to reformulate the processes themselves.
Terrarium.jl addresses this limitation by formulating all processes in terms of continuous-time dynamics and enforcing a clear separation between the instantaneous tendencies of the system and the numerical scheme used to discretize the corresponding governing equations in time.

# State of the field

There have been numerous recent efforts to reformulate land models in modern programming languages, such as DifferLand [@fangDifferentiableLandModel2026] and JAX-CanVeg [@jiangJAXCanVegDifferentiableLand2025], both of which enable GPU-accelerated and AD-compatible land modeling using Python/JAX.
However, Terrarium.jl distinguishes itself from both of these models in its support for efficient global simulations (unlike JAX-CanVeg) and its focus on providing only process-based components (unlike DifferLand) while enabling ML-based extensions via its GPU and AD capabilities.
Recent work on the largely process-based ClimaLand.jl model [@deckClimaLandLandSurface2026] has also demonstrated the promise of GPU-accelerated land modeling in Julia.
In contrast to ClimaLand.jl, Terrarium.jl enables not only GPU-accelerated land simulations but also full compatibility with both forward- and reverse-mode AD via Enzyme.jl and Reactant.jl [@mosesInsteadRewritingForeign2020; @mosesDJ4EarthDifferentiablePerformancePortable2026].

# Software design

![Overview of the Terrarium.jl software design. Models (blue) are built from one or more `Process`es (purple) which may contain additional parameterization types that determine which equations and parameters are used. Models and any required external inputs (green) are passed to `initialize` which allocates the model state and configures the time stepper. Each `Process` defines device-agnostic compute kernels that are "launched" over the underlying spatial grid within each time step. \label{fig:design}](assets/figure1.png)

Terrarium.jl is designed as a modular framework of land model components rather than a monolithic model.
This allows Terrarium.jl to serve as the basis for a wide range of downstream land and ecosystem modeling use cases, spanning both highly detailed local-scale simulations with prescribed boundary conditions and intermediate-complexity global-scale simulations within a larger, coupled Earth system model.

Models in Terrarium.jl are composed of four basic components: one or more `Process`es (subtyping `AbstractProcess`) implementing specific physical relationships and governing equations, a `grid` that describes the discretization of the underlying spatial domain, a `timestepper` that determines how the prognostic variables of the model are advanced in time, and an `initializer` that specifies how the initial state of the prognostic variables is determined (\autoref{fig:design}).
All state variables are classified as *prognostic* (evolved in time), *auxiliary* (derived from the prognostic state), or *inputs* (externally prescribed).
Constitutive relationships between prognostic and auxiliary variables are explicitly declared and evaluated separately from the tendency and timestepping scheme.
This design facilitates the implementation and comparison of new solvers and time discretization schemes while also helping to maintain mathematical consistency between model components.

Terrarium.jl currently provides four primary model types that can be used to run simulations: `SoilModel`, `VegetationModel`, `SnowModel`, and `LandModel`. The first three represent standalone models of soil, snow, and vegetation processes while the `LandModel` couples together all three of these components to enable realistic, multi-physics simulations of the land surface.
Each model type combines two or more `Process`es by defining the orchestration logic needed to wire together the various components.
Users can implement their models by defining subtypes of the `AbstractModel` and `AbstractProcess` types that satisfy their documented *method interfaces* ("method" here refers to an instance of a Julia function defined for a particular set of input argument types).
Processes may also consist of zero or more *parameterization* types that act as both parameter containers and configuration switches (via dispatch) for varying formulations of the constitutive relationships needed to fully define the underlying governing equations.

Terrarium.jl's modular design, while flexible, comes with trade-offs.
Terrarium.jl simultaneously prioritizes support for multi-architecture parallelization via KernelAbstractions.jl [@churavyKernelAbstractionsjl2026] and AD via Enzyme.jl and Reactant.jl.
Maximizing modularity would necessitate each `Process` launching its own GPU *kernel*, i.e. a function executed in parallel over the spatial field, which can incur significant overhead in models that involve coupling together many interdependent physical processes.
To balance modularity with efficient parallelization, Terrarium.jl `Process`es are typically implemented at three different levels of abstraction: a top-level interface which can be used to evaluate and test each `Process` type standalone, a grid-aware kernel function interface called at the given grid indices within a device-agnostic compute kernel, and primitive functions that implement scalar equations independent of the underlying spatial discretization.
This pattern allows for most process code to be reused across different kernels.

![Performance benchmarks of Terrarium.jl's `LandModel` in "simulated years per wallclock day" (SYPD) vs. the number of grid points on both CPU (single core, AMD EPYC 9554) and GPU (single NVIDIA H100) with and without Reactant \label{fig:benchmark}](assets/figure2.png)

\autoref{fig:benchmark} shows the performance of a full Terrarium.jl `LandModel` across multiple horizontal resolutions. On a single GPU, throughput stays constant at around 2000 simulated years per day (SYPD) until reaching nearly $10^5$ grid cells (~1° x 1° resolution). Beyond this, the GPU remains roughly 3000 times faster than single-core CPU which translates to a 25-fold performance improvement over a single, 128-core compute node. Reactant also shows a considerable improvement in performance at lower resolutions where kernel launch overhead dominates the per-step cost.

![Land surface variables from a five-year simulation with Terrarium.jl coupled to the SpeedyWeather.jl atmosphere model with slab ocean at T64 spectral resolution (about 1.9°), starting on 1 October 2024. The simulation includes soil thermodynamics and hydrology with prescribed vegetation based on a leaf area index climatology calculated from ERA5-Land (1980-2010). All panels show means over the simulation period, except for snow water equivalent, which shows the mean of the annual maxima over all snow years (October to September); Antarctica is included in the simulation as snow-covered land but not shown.\label{fig:coupled}](assets/figure3.png)

Currently, Terrarium.jl implements a soil energy-hydrology scheme that is largely based on the equations and parameterizations of CryoGrid [@westermannCryoGridCommunityModel2023; @langerEvolutionArcticPermafrost2024; @groenkeCryoGridJL2024] with pedotransfer functions taken from SURFEX/ISBA [@massonSURFEXv72LandOcean2013; @noilhanISBALandSurface1996].
The surface energy balance, evapotranspiration schemes, and vegetation/canopy processes are based primarily on PALADYN [@willeitPALADYNV10Comprehensive2016] which presently serves as the land component of the CLIMBER-X intermediate-complexity Earth system model [@willeitEarthSystemModel2022].
A basic 1-pool soil carbon model, with vertical transport, litter input, and decomposition is presently implemented, with a more comprehensive approach based on @ahrensCombinationEnergyLimitation2020 under active development.
Example outputs from a prototype (uncalibrated) Terrarium-SpeedyWeather coupled simulation are shown in \autoref{fig:coupled}.

# Research impact statement

![Sensitivities of the integrated latent heat flux (from evaporation) after 24 hours at the marked grid cell to the initial atmospheric humidity field. Arrows show the average wind direction over 24 hour simulation period. Negative sensitivities show how evaporation decreases with higher upwind humidity. The faint ring patterns are numerical artifacts from SpeedyWeather.jl's spectral discretization.\label{fig:sensitivity}](assets/figure4.png)

Terrarium.jl's modular design, device-agnostic parallelization architecture, and first-class support for AD make it ideal for cutting-edge research on so-called "hybrid" land-atmosphere modeling, which aims to combine data-driven components, such as neural networks, with physics-based simulation [@shenDifferentiableModellingUnify2023; @gelbrechtDifferentiableProgrammingEarth2023].
Terrarium is being actively used to develop and evaluate the capabilities of such methods to improve global land and climate simulations. One use-case of AD in physics-based modeling is local sensitivity analysis of the simulated outputs to the initial state; an example of this is shown in \autoref{fig:sensitivity}. Another use case is enabling online downscaling of a coarse resolution atmosphere to a higher resolution land surface via machine learning, which is currently underway as part of the Past2Future (P2F) project.

Terrarium.jl is designed to allow for efficient coupling with other global or regional, GPU-accelerated ESM components such as the SpeedyWeather.jl [@klowerSpeedyWeatherjlReinventingAtmospheric2024] and Breeze.jl [@wagnerBreezejl2026] atmospheric models via the NumericalEarth.jl [@wagnerNumericalEarthjl2026] framework. The NumericalEarth developer community maintains an extension module for Terrarium.jl that allows its models to be plugged in as the land component in `EarthSystemModel` simulations, thereby broadening the scope of Terrarium.jl's impact beyond its immediate developer and user community.

# AI usage disclosure

All text in this manuscript was drafted and reviewed by the authors.
Generative AI (Claude Sonnet 5, Opus 5.5, and Fable 5.1 via Claude Code) was used to assist with developing the scripts that run the example simulation and produce the figures as well as for editorial feedback.
Some parts of the software, including tests, benchmarks, documentation, and the source code itself, are developed with AI assistance (code generation, refactoring, and debugging) under careful human oversight.
AI-assisted development is transparently documented in markdown-based implementation plans that are tracked in the `docs/dev/plans` directory of the repository.
The authors made all core design decisions, and all AI-assisted outputs and code changes are reviewed, validated, and signed off by human developers.

# Acknowledgements

BG and NB acknowledge the support of the Past2Future (P2F) project; The Past to Future (P2F) project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101184070.
MG and NB acknowledge funding from the Volkswagen Foundation and the European Union’s Horizon Europe research and innovation programme ClimTip (grant agreement no. 101137601), this is ClimTip contribution #215.
PLV acknowledges funding from the National Centre for Atmospheric Science and from Schmidt Futures (LEMONTREE project, #XXX)
OB acknowledges funding from Research Foundation - Flanders (V436726N) for the research stay at the Potsdam Institute for Climate Impact Research.
Funded by the European Union.
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Climate, Infrastructure and Environment Executive Agency (CINEA).
Neither the European Union nor the granting authority can be held responsible for them.
None of the funders had any role in the design of the software or preparation of this manuscript.

# References
