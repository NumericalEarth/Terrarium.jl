# Contributing

We gladly welcome any and all contributions to Terrarium.jl. Building a new land model is a huge undertaking that no single developer or scientist can hope to achieve alone. Collaboration always has been and always will be key to building good geoscientific models. Terrarium is no exception to this, and its success will depend on the contributions of the broader community.

There are multiple ways in which you could consider contributing to the project:
- If you have a question or an idea, raise an issue or start a discussion on the GitHub repository.
- If you want to try implementing something, clone the repository and make a pull request. Please take note of our [Software development practices](@ref) below.
- If you're not able to directly contribute yet but would like to support our work, consider sharing the GitHub repository (or this documentation) with others who you think might be interested.

Regardless of how you choose to contribute, we thank you for your participation, and we look forward to working with you!

## Software development practices

### Automated testing and continuous integration

Terrarium.jl adheres to software development standards for automated testing via continuous integration. We write unit tests for every function of our model. In some cases this might appear trivial, but we still want to achieve a near complete coverage of our code in the tests. The majority of the tests should cover the smallest possible units over different input arguments and types (if applicable). Unit tests should typically call the tested functions in a way that is representative for their use in the model, but try to reduce the computational complexity (e.g. by choosing very low dimensional inputs) to keep the overall CI time manageable. Additionally, we have some tests that ensure top-level functionality and stability of the model as well. Every additional proposed feature in a Pull Request has to come with unit tests. Tests verifying differentiability and GPU compatibility will also be requried.

### Kernel programming

Terrarium.jl is a device-agnostic model that runs on CPUs and GPUs by using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). This means that most computations need to be implemented within kernel functions that follow KernelAbstractions’ syntax. Each thread in the CPU or GPU will then compute the discretized equations for a single grid point.

For writing kernels we follow a strategy closely inspired by Oceaninanigans.jl: we want to fuse kernels as much as possible. Kernel fusion means that in practice we write very “large” kernels that fuse as many operations as possible together in one kernel. Kernel fusion leads to more efficient GPU computations, especially by reducing memory demand ([Wang et. al 2000](https://ieeexplore.ieee.org/document/5724850/)). In order to still keep our code well structured and modular, our approach relies on implementing most processes as inlined functions that can be called from a GPU kernel. We have not yet found any significant limitations to this approach.

How this looks in action in Terrarium.jl you can already see in the prototype code, e.g. for the [SoilEnergyBalance](https://github.com/NumericalEarth/Terrarium.jl/blob/main/src/processes/soil/soil_energy.jl): There `compute_tendencies!` is the mandatory function for the model component, it launches exactly one kernel `compute_energy_tendency!`, which includes several `@inline` function to compute individual contributions to the energy balance such as `compute_energy_tendency`, `compute_thermal_conductivity` and the `diffusive_heat_flux`.

It's also worth checking out the [Simulation tips](https://clima.github.io/OceananigansDocumentation/stable/simulation_tips/) provided in the documentation for Oceananigans.

### Automatic Differentiation with Enzyme 

For AD, we rely primarily on reverse mode differentiation via [Enzyme.jl](https://enzyme.mit.edu/julia/stable/). In contrast to many other AD systems, Enzyme doesn’t put particularly strong restrictions on coding style. For example, Array mutations are not only allowed, they are even encouraged!

There is one thing however, that is crucial for Enzyme to work: [type stability](https://docs.julialang.org/en/v1/manual/faq/#man-type-stability). The code absolutely has to be type stable, even in parts that are performance non-critical. Every occurrence of type instability may break differentiability with Enzyme. If this happens, we recommend first quickly checking for type instability of core function calls using `@code_warntype` and `@inferred` in unit tests. Then, if the issue is not yet apparent, debug the code using [Cthulhu.jl](https://github.com/JuliaDebug/Cthulhu.jl). Cthulhu is an awesome tool to interactively inspect your code for type instabilities and it is extremely easy to run: just preempt the function that you are trying to differentiate with `@descend` after loading the `Cthulhu` package.

Enzyme does, however, have some disadvantages; it is still not fully mature and bugs do occur. As of the time of writing (August 2025), this is especially the case for Julia 1.11. We currently recommend staying on Julia 1.10.10 (LTS) for the time being. Other cryptic Enzyme error messages have become rarer with time, but they do still occasionally happen. In these cases, we, along with our AD team led by Valentin in the DELTA-ESM project, are happy to offer support to the best of our abilities.

### Code formatting

We use [Runic.jl](https://github.com/fredrikekre/Runic.jl) for automatted code formatting. If you submit a PR you probably have seen a comment from our CI that the code is not formatted according to the Runic style. To use runic, you have to run the install Runic install script and then you can choose to either format directly from the command line with `runic --inplace .` (don't forget the `.` at the end!), or configure it in your editor of choice or use a Git hook. All details are described in [Runic's readme](https://github.com/fredrikekre/Runic.jl). We recommend setting up Runic as a Git hook to automatically format your code on commit. For that purpose, we provide the Git hook in the `.githooks` directory of the repository. You can copy the hook from there to your local git hooks directory and make it executable to use like so: 
```
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```