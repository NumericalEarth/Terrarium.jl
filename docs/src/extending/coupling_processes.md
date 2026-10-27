# Coupling processes

```@meta
CurrentModule = Terrarium
```

Most realistic land surface simulations require multiple processes that must share information at each time step: for example, a soil energy balance may apply the ground heat flux at the land surface estimated by the surface energy balance as an upper boundary condition, or a canopy interception scheme might use the leaf area index produced by the vegetation module to determine the amount of rate at which precipitation reaches the ground. Terrarium allows for two distinct coupling strategies that can be chosen depending on how tightly the coupled processes are bound to one another.

## Indirect coupling

The simplest approach to coupling processes in Terrarium is to **indirectly** link the outputs of one process to the inputs another via [`input`](@ref) variables. This strategy takes advantage the variable promotion rules described on the [State variables](@ref) page. The basic workflow looks like this:

1. Process A declares a variable `X` as an `input`. Input variables are agnostic to their source and are always treated as strictly read-only.
2. Process B declares the same variable as `prognostic` or `auxiliary`.
3. When the variables of both processes are combined in a [`Variables`](@ref) container, the `input` declaration from A is automatically promoted to reference B's `prognostic`/`auxiliary` field. A single `Field` is allocated for the shared variable; both processes read from it without either one knowing about the other's type.

The main advantage of indirect coupling is that the implementations of processes A and B remain completely **independent** of each other. They can therefore maintain a strict separation of concerns allowing each process to be easily developed, tested, and reused in isolation.

### Example: Ground surface temperature

A concrete example of indirect coupling in Terrarium is the `ground_temperature` variable which represents the temperature of the uppermost subsurface layer (not to be confused with [skin temperature](@ref "Skin temperature")). The `ground_temperature` is defined as a derived auxiliary variable by [`SoilThermodynamics`](@ref), with the resulting `Field` being simply a view of the uppermost soil `temperature` layer:

```julia
variables(energy::SoilThermodynamics) = (
    prognostic(:internal_energy, Ground(XYZ()); closure = energy.closure, units = u"J/m^3", desc = "Internal energy of the soil volume, including both latent and sensible components"),
    auxiliary(:ground_temperature, Ground(Top(z = Center())), ground_temperature, energy, units = u"°C", desc = "Temperature of the uppermost ground or soil grid cell in °C"),
)

function ground_temperature(grid, clock, fields, energy::SoilThermodynamics)
    ground_grid = ground_domain(grid)
    # Use uppermost soil layer as ground temperature
    return @view fields.temperature[:, :, ground_grid.Nz]
end
```

This `ground_temperature` is then consumed by several other processes such as the [surface energy balance](@ref surface_energy_balance_docs), [ evapotranspiration](@ref "Evapotranspiration"), and the vegetation stress factors computed for [autotrophic respiration](@ref "Autotrophic respiration"). These processes can simply declare `ground_temperature` as an input variable:

```julia
input(:ground_temperature, Ground(Top(z = Center())), default = 10.0, units = u"°C")
```

When used standalone, processes declaring `ground_temperature` as an input in this manner will allocate it as an independent input `Field` (here with a uniform initial value of 10°C across space). This can be very helpful for isolated testing of such components under a range of different input values for `ground_temperature`.

## Direct coupling

The primary strength of the indirect coupling approach is its flexibility and ability to maintain a strict separation of concerns. However, many (if not most) physical processes are not so easy to cleanly separate. This is especially the case when defining multiple processes that operate within the same physical domain, e.g. the transport of energy, water, and carbon within the soil or vegetation canopy. Coupling such processes together may require entirely new implementations of functions with new sets of equations. In Terrarium, this is referred to as **direct coupling**.

More concretely, direct coupling is preferred when processes need to access each others parameters or share kernel function dispatches in their call graphs. An example of the latter is the coupling between [`SoilThermodynamics`](@ref) and [`SoilHydrology`](@ref); both processes depend on common methods like [`soil_composition`](@ref) to compute the volumetric fractions of each constituent material in the soil volume. These methods require access to the parameters and state variables for both processes, as well as the [`AbstractStratigraphy`](@ref), and thus require direct coupling.

!!! note "Why not just make everything a state variable?"
    You might wonder why we can't just make the aforementioned soil volumetric fractions themselves state variables, thereby enabling the use of indirect coupling. This is indeed possible; however, it comes with a cost. Every state variable included in the coupled system adds memory and kernel launching overhead. This will become especially problematic for high resolution simulations where each `Field` might consist of millions of grid cells. Since Terrarium is primarily oriented towards spatially distributed land simulations at both regional and global scales, we mitigate this problem by adopting the Oceananigans "lazy computation" philosophy where recomputation (in kernel functions) is preferred over excessive caching of intermediate outputs. This approach tends to favor direct over indirect coupling.

### Coupling interfaces

Terrarium is designed around the philosophy that relationships between different processes or model components should be made explicit and readily apparent wherever possible. Thus, instead of accepting monolithic objects (like `AbstractModel`) and then unpacking them internally, methods defined for specific subtypes of `AbstractProcess` must explicitly declare their (direct) dependencies by adding them as positional arguments to their top-level process interface methods, such as `initialize!`, `compute_auxiliary!`, `compute_tendencies!`.

Consider a simple conceptual example where we are implementing `compute_auxiliary!` for a specific implementation of a process `ProcessA <: AbstractProcessA`. Suppose the auxiliary variables for this process directly depend one or more parameters or state variables from another process of type `AbstractProcessB`. We then simply add process B as an argument of `compute_auxiliary!`:
```julia
function compute_auxiliary!(
        state, grid,
        procA::ProcessA,
        procB::AbstractProcessB,   # direct dependency
        args...
    )
    out = auxiliary_fields(state, procA)
    fields = get_fields(state, procA, procB; except = out)  # includes procB's Fields
    launch!(grid, XY, compute_auxiliary_kernel!, out, fields, procA, procB, args...)
    return nothing
end
```

Note the use of `AbstractProcessB` instead of a concrete type. This should generally be preferred to make the coupled implementation compatible with an arbitrary number of possible implementations of process B. However, this type can be made concrete if the coupling is genuinely specific to those concrete types. The trailing `args...` make this function forward compatible with alternative coupling interfaces that might include additional dependencies.

Inside a kernel function, `procB` can be used both for dispatch (selecting the right method specialization) and for accessing its parameters. Suppose we have a function `compute_flux!` which is called by the `compute_auxiliary_kernel!` kernel method for `ProcessA`:

```julia
@propagate_inbounds function compute_flux!(out, i, j, grid, fields, procA::ProcessA, procB::AbstractProcessB, args...)
    # Access a parameter in procB
    κ = procB.κ

    # Assume that `x` is a state variable defined by procB
    x = fields.x[i, j]

    # Optionally delegate to proc_b's own kernel functions; dispatch on concrete type
    val = some_kernel_function(i, j, grid, fields, proc_b)
    out.result[i, j, 1] = κ * x * val
    return out
end
```

The enclosing model for any given process is then responsible for supplying the coupled argument when it calls the interface methods in its own `compute_auxiliary!` / `compute_tendencies!` implementations, e.g:

```julia
function compute_auxiliary!(state, model::ABModel)
    # First compute auxiliaries for process B
    compute_auxiliary!(state, model.grid, model.procB)
    # ...then for process A
    compute_auxiliary!(state, model.grid, model.procA, model.procB)
end
```

### Conventions for direct coupling

Direct coupling introduces ordering constraints that do not exist with indirect coupling. To keep these constraints manageable and consistent, Terrarium adopts some conventions regarding the design of coupling interfaces.

#### Evaluation order

When a process accepts another process as a trailing argument to one of its top-level interface methods (`initialize!`, `compute_auxiliary!`, `compute_tendencies!`), the convention is that **the same method must have already been called on the dependency before it is called on the dependent process**. In the `ABModel` example above, `compute_auxiliary!` is called on `procB` first — before it is passed as an argument to `procA`'s `compute_auxiliary!`. This ensures that any auxiliary variables computed by `procB` are up to date when `procA` reads them.

This is also an important physical constraint for ensuring that the dynamics are well defined differential equations (see [Mathematical formulation](@ref)). Violations of this convention — i.e. cases where a process reads state from a dependency that has not yet been updated in the current time step — introduce temporal lag and must be clearly documented and justified wherever they occur.

#### Argument order

The trailing process arguments in a directly coupled interface method should follow a consistent ordering convention:

1. **Sibling processes** from the same physical domain or coupled process group (e.g. other soil sub-processes when implementing a soil process, or other vegetation sub-processes when implementing a vegetation process). In some cases, it may be reasonable to simply to depend directly an `AbstractCoupledProcesses` type if all of the respective sub-processes are very tightly coupled, as is done with soil processes.
2. **[`AbstractAtmosphere`](@ref)** for atmospheric input variables, if required.
3. **[`PhysicalConstants`](@ref)** providing access to general physical constants, if required.
4. **Foreign processes** from other modules/domains in whatever order is natural for the coupling interface.

As an example, consider the `compute_auxiliary!` method for [`PALADYNCanopyEvapotranspiration`](@ref) which follows this convention:

```julia
function compute_auxiliary!(state, grid,
        evap::PALADYNCanopyEvapotranspiration,
        canopy_interception::AbstractCanopyInterception,  # 1. sibling (surface hydrology)
        atmos::AbstractAtmosphere,                        # 2. atmospheric inputs
        constants::PhysicalConstants,                     # 3. constants
        soil::Optional{AbstractSoil} = nothing            # 4. foreign (soil)
    )
```

### Coupled process types

In many cases, it may be helpful to group sets of tightly coupled processes together in a single process type. This can be accomplished by creating subtypes of [`AbstractCoupledProcesses`](@ref) that compose multiple sub-processes into one interface:

```@docs; canonical = false
AbstractCoupledProcesses
```

It is important to note that `AbstractCoupledProcesses` is itself a subtype of `AbstractProcess` and thus subtypes must also implement the same interface described in [Core interfaces](@ref).

Coupled process types allow for the design of *hierarchical* coupling interfaces which can simplify function signatures; e.g. vegetation processes can simply accept coupled [`AbstractSoil`](@ref) types instead of specific soil processes wherever appropriate. Note, however, that such dependencies should be carefully considered since each of these coupled process types carries with it more overhead in the form of larger numbers of state variables.

These coupling types also have another key benefit: they can define their own custom kernel functions that fuse together kernels from each of their sub-processes wherever possible, thereby resulting in potentially significant efficiency gains. This pattern enables for computationally efficient process coupling while still allowing for each process to be usable standalone during testing and development.

Current examples of coupled process types include:
- [`SoilEnergyWaterCarbon`](@ref) which couples soil energy, hydrology, and biogeochemistry processes,
- [`VegetationCarbonCycle`](@ref) which couples vegetation biochemical processes,
- [`SurfaceHydrology`](@ref) which couples surface hydrological processes like canopy interception, evapotranspiration, and runoff,
- [`SurfaceEnergyBalance`](@ref) which couples the various flux terms of the surface energy balance.

## When to use direct vs. indirect coupling

Prefer **indirect coupling** (shared `input`/`prognostic` variable names) when a process only needs the *scalar value* of another process's output. The processes remain independent and can each be run and tested without the other.

Use **direct coupling** when:
- A process needs to call **kernel functions that dispatch on the other process's concrete type**
- A process needs direct access to the **parameters** (struct fields) of the other process
- Two processes share complex computations that cannot easily (or economically) be expressed via a single shared `Field`
