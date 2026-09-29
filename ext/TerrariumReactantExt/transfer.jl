# Device initialization.
#
# A model whose grid lives on `ReactantState` is initialized on the device: the state `Field`s are
# allocated directly on the device grid and `initialize!` populates them in place. This works
# because the eager KernelAbstractions launches invoked by `initialize!` (halo fills,
# closure/auxiliary kernels) run on the Reactant backend, and Oceananigans' `set_to_function!` for
# a device field internally detours through the CPU. Only `run!`/`timestep!` are traced and
# compiled (see integrator.jl).
#
# The sole device-specific ingredient is the clock: a traced `ConcreteRNumber`-backed
# `Clock(field_grid)` so that time advances *inside* the compiled step. We supply it by overriding
# `default_clock` for device models; the generic `Terrarium.initialize` then needs no
# specialization.

Terrarium.default_clock(model::ReactantModel) =
    Oceananigans.TimeSteppers.Clock(ground_domain(get_grid(model)))
