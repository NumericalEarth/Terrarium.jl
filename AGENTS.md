# Terrarium.jl — Agent Rules

Note: This `AGENTS.md` file is adapted from [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl/blob/main/AGENTS.md). See relevant copyright notices therein.

## Project Overview

Terrarium.jl is a Julia package for fast, friendly, flexible, and process-based land and
terrestrial ecosystem modeling on CPUs and GPUs. It solves the coupled heat and Richards
equations in 1D for the soil along with common 0D parameterizations of vegetation and surface
hydrology processes. Terrarium is designed to be modular in the sense that (almost) all components
should be exchangeable with alternative implementations. Terrarium is also intended to be a fully
differentiable land model with continuous-time dynamics. All process implementations must be defined
in terms of well-formed ordinary or partial differential equations. No instantaneous or discrete-time
dynamics are allowed except in very special cases where they must be clearly documented and justified.

## Language & Environment

- **Julia 1.10+** | CPU and GPU (CUDA)
- **Key packages**: KernelAbstractions.jl, CUDA.jl, Enzyme.jl
- **Style**: ExplicitImports.jl for source code; `using Terrarium` for examples/tests

## Testing

- Test-only dependencies (e.g. `SpecialFunctions`, `CUDA`) live in `test/Project.toml`, not the
  root project, so the test files cannot be `include`d from the bare `--project=.` environment.
- To run individual test files, activate the test environment from the project environment with
  [TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl):
  ```julia
  # started with: julia --project=.
  using TestEnv; TestEnv.activate()
  include("test/soil/soil_energy_tests.jl")
  ```
- To run the full suite, use `julia --project=. -e 'using Pkg; Pkg.test()'` (Enzyme/AD tests run
  via `Pkg.test(; test_args=["enzyme"])`).
- Julia errors and associated stack traces are often very long due to long type signatures. To
  mitigate this, always write test output to temporary files and analyze this output using `grep`
  and similar tools.
- After running the full test suite, always run a draft doc build with ``julia --project=docs docs/make.jl --local --draft` and check for errors due to, e.g. stale docstring entires. Ignore errors from missing files due to the example scripts not being executed.

## Critical Rules

### Kernels (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl) to define device-agnostic kernels
- Functions marked with `@kernel` are called **kernels** which then invoke **kernel functions** with call pattern `compute_something(i, j, k, grid, fields, process::ProcessType, args...)` or `compute_something!(out, i, j, k, grid, fields, proces::ProcessType)`
    - Kernel functions defined for 2D kernels instead have `i, j` instead of `i, j, k`
- Kernels and their subsequent call graph must be fully **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` in kernels. `ifelse` evaluates **both**
  branches, so never index in a branch that may be out of range: select the *index*, not the
  loads (write `x[ifelse(cond, i, n)]`, not `ifelse(cond, x[i], x[n])`)
- No error messages, no `AbstractModel`s, and no `state` inside kernels. This bans **any reachable
  throw path** — `@assert`, bounds errors from un-elided `@inbounds`, `InexactError` from
  `convert`/`round(Int, …)`, integer `DivideError` — because under Reactant these lower to
  `llvm.intr.trap`, which cannot be raised to StableHLO and fails compilation *even if never
  triggered*. Put argument validation in host-side (keyword) constructors, not in the positional
  constructors that kernels call
- Always extract relevant input/output `Field`s with `get_fields` and related methods
- Favor explicit enumeration of process types when invoking kernels rather than passing `AbstractCoupledProcesses` types
- Mark functions called inside kernels with `@inline` or `@propagate_inbounds` when including indices
- **Never loop over grid points outside kernels** — use `launch!`

### Differentiability & Enzyme.jl

Terrarium targets full differentiability for inverse modeling and sensitivity analysis. Compatibility with Enzyme.jl
is a top priority and must be continuously tested.

- **Ensure type stability**: All code in kernels and within state-mutating methods (e.g. `initialize!`, `compute_tendencies!`, `compute_auxiliary!`) must be fully type stable.
- **Minimize allocations**: Allocations should be avoided wherever possible. Prefer mutation of output `Field`s. Never mutate input or prognostic `Field`s outside of `update_inputs!` or timestepper `timestep!` respectively.
- **No global state**: Initialize all parameters explicitly; never rely on global variables or implicit state
- **Test differentiability**: Use Enzyme to test that critical functions compute valid adjoints; include in test suite. See existing tests in `test/differentiability` for reference.
- **Document AD limitations**: If a function cannot be differentiated, mark it clearly with comments and docstrings

### Reactant compatibility

Terrarium runs through Reactant.jl (trace → MLIR/StableHLO → XLA). All Reactant-specific code lives
in `TerrariumReactantExt`; the user's only knob is the architecture, `ReactantState()`. A model on a
`ReactantState` grid allocates its state directly on the device grid and is initialized *eagerly*
on the device (eager KernelAbstractions launches run on the Reactant backend; Oceananigans'
`set_to_function!` for a device field detours through the CPU internally). Only
`run!`/`timestep!`/`run_timesteps!` are traced and compiled by XLA. Correctness is tested in
`test/reactant/` (own `Project.toml`, CI `Reactant_CI.yml`).

- **No reachable throw paths in kernels** — see the Kernels rule above (this is the single most
  common Reactant compile failure).
- **`using CUDA` is required alongside Reactant, even on CPU**: it provides the KernelAbstractions↔Reactant
  glue. Without it, kernel launches fail with `MethodError: ka_with_reactant(::Nothing, …)`.
- **Closures compiled into kernels must capture only `isbits` values.** A boundary-condition
  function that closes over a `Type` (e.g. `NF`/`Float32`) becomes a non-`isbits` kernel argument
  and fails to compile — hoist numeric constants out (`amplitude = NF(5); bc(x, t) = amplitude * …`).
- **Reverse-mode AD**: differentiate `run_timesteps!` with `Enzyme.autodiff(set_strong_zero(ReverseWithPrimal),
  …, Duplicated(integrator, dintegrator), …)` inside `@compile raise=true raise_first=true sync=true`.
  Pass a `checkpointing` scheme (`Reactant.Periodic(n)`) to `run!`/`run_timesteps!` to bound reverse-pass
  memory; it must not change the gradient. See `test/reactant/autodiff.jl`.
- **Do not run `test/reactant` under `--check-bounds=yes`**: forced bounds checks make every kernel
  un-raisable (see the trap rule); `runtests.jl` guards against this.

### Type Stability & Memory

- All structs must be concretely typed
- Minimize allocation; favor inline computation
- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels or constructors.
  Use `zero(grid)`, `one(grid)`, `NF(x)` where `x` is a number, `convert(FT, 1//2)`, or rational literals

### Type Annotations

- Type annotations are used to **dispatch to relevant types**, restrict method signatures, and enable compiler optimizations
- Type annotations express intent and document assumptions
- Annotate function arguments and struct fields to be as specific as possible but not more specific than necessary
- Avoid use of overly broad types like `Any` unless absolutely necessary; instead, use Union types or create a common abstract type. Use of `Any` is permitted for generic containers such as `state` and `grid`.
- Use `where` clauses to express type constraints that improve clarity and dispatch precision

### Imports

- Source code: explicit imports (checked by tests)
- Examples/docs: rely on `using Terrarium`; never explicitly import exported names

### Docstrings

- Use DocStringExtensions.jl with `$TYPEDSIGNATURES`
- **ALWAYS `jldoctest` blocks, NEVER plain `julia` blocks** — doctests are tested; plain blocks rot
- Include `# output` with verifiable output; prefer `show` methods over boolean comparisons
- Use unicode for math (`Δt`, `η`, `ρ`), not LaTeX — LaTeX doesn't render in the REPL
- Use parentheses instead of brackets for units (e.g. (m/s) instead of [m/s])
- Use DocumenterCitations.jl for references, e.g. `[GoerzQ2022](@cite)`. Make sure to include the Bibtex entry in `docs/src/references.bib`. Also add a `# References` section to the docstring, where you add e.g. `* [GoerzQ2022](@cite) Goerz et al. Quantum 6, 871 (2022)`

### Documentation pages

- Doc pages for processes and models should always consist of the following sections:
    - **Overview**: General overview of of the physical process, what the main inputs and output variables typically are, and general equations relevant for understanding the implementations. This section should not contain implementation-specific details.
    - **Implementations**: There should be sections for each implementation of the abstract process type, describing the general theory and any relevant implementation details. These sections should also each include a non-canonical docstring of the concrete process type.
        - For each concrete process implementation, provide docstrings for the implementation-specific dispatches of `initialize!`, `compute_auxiliary!`, and `compute_tendencies!` corresponding to each coupling interface.
    - **Methods**: Enumeration of process-specific methods.
    - **Kernel functions**: Enumeration of **kernel functions** of the form `compute_something(i, j, k, grid, fields, ...)` or `compute_something!(out, i, j, k, grid, fields, ...)`; do NOT include **kernels** (i.e. functions annotated with `@kernel`)
- All functions referenced in doc pages should be marked with `canonical = false` since the canonical versions of the docstrings are defined in a separate `@autodocs` block in the index
- All types and functions referenced in the doc pages must have docstrings otherwise the doc build will fail. Ensure docstrings are defined and add them if they are missing.
- Doc pages should always be prefaced with appropriate `@meta` and `@setup` blocks
- If a model or process is not fully implemented, an appropriate warning should be displayed on the doc page
- Do not use brackets for expressing units as this conflicts with Markdown link syntax; use parentheses instead
- All code examples should be given as `@example name` blocks, replacing `name` with an appropriate identifier for the page, which are executed by Documenter.jl.
- A fast version of the doc build can be executed with `julia --project=docs docs/make.jl --local --draft`

### Model Constructors

- `grid` is positional: `SoilModel(grid; initializer)`
- Omit semicolon when there are no keyword arguments: `SoilModel(grid)`

## Naming Conventions

- **Files**: snake_case matching the type they define — `soil_hydrology.jl`
- **Types/Constructors**: PascalCase **only for true constructors** — `SoilHydrology`
- **Functions**: snake_case — `compute_tendencies!` unless commonly combined in English, e.g. `timestep!`. This is not a hard rule, exceptions are permitted.
- **Kernels**: should always be suffixed with `_kernel!` — `compute_tendencies_kernel!`
- **Kernel functions**: should always be prefixed with `compute_`; mutating variants should use the standard bang `!` convention.
- **Variables**: Prefer English long name or readable unicode math notation — do not use abbreviations that may introduce ambiguity, e.g. `cond` could be either "condition" or "conductivity"; be as specific as possible.

## Physics & Process Equations

All dynamical processes must be grounded in physics:

- **Equation reference**: Every process implementation should cite the governing equations in code comments or docstrings
- **Continuous time**: Discrete-time updates (e.g., "update this once per day") are prohibited
- **Conservation where applicable**: If a process conserves a quantity (mass, energy), verify conservation in tests
- **Nondimensionalization**: Consider whether nondimensionalization would improve solver stability; document choices if used

## Module Structure

```
src/
├── Terrarium.jl                    # Main module, exports
├── abstract_model.jl               # CPU/GPU architecture abstractions
├── diagnostics/                    # Diagnostic and debugging utilities
├── grids/                          # Grid types and discretizations
├── input_output/                   # Types and functions for managing inputs and outputs
├── models/                         # Model implementations
├── processes/                      # Process implementations
├── timesteppers/                   # Time stepping schemes, integrators, and integration with Oceananigans `AbstractModel`
├── utils/                          # Miscellaneous utilities
```

## Common Pitfalls

1. **Type instability** in kernels ruins GPU performance
2. **Missing imports**: tests will catch this — add to `using` statements
3. **Plain `julia` blocks in docstrings**: always use `jldoctest`
4. **Subtle bugs from missing method imports**, especially in extensions
5. **Expecting unexported names**: consider exporting them rather than changing user scripts
6. **Extending `getproperty` to fix undefined property bugs**: fix on the caller side instead
7. **"Type is not callable" errors**: variable name shadows a function — rename or qualify
8. **Quick fixes that break correctness**: if a test fails after a change, revisit the original edit
9. **Commented-out code**: delete it. Git is the journal — don't leave commented code, debugging artifacts, or stale copy-paste remnants
10. **2D indexing on fields**: always use 3D indexing (`field[i, j, k]`). 2D indexing works by coincidence on some fields but is unsupported and will break
11. **Hardcoded Float64**: never use `0.0`, `1.0` in kernels or constructors; use `zero(grid)` etc.
12. **Scope creep in PRs**: keep changes focused on a single concern. Unrelated cleanup goes in a separate PR
13. **Modifying Project.toml dependencies**: never add, remove, or change `[deps]` or `[weakdeps]` in the root `Project.toml` unless the task absolutely requires it. Dependency changes have wide-reaching consequences — they affect CI, load time, and downstream compatibility. Only touch `[compat]` bounds when explicitly asked.
14. **Mutable function closures in kernels**: function closures that capture mutable state will not differentiate correctly — use explicit parameters instead
15. **Non-local dependencies in process equations**: process functions must not depend on global state; pass all dependencies as arguments for traceability and differentiability

## Benchmarks

Performance benchmarks live in `benchmark/` and are run manually, one architecture at a time:

```bash
cd benchmark
julia --project=. manual_benchmarking.jl                # CPU (auto-labelled cpu-arm or cpu-x86)
julia --project=. manual_benchmarking.jl gpu            # CUDA GPU
julia --project=. manual_benchmarking.jl reactant-cpu   # Reactant/XLA, CPU backend
julia --project=. manual_benchmarking.jl reactant-gpu   # Reactant/XLA, CUDA backend
```

A second argument sets the duration: `quick` (0.25x time steps, resolution sweeps capped — use this to
check that the harness and configurations still run), `long` (10x, for numbers worth publishing), or a
numeric multiplier.

Each run merges its results into `benchmark/assets/benchmark_results.json` and regenerates
`benchmark/README.md` from the whole store, so running on one machine never overwrites another
architecture's numbers. The documentation page `docs/src/benchmarks.md` is generated from the same JSON
at doc-build time.

The committed README numbers are the regression baseline: deviations of ±20% are normal, larger ones
should be reported. Model configurations are defined in `benchmark/model_configurations.jl`; add one by
adding a `build_model(::Val{:name}, arch, NF; nlat_half, nz, model_kwargs)` method, following the same
registry pattern as `test/reactant/setup.jl`. 

Only ever run these benchmarks when explicilty instructed to or after a major revisions of the model compute functions. 

## Git Workflow

Follow [ColPrac](https://github.com/SciML/ColPrac). Feature branches, descriptive commits, update tests and docs with code changes, check CI before merging.

## Design Principles

- **Dispatch over conditionals**: use Julia's type system and multiple dispatch instead of `if`/`else` branching. Backend-specific code goes in `ext/` extensions, not `if` branches in `src/`
- **Use `on_architecture` for data transfers** — never manual `Array()` / `CuArray()` calls
- **Defaults serve the common case**: avoid `nothing` defaults when a concrete default (like `CPU()`) covers 80% of usage. Minimize boilerplate for the typical user.
- **Keyword argument names must be consistent** across related types and constructors
- **Always use explicit `return`** in functions longer than one expression
- **One operation per line** as default; break long expressions across lines

## Agent Behavior

- Prioritize type stability, GPU compatibility, and differentiability
- Follow established patterns in existing code
- Add tests for new functionality; update exports when adding public API
- Reference physics equations in comments when implementing dynamics
- Do not make unsolicited changes; focus on specific tasks
- When extending Enzyme.jl compatibility, verify adjoints with `Enzyme.autodiff`
- Ensure type annotations are restrictive enough to guide dispatch and minimize misuse

## Implementation plans

All major feature additions, bug fixes, or refactoring that requires substantial changes to the existing code must be prefaced with an **implementation plan** that is **reviewed and signed off by a human**. These plan documents should be organized by date and stored in `docs/dev/YYYY-MM`. You are **ABSOLUTELY REQUIRED** to get the explicit approval of a human developer before executing the implementation.

Each document should be prefaced by the following template:

```md
# Descriptive title

> Status: **planned**/**in progress**/**completed**. One sentence summary of current status.

Date of initial draft: YYYY-MM-dd

Base revision: <SHA1 of HEAD when plan was drafted>

## Originating prompt

> User prompts here

## Revision log

> User prompts here

N.B: Make sure that each revision is given a number and a date.

## Problem description

## Background

```

The revision log should, to the greatest extent possible, briefly summarize changes to the plan that are made on-the-fly during development. Make sure that each revision is given a number and a date. **A human must approve each revision before implementation**.

The remainder of the plan document may be adapted on a case-by-case basis but should generally follow this structure:

```md
## Summary of changes

## Testing and verification

## Documentation changes

## Known limitations

## Future work
```

## Further Reading

For detailed guidance on specific workflows:
- `test/runtests.jl` — test organization and running conventions
