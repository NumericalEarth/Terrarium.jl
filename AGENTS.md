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

## Critical Rules

### Kernels (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl) to define device-agnostic kernels
- Functions marked with `@kernel` are called **kernels** which then invoke **kernel functions** with call pattern `compute_something(i, j, k, grid, fields, process::ProcessType, args...)` or `compute_something!(out, i, j, k, grid, fields, proces::ProcessType)`
    - Kernel functions defined for 2D kernels instead have `i, j` instead of `i, j, k`
- Kernels and their subsequent call graph must be fully **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` in kernels
- No error messages, no `AbstractModel`s, and no `state` inside kernels
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
- **Test differentiability**: Use Enzyme to test that critical functions compute valid adjoints; include in test suite
- **Document AD limitations**: If a function cannot be differentiated, mark it clearly with comments and docstrings

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

### Documentation pages

- Doc pages for processes and models should always consist of the following sections:
    - **Theory**: General overview of of the physical process, what the main inputs and output variables typically are, and general equations relevant for understanding the implementations.
    - **Abstract types**: Enumeration of all abstract types relevant for the process, both subtypes of `AbstractProcess` and parameterization types.
    - **Concrete types**: Enumeration of all concrete implementations of the aforementioned abstract types. For implementations `AbstractProcess`es, this should also include an enumeration of signatures for `compute_auxiliary!` and `compute_tendencies!` for each type.
    - **Methods**: Enumeration of process-specific methods.
    - **Kernel functions**: Enumeration of **kernel functions**; do NOT include **kernels** (i.e. functions annoated with `@kernel`)
- All functions referenced in doc pages should be marked with `canonical = false` since the canonical versions of the docstrings are defined in a separate `@autodocs` block in the index
- All types and functions referenced in the doc pages must have docstrings otherwise the doc build will fail. Ensure docstrings are defined and add them if they are missing.
- Doc pages should always be prefaced with appropriate `@meta` and `@setup` blocks
- If a model or process is not fully implemented, an appropriate warning should be displayed on the doc page
- Do not use brackets for expressing units as this conflicts with Markdown link syntax; use parentheses instead

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
├── timesteppers/                   # Time stepping schemes, integrators, and integration with Oceanangians `AbstractModel`
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
9. **Commented-out code**: delete it. Git is the journal — don't leave commented code, debugging
    artifacts, or stale copy-paste remnants
10. **2D indexing on fields**: always use 3D indexing (`field[i, j, k]`). 2D indexing works by
    coincidence on some fields but is unsupported and will break
11. **Hardcoded Float64**: never use `0.0`, `1.0` in kernels or constructors; use `zero(grid)` etc.
12. **Scope creep in PRs**: keep changes focused on a single concern. Unrelated cleanup goes
    in a separate PR
13. **Modifying Project.toml dependencies**: never add, remove, or change `[deps]` or `[weakdeps]`
    in the root `Project.toml` unless the task absolutely requires it. Dependency changes have
    wide-reaching consequences — they affect CI, load time, and downstream compatibility.
    Only touch `[compat]` bounds when explicitly asked.
14. **Mutable closures in kernels**: closures that capture mutable state will not differentiate correctly — use
    explicit parameters instead
15. **Non-local dependencies in process equations**: process functions must not depend on global state; pass all
    dependencies as arguments for traceability and differentiability

## Git Workflow

Follow [ColPrac](https://github.com/SciML/ColPrac). Feature branches, descriptive commits,
update tests and docs with code changes, check CI before merging.

## Design Principles

- **Dispatch over conditionals**: use Julia's type system and multiple dispatch instead of
  `if`/`else` branching. Backend-specific code goes in `ext/` extensions, not `if` branches in `src/`
- **Use `on_architecture` for data transfers** — never manual `Array()` / `CuArray()` calls
- **Defaults serve the common case**: avoid `nothing` defaults when a concrete default (like `CPU()`)
  covers 80% of usage. Minimize boilerplate for the typical user.
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

## Further Reading

For detailed guidance on specific workflows:
- `test/runtests.jl` — test organization and running conventions
