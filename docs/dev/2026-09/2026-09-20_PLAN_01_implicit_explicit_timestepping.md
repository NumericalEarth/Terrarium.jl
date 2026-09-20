# Implicit-explicit (IMEX) time stepping: Jacobian-free Newton–Krylov and Oceananigans-style tridiagonal solves

> Status: **planned**. Assessment of two implicit paths (a fully coupled Jacobian-free Newton–Krylov
> solver using Enzyme forward-mode JVPs, and an Oceananigans-style linearly implicit vertical
> diffusion solve built on `BatchedTridiagonalSolver`), the refactoring of the timestepping core that
> both need, and a phased implementation path. Recommendation: build the tridiagonal path first and
> then add the Newton–Krylov path as a second implicit solver that reuses the tridiagonal operator as
> its preconditioner. Awaiting human review; nothing has been implemented.

Date of initial draft: 2026-09-20

Base revision: 096bacc882bf5b21950abb0bc639aa457adc30ba

## Originating prompt

> Please draft a plan that outlines a path for implementing implicit-explicit timestepping in
> Terrarium. We would like to explore two paths, possibly implementing both as separate options if
> feasible:
> 1. Leverage differentiability via Enzyme to evaluate JVPs in the quasi-Newton solver fully
>    Jacobian-free. Ideally, the solve should run over all variables simultaneously, including
>    explicit terms, though operator splitting can also be considered as a viable option.
> 2. Follow a pattern closer to the way Oceananigans currently implements implicit-explicit time
>    stepping in its own model timesteppers. For this path, we will need to do a thorough
>    investigation of the existing timesteppers in Oceananigans and assess how much of the
>    functionality and patterns can be transferred to Terrarium. At bare minimum, we would want to
>    reuse certain key elements like `BatchedTridiagonalSolver` for soil operators.
>
> Give a thorough assessment of both approaches and sketch an implementation path for each, along
> with any necessary refactoring of the timestepping system. Give a recommendation of which path to
> take, or whether both should be provided as separate timestepping options to the user. You may
> find the previous plan on adaptive timestepping useful as a reference.

## Revision log

> **Revision 1, 2026-09-20.** Initial draft. Not yet reviewed.

## Problem description

Terrarium integrates every prognostic variable with fixed-step explicit schemes (`ForwardEuler`,
`Heun`). The `IMEX` type in `src/timesteppers/imex.jl` exists, but only as routing scaffolding: it
assigns each prognostic *variable* to an explicit or an implicit sub-stepper, and no implicit
sub-stepper exists. The soil operators are the stiff part of the model:

- Heat conduction has a per-cell diffusive timescale `τ = Δz² C / κ`. For a 2 cm surface layer with
  `C ≈ 2×10⁶ J m⁻³ K⁻¹` and `κ ≈ 1 W m⁻¹ K⁻¹` this is roughly 800 s, which is the same order as the
  300 s default step. Finer near-surface layers (needed to resolve the diurnal cycle in permafrost
  applications) make it quadratically worse.
- The Richardson–Richards equation has `τ = Δz² (∂θ/∂ψ) / K`, and `∂θ/∂ψ → 0` as the soil saturates,
  so the explicit step size collapses toward zero near saturation. This was documented as the
  motivating case in the adaptive-timestepping plan
  (`docs/dev/2026-08/2026-08-11_PLAN_01_adaptive_timestepping_cell_diffusion_timescale.md`), which
  made the timescale an explicit, testable diagnostic but did not remove the restriction.

The goal is an implicit-explicit stepping system in which the stiff vertical diffusion operators can
be integrated implicitly while sources, boundary fluxes, vegetation, snow, and surface hydrology stay
explicit, on CPU, CUDA, and Reactant, without giving up Enzyme differentiability.

## Background

### B1. The current Terrarium timestepping system

| File | Role |
|---|---|
| `src/timesteppers/abstract_timestepper.jl` | `AbstractTimeStepper{NF}`, `AbstractTimeStepperCache`, the `Timestepping` trait (`Explicit`/`Implicit`), `timestepping(var, model, ts)` (defaults to `Explicit()`), `initialize(ts, state, progvars, model)`, `get_cache`, and the `explicit_step!` kernels (`u += ∂u∂t Δt`) for XYZ, XY, and vertically sliced fields |
| `src/timesteppers/forward_euler.jl`, `heun.jl` | The two explicit schemes. Each `timestep!(integrator, ts, Δt, names)` calls `update_state!` itself, steps the fields in `names`, calls the model hook `timestep!(state, model, ts, Δt)`, then `closure!` |
| `src/timesteppers/imex.jl` | `AbstractIMEX`, `IMEX(explicit, implicit)`, `IMEXCache{classes}`; `timestep!` splits `prognostic_names` by resolved class (a `@generated` compile-time split) and calls each sub-stepper's `timestep!(integrator, ts, Δt, names)` in sequence, then ticks the clock once |
| `src/timesteppers/model_integrator.jl` | `ModelIntegrator <: Oceananigans.AbstractModel`; `timestep!(integrator, Δt)`, `run!`, `run_timesteps!` (host loop, overridden in `TerrariumReactantExt`), the Oceananigans `Simulation` glue |
| `src/timesteppers/cell_diffusion_timescale.jl`, `src/processes/soil/soil_diffusion_timescales.jl` | The `TimeStepWizard` diagnostics from the adaptive-timestepping plan |
| `src/solvers/` | Per-point nonlinear solvers used *inside* kernels: `ObjectiveFunction`, `FixedPointSolver`, `NewtonSolver{NF, iterations}` (fixed trip count, finite-difference derivative, Reactant-raisable), `RootSolver` (RootSolvers.jl). Used today by `ImplicitSkinTemperature` in `src/processes/surface/skin_temperature.jl` |

The step sequence of `update_state!` (`src/state_variables.jl`) is: reset tendencies, `update_inputs!`
at the clock time, `compute_auxiliary!`, `compute_boundary_conditions!` (halo fills, the surface energy
balance solve, and `compute_z_bcs!` which *adds* flux boundary conditions into the tendency fields),
then `compute_tendencies!`.

Structural facts that constrain any implicit scheme:

1. **Prognostic variables are conserved quantities with closures.** Soil energy is stepped in internal
   energy `U` with the closure `U ↦ T` (`SoilEnergyTemperatureClosure`), and water in saturation with
   the closure `saturation ↦ pressure_head` (`SoilSaturationPressureClosure`). The diffusive operators
   act on the closure variables (`∂z(κ ∂z T)`, `∂z(K ∂z ψ)`), not on the prognostics. The `closure!`
   call happens *after* the explicit step (see `forward_euler.jl`), and the tendency kernels read the
   closure fields (`fields.temperature`, `fields.pressure_head`) computed in the previous `closure!`.
2. **Operators are expressed as flux divergences on the Oceananigans ground grid** via `∂zᵃᵃᶜ`,
   `∂zᵃᵃᶠ`, `ℑzᵃᵃᶠ` (`src/processes/thermodynamics/heat_conduction.jl`,
   `src/processes/soil/hydrology/soil_hydrology_rre.jl`). Hydraulic conductivity is upwinded as the
   minimum of the neighboring cell values (`darcy_flux`), which makes the Richards operator nonsmooth
   in the state.
3. **Boundary conditions.** Flux conditions enter the tendency through `compute_z_bcs!` and are
   therefore naturally explicit right-hand-side terms. Value and gradient conditions enter through
   halo fills of the closure fields. The soil top flux is the ground heat flux produced by the
   surface energy balance, which is itself a per-column Newton solve for skin temperature at each
   `compute_boundary_conditions!`.
4. **Discrete corrections exist and are applied post-step**: `adjust_saturation_profile!` (pushes
   oversaturated water upward and into `surface_excess_water`, called inside the hydrology
   `closure!`), and the snow model's `enforce_snow_constraints!` clamp in the model hook
   `timestep!(state, model, ts, Δt)` (`src/models/snow/snow_model.jl`, `land_model.jl`).
5. **Architectures.** CPU and CUDA run kernels eagerly through `launch!`. Under Reactant only
   `timestep!(integrator, timestepper, Δt)` is traced (inside a `@trace for` loop in
   `ext/TerrariumReactantExt/integrator.jl`), so every loop inside a step must have a trip count that
   is known at trace time, kernels must have no reachable throw path, and host-side reductions
   (norms for convergence tests) force synchronization. `NewtonSolver{NF, iterations}` was written
   precisely for this.
6. **Differentiability.** Enzyme reverse mode through `timestep!` is tested on CPU
   (`test/differentiability/*.jl`, `Duplicated(integrator, make_zero(integrator))`) and under
   Reactant (`test/reactant/autodiff.jl`, with `Reactant.Periodic` checkpointing). Enzyme is a *test*
   dependency only; `src/` never references it.

Limitations of the present `IMEX` scaffolding that the new design has to fix:

- It splits by **variable**, not by **term**. Real IMEX splits terms within one variable: the
  diffusive part of the soil energy equation is stiff, its boundary flux and source terms are not.
- Each sub-stepper calls `update_state!` on its own, so a two-sub-stepper `IMEX` evaluates all
  tendencies twice per step and the second sub-stepper sees a state that the first has already
  advanced (an uncontrolled Gauss–Seidel splitting).
- `is_adaptive` is declared but unused, and the clock is ticked only after both sub-steps, so no
  sub-stepper can evaluate inputs at `tⁿ⁺¹`.
- Multi-stage explicit schemes (`Heun`) have no hook where an implicit correction could be applied
  per stage.

### B2. What Oceananigans provides (investigation summary)

Investigated against the local checkout at `e8975f17d` (2026-09-18) and cross-checked against the
released 0.111.0 that Terrarium pins (`Manifest.toml`; compat `0.110.15, 0.111`).

**Generic and directly reusable**

- `Oceananigans.Solvers.BatchedTridiagonalSolver` (`src/Solvers/batched_tridiagonal_solver.jl`).
  Constructed as `BatchedTridiagonalSolver(grid; lower_diagonal, diagonal, upper_diagonal, scratch,
  parameters, tridiagonal_direction = ZDirection())`. It allocates exactly one `Nx×Ny×Nz` scratch
  array. `solve!(ϕ, solver, rhs, args...)` launches one KernelAbstractions work item per `(i, j)`
  column and runs a serial Thomas sweep along `k` inside the kernel, in place (Oceananigans passes
  `ϕ === rhs`, which destroys the right-hand side). Coefficients are fetched through
  `get_coefficient(i, j, k, grid, coeff, p, direction, args...)`; 1D and 3D arrays work out of the
  box and **arbitrary coefficient functions are attached by defining `get_coefficient` on a singleton
  marker struct**, which is how `VerticallyImplicitDiffusionDiagonal` and friends work. A
  `abs(β) > 10 eps` guard skips the update when the system is not diagonally dominant. There are no
  CUDA-specific code paths; GPU support is entirely via `launch!`. The same pattern also underlies
  `ConjugateGradientPoissonSolver`'s `ColumnwiseTridiagonal*Diagonal` coefficients.
- `Oceananigans.Solvers.KrylovSolver` (`src/Solvers/krylov_solver.jl`, present in 0.111.0): a
  Krylov.jl wrapper (`:cg`, `:gmres`, `:bicgstab`, `:fgmres`, ...) over a single `AbstractField`,
  with matrix-free `linear_operator(y, x, args...)` and optional preconditioner callables. The
  `KrylovField` vector wrapper (kdot, knorm, kaxpy on `Field`s) is the template for a multi-field
  vector type. Krylov.jl reaches Terrarium only as a transitive dependency of Oceananigans.
- `IMEXFluxBoundaryCondition(Fₑ, λ)` (`src/BoundaryConditions/implicit_explicit_flux_boundary_condition.jl`):
  an affine flux `J = Fₑ + λ φ_boundary` whose linear coefficient `λ` is folded into the boundary cell
  diagonal by `boundary_flux_diagonal`. This is a ready-made linearized surface flux coupling.
  It is present in 0.111.0 (the pinned version) but absent in 0.110.15, which the compat entry still
  admits; the lower compat bound is raised to `0.111` in Phase 1 (older versions need no support).
- The vertically implicit coefficient construction in
  `src/TurbulenceClosures/vertically_implicit_diffusion_solver.jl` (`ivd_upper_diagonal`,
  `ivd_lower_diagonal`, `ivd_diagonal`, `implicit_linear_coefficient`): off-diagonals are
  `−Δt κ_face / (Δz_center Δz_face)` masked to zero at `peripheral_node`s, and the diagonal is
  `1 − Δt L − (upper + lower)`, which makes the operator conservative and an M-matrix and gives a
  no-flux boundary for free. Explicit vertical fluxes are zeroed in the interior for closures with
  `VerticallyImplicitTimeDiscretization` and kept only at the two boundary faces
  (`abstract_scalar_diffusivity_closure.jl`), which is how double counting is avoided. These
  functions are short and tied to the closure argument convention (`closure, K, Val(id), clock,
  fields`), so they should be **copied and adapted**, not imported.
- `implicit_step!(field::Field, solver::BatchedTridiagonalSolver, ...)` is generic over `Field`, and
  `implicit_step!(field, ::Nothing, ...) = nothing` gives a free "no implicit solver" fallback.
- The CATKE sub-cycling loop (`time_step_catke_equation.jl`): recompute frozen coefficients and a
  linear sink coefficient, explicit sub-step, tridiagonal solve, repeat. It is the closest in-repo
  template for a nonlinear column (Richards, freeze–thaw) and confirms that Oceananigans offers **no
  Newton or Jacobian infrastructure**: all nonlinearity is handled by lagging coefficients.

**Not transferable as-is**

- The time stepper structs (`QuasiAdamsBashforth2TimeStepper`, `RungeKutta3TimeStepper`,
  `SplitRungeKuttaTimeStepper`) are generic, but every hook they call (`ab2_step!`, `rk_substep!`,
  `cache_previous_tendencies!`, `cache_current_fields!`) errors by default and is implemented per
  model under `src/Models/*`, with ocean-specific details (z-star scaling, free surface, CATKE
  skips). The "explicit update then `implicit_step!` on the same field with the same stage `Δt`"
  loop is written by each model, never provided generically. Terrarium's `ForwardEuler`/`Heun`
  already play the role of these structs; adopting Oceananigans' would mean replacing the
  `AbstractTimeStepper` hierarchy for no gain. What *is* worth borrowing is the
  `SplitRungeKuttaTimeStepper` stage structure (every stage is an Euler step from the cached
  state with `Δτ = Δt/β`, and the implicit solve uses that same `Δτ`), which is the cleanest host
  for a second-order IMEX scheme.
- The `Reactant` extension replaces AB2's `time_step!` because `Δt != clock.last_Δt` becomes a
  traced boolean; `BatchedTridiagonalSolver` has **no** Reactant specialization and is traced as a
  plain serial loop. The `Enzyme` extension only marks grid helpers inactive; **nothing in the
  implicit machinery is exercised under Enzyme or Reactant in Oceananigans' own tests**, and there is
  no adjoint rule for the tridiagonal solve. Its `ϕ === rhs` in-place use is a classic Enzyme
  aliasing hazard, so Terrarium should keep a separate right-hand-side field.
- `Δt` never lives in an Oceananigans timestepper; it flows `sim.Δt → time_step!(model, Δt) →
  implicit_step!(..., Δt) → get_coefficient(..., Δt, ...)`. Terrarium's
  `timestep!(integrator, ts, Δt, names)` already has this shape.
- NumericalEarth.jl's land components are hand-rolled explicit forward Euler kernels with clamping;
  they reuse only the `Simulation`/`Clock`/`AbstractModel` interface and no implicit machinery.

### B3. The stiff operators and how they linearize

**Heat conduction.** `∂U/∂t = ∂z(κ(T) ∂z T)` with `U = U(T)` given by the energy closure. Define the
apparent heat capacity `C_app = dU/dT = C + ρL θ dF/dT`, where `F(T)` is the liquid water fraction.
Two ways to make the diffusion implicit:

- *Solve in temperature with lagged apparent capacity* (Path 2):
  `C_app,k (T_k^{n+1} − T_k^n) / Δt = [∂z(κⁿ ∂z T^{n+1})]_k + G_k^{exp}`, a tridiagonal system in `T`.
  The energy is then updated **conservatively** from the implicit flux divergence,
  `U^{n+1} = U^n + Δt ([∂z(κⁿ ∂z T^{n+1})] + G^{exp})`, and `closure!` recovers the `T` consistent with
  `U^{n+1}`. When `C_app` varies inside the step the two temperatures differ; a small fixed number of
  Picard iterations (re-evaluate `C_app`, `κ` at the new iterate, re-solve) reduces the mismatch.
  The `FreeWater` freeze curve (Terrarium's default) has a Dirac `C_app` at 0 °C, so this form
  requires either a regularized capacity `C_app ≈ ρLθ / ΔT_reg` over a small interval (as several
  land models do) or a smooth freeze curve (`SFCC` types from FreezeCurves.jl). The
  energy-conserving Newton scheme of [dallamicoEnergyConservingFreezingSoil2011](@cite) (already in
  `references.bib`) is the reference for the smooth case.
- *Solve in energy with the closure inside the residual* (Path 1): `R(U) = U − Uⁿ − Δt f(T(U))`. The
  Jacobian `∂f/∂U = ∂f/∂T · ∂T/∂U` is well defined everywhere for `FreeWater` (`∂T/∂U = 0` in the
  phase-change interval, so the Jacobian reduces to the identity there), which is exactly why the
  enthalpy formulation is the robust choice for Newton.

**Richards' equation.** Mixed form `∂θ/∂t = ∂z(K(θ) ∂z Ψ)` with `Ψ = ψ_m(θ) + ψ_z + ψ_h`. The
standard mass-conservative implicit scheme is the modified Picard iteration of Celia, Bouloutas, and
Zarba (1990): with `C = ∂θ/∂ψ` lagged at iterate `m`,
`C^m (ψ^{m+1} − ψ^m) / Δt + (θ^m − θⁿ) / Δt = ∂z(K^m ∂z ψ^{m+1}) + G^{exp}`, tridiagonal in `ψ`,
followed by the conservative update `θ^{n+1} = θⁿ + Δt (∂z(K^m ∂z ψ^{m+1}) + G^{exp})`. The gravity
and hydrostatic parts of `Ψ` are known and go to the right-hand side. The saturated limit `C → 0`
is harmless here because the diagonal keeps its `Δt K / Δz²` contributions as long as `K > 0`.
[farthingNumericalSolutionRichards2017](@cite) and [kavetskiModelSmoothingStrategies2007](@cite)
(both already cited) cover the smoothing and convergence issues of this iteration.

**Cross-couplings.** Energy and water are coupled through the freeze curve, the thermal properties,
and evapotranspiration sinks. In both paths these couplings are treated by operator splitting
between variables (evaluated at the lagged state) except in the fully coupled Newton option of
Path 1.

**Surface coupling.** The ground heat flux from the surface energy balance is a flux boundary
condition and stays explicit in the first implementation. `IMEXFluxBoundaryCondition` (with
`λ = ∂G/∂T_ground` from the skin-temperature solve) is the later route to an implicitly coupled
surface, and is standard practice in land surface models such as CLM and JSBACH.

## Approach 1: Jacobian-free Newton–Krylov (JFNK) with Enzyme JVPs

### Formulation

Backward Euler over the set `u` of prognostic fields assigned to the implicit solver (by default
*all* of them, so that explicit terms, closures, and the surface energy balance are inside the
solve):

```
R(u) = u − uⁿ − Δt f(u, tⁿ⁺¹) = 0,        f = tendencies produced by update_state!
J_k δ = −R(u_k),   u_{k+1} = u_k + δ,      J v = v − Δt (∂f/∂u) v
```

The Jacobian–vector product `(∂f/∂u) v` is computed without forming `J`:

- **Enzyme forward mode** (the requested path):
  `Enzyme.autodiff(Forward, update_state!, Duplicated(state, dstate), Const(model), Const(inputs))`
  with `dstate.prognostic := v` and the result read from `dstate.tendencies`. The shadow `dstate` is
  a second, preallocated `StateVariables` tree held in the stepper cache. Because `closure!` maps
  `U ↦ T` inside the residual, the shadow must also carry the closure fields, which `make_zero(state)`
  gives.
- **Finite-difference JVP** as a backend-independent fallback:
  `(f(u + εv) − f(u)) / ε`, one extra tendency evaluation, following the pattern already used in
  `NewtonSolver`. This is the standard JFNK approximation (Knoll and Keyes, 2004) and is what makes
  the method available on every backend before the Enzyme extension is proven.

The linear solve is a right-preconditioned restarted GMRES(m) (or BiCGSTAB); `J` is nonsymmetric
because of the upwinded hydraulic conductivity, the closure Jacobians, and the multi-variable
coupling. Two implementation options for the Krylov iteration:

1. Krylov.jl through a Terrarium `StateVector` wrapper (a `NamedTuple` of `Field`s with `kdot`,
   `knorm`, `kaxpy!` summing over the tree), modeled on Oceananigans' `KrylovField`. This needs
   Krylov.jl as a **direct** dependency (it is currently transitive), and Krylov.jl's convergence
   loops are not Reactant-raisable.
2. A hand-written fixed-iteration GMRES(m) on `Field`s with KernelAbstractions kernels for the vector
   operations and `mapreduce` for dot products. More code, but no dependency change and raisable
   under Reactant (fixed trip counts, no host branching, as in `NewtonSolver`).

The plan proposes option 1 for CPU/CUDA if the dependency is approved, otherwise option 2 for
everything; option 2 is required for Reactant in any case.

**Preconditioning is not optional.** For diffusion-dominated columns the unpreconditioned Krylov
iteration count grows like `√(Δt/τ)`, which defeats the purpose. The natural preconditioner is the
per-column tridiagonal operator of Path 2 (block Jacobi over columns, exact for the lagged 1D
diffusion), applied with `BatchedTridiagonalSolver`. With it, 2–5 Krylov iterations per Newton step
are expected. This dependency is the main reason Path 2 should be built first.

### Assessment

Strengths:

- **Fully consistent nonlinear treatment**: phase change, saturation, evapotranspiration sinks, and
  the surface energy balance are solved together at `tⁿ⁺¹`, so there is no splitting error and no
  need for per-operator implicit code. Any process that provides `compute_tendencies!` is
  automatically eligible.
- Extends naturally to higher-order stiffly accurate schemes (trapezoidal, SDIRK) once the residual
  and JVP exist.
- The enthalpy formulation is the robust choice for `FreeWater` freeze–thaw.

Risks and costs:

- **Enzyme forward mode through `update_state!` is untested.** The suite tests reverse mode on CPU
  only. Forward mode through KernelAbstractions kernels on CUDA relies on KernelAbstractions' Enzyme
  extension and CUDA.jl's, which are less exercised than the CPU path. The `RootSolver`
  (RootSolvers.jl) inside the surface energy balance has data-dependent loops, which Enzyme handles
  but Reactant does not. A spike (Phase 0) must settle this before the path is committed to.
- **Nested AD.** Differentiating a JFNK step in reverse mode for inverse modeling means reverse over
  forward. Enzyme LLVM supports this in principle, but it is a fragile combination; the correct
  long-term answer is a custom rule implementing the implicit-function-theorem adjoint (solve
  `Jᵀ λ = ∂L/∂u`), which is Phase 5 work.
- **Reactant.** Convergence-tested Newton and Krylov loops cannot be raised. Fixed-iteration variants
  (`NewtonKrylov{NF, newton_iterations, krylov_iterations}`) are needed, as `NewtonSolver` already
  demonstrates, and every host-side norm inside the compiled program becomes a synchronization
  point. Enzyme forward mode inside a Reactant-compiled program uses Enzyme-MLIR, which is
  supported but unverified for this code.
- **Cost.** Per step, roughly `N_newton × (N_krylov × c_JVP + 1)` tendency evaluations with
  `c_JVP ≈ 2–3` for an Enzyme forward pass. With a good preconditioner and 2–3 Newton iterations this
  is 10–30 explicit-step equivalents per implicit step, so the step must be at least that much
  longer than the explicit limit to pay off. For Richards near saturation it always does; for heat
  conduction on coarse grids it may not.
- **Memory.** A full shadow state, `uⁿ` copies, the residual, and the Krylov basis (`m` state-sized
  vectors, `m ≈ 10–20`).
- **Nonsmooth residuals.** `adjust_saturation_profile!` and the snow clamps are discrete corrections;
  inside a residual they stall Newton. They must stay post-step (as now), and the upwinded `K` and
  `max(0, ·)` hydrostatic terms will still degrade Newton to linear convergence in places.
- **Dependency change.** Enzyme JVPs require Enzyme (or EnzymeCore) in `[weakdeps]` and a new
  `TerrariumEnzymeExt`. Per the repository rules this needs explicit approval.

### Implementation sketch

New files under `src/timesteppers/implicit/`:

- `newton_krylov.jl`: `NewtonKrylov{NF, JVP, Lin, Pre} <: AbstractImplicitSolver` with fields
  `jvp` (`FiniteDifferenceJVP()` or `EnzymeJVP()`), `linear_solver` (`GMRES(m; iterations)` settings),
  `preconditioner` (`nothing` or `TridiagonalPreconditioner()`), `newton_iterations`, and, for the
  host-only variant, tolerances. The Reactant-compatible variant carries all iteration counts as type
  parameters.
- `newton_krylov_cache.jl`: `NewtonKrylovCache` holding `u₀` and residual trees mirroring the state
  (same recursion as `HeunCache`), the Krylov basis, and the shadow state slot (filled by the Enzyme
  extension, `nothing` otherwise).
- `residual.jl`: `evaluate_residual!(residual, cache, integrator, u, Δt)`: write `u` into the
  prognostic fields, `closure!`, `update_state!` with the clock at `tⁿ⁺¹`, then
  `R = u − u₀ − Δt · tendencies` over the selected names via a generic axpy kernel.
- `jvp.jl`: `jvp!(out, ::FiniteDifferenceJVP, ...)` in base; `jvp!(out, ::EnzymeJVP, ...)` in
  `ext/TerrariumEnzymeExt/`.
- `state_vector.jl`: tree-wide `dot`, `norm`, `axpy!`, `copy!` over the selected prognostic names,
  built from `fastiterate` and `launch!`, allocation free.
- The tridiagonal preconditioner reuses Path 2's coefficient kernel functions and
  `BatchedTridiagonalSolver` unchanged.

## Approach 2: Oceananigans-style linearly implicit vertical diffusion

### Formulation

Per implicit variable and per stage (IMEX Euler with `ForwardEuler` as the explicit partner):

1. `update_state!` once. Processes whose operator is marked implicit contribute **only their
   non-diffusive terms** to the tendency (sources, sinks, the two boundary-face fluxes via
   `compute_z_bcs!`), mirroring Oceananigans' zeroing of the interior explicit flux.
2. Explicit predictor for all variables with the existing `explicit_step!` kernels:
   `u* = uⁿ + Δt G^{exp}`.
3. For each implicit variable, `M ≥ 1` Picard iterations of: assemble lagged coefficients
   (`κ` or `K` at faces, `C_app` or `C = ∂θ/∂ψ` at centers), solve the tridiagonal system for the
   closure variable (`T` or `ψ`), update the prognostic conservatively from the implicit flux
   divergence, and re-evaluate the closure. `M = 1` is the linearly implicit scheme used by most
   land surface models; `M = 2–3` is the Celia modified Picard for Richards.
4. Model post-step hooks and `closure!` as today.

Coefficient assembly follows the Oceananigans `ivd_*` construction with the capacity on the
diagonal:

```
lower_k = −Δt κ_{k−½} / (Δz^c_k Δz^f_{k−½})          (0 at the bottom boundary)
upper_k = −Δt κ_{k+½} / (Δz^c_k Δz^f_{k+½})          (0 at the top boundary)
diag_k  = C_app,k − lower_k − upper_k
rhs_k   = C_app,k Tⁿ_k + Δt G^{exp}_k
```

The masked off-diagonals give a no-flux boundary; the actual boundary fluxes are already in
`G^{exp}` through `compute_z_bcs!`. Value/gradient boundary conditions on the closure variable are
not supported in the first version (Terrarium's soil boundary conditions are flux conditions in
practice) and would need a Dirichlet row treatment later. With `IMEXFluxBoundaryCondition` the top
flux can later be split into an explicit part and a `λ T_top` part folded into `diag_Nz`.

### Reuse from Oceananigans

| Element | How |
|---|---|
| `BatchedTridiagonalSolver` | Used as-is, constructed on `ground_domain(grid)` with three Terrarium marker structs (`ImplicitDiffusionLowerDiagonal`, `ImplicitDiffusionDiagonal`, `ImplicitDiffusionUpperDiagonal`) and `Oceananigans.Solvers.get_coefficient` methods that call Terrarium kernel functions with `(fields, process, Δt, ...)` in `args`. Separate `rhs` field, never `ϕ === rhs` |
| `ivd_upper_diagonal`, `ivd_lower_diagonal`, `ivd_diagonal` | Copied and adapted (capacity on the diagonal, Terrarium argument convention); not imported |
| `implicit_step!(field, ::Nothing) = nothing` idiom | Adopted as `implicit_solve!(…, ::Nothing, …) = nothing` for processes with no implicit operator |
| `IMEXFluxBoundaryCondition`, `boundary_flux_diagonal` | Phase 3 (available in the pinned 0.111.0) |
| `SplitRungeKuttaTimeStepper` stage pattern | Design template for the second-order IMEX (Phase 3), not imported |
| `KrylovSolver` / `KrylovField` | Template for Path 1's multi-field Krylov vector |

### Assessment

Strengths:

- **Proven and cheap**: one Thomas sweep per column per implicit variable, `O(Nz)` work, no
  reductions, no data-dependent loops. Fits the land grid (independent columns) perfectly.
- **GPU and Reactant friendly**: fixed trip counts, no throw paths, no host synchronization; the
  Reactant raise of the serial `k` loop needs verification (Phase 0) but has no structural blocker.
- **Enzyme friendly**: straight-line code per column, so reverse mode works without custom rules
  once the `ϕ === rhs` aliasing is avoided.
- **No new dependencies** for the core.
- Unconditionally stable for the linear diffusion; the step is then limited by accuracy and by the
  nonlinearity, not by `Δz²`.

Weaknesses:

- **Linearly implicit**: first order in time with lagged coefficients. Freeze–thaw and saturation
  nonlinearities are handled by Picard iterations, which converge linearly and can stall for the
  Dirac `FreeWater` capacity without regularization.
- **Per-process implicit code**: every process that wants implicit treatment must implement the
  coefficient kernel functions and the split of its tendency into explicit and implicit parts.
  Initially only soil heat conduction and Richards flow.
- **Only the vertical diffusion is implicit**; cross-couplings and the surface flux remain explicit
  (splitting error, and the surface energy balance can still limit `Δt` for thin top layers until
  the IMEX flux boundary condition is added).

## Common refactoring of the timestepping system

Both paths need the same changes to the core. These are the substantive refactoring items.

- **R1. Term-level splitting through operator classes.** Give operators a `Timestepping` trait:
  `timestepping(::ExplicitTwoPhaseHeatConduction) = Explicit()`, and new
  `ImplicitTwoPhaseHeatConduction <: AbstractHeatOperator` and `RichardsEq{ImplicitFlow}` (or a
  parallel `ImplicitRichardsEq`) with `Implicit()`. The process `compute_tendencies!` dispatches on
  the operator class so that implicit operators omit their interior diffusive flux. New kernel
  function interface for implicit operators: `compute_implicit_diffusivity(i, j, k, grid, fields,
  proc, args...)` (face), `compute_implicit_capacity(i, j, k, grid, fields, proc, args...)` (center),
  and `compute_implicit_linear_coefficient` (for linear sinks, zero by default).
- **R2. Variable routing derived from operators.** `timestepping(var, model, imex)` keeps its user
  override role, but `SoilModel`/`LandModel` define it for `:internal_energy` and
  `:saturation_water_ice` from the class of the corresponding operator, so a user only picks the
  operator type and the implicit stepper.
- **R3. Single `update_state!` per stage, orchestrated by the IMEX.** `timestep!(integrator,
  ::AbstractIMEX, Δt)` performs `update_state!`, calls the explicit sub-stepper's stage for its names
  *without* re-evaluating tendencies, then `implicit_solve!(integrator, implicit_ts, Δt, names)`.
  This requires splitting the explicit steppers' `timestep!` into `explicit_stage!` (the update
  only) and the outer driver that calls `update_state!`, hooks, and `closure!`. The mock-based tests
  in `test/timestepping/imex.jl` are updated accordingly.
- **R4. Stage hook for multi-stage explicit schemes.** Phase 1 supports only `ForwardEuler` as the
  explicit partner (IMEX Euler). Phase 3 adds a per-stage `implicit_solve!` call inside `Heun` under
  an IMEX (predictor stage with implicit correction, corrector with averaged explicit tendencies and
  a second implicit correction), following the Oceananigans "Euler from cached state with the stage
  `Δτ`" pattern.
- **R5. Caches.** `initialize(ts, state, progvars, model)` already receives the model, so implicit
  caches can allocate the tridiagonal scratch on `ground_domain(get_grid(model))`, a right-hand-side
  field and coefficient buffers per implicit variable, and, for Path 1, the shadow state and Krylov
  basis. All caches must be `Adapt`-able as `HeunCache` is.
- **R6. `Δt` plumbing.** `Δt` is passed down to the coefficient functions (via
  `BatchedTridiagonalSolver` `args`) rather than stored, matching Oceananigans and the existing
  `timestep!(integrator, ts, Δt, names)` signature.
- **R7. Clock semantics.** Path 2 evaluates explicit terms at `tⁿ` (Oceananigans convention). Path 1
  evaluates the residual at `tⁿ⁺¹`, so the IMEX driver must be able to `tick!` before the residual
  evaluation and the traced Reactant clock must tolerate that ordering. The single `tick!` in
  `timestep!(integrator, ts, ::Timestepping, Δt)` moves into the IMEX driver.
- **R8. Post-step hooks stay outside the solve.** `timestep!(state, model, ts, Δt)` (snow clamps) and
  `closure!` (including `adjust_saturation_profile!`) run after the implicit solve, never inside a
  residual.
- **R9. Wizard interplay.** `cell_diffusion_timescale` returns `Inf` for operators of class
  `Implicit()` so the `TimeStepWizard` no longer throttles `Δt` by the stiff limit; a later diagnostic
  may bound `Δt` by the Picard/Newton convergence instead. `is_adaptive` gets a definition
  (`true` for tolerance-controlled Newton variants) or is removed.
- **R10. Naming.** The Terrarium function is `implicit_solve!` to avoid confusion with
  `Oceananigans.TimeSteppers.implicit_step!`, which has a different signature.

Proposed user-facing API (both paths behind one implicit stepper with a pluggable solver):

```julia
# Path 2 (default): lagged-coefficient tridiagonal solve, M Picard iterations
ts = IMEX(ForwardEuler(NF), ImplicitEuler(NF; solver = TridiagonalPicard(iterations = 1)))

# Path 1: fully coupled Newton–Krylov, tridiagonal preconditioner, Enzyme or finite-difference JVP
ts = IMEX(ForwardEuler(NF), ImplicitEuler(NF; solver = NewtonKrylov(jvp = EnzymeJVP(), preconditioner = TridiagonalPreconditioner())))

soil = SoilEnergyWaterCarbon(NF; energy = SoilThermodynamics(NF; operator = ImplicitTwoPhaseHeatConduction()))
model = SoilModel(grid; soil, timestepper = ts)
```

`ImplicitEuler` declares `timestepping(::ImplicitEuler) = Implicit()` and slots into the existing
`IMEXCache` routing. `AbstractImplicitSolver` is the new extension point, and `TridiagonalPicard`
and `NewtonKrylov` are its two implementations.

## Recommendation

**Implement both, in sequence, as two solvers of the same `ImplicitEuler` stepper. Build Path 2 first.**

1. Path 2 is low risk, adds no dependencies, runs on all three backends with no structural obstacle,
   differentiates in reverse mode without custom rules, and removes the dominant stiffness (vertical
   diffusion in the soil column). It is the right default for users.
2. Path 2 produces exactly the preconditioner that makes Path 1 practical. Without it, Path 1 would
   spend tens of Krylov iterations per Newton step on diffusion-dominated columns.
3. Path 1's feasibility hinges on questions that cannot be answered on paper: Enzyme forward mode
   through `update_state!` on CUDA, Enzyme forward inside a Reactant program, and nested reverse over
   forward. These are settled by the Phase 0 spikes, and the Enzyme weak dependency needs a decision.
   Path 1 then becomes the advanced option for fully coupled freeze–thaw, saturated infiltration, and
   implicitly coupled surface energy balance, and the natural host for higher-order stiff schemes.

If only one path can be funded, it is Path 2.

## Phased implementation

**Phase 0: spikes (no production code).** Decide Path 1 scope and the Reactant story.

- (a) Enzyme forward-mode JVP through `update_state!` for a `SoilModel` column on CPU; compare with
  a finite-difference JVP.
- (b) The same on CUDA.
- (c) A `BatchedTridiagonalSolver` solve on a `ColumnRingGrid` compiled with `@compile raise = true`
  under Reactant; measure compile time for `Nz = 10, 50`.
- (d) Enzyme reverse mode through a `BatchedTridiagonalSolver` solve with a separate `rhs` field.

**Phase 1: core refactor and implicit heat conduction (Path 2).**

- R1–R3, R5–R10. `ImplicitEuler`, `AbstractImplicitSolver`, `TridiagonalPicard`, marker structs and
  `get_coefficient` methods, `implicit_solve!`.
- Raise the Oceananigans compat lower bound to `0.111` (drop `0.110.15`); no other dependency
  changes in this phase.
- `ImplicitTwoPhaseHeatConduction` with coefficient kernel functions; explicit tendency reduced to
  boundary faces and sources; apparent heat capacity kernel function with a documented regularization
  parameter for `FreeWater` and the analytic `C_app` for `SFCC` curves; conservative energy update.
- `SoilModel` routing; tests; docs.

**Phase 2: implicit Richards flow and the coupled land model.**

- `RichardsEq` implicit variant with Celia modified Picard (`iterations = 2` default), gravity and
  hydrostatic terms on the right-hand side, conservative saturation update, interaction with
  `adjust_saturation_profile!` (post-step) verified by mass balance.
- `LandModel` routing (snow, vegetation, surface hydrology stay explicit); GPU tests; Reactant
  registry configuration in `test/reactant/setup.jl` if Phase 0(c) passes.

**Phase 3: second order and implicit surface coupling.**

- R4: `Heun` as explicit partner with per-stage implicit corrections (IMEX trapezoidal).
- `IMEXFluxBoundaryCondition` for the ground heat flux with `λ = ∂G/∂T_ground` from the skin
  temperature solve.
- Wizard diagnostics for implicit runs.

**Phase 4: Newton–Krylov (Path 1).**

- `NewtonKrylov` with `FiniteDifferenceJVP` in base, GMRES(m) (Krylov.jl if the dependency is
  approved, otherwise the hand-written fixed-iteration version), `TridiagonalPreconditioner` reusing
  Phase 1–2 coefficients, state-tree vector operations.
- `TerrariumEnzymeExt` with `EnzymeJVP` (weak dependency, to be approved).
- CPU/CUDA tests: JVP agreement, Newton convergence rates, agreement with Path 2 in the linear limit.

**Phase 5: Reactant and AD for the implicit steppers.**

- Fixed-iteration `NewtonKrylov` variant, raise tests.
- Enzyme reverse tests for both solvers; implicit-function-theorem adjoint rule for `implicit_solve!`
  if nested AD proves fragile.

Each phase ends with the full test suite, the Enzyme test set, and a draft doc build.

## Testing and verification

- **Linear consistency**: for constant `κ`, `C` on a small column, assemble the dense
  `I − Δt L` from finite differences of the explicit operator and check that one implicit step
  matches the dense solve to roundoff (both solvers).
- **Order and stability**: the existing `examples/extending/linear_heat_conduction.jl` analytic
  solution; first-order convergence in `Δt` for IMEX Euler, second-order for the Phase 3 scheme;
  stable and accurate runs with `Δt` 100× the explicit limit.
- **Conservation**: total column energy and water change equals the integrated boundary fluxes to
  roundoff, for every solver, with and without Picard iterations, and with
  `adjust_saturation_profile!` active.
- **Freeze–thaw**: a Stefan-type freezing column compared against a fine-step explicit reference
  with `FreeWater` (regularized) and an `SFCC` curve; the Newton path must show quadratic convergence
  on the smooth curve.
- **Richards**: the Celia et al. (1990) infiltration test against a fine explicit reference,
  including a near-saturated column where explicit stepping is impractical.
- **Routing**: `test/timestepping/imex.jl` updated for the single-`update_state!` orchestration.
- **JVP**: Enzyme versus finite-difference JVP agreement on `SoilModel` and `LandModel` (Phase 4).
- **Differentiability**: `test/differentiability/` gains implicit `SoilModel` cases for both solvers;
  gradients cross-checked with FiniteDifferences as in the existing tests.
- **Reactant**: `test/reactant/correctness.jl` gains implicit configurations; compiled results match
  the CPU run.
- **GPU**: implicit configurations in the CUDA test set.
- Benchmarks are added to `benchmark/model_configurations.jl` only when explicitly requested.

## Documentation changes

- `docs/src/running/time_stepping.md`: new "Implicit-explicit time stepping" section (formulation,
  choosing operators and solvers, Picard iterations, the `FreeWater` regularization caveat, cost
  guidance, Reactant and AD notes); update the "planned" warning in the adaptive section.
- `docs/src/extending/implementing_processes.md`: the implicit operator interface (R1) with the
  kernel-function signatures.
- `docs/src/solvers/solvers.md`: `AbstractImplicitSolver`, `TridiagonalPicard`, `NewtonKrylov`.
- Process pages for soil energy and soil hydrology: implementation sections for the implicit
  operators with non-canonical docstrings and kernel-function lists.
- `docs/src/references.bib`: add Celia, Bouloutas, and Zarba (1990, Water Resources Research 26,
  1483–1496); Knoll and Keyes (2004, Journal of Computational Physics 193, 357–397); Ascher, Ruuth,
  and Spiteri (1997, Applied Numerical Mathematics 25, 151–167) for the IMEX Runge–Kutta background.
  Dall'Amico et al. (2011), Farthing and Ogden (2017), and Kavetski and Kuczera (2007) are already
  present.

## Known limitations

- Path 2 is first order with lagged coefficients; Picard iterations reduce but do not remove the
  nonlinear error within a step. The Dirac `FreeWater` capacity requires regularization in the
  temperature form.
- Only vertical diffusion in the soil is implicit in Phases 1–3. Snow, vegetation, surface
  hydrology, and the energy–water cross-couplings remain explicit, and the surface energy balance
  stays an explicit flux until Phase 3.
- Value/gradient boundary conditions on the closure variables are not supported by the tridiagonal
  operator in the first version.
- Path 1 on Reactant requires fixed iteration counts and is not adaptive; Path 1's Enzyme JVP
  requires a new weak dependency; nested AD through Path 1 is not guaranteed until the adjoint rule
  of Phase 5 exists.
- Adaptive `Δt` under Reactant remains out of scope, as in the adaptive-timestepping plan.

## Future work

- Implicit-function-theorem adjoints for `implicit_solve!` (efficient inverse modeling through
  implicit steps).
- Stiffly accurate higher-order IMEX Runge–Kutta pairs (ARS(2,2,2), ARS(4,4,3)) on top of the stage
  hook.
- Implicit treatment of snow energy and of the lateral/2D couplings if they are ever added.
- A convergence-based step controller for the implicit solvers to replace the diffusive CFL wizard.

## Decisions requested from the reviewer

1. Approve the phased sequencing (Path 2 first, Path 1 second, both exposed as solvers of
   `ImplicitEuler`), or select only one.
2. Enzyme (or EnzymeCore) as a `[weakdeps]` entry with `TerrariumEnzymeExt` for `EnzymeJVP`
   (Phase 4).
3. Krylov.jl as a direct dependency for Path 1's GMRES on CPU/CUDA, versus a hand-written
   fixed-iteration GMRES(m) for all backends.
4. The default treatment of `FreeWater` under the implicit heat operator: regularized apparent
   capacity with a documented `ΔT_reg`, or requiring an `SFCC` curve and erroring otherwise in the
   host-side constructor.
5. Type names (`ImplicitEuler`, `TridiagonalPicard`, `NewtonKrylov`, `ImplicitTwoPhaseHeatConduction`).
