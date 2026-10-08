# Implicit-explicit (IMEX) time stepping: Jacobian-free Newton–Krylov and Oceananigans-style tridiagonal solves

> Status: **in progress** (approved at revision 4; Phase 1a core refactor started 2026-10-08). Assessment of two implicit paths (a fully coupled Jacobian-free Newton–Krylov
> solver using Enzyme forward-mode JVPs, and an Oceananigans-style linearly implicit vertical
> diffusion solve built on `BatchedTridiagonalSolver`), the refactoring of the timestepping core that
> both need, and a phased implementation path. **Scope of this plan and PR: the tridiagonal path
> (Path 2) and the core refactoring**, followed by feasibility spikes for the Newton–Krylov path.
> Path 1 itself is assessed here but deferred in its entirety to a future plan and PR, where it will
> reuse the Path 2 tridiagonal operator as its preconditioner. Revision 2 (2026-10-08) re-checked the
> plan against the repository; revision 3 (2026-10-08) narrowed the scope as described. Awaiting
> human review; nothing has been implemented.

Date of initial draft: 2026-09-20

Base revision: 096bacc882bf5b21950abb0bc639aa457adc30ba

Revision 2 reviewed against: e9b9ef82a706492972a70f7a9892a7cd13afdd4e (branch `bg/implicit-timestepping`,
Terrarium 0.1.9-DEV, Oceananigans 0.113.5)

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
>
> **Revision 2, 2026-10-08.** Re-reviewed against the current repository (see "B0. Repository state
> at revision 2"). Changes: the explicit heat operator was renamed to `TwoPhaseHeatTransport`
> (commit `e9b9ef82a`); per reviewer feedback the operator types are left untouched and R1 instead
> passes the resolved `Timestepping` trait into the soil energy and hydrology `compute_tendencies!`,
> and the implicit entry point is named `implicit_step!`; Oceananigans is pinned at 0.113.5, so the compat item and the
> `IMEXFluxBoundaryCondition` availability caveat are dropped; state fields may now live on plain
> `AbstractGrid`s and may carry non-scalar element types, which adds two constraints to the implicit
> caches (R5); the soil kernel fusion that R1 would have interacted with was abandoned; the ground
> heat flux is now its own SEB sub-process, which is where the Phase 3 linearized surface flux hooks
> in; the Reactant extension now converts the traced clock time, which R7 relies on. The
> recommendation and phasing are unchanged. Not yet reviewed.
>
> **Revision 3, 2026-10-08.** Reviewer feedback: the implicit entry point is `implicit_step!`; the
> operator types are left unchanged and the resolved `Timestepping` trait is passed into the soil
> energy and hydrology `compute_tendencies!` instead; the finer `Implicit{…}` trait distinction is
> deferred as a Path 1 consideration; the feasibility spikes (formerly Phase 0) move after the Path 2
> implementation; and Path 1 is deferred in its entirety to a future plan and PR. Its assessment
> and sketch stay in this document as the record for that future work. Decisions on the Enzyme weak
> dependency and Krylov.jl are deferred with it.
>
> **Revision 4, 2026-10-08. Approved for implementation.** Reviewer decisions: scope (Path 2 and the
> core refactoring in this PR, spikes in Phase 4, Path 1 deferred) approved; type names
> `ImplicitEuler`, `AbstractImplicitSolver`, `TridiagonalPicard` approved; the implicit heat operator
> uses a **regularized apparent heat capacity** for `FreeWater` (with a documented `ΔT_reg`
> parameter) rather than requiring a smooth freeze curve.

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

### B0. Repository state at revision 2 (2026-10-08)

Changes since the base revision that bear on this plan:

- **Operator naming.** `ExplicitTwoPhaseHeatConduction` was renamed to `TwoPhaseHeatTransport`
  (`src/processes/thermodynamics/heat_conduction.jl`). The operator types are not modified by this
  plan; the explicit/implicit split is carried by the `Timestepping` trait passed into
  `compute_tendencies!` (R1).
- **Oceananigans 0.113.5** is pinned (`Project.toml` compat `0.113.5`, both Manifests). The
  `BatchedTridiagonalSolver` constructor, `solve!`, `get_coefficient`, `KrylovSolver`,
  `implicit_step!`, and `IMEXFluxBoundaryCondition` are all present with the signatures described in
  B2 (verified in the 0.113.1 through 0.113.6 depot copies). No compat work remains.
- **Grids.** `ModelIntegrator` and `StateVariables` no longer require an `AbstractLandGrid`; fields
  are allocated on whatever grid is given, and `variable_grid`/`ground_domain` are identity on a plain
  `AbstractGrid`. The tridiagonal scratch and coefficient buffers must therefore be allocated on
  `ground_domain(get_grid(model))`, which covers both cases.
- **Variable element types.** `Variable` gained an `eltype` (e.g. `SVector{3, NF}`), the explicit
  step kernels convert `Δt` with `eltype(eltype(tendency))`, and `test/timestepping/vector_eltype.jl`
  covers it. The implicit solvers are scalar only in this plan; the host-side constructor must reject
  routing a non-scalar variable to the implicit sub-stepper.
- **`Top()` semantics.** A `Top()` variable now defaults to `z = Face()` and resolves to `k = Nz + 1`;
  `Top(z = Center())` gives the uppermost cell. The XY explicit kernel indexes `k = 1`. Any implicit
  right-hand-side assembly that touches surface pools must follow these conventions.
- **Kernel fusion.** The fused-kernels plan (rev 7) abandoned soil tendency fusion after
  benchmarking and fuses auxiliaries only, so the implicit split of the soil tendency in R1 touches
  only the per-process `compute_tendencies!` launches, not a fused kernel.
- **Evapotranspiration sink.** `LandModel` now passes `surface_hydrology` into the soil
  `compute_tendencies!`, so the evapotranspiration sink is an ordinary explicit source term in the
  Richards tendency. It stays on the right-hand side in Path 2 and inside the residual in Path 1.
- **Surface energy balance.** The ground heat flux is now a first-class SEB sub-process with
  `Diagnosed` and `Prescribed` implementations (plan `2026-09-01_PLAN_prescribed_surface_energy_balance.md`,
  completed). The Phase 3 linearized surface flux (`λ = ∂G/∂T_ground`) attaches to the `Diagnosed`
  ground heat flux implementation.
- **Reactant.** `TerrariumReactantExt` now defines `convert_dt` for a `TracedRNumber` clock time, and
  the Reactant test registry gained time-varying input configurations. R7 (ticking the clock before a
  residual evaluation) is therefore supported inside the compiled loop.
- **Heun.** `HeunCache` allocates with `similar` rather than `deepcopy`; implicit caches should do the
  same.
- The `IMEX` routing tests now use plain `XY()` declarations on an `AbstractGrid`; the planned
  rewrite of those tests under R3 applies to the current versions.

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
released 0.111.0 pinned at drafting time and, at revision 2, against the now-pinned 0.113.5.

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
- `Oceananigans.Solvers.KrylovSolver` (`src/Solvers/krylov_solver.jl`, present in 0.111.0 and 0.113.5): a
  Krylov.jl wrapper (`:cg`, `:gmres`, `:bicgstab`, `:fgmres`, ...) over a single `AbstractField`,
  with matrix-free `linear_operator(y, x, args...)` and optional preconditioner callables. The
  `KrylovField` vector wrapper (kdot, knorm, kaxpy on `Field`s) is the template for a multi-field
  vector type. Krylov.jl reaches Terrarium only as a transitive dependency of Oceananigans.
- `IMEXFluxBoundaryCondition(Fₑ, λ)` (`src/BoundaryConditions/implicit_explicit_flux_boundary_condition.jl`):
  an affine flux `J = Fₑ + λ φ_boundary` whose linear coefficient `λ` is folded into the boundary cell
  diagonal by `boundary_flux_diagonal`. This is a ready-made linearized surface flux coupling.
  It is present in the pinned 0.113.5.
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
  but Reactant does not. A Phase 4 spike must settle this before the path is committed to.
- **Nested AD.** Differentiating a JFNK step in reverse mode for inverse modeling means reverse over
  forward. Enzyme LLVM supports this in principle, but it is a fragile combination; the correct
  long-term answer is a custom rule implementing the implicit-function-theorem adjoint (solve
  `Jᵀ λ = ∂L/∂u`), which belongs to the future Path 1 plan.
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
- **Trait distinction (deferred from R1).** The Newton residual needs the full tendency including
  the interior diffusion term, whereas the Path 2 tendency omits it under `Implicit()`. The future
  plan must decide how `compute_tendencies!` learns which it is: a parameterized trait such as
  `Implicit{LinearlyImplicit}()` versus `Implicit{FullyImplicit}()` (preferred, since process methods
  then dispatch on one argument), or a query `omits_interior_diffusion(::AbstractImplicitSolver)`
  consulted when the model-level `compute_tendencies!` resolves the trait.

This sketch, the assessment above, and the spikes of Phase 4 are the inputs to the future Path 1
plan. Nothing in this section is implemented in the present PR.

## Approach 2: Oceananigans-style linearly implicit vertical diffusion

### Formulation

Per implicit variable and per stage (IMEX Euler with `ForwardEuler` as the explicit partner):

1. `update_state!` once. Processes whose operator is marked implicit contribute **only their
   non-diffusive terms** to the tendency (sources, sinks, the two boundary-face fluxes via
   `compute_z_bcs!`), mirroring Oceananigans' zeroing of the interior explicit flux.
2. Explicit update for all variables with the existing `explicit_step!` kernels:
   `u* = uⁿ + Δt G^{exp}`. For implicitly stepped variables this is not a predictor in the
   predictor-corrector sense; `u*` is simply the right-hand side of the backward Euler system that
   step 3 solves. The scheme is a straight implicit solve with one tendency evaluation per step; no
   corrector pass follows. (The two-stage `Heun` pairing of Phase 3 is where a genuine
   predictor-corrector structure appears.)
3. For each implicit variable, `M ≥ 1` Picard iterations of: assemble lagged coefficients
   (`κ` or `K` at faces, `C_app` or `C = ∂θ/∂ψ` at centers), solve the tridiagonal system for the
   closure variable (`T` or `ψ`), update the prognostic conservatively from the implicit flux
   divergence, and re-evaluate the closure. `M = 1` is the linearly implicit scheme used by most
   land surface models; `M = 2–3` is the Celia modified Picard for Richards. The Picard loop is a
   fixed-point iteration on the nonlinear coefficients within one backward Euler step, not a
   multi-stage time scheme.
4. Model post-step hooks and `closure!` as today.

Coefficient assembly follows the Oceananigans `ivd_*` construction with the capacity on the
diagonal:

```
lower_k = −Δt κ_{k−½} / (Δz^c_k Δz^f_{k−½})          (0 at the bottom boundary)
upper_k = −Δt κ_{k+½} / (Δz^c_k Δz^f_{k+½})          (0 at the top boundary)
diag_k  = C_app,k − lower_k − upper_k
rhs_k   = C_app,k Tⁿ_k + Δt G^{exp}_k
```

**Flux boundary conditions are the primary case and are required from Phase 1.** In Terrarium they
are attached to the conserved prognostic (`SoilHeatFlux` and `GeothermalHeatFlux` on
`internal_energy`, `InfiltrationFlux` on `saturation_water_ice`) and `compute_z_bcs!` adds them into
the tendency during `update_state!`. The implicit step takes that tendency as `G^{exp}`, so the
ground heat flux enters the top row through the right-hand side while the masked off-diagonals keep
the matrix itself flux-free, exactly as in Oceananigans. The R1 split must therefore omit only the
*interior* diffusive flux under `Implicit()` and leave the `compute_z_bcs!` contributions in place;
the conservation tests check this by requiring the column energy and water change to equal the
integrated boundary fluxes. The boundary flux is evaluated at `tⁿ` (from the surface energy balance
of that step) and held fixed over the step; Phase 3 adds the `λ T_top` linearization on top of this
flux form for cases where the explicit surface coupling limits `Δt`. Value and gradient conditions are instead attached to the
closure variables (`temperature`, `pressure_head`; e.g. `PrescribedSurfaceTemperature` and
`PrescribedBottomTemperature` in `src/models/soil/soil_model_bcs.jl`) and reach the explicit operator
through halo fills, which the implicit operator does not read. They are handled as follows:

- A **gradient** condition on `T` at a boundary face is a known flux `−κ ∂T/∂z`, so it is added to
  the right-hand side exactly like a flux condition (Phase 1).
- A **value** condition `T_bc` at the top gives the boundary flux `κ (T_bc − T_Nz) / (Δz_Nz / 2)`,
  which is affine in the unknown `T_Nz`: the coefficient `κ / (Δz_Nz / 2)` is folded into `diag_Nz`
  and the `T_bc` part into the right-hand side (and symmetrically at the bottom). This is the same
  structure as `IMEXFluxBoundaryCondition`, so it shares the boundary-diagonal code introduced in
  Phase 3 for the linearized surface flux `λ T_top`. Until Phase 3, a value condition on a closure
  variable routed to the implicit stepper is rejected in the host-side constructor.

### Reuse from Oceananigans

| Element | How |
|---|---|
| `BatchedTridiagonalSolver` | Used as-is, constructed on `ground_domain(grid)` with three Terrarium marker structs (`ImplicitDiffusionLowerDiagonal`, `ImplicitDiffusionDiagonal`, `ImplicitDiffusionUpperDiagonal`) and `Oceananigans.Solvers.get_coefficient` methods that call Terrarium kernel functions with `(fields, process, Δt, ...)` in `args`. Separate `rhs` field, never `ϕ === rhs` |
| `ivd_upper_diagonal`, `ivd_lower_diagonal`, `ivd_diagonal` | Copied and adapted (capacity on the diagonal, Terrarium argument convention); not imported |
| `implicit_step!(field, ::Nothing) = nothing` idiom | Adopted as `implicit_step!(…, ::Nothing, …) = nothing` for processes with no implicit operator |
| `IMEXFluxBoundaryCondition`, `boundary_flux_diagonal` | Phase 3 (available in the pinned 0.113.5) |
| `SplitRungeKuttaTimeStepper` stage pattern | Design template for the second-order IMEX (Phase 3), not imported |
| `KrylovSolver` / `KrylovField` | Template for Path 1's multi-field Krylov vector |

### Assessment

Strengths:

- **Proven and cheap**: one Thomas sweep per column per implicit variable, `O(Nz)` work, no
  reductions, no data-dependent loops. Fits the land grid (independent columns) perfectly.
- **GPU and Reactant friendly**: fixed trip counts, no throw paths, no host synchronization; the
  Reactant raise of the serial `k` loop is verified in Phase 2 and has no structural blocker.
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

- **R1. Term-level splitting by passing the `Timestepping` trait into `compute_tendencies!`.**
  The operator types are left unchanged. Instead, the resolved class of each prognostic variable,
  which the `IMEXCache{classes}` type parameter already holds, is handed to the process tendency
  methods: a type-stable helper `timestepping(state, var)` (or `timestepping(state, ::Val{name})`)
  reads it from `state.timestepper_cache` and returns `Explicit()` when the cache is not an
  `IMEXCache`. The model-level `compute_tendencies!(state, model::SoilModel)` and
  `compute_tendencies!(state, model::LandModel)` resolve the trait for `:internal_energy` and
  `:saturation_water_ice` and pass it down through `SoilEnergyWaterCarbon` to the soil energy and
  hydrology `compute_tendencies!`, which dispatch on it: under `Implicit()` the interior diffusive
  flux is omitted from the tendency (mirroring Oceananigans' vertically implicit closures) and only
  the boundary-face fluxes and sources remain. Process methods default the trait argument to
  `Explicit()`, so standalone use and the existing tests are unaffected. Only the trait is passed,
  never the timestepper itself, so kernels stay free of `AbstractTimeStepper` objects (the trait is
  `isbits`). New kernel function interface for implicit operators:
  `compute_implicit_diffusivity(i, j, k, grid, fields, proc, args...)` (face),
  `compute_implicit_capacity(i, j, k, grid, fields, proc, args...)` (center), and
  `compute_implicit_linear_coefficient` (for linear sinks, zero by default).
  Two clarifications on what "omit the interior diffusive flux" means:
    - In this plan `Implicit()` always means the linearly implicit tridiagonal treatment, so the
      interior diffusion term is omitted from the explicit tendency whenever the trait is
      `Implicit()`. A fully implicit Newton solver would instead need the full right-hand side; how
      the trait expresses that distinction is a Path 1 consideration (see Approach 1, Implementation
      sketch) and is not part of this PR.
    - Path 2 still evaluates the diffusive flux divergence once per Picard iteration, *after* the
      tridiagonal solve and at the implicit closure variable (`T^{n+1}` or `ψ^{m+1}`), to update the
      conserved prognostic. The existing flux-divergence kernel functions (`compute_energy_tendency`,
      `compute_volumetric_water_content_tendency`) are refactored so that `compute_tendencies!` and
      `implicit_step!` share them; the operator code is reused, it is only skipped inside
      `compute_tendencies!`.
- **R2. Routing has a single source of truth.** `timestepping(var, model, imex)` remains the only
  place a variable is assigned to the implicit sub-stepper. `SoilModel`/`LandModel` define it as
  `Implicit()` for `:internal_energy` and `:saturation_water_ice` whenever the model's timestepper
  is an `AbstractIMEX` with an implicit sub-stepper, and users override it per variable if they want,
  e.g., implicit heat but explicit water. The user therefore selects only the timestepper; no
  operator or process type changes hands.
- **R3. Single `update_state!` per stage, orchestrated by the IMEX.** `timestep!(integrator,
  ::AbstractIMEX, Δt)` performs `update_state!`, calls the explicit sub-stepper's stage for its names
  *without* re-evaluating tendencies, then `implicit_step!(integrator, implicit_ts, Δt, names)`.
  This requires splitting the explicit steppers' `timestep!` into `explicit_stage!` (the update
  only) and the outer driver that calls `update_state!`, hooks, and `closure!`. The mock-based tests
  in `test/timestepping/imex.jl` are updated accordingly.
- **R4. Stage hook for multi-stage explicit schemes.** Phase 1 supports only `ForwardEuler` as the
  explicit partner (IMEX Euler). Phase 3 adds a per-stage `implicit_step!` call inside `Heun` under
  an IMEX (predictor stage with implicit correction, corrector with averaged explicit tendencies and
  a second implicit correction), following the Oceananigans "Euler from cached state with the stage
  `Δτ`" pattern.
- **R5. Caches.** `initialize(ts, state, progvars, model)` already receives the model, so implicit
  caches can allocate the tridiagonal scratch on `ground_domain(get_grid(model))`, a right-hand-side
  field and coefficient buffers per implicit variable, and, for Path 1, the shadow state and Krylov
  basis. All caches must be `Adapt`-able as `HeunCache` is, allocate with `similar` as `HeunCache`
  now does, work on a plain `AbstractGrid` as well as a `LandGrid`, and the host-side constructor
  must reject implicit routing of variables with non-scalar element types (B0).
- **R6. `Δt` plumbing.** `Δt` is passed down to the coefficient functions (via
  `BatchedTridiagonalSolver` `args`) rather than stored, matching Oceananigans and the existing
  `timestep!(integrator, ts, Δt, names)` signature.
- **R7. Clock semantics.** Path 2 evaluates explicit terms at `tⁿ` (Oceananigans convention). Path 1
  evaluates the residual at `tⁿ⁺¹`, so the IMEX driver must be able to `tick!` before the residual
  evaluation and the traced Reactant clock must tolerate that ordering; the extension's `convert_dt`
  for traced times (B0) makes input updates at a traced `tⁿ⁺¹` possible. The single `tick!` in
  `timestep!(integrator, ts, ::Timestepping, Δt)` moves into the IMEX driver.
- **R8. Post-step hooks stay outside the solve.** `timestep!(state, model, ts, Δt)` (snow clamps) and
  `closure!` (including `adjust_saturation_profile!`) run after the implicit solve, never inside a
  residual.
- **R9. Wizard interplay.** `cell_diffusion_timescale` returns `Inf` for operators of class
  `Implicit()` so the `TimeStepWizard` no longer throttles `Δt` by the stiff limit; a later diagnostic
  may bound `Δt` by the Picard/Newton convergence instead. `is_adaptive` gets a definition
  (`true` for tolerance-controlled Newton variants) or is removed.
- **R10. Naming.** The Terrarium function is `implicit_step!`, defined as Terrarium's own function
  (Terrarium imports `Oceananigans.TimeSteppers` selectively and does not import its
  `implicit_step!`, so there is no method clash; the Oceananigans function keeps its own signature).

Proposed user-facing API (both paths behind one implicit stepper with a pluggable solver):

```julia
# Path 2 (default): lagged-coefficient tridiagonal solve, M Picard iterations
ts = IMEX(ForwardEuler(NF), ImplicitEuler(NF; solver = TridiagonalPicard(iterations = 1)))

# Path 1: fully coupled Newton–Krylov, tridiagonal preconditioner, Enzyme or finite-difference JVP
ts = IMEX(ForwardEuler(NF), ImplicitEuler(NF; solver = NewtonKrylov(jvp = EnzymeJVP(), preconditioner = TridiagonalPreconditioner())))

# Operators and processes are unchanged; the IMEX routes soil energy and water to the implicit stepper (R2)
model = SoilModel(grid; timestepper = ts)

# Optional per-variable override: keep soil water explicit
Terrarium.timestepping(::AbstractVariable{:saturation_water_ice}, ::SoilModel, ::AbstractIMEX) = Explicit()
```

`ImplicitEuler` declares `timestepping(::ImplicitEuler) = Implicit()` and slots into the existing
`IMEXCache` routing. `AbstractImplicitSolver` is the new extension point, and `TridiagonalPicard`
and `NewtonKrylov` are its two implementations.

## Recommendation

**Implement Path 2 in this PR. Defer Path 1 to a future plan and PR, with both eventually exposed as
solvers of the same `ImplicitEuler` stepper.**

1. Path 2 is low risk, adds no dependencies, runs on all three backends with no structural obstacle,
   differentiates in reverse mode without custom rules, and removes the dominant stiffness (vertical
   diffusion in the soil column). It is the right default for users.
2. Path 2 produces exactly the preconditioner that makes Path 1 practical. Without it, Path 1 would
   spend tens of Krylov iterations per Newton step on diffusion-dominated columns, so Path 1 cannot
   sensibly be built first.
3. Path 1's feasibility hinges on questions that cannot be answered on paper: Enzyme forward mode
   through `update_state!` on CUDA, Enzyme forward inside a Reactant program, and nested reverse over
   forward. The spikes of Phase 4 settle them once Path 2 exists to compare against, and their results
   feed the future Path 1 plan together with the Enzyme weak-dependency and Krylov.jl decisions.
   Path 1 then becomes the advanced option for fully coupled freeze–thaw, saturated infiltration, and
   implicitly coupled surface energy balance, and the natural host for higher-order stiff schemes.

## Phased implementation

Phases 1 through 4 are the scope of this PR. Path 1 is deferred to a future plan.

**Phase 1: core refactor and implicit heat conduction (Path 2).**

- R1–R3, R5–R10. `ImplicitEuler`, `AbstractImplicitSolver`, `TridiagonalPicard`, marker structs and
  `get_coefficient` methods, `implicit_step!`, the `timestepping(state, var)` helper.
- Implicit heat conduction: `compute_tendencies!` for `SoilThermodynamics` dispatching on the trait,
  coefficient kernel functions for `TwoPhaseHeatTransport`; explicit tendency reduced to
  boundary faces and sources; apparent heat capacity kernel function with a documented regularization
  parameter for `FreeWater` and the analytic `C_app` for `SFCC` curves; conservative energy update.
- Flux boundary conditions (`SoilHeatFlux`, `GeothermalHeatFlux`) verified to enter the implicit
  step through the right-hand side, with conservation tests against the integrated boundary fluxes.
- `SoilModel` routing; tests; docs.

**Phase 2: implicit Richards flow and the coupled land model.**

- `RichardsEq` implicit variant with Celia modified Picard (`iterations = 2` default), gravity and
  hydrostatic terms on the right-hand side, conservative saturation update, interaction with
  `adjust_saturation_profile!` (post-step) verified by mass balance.
- `LandModel` routing (snow, vegetation, surface hydrology stay explicit); GPU tests; Reactant
  registry configuration in `test/reactant/setup.jl` (the tridiagonal solve has fixed trip counts
  and no throw paths, so no structural obstacle is expected; compile time is measured here).

**Phase 3: second order and implicit surface coupling.**

- R4: `Heun` as explicit partner with per-stage implicit corrections (IMEX trapezoidal).
- `IMEXFluxBoundaryCondition` for the ground heat flux with `λ = ∂G/∂T_ground` from the skin
  temperature solve, attached to the `Diagnosed` ground heat flux sub-process of the SEB (B0).
- Value boundary conditions on the closure variables under the implicit operator, sharing the
  boundary-diagonal code with the IMEX flux condition.
- Wizard diagnostics for implicit runs.

**Phase 4: feasibility spikes for Path 1 (no production code).** Run after Path 2 is complete so the
results can be compared against a working implicit stepper; recorded in the future Path 1 plan.

- (a) Enzyme forward-mode JVP through `update_state!` for a `SoilModel` column on CPU; compare with
  a finite-difference JVP.
- (b) The same on CUDA.
- (c) Enzyme forward mode inside a Reactant-compiled program for the same JVP.
- (d) Enzyme reverse mode through the Phase 1 `implicit_step!` (tridiagonal solve with a separate
  `rhs` field), as a baseline for the nested-AD question.

Each phase ends with the full test suite, the Enzyme test set, and a draft doc build.

**Deferred to a future plan and PR (Path 1).** For the record, the work previously planned here:

- `NewtonKrylov` with `FiniteDifferenceJVP` in base, GMRES(m) (Krylov.jl if the dependency is
  approved, otherwise a hand-written fixed-iteration version), `TridiagonalPreconditioner` reusing
  the Phase 1–2 coefficients, state-tree vector operations, and the trait distinction described in
  the Approach 1 implementation sketch.
- `TerrariumEnzymeExt` with `EnzymeJVP` (weak dependency, to be approved then).
- CPU/CUDA tests: JVP agreement, Newton convergence rates, agreement with Path 2 in the linear limit.
- Fixed-iteration `NewtonKrylov` variant for Reactant, raise tests, Enzyme reverse tests, and an
  implicit-function-theorem adjoint rule for `implicit_step!` if nested AD proves fragile.

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
- Value conditions on the closure variables (`temperature`, `pressure_head`), such as
  `PrescribedSurfaceTemperature`, are supported by the tridiagonal operator only from Phase 3; flux
  conditions on the conserved prognostics and gradient conditions on the closure variables work from
  Phase 1.
- Path 1 on Reactant requires fixed iteration counts and is not adaptive; Path 1's Enzyme JVP
  requires a new weak dependency; nested AD through Path 1 is not guaranteed until the adjoint rule
  from the future Path 1 plan exists.
- Adaptive `Δt` under Reactant remains out of scope, as in the adaptive-timestepping plan.

## Future work

- Implicit-function-theorem adjoints for `implicit_step!` (efficient inverse modeling through
  implicit steps).
- Stiffly accurate higher-order IMEX Runge–Kutta pairs (ARS(2,2,2), ARS(4,4,3)) on top of the stage
  hook.
- Implicit treatment of snow energy and of the lateral/2D couplings if they are ever added.
- A convergence-based step controller for the implicit solvers to replace the diffusive CFL wizard.

## Reviewer decisions

1. **Approved:** scope is Path 2 and the core refactoring in this PR (Phases 1–3), spikes in
   Phase 4, Path 1 deferred to a future plan.
2. **Decided:** `FreeWater` under the implicit heat operator uses a regularized apparent heat
   capacity `C_app ≈ C + ρLθ / ΔT_reg` over a phase-change interval of width `ΔT_reg`, exposed as a
   documented parameter of the implicit solver; `SFCC` curves use their analytic `C_app`.
3. **Approved:** type names `ImplicitEuler`, `AbstractImplicitSolver`, `TridiagonalPicard`.

Deferred with Path 1: Enzyme (or EnzymeCore) as a weak dependency with `TerrariumEnzymeExt`, and
Krylov.jl as a direct dependency versus a hand-written fixed-iteration GMRES(m).
