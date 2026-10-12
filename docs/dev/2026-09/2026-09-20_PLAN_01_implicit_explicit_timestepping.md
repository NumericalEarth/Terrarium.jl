# Add support for implicit-explicit (IMEX) time stepping

> Status: **in progress**. Phases 1–3 (implicit heat conduction, implicit Richards flow, energy advection, and the wizard diagnostics) and IMEX as the default of `SoilModel` and `LandModel` are implemented; revision 20 (implicit snow meltwater drainage) awaits approval, and Phase 4 (feasibility spikes for Path 1) remains.

Date of initial draft: 2026-09-20

Base revision: 096bacc882bf5b21950abb0bc639aa457adc30ba

Revision 2 reviewed against: e9b9ef82a706492972a70f7a9892a7cd13afdd4e (branch `bg/implicit-timestepping`, Terrarium 0.1.9-DEV, Oceananigans 0.113.5)

## Originating prompt

> Please draft a plan that outlines a path for implementing implicit-explicit timestepping in Terrarium.
> We would like to explore two paths, possibly implementing both as separate options if feasible: 1. Leverage differentiability via Enzyme to evaluate JVPs in the quasi-Newton solver fully Jacobian-free.
> Ideally, the solve should run over all variables simultaneously, including explicit terms, though operator splitting can also be considered as a viable option. 2. Follow a pattern closer to the way Oceananigans currently implements implicit-explicit time stepping in its own model timesteppers.
> For this path, we will need to do a thorough investigation of the existing timesteppers in Oceananigans and assess how much of the functionality and patterns can be transferred to Terrarium.
> At bare minimum, we would want to reuse certain key elements like `BatchedTridiagonalSolver` for soil operators.
>
> Give a thorough assessment of both approaches and sketch an implementation path for each, along with any necessary refactoring of the timestepping system.
> Give a recommendation of which path to take, or whether both should be provided as separate timestepping options to the user.
> You may find the previous plan on adaptive timestepping useful as a reference.

## Revision log

> **Revision 1, 2026-09-20.** Initial draft.
>
> **Revision 2, 2026-10-08.** Re-reviewed against the current repository (see "B0. Repository state at revision 2").
> Changes: the explicit heat operator was renamed to `TwoPhaseHeatTransport` (commit `e9b9ef82a`); per reviewer feedback the operator types are left untouched and R1 instead passes the resolved `Timestepping` trait into the soil energy and hydrology `compute_tendencies!`, and the implicit entry point is named `implicit_step!`; Oceananigans is pinned at 0.113.5, so the compat item and the `IMEXFluxBoundaryCondition` availability caveat are dropped; state fields may now live on plain `AbstractGrid`s and may carry non-scalar element types, which adds two constraints to the implicit caches (R5); the soil kernel fusion that R1 would have interacted with was abandoned; the ground heat flux is now its own SEB sub-process, which is where the Phase 3 linearized surface flux hooks in; the Reactant extension now converts the traced clock time, which R7 relies on.
> The recommendation and phasing are unchanged.

>
> **Revision 3, 2026-10-08.** Reviewer feedback: the implicit entry point is `implicit_step!`; the operator types are left unchanged and the resolved `Timestepping` trait is passed into the soil energy and hydrology `compute_tendencies!` instead; the finer `Implicit{…}` trait distinction is deferred as a Path 1 consideration; the feasibility spikes (formerly Phase 0) move after the Path 2 implementation; and Path 1 is deferred in its entirety to a future plan and PR.
> Its assessment and sketch stay in this document as the record for that future work.
> Decisions on the Enzyme weak dependency and Krylov.jl are deferred with it.
>
> **Revision 4, 2026-10-08. Approved for implementation.** Reviewer decisions: scope (Path 2 and the core refactoring in this PR, spikes in Phase 4, Path 1 deferred) approved; type names `BackwardEuler`, `AbstractImplicitSolver`, `TridiagonalPicard` approved; the implicit heat operator uses a **regularized apparent heat capacity** for `FreeWater` (with a documented `ΔT_reg` parameter) rather than requiring a smooth freeze curve.
>
> **Revision 5, 2026-10-08. Energy advection added to Phase 2.** Reviewer request: the soil energy operator gains advection of sensible heat by the soil water flux, `−∂z(ρ_w c_w T q)`, with `q` the Darcy flux at faces.
> Rationale from experience with coupled energy–water runs: without it, water moves between cells while its heat does not, so the latent-heat budget `ρLθ` of the receiving cell changes at fixed `U`, the diagnosed `T` and liquid fraction jump, and the run destabilizes near the freezing front.
> Decisions: the advective flux is **explicit, first-order upwind** (only vertical diffusion is implicit); it lands in **Phase 2** with the Richards work since the flux requires `RichardsEq` hydrology (zero under `NoFlow`); infiltrating water enters at the **upwind (top cell) temperature**.
> The explicit heat operator (`TwoPhaseHeatTransport`) carries the term in both the explicit and the implicit configuration; it is part of `G^{exp}` in the implicit case.
>
> **Revision 6, 2026-10-08. Phase 1 review fixes.** Reviewer requests applied to the Phase 1 implementation: `phase_change_interval` (ΔT_reg) moved from `TridiagonalPicard` to `SoilThermalProperties` (the solver keeps only `iterations`); `Uⁿ` is stored exactly by a new IMEX pre-stage hook `update_cache!` (before the explicit stage) instead of being reconstructed as `u* − Δt G`; gradient boundary conditions on `temperature` are supported (only value conditions are rejected); a generic `compute_implicit_linear_coefficient` (zero by default) enters the tridiagonal diagonal as `−Δt L`; the unused `BackwardEuler.Δt` was removed (the IMEX takes `Δt` from its explicit partner); `IMEX` validates the order of its sub-steppers and `check_implicit_operator` rejects variables without an implicit operator at `initialize`; the implicit heat step ends with the energy closure.
> The requested extension of the regularized capacity to fully frozen or thawed cells with `|T| < ΔT_reg / 2` was evaluated and not adopted: it increased the maximum freeze–thaw temperature error from 0.30 K to 3.6 K (`M = 1`) and from 0.022 K to 4.0 K (`M = 2`).
> Reviewer decision (2026-10-08): the `0 < F < 1` criterion is kept, and the default number of Picard iterations of `TridiagonalPicard` is now 2.
>
> **Revision 7, 2026-10-08. Phase 2 implementation notes.** Implemented as planned (Celia modified Picard in the total head, conservative saturation update, hydrology closure after every iteration, explicit first-order upwind energy advection, `cell_advection_timescale`), with these deviations, pending reviewer decision: (a) the lagged moisture capacity is bounded from below by a new `minimum_moisture_capacity` (default `1e-5` 1/m) of the hydraulic properties, since frozen saturated cells (`K = 0` and `∂θ/∂ψ = 0`) otherwise give a zero row and NaNs; (b) intermediate Picard iterations discard surface excess water and only the last one routes it to the runoff pool, since every iteration restarts from `s*`; (c) the face conductivity of the implicit operator is the same direction-dependent minimum as the explicit `darcy_flux` (`darcy_conductivity`), with the direction lagged; (d) `C_w` is the `water` volumetric heat capacity of `SoilThermalProperties` rather than `ρ_w c_w` from `PhysicalConstants`, so that moving water of the cell temperature leaves the temperature unchanged (the two differ by 0.5 %); (e) the heat of water crossing the column boundary through flux conditions on `saturation_water_ice` is added from the saturation tendency of those conditions (in `compute_boundary_conditions!` of `SoilEnergyWaterCarbon`), since that flux is not part of the Darcy flux; (f) the advective timescale is `C Δz / (C_w |q|)`, the actual CFL limit of the energy equation, rather than `Δz / |q|`, and the thermal front moves at `C_w q / C`, not at the pore velocity `q / θ` as the testing section stated; (g) a pre-existing bug was fixed: `adjust_saturation_profile!` routed the surface excess to the runoff pool without the porosity factor, creating water whenever the surface saturated.

> **Revision 8, 2026-10-08. One-sided boundary condition fix.** Fixed a pre-existing bug found by the Phase 1 review: `create_field` built `FieldBoundaryConditions` from a user-supplied `NamedTuple` with a hardcoded `(Center(), Center(), nothing)` location, so a condition given on one vertical side only (e.g. a bottom `GradientBoundaryCondition` or `PrescribedBottomTemperature`) left the other side `nothing`, its halo was never filled, and the zero halo produced a spurious top-face conductive flux. `create_field` now passes the field's actual location and indices, so unspecified sides take the Oceananigans default (no flux); regression tests are in `test/boundary_conditions.jl`.
> The workaround previously needed in `test/timestepping/implicit_heat_conduction.jl` (since revision 15 part of `test/soil/soil_energy_tests.jl`) (an explicit no-flux top condition in the gradient-BC energy conservation test) has been removed.

> **Revision 9, 2026-10-09.** Reviewer requests: the two IMEX validation checks are merged into a single unexported `validate_timestepper` (one-argument form for a stepper on its own, two-argument form for a sub-stepper's role under an `AbstractIMEX`), and the unused `is_adaptive` trait and export are removed since adaptive stepping is provided by the Oceananigans `TimeStepWizard`.

> **Revision 10, 2026-10-09.** The pre-stage hook `prepare_implicit_step!` is renamed `update_cache!` (by the reviewer) and generalized to any timestepper cache, with signature `update_cache!(cache::AbstractTimeStepperCache, integrator, Δt, names)`: it dispatches on the cache type, so the timestepper argument is dropped; the default is a no-op.

> **Revision 11, 2026-10-09.** Reviewer request: the models no longer resolve the per-variable `Timestepping` traits themselves.
> `SoilModel` and `LandModel` pass their timestepper to the `SoilEnergyWaterCarbon` methods (`compute_tendencies!` and `cell_diffusion_timescale`), which extract the classes of `internal_energy` and `saturation_water_ice` with the new `timestepping(timestepper, state, Val(name))`.
> That method dispatches on the timestepper: an `AbstractIMEX` reads the classes resolved in its cache, a single timestepper returns its own class, and `nothing` gives `Explicit()`.

> **Revision 12, 2026-10-09. Approved for implementation: operators queried from processes, solvers decoupled from the linear form.**
> Reviewer request: instead of validating whether a solver supports each routed variable (`check_implicit_operator`), the implicit timestepper queries each routed prognostic variable for the form of its linear operator, stores the operators in its cache, and the solver dispatches on the operator type.
> Proposed design:
>
> 1. **`AbstractLinearOperator`** is a new abstract type for the linearized implicit operator of one prognostic variable.
>    Its first implementation, `TridiagonalOperator`, represents a batched vertical tridiagonal operator.
>    It owns the `BatchedTridiagonalSolver`, the three marker coefficient structs, and the scratch fields `diagonal_coefficient` (Center) and `face_diffusivity` (Face) on the ground grid, all of which currently live in `TridiagonalPicardCache`.
>    The `get_coefficient` methods, the generic coefficient kernel functions, and the flux-divergence stencil move with it.
> 2. **Query.** `implicit_operator(model, state, ::Val{name})` returns the operator for a routed variable.
>    `SoilModel` and `LandModel` forward to `implicit_operator(soil::SoilEnergyWaterCarbon, state, grid, ::Val{name})`, and the coupled soil type maps `:internal_energy` to its energy process and `:saturation_water_ice` to its hydrology process, mirroring revision 11.
>    `SoilThermodynamics` and `SoilHydrology{NF, RichardsEq}` each return a `TridiagonalOperator` allocated on the ground grid.
>    The host-side validation now done by `check_implicit_operator` (only `FreeWater`, no value boundary conditions on the closure variable, `RichardsEq` with `SoilSaturationPressureClosure`) moves into these methods.
>    The fallback throws the existing "no implicit operator is defined for variable" `ArgumentError`, so a misrouted variable still fails at `initialize`.
> 3. **Cache.** `BackwardEulerCache` gains `operators::NamedTuple`, keyed by the routed prognostic names, alongside `rhs`, `previous_state`, and `explicit_state`.
>    `AbstractImplicitSolverCache`, `EmptyImplicitSolverCache`, `TridiagonalPicardCache`, and `check_implicit_operator` are removed.
>    Each variable gets its own operator, so with both soil variables implicit the scratch fields are allocated twice; this costs two ground-grid fields and one tridiagonal scratch array per variable.
> 4. **Solver.** `TridiagonalPicard` is renamed `Picard` and keeps only `iterations` (default 2).
>    `BackwardEuler` calls `implicit_step!(integrator, timestepper, solver, operators[name], Δt, Val(name))`.
>    The `Picard` method owns the iteration loop and is generic over the operator.
>    Each iteration calls four process hooks, reached through the same model-to-coupled-soil forwarding as the query: `linearize!` (fill the operator's lagged coefficients and the right-hand side at the current iterate), `solve!(unknown, operator, rhs, Δt)` (dispatch on the operator type; for `TridiagonalOperator` the batched Thomas sweep), `implicit_update!` (conservative update of the prognostic from the operator's flux divergence), and the closure.
>    The heat step's use of `sⁿ` while saturation is also implicit (`use_previous_saturation!`) becomes part of the energy process's hooks.
>    A future Newton–Krylov solver (Path 1) would add methods for the same operators instead of a parallel solver type per linear form.
> 5. **Tests and docs.** The heat and Richards tests keep their numerical assertions unchanged, so the refactor must be behavior preserving to roundoff.
>    The IMEX routing tests replace the `check_implicit_operator` cases with `implicit_operator` cases, and the docs pages are updated for the new names.
>
> Call chain under this design: the IMEX `timestep!` calls `implicit_step!(integrator, ::BackwardEuler, Δt, names)` as now.
> That method calls, per routed variable, `implicit_step!(integrator, timestepper, solver, operators[name], Δt, Val(name))`, which is the solver's method and contains the iteration loop.
> The hooks are the process-specific pieces called inside that loop; today the same four steps are written out inline in the per-variable `implicit_step!` methods for heat and water.
>
> Reviewer decisions so far: (a) Terrarium keeps its own operator type, named `TridiagonalOperator`, under `AbstractLinearOperator`; `LinearAlgebra.Tridiagonal` is not used, because the operator must own the batched solver, its scratch array, and the coefficient fields, and no `[deps]` change is needed.
> (b) One operator per variable, as proposed.
> (c) The hook names `linearize!` and `implicit_update!` are kept as proposed.
> Preliminary step, done first: the `operator` field is removed from `SoilThermodynamics`, together with `AbstractHeatOperator` and `TwoPhaseHeatTransport`.
> The field carried no data, had a single implementation, and was never dispatched on; it also clashed with the operator concept above.
> The energy equation it documented now lives in the `SoilThermodynamics` docstring, and `SoilThermodynamics` has one type parameter fewer.
> Earlier revisions that mention `TwoPhaseHeatTransport` are kept as written.
>
> Implementation notes (2026-10-09), pending reviewer decision on the points marked as additions:
> `AbstractLinearOperator`, `TridiagonalOperator` (with the constructor `TridiagonalOperator(grid)`, its `Adapt` rule, the marker coefficients, the `get_coefficient` methods, and the coefficient and flux-divergence kernel functions), and the method `solve!(unknown, operator::TridiagonalOperator, rhs, Δt, args...)` live in `src/timesteppers/implicit/tridiagonal_operator.jl`; `Picard` lives in `picard.jl` (both moved to `src/solvers/` by revision 13).
> The model-level hook signatures are `linearize!(operator, rhs, state, model, explicit_stage, Δt, ::Val{name})`, `implicit_update!(state, model, operator, explicit_stage, Δt, ::Val{name})`, and `implicit_closure!(state, model, final::Val, ::Val{name})`, following the "outputs first, then state and model" convention of the process interface; `explicit_stage` is the field holding `u*` of the variable.
> `SoilModel` and `LandModel` forward each hook to `SoilEnergyWaterCarbon` (adding the grid, the physical constants, and, for the closure, the surface runoff process), which forwards to the energy or hydrology process (e.g. `linearize!(operator, rhs, state, grid, process, soil, constants, explicit_stage, Δt)`).
> Addition (a): a fifth query, `implicit_unknown(model, state, ::Val{name})`, returns the field the linear system is solved for (`temperature` or `pressure_head`), since `solve!(unknown, operator, rhs, Δt)` needs it and `Picard` must not know which variable it is.
> Addition (b), superseded by revision 13: the setup and teardown that are not part of an iteration are split as follows: `Picard` itself copies `u*` into `explicit_state` and resets the prognostic field to `uⁿ`, and an optional pair of hooks, `begin_implicit_step!` and `end_implicit_step!(state, model, previous_state, explicit_state, ::Val{name})` (no-ops by default), is called before and after the iterations.
> The heat step's use of `sⁿ` (`use_previous_saturation!` and `restore_explicit_saturation!`, now taking the `previous_state` and `explicit_state` `NamedTuple`s instead of the cache) is implemented as this pair on `SoilEnergyWaterCarbon` for `internal_energy`, since it concerns the hydrology variable rather than the energy process alone.
> Revision 13 removes this pair of hooks and moves the copy of `u*` and the reset to `uⁿ` from `Picard` to `BackwardEuler`.
> Addition (c): `implicit_closure!` receives `final` as `Val(true)` or `Val(false)`, and `Picard` peels off its last iteration, so that the Richards closure routes surface excess water to the runoff pool only in the last iteration without a type instability.
> Addition (d): `linearize!` has a fallback that throws an `ArgumentError` for a variable that has an operator but no hooks; the IMEX routing test now checks this message instead of the former "no implicit operator" message of the per-variable `implicit_step!` fallback, which is replaced by a fallback for an unsupported solver and operator pair.
> The refactor is bitwise behavior preserving: the final states of a freezing heat column and of `LandModel`s with both soil variables implicit (with and without snow) are identical to those before the change, and no numerical test assertion was modified.

> **Revision 13, 2026-10-09. Saturation substitution in the heat step, and reorganization of the implicit code.**
> The heat conduction step no longer swaps the saturation field in place with `begin_implicit_step!` and `end_implicit_step!` (`use_previous_saturation!` and `restore_explicit_saturation!`); these hooks are removed.
> Instead, when the saturation is also integrated implicitly, the heat step substitutes `sⁿ` from the `previous_state` of the `BackwardEulerCache` for the saturation field in the fields it passes to its kernels and to the energy closure (`implicit_heat_conduction_fields` and `implicit_heat_conduction_substitutions`), without modifying the state.
> The public `closure!` API is unchanged; it calls the new internal `compute_energy_closure!`, which takes the substitutions as an additional `NamedTuple` argument.
> Reviewer request: solver code lives in `src/solvers/`, and soil physics lives in the soil process directories rather than in the generic timestepping code.
> `AbstractLinearOperator`, `TridiagonalOperator`, and its coefficient kernel functions move to `src/solvers/linear_operators.jl`, the implicit solver interface (`AbstractImplicitSolver`, `implicit_operator`, and the process hooks) to `src/solvers/implicit_solvers.jl`, and `Picard` to `src/solvers/picard.jl`.
> `Picard` no longer references `ModelIntegrator` or `BackwardEuler`: it implements the solver-level `implicit_step!(solver, operator, state, model, rhs, explicit_stage, Δt, ::Val{name})`, which contains only the iteration loop, and the fallback for an unsupported solver and operator pair moves to this method.
> `BackwardEuler` keeps the bookkeeping of the step in `src/timesteppers/implicit/implicit_euler.jl`: its per-variable `implicit_step!(integrator, ::BackwardEuler, Δt, ::Val{name})` copies `u*` into `explicit_state`, resets the prognostic field to `uⁿ`, and calls the solver-level method.
> The soil energy and hydrology hooks move to `src/processes/soil/energy/soil_energy_implicit.jl` and `src/processes/soil/hydrology/soil_hydrology_implicit.jl`, together with the `SoilEnergyWaterCarbon` methods that map each variable to its process; the coupled soil fallback of `implicit_operator` moves to `soil_coupled.jl`.
> The `SoilModel` and `LandModel` forwarding methods move to `soil_model.jl` and `land_model.jl`, next to their `timestepping` routing methods.
> Both changes are bitwise behavior preserving.

> **Revision 14, 2026-10-09. Approved for implementation: one implicit operator per model block instead of per variable.**
> Reviewer concern: the per-variable routing bakes operator splitting into the timestepping infrastructure, with no clear way to solve the implicit variables together.
> The reviewer considered dropping the routing classes and moving `implicit_step!` and the solver caches into each model.
> Assessment: the splitting is not caused by the routing classes but by two narrower choices.
> `BackwardEuler` steps the implicit variables one at a time, and its cache holds one operator per variable, so energy and water are always solved in sequence.
> The routing classes themselves only decide which variables are implicit and tell processes to omit their interior fluxes, which any IMEX design needs.
> Moving everything into the models would delete about 150 to 200 lines of routing but duplicate the beginning-of-step bookkeeping, cache allocation, and trait plumbing in every model, and drop the per-variable user override.
> Proposed instead: keep the routing and the IMEX driver, and make the unit of the implicit solve the model's whole set of implicit variables.
>
> 1. **Block query.** `implicit_operator(model, state, names::Tuple)` is called once in `initialize` with the routed names and returns a single `AbstractLinearOperator` for all of them.
>    The per-variable query `implicit_operator(model, state, ::Val{name})` stays as the building block that models use to assemble it.
> 2. **Composite operator for splitting.** A new `SplitOperator <: AbstractLinearOperator` holds a `NamedTuple` of per-variable operators, in the order in which they are applied.
>    `SoilModel` and `LandModel` return `SplitOperator((internal_energy = …, saturation_water_ice = …))` for the routed subset, so today's sequential splitting becomes one model choice rather than a property of the timestepper.
>    The order is set by the model (energy, then water) rather than by the order of the prognostic variables.
>    A fully coupled operator (for example a block operator for a future Newton–Krylov solver) is then just another operator type returned by the same query.
> 3. **Cache.** `BackwardEulerCache` holds one `operator` instead of the `operators` `NamedTuple`; `rhs`, `previous_state`, and `explicit_state` are unchanged.
> 4. **Bookkeeping for the whole block.** `BackwardEuler`'s `implicit_step!` first copies `u*` into `explicit_state` and resets the prognostic field to `uⁿ` for every implicit variable, and then calls the solver once, as `implicit_step!(solver, cache.operator, state, model, cache.rhs, cache.explicit_state, Δt, names)`.
>    Today the copy and reset happen per variable, just before that variable's solve.
> 5. **Solver dispatch.** `Picard` gets a method for `SplitOperator` that applies the existing per-variable Picard loop to each component in order, and keeps the per-variable method for `TridiagonalOperator`.
>    The fallback for an unsupported solver and operator pair is unchanged.
> 6. **Simplification of the heat step.** Because every implicit variable is reset to `uⁿ` before any solve, the saturation field already holds `sⁿ` when the heat step runs.
>    The field substitution of revision 13 (`implicit_heat_conduction_fields`, `implicit_heat_conduction_substitutions`, and the internal `compute_energy_closure!`) is therefore removed, and the heat hooks call `closure!` and `get_fields` directly again.
>    With explicit hydrology the saturation field holds `sⁿ⁺¹`, as now.
>    The water step reads the temperature and liquid water fraction produced by the heat step, as now.
>    The results should therefore be bitwise identical to the current code, which the implementation verifies with a before-and-after snapshot.
> 7. **Unknown from the closure metadata (reviewer request).** `implicit_unknown` is removed, together with its model, coupled-soil, and process methods.
>    The field the linear system is solved for is derived from the prognostic variable's metadata instead: the closure variable if the variable has a closure, and the prognostic variable itself otherwise.
>    The documented `varname(closure)` interface (the name of the variable a closure relation returns) is implemented for `SoilEnergyTemperatureClosure` (`:temperature`) and `SoilSaturationPressureClosure` (`:pressure_head`); today no closure type implements it, and the energy closure declares two output variables, so the primary one must be named explicitly.
>    `BackwardEuler` resolves the names in `initialize` and stores them in its cache as type-level constants, so the solver retrieves the unknown field type-stably.
>    This fixes the convention that implicit operators are linearized in the closure variable, which holds for both current schemes (temperature and pressure head); a solver that works on the prognostic variable directly can ignore the stored name.
> 8. **Tests and docs.** The numerical tests are unchanged.
>    The IMEX cache-structure tests change from `operators` to `operator` and gain checks for the `SplitOperator` components and their order.
>    The solver, extending, and time-stepping docs describe the block query and `SplitOperator`.
>
> Reviewer decisions: (a) the name is `SplitOperator`;
> (b) the revision 13 field substitution is removed as part of this change (item 6);
> (c) `SplitOperator` is not exported;
> (d) the unknown is identified by explicit `varname` methods on the two soil closures (item 7).
>
> Implementation notes (2026-10-09):
> `SplitOperator{Operators <: NamedTuple} <: AbstractLinearOperator` (field `operators`, with an `Adapt` rule) lives in `src/solvers/linear_operators.jl` and is not exported; it has a `canonical = false` entry on the solvers page, following the other unexported types documented there.
> The block query has a generic fallback, `implicit_operator(model::AbstractModel, state, names::Tuple{Vararg{Symbol}})`, which returns a `SplitOperator` of the per-variable operators in the order of `names`; models that do not define the block query (e.g. the mock models of the IMEX tests) therefore keep working.
> `SoilModel` and `LandModel` forward the block query to `implicit_operator(soil::AbstractSoil, state, grid, names::Tuple{Vararg{Symbol}})` in `soil_coupled.jl`, which orders `internal_energy` before `saturation_water_ice` and appends any other names, which the per-variable fallback then rejects.
> The solver-level step has two forms, distinguished by arity: the block step `implicit_step!(solver, operator, state, model, rhs::NamedTuple, explicit_state::NamedTuple, Δt, unknowns::NamedTuple)`, called once per step by `BackwardEuler`, and the per-variable step `implicit_step!(solver, operator, state, model, rhs, explicit_stage, Δt, ::Val{name}, ::Val{unknown})`, which the block step applies to each component of a `SplitOperator`.
> Instead of the `names` tuple of item 4, the block step receives `unknowns`, a `NamedTuple` keyed by the routed names whose values are `Val`s of the unknown field names; its keys carry the same information as `names`.
> Both forms have a fallback that throws the unsupported solver and operator `ArgumentError`; `Picard` implements the block step for `SplitOperator` and keeps the per-variable step for any `AbstractLinearOperator`, as in revision 12.
> `BackwardEulerCache` holds `operator` and the new field `unknowns`, e.g. `(internal_energy = Val(:temperature), saturation_water_ice = Val(:pressure_head))`.
> The names are resolved in `initialize` by the unexported host-side helper `resolve_implicit_unknown(var::PrognosticVariable, state)`, which returns `varname(var.closure)` if `hasclosure(var)` and `varname(var)` otherwise, and throws an `ArgumentError` if the closure does not implement `varname` or the state has no field of that name.
> `varname(::SoilEnergyTemperatureClosure) = :temperature` and `varname(::SoilSaturationPressureClosure) = :pressure_head` live next to the `variables` methods of the closures in `soil_energy_closures.jl` and `soil_hydraulic_closures.jl`.
> The per-variable `implicit_step!(integrator, ::BackwardEuler, Δt, ::Val{name})` is removed; the block method copies `u*` and resets `uⁿ` for all variables and calls the solver once.
> `implicit_unknown` (all methods and doc entries), `implicit_heat_conduction_fields`, `implicit_heat_conduction_substitutions`, and `compute_energy_closure!` are removed; the heat hooks call `get_fields` and `closure!` directly, and the body of `closure!` is again identical to the base revision.
> The change is bitwise behavior preserving: the final states of a freezing heat column (50 steps), and of `LandModel`s with both soil variables implicit and rain, without snow and with a `SingleLayerSnow` pack that survives the 24 steps, are identical (`==`) to those before the change for all compared fields, and no numerical test assertion was modified.

> **Revision 15, 2026-10-09. Approved for implementation: heat advected with the water that actually moves.**
> Reviewer question: does the implicit scheme actually increase the permissible time step for Richards flow?
> Experiments (one day, 50 cells of 2 cm, van Genuchten soils, compared with forward Euler at Δt = 1 s) gave a mixed answer.
>
> - **Water alone: yes.** In a nearly saturated column with gravity drainage, `IMEX` with two Picard iterations stays within `2.5e-5` in saturation up to Δt = 1 h, while forward Euler is accurate at 10 s and wrong (error 0.25) from 30 s.
>   In a dry column with infiltration, `IMEX` stays within 0.01 to 0.02 at 30 min to 1 h, while forward Euler already has an error of 0.08 at 5 s.
>   Forward Euler never produces NaNs, because the closure clips the saturation to `[0, 1]`; its failures show up as large errors instead.
> - **Coupled energy and water: no, because of two defects in the energy advection of revisions 5 and 7.**
>   Both appear with a uniform initial temperature of 5 °C and no heat flux at the boundaries, where the temperature must stay at 5 °C.
>
> **Defect 1: under `IMEX`, heat and water move with different fluxes.**
> The interior advective heat flux `C_w T_up q` is explicit, so it uses the Darcy flux `qⁿ` at the start of the step, while the water moves with the implicit flux of the Picard solve.
> During wetting the two differ greatly.
> In a sandy column (`Ks = 1e-4` m/s, initial saturation 0.2, infiltration `2e-5` m/s, free drainage), one 300 s step adds 6 mm of water to a top cell that holds about 1.6 mm, and the implicit solve passes most of it downward, while `qⁿ` at the face below is still the small flux of the dry soil.
> The heat of that water therefore stays in the top cell, which warms; since infiltrating water enters at the temperature of the top cell, the warming feeds back.
> The column ends at a uniform 6.97 °C after one day, while forward Euler stays at 5.001 °C; the energy budget of each step closes exactly, so the scheme is consistent but advects heat with the wrong flux.
> At Δt ≥ 1800 s the same column diverges (temperatures of order `−10⁹` °C).
> The advective time scale `cell_advection_timescale` does not detect this, since it is also computed from `qⁿ`.
>
> **Defect 2: water removed by the hydrology closure carries no heat.**
> `adjust_saturation_profile!` moves oversaturated water upward through the column and removes the excess at the surface (discarded by `SoilModel`, routed to `surface_excess_water` by `LandModel`), without moving or removing its heat.
> Infiltrating water, on the other hand, brings the heat `C_w T_top` into the top cell.
> Once the column is full, every step adds heat with water that is then removed without it, which heats the top cell exponentially with an e-folding time of about `C Δz / (C_w q)` (about 50 min for `q = 5e-6` m/s and `Δz = 2` cm).
> This affects explicit time stepping too: a closed, nearly saturated column reaches 194 °C after one day with forward Euler at Δt = 1 s (179 °C with `IMEX` at 300 s), and a closed sandy column reaches 41 °C (forward Euler) and `3.7e12` °C (`IMEX`).
> The limitation "water routed to the surface excess pool carries no heat" in the time stepping docs understates this; it is an instability, not just a bias.
>
> Proposed changes:
>
> 1. **Water before energy.** `implicit_operator(soil, state, grid, names)` orders `saturation_water_ice` before `internal_energy` in the `SplitOperator`, so that the heat step sees the water flux of the current step.
>    The water step then reads the temperature and liquid water fraction at `tⁿ` (lagged by one step) rather than those of the heat step, so the coupling of frozen ground to hydraulic conductivity is lagged; this is the usual choice of sequentially split land models.
>    To quantify the effect of the lag, a Stefan problem benchmark with an analytical solution (freezing or thawing front in a soil column with phase change) is added, comparing the front position and temperature profile of the fully explicit scheme and of `IMEX` with water and energy implicit; CryoGrid.jl's Stefan benchmark serves as the reference for its setup.
> 2. **Record the water flux of the step.** The final water iteration stores the interior face Darcy flux it applied in its conservative update, `q = −K^m ∂zΨ^{m+1}`, in a new auxiliary variable `water_flux` (m/s) of `SoilHydrology` with `RichardsEq` on cell faces, positive upward and zero at the boundary faces.
>    Under explicit hydrology the same field holds the Darcy flux at `tⁿ` used by the explicit update, so the heat step always uses the flux that moved the water.
> 3. **Implicit upwind advection in the heat operator.** When `internal_energy` is implicit, the interior advective flux leaves `G_exp` and enters the heat system as first-order upwind advection with the recorded flux:
>    `C_app T^{m+1} − Δt ∂z(κ ∂z T^{m+1}) + Δt ∂z(C_w q T_up^{m+1}) = C_app T^m − U^m + u*`.
>    For a fixed `q`, upwind advection keeps the system tridiagonal and an M-matrix, so the temperature still obeys a discrete maximum principle and the advective CFL limit disappears.
>    `TridiagonalOperator` gains an optional `face_velocity` field that is `nothing` when unused; the upwind terms of its three diagonals read the velocity through a small inlined accessor whose `Nothing` method returns `zero(grid)`, so heat-only and Richards operators compile to the same code as now; the conservative energy update uses the same interior fluxes, `U = u* + Δt [∂z(κ ∂z T) − ∂z(C_w q T_up)]`, and so remains exactly conservative.
>    The heat of water crossing the boundaries through flux conditions on `saturation_water_ice` stays explicit (`compute_boundary_advected_heat!`), since it is tied to an explicit boundary flux.
>    `cell_advection_timescale` returns `Inf` when `internal_energy` is implicit.
> 4. **Heat moves with the water redistributed by the closure.** `adjust_saturation_profile!` moves the heat `C_w T` of the water it shifts between cells, and removes `C_w T_top` times the excess it discards or routes to `surface_excess_water`, from `internal_energy`, using the temperature of the cell the water leaves.
>    The hydrology closure runs before the energy closure in `closure!` of `SoilEnergyWaterCarbon`, so the temperature is recomputed from the adjusted energy.
>    This applies to explicit and implicit time stepping alike and leaves `closure!` signatures unchanged; the surface excess pool still stores no heat, i.e. the heat of runoff leaves the system, which is documented.
> 5. **Implicit water with explicit energy.** With Richards hydrology, implicit `saturation_water_ice` together with explicit `internal_energy` reintroduces defect 1, since the explicit advection then uses the flux at `tⁿ`.
>    The combination remains allowed, and the time stepping docs warn against it; the default routing of `SoilModel` and `LandModel` (both implicit) is unaffected.
> 6. **Tests.** New tests, for forward Euler and for `IMEX` at Δt from 300 s to 3 h:
>    a uniform-temperature column stays at its initial temperature (to `1e-6` K) under infiltration, wetting fronts, free drainage, and filling of a closed column;
>    the column energy budget closes with the heat carried by infiltration, drainage, and surface excess water;
>    a sandy column with infiltration and free drainage stays bounded at large Δt with `IMEX`;
>    the Stefan problem benchmark of item 1, with tolerances on the front position set from the explicit reference;
>    the existing advection tests for the explicit path are kept, and the IMEX advection test checks agreement with a fine explicit reference instead of the CFL limit.
>    Revision 14 snapshots are not expected to be bitwise identical, since the order of the split and the advection change.
> 7. **Docs.** The time stepping page replaces the explicit advection paragraph and the "carries no heat" limitation with the implicit upwind scheme, the order of the split, and the lagged frozen-ground coupling; the soil energy and hydrology pages document `water_flux` and the heat moved by `adjust_saturation_profile!`.
>
> Not part of this revision, but found while investigating (proposed for separate PRs):
> (a) `FreeDrainage()` prescribes a zero gradient of `pressure_head`, which holds the total head, so it is a no-flux condition; free drainage needs `∂zΨ = 1`.
> (b) The explicit Richards `cell_diffusion_timescale`, `Δz² (∂θ/∂ψ) / K`, is zero as soon as one cell saturates, so a `TimeStepWizard` driving explicit Richards flow would collapse the step.
> (c) Explicit boundary fluxes (e.g. free drainage at `tⁿ`) are not limited by the water available in the boundary cell, so a large step can overdraw it; with the fixes above, this is the next constraint on Δt for draining columns.
>
> Reviewer decisions: (a) water before energy (item 1), with the Stefan problem benchmark to quantify the difference from the fully explicit scheme;
> (b) `water_flux` is an auxiliary variable of the soil hydrology (item 2);
> (c) implicit water with explicit energy is allowed, with a warning in the docs (item 5);
> (d) the face velocity of `TridiagonalOperator` is `nothing` when unused and resolved to zero by dispatch (item 3).
>
> Implementation notes (2026-10-09):
> `implicit_operator(soil, state, grid, names)` orders `saturation_water_ice` before `internal_energy`; the heat operator is allocated by `implicit_operator(energy::SoilThermodynamics, hydrology, state, grid)`, which now takes the hydrology to decide whether the operator advects.
> `TridiagonalOperator` has a fourth field `face_velocity` (`nothing` unless constructed with `TridiagonalOperator(grid; advection = true)`), read through `compute_implicit_face_velocity`, whose `Nothing` method returns `zero(grid)`; the upwind terms are separate kernel functions (`compute_implicit_advection_{upper,lower,}diagonal`, `compute_implicit_advective_flux`, `compute_implicit_advection_divergence`, the latter zero for `nothing` by dispatch), and `face_velocity` is passed to `get_coefficient` after `face_diffusivity`.
> `face_velocity` stores `C_w q` (W/m²/K), i.e. the velocity in units of the capacity times a velocity, so the operator stays generic and the flux is `u ϕ_up`; it is filled by the heat `linearize!` from `water_flux` (`compute_implicit_advective_velocity!`).
> Deviation: `water_flux` is not zero at the boundary faces; it holds the boundary Darcy fluxes at `tⁿ` (zero for no-flux conditions), because the heat step must use the boundary flux that moved the water and cannot recompute it from the `pressure_head` halos, which are filled at `tⁿ` while the interior holds `Ψⁿ⁺¹` after the water step (doing so removed spurious heat and froze a draining column).
> Consequently `water_flux` is computed in the Richards `compute_tendencies!` (after the halo fill, for every face and both timestepping classes) rather than in `compute_auxiliary!`, and the implicit update overwrites only the interior faces, in every Picard iteration (the last one remains, which is equivalent to storing it in the final iteration only).
> Deviation: under implicit energy, the advective heat of water leaving the column through a boundary face (e.g. gravity drainage) is part of the heat operator (outflow added to the diagonal, upwind value of the boundary cell at `tⁿ⁺¹`), and only the inflow through the boundary faces stays in `G_exp` (`compute_boundary_advective_energy_tendency`).
> Without this, the explicit drainage flux at `tⁿ`, which overdraws the bottom cell at large Δt (proposal (c) above), removed more heat than the cell held and made the bottom temperature unstable (with a perturbation of 0.01 K, the bottom cell of a draining sandy column warmed from 5 °C to 20.8 °C in 6 days at Δt = 1 h); with it, the system stays an M-matrix and the perturbation decays.
> The heat `linearize!` re-evaluates the energy closure before computing the coefficients, so that the first iterate `T⁰` is consistent with the internal energy and the saturation `sⁿ⁺¹` left by the water step (or by the explicit stage of an explicit saturation); without it, a uniform temperature is not preserved.
> When both variables are implicit, the final water closure adjusts `internal_energy` while it holds `Uⁿ`, which the heat step's conservative update `u* + …` would discard; `Picard`'s `SplitOperator` step therefore re-anchors the explicit stage of every component after the first at its field just before it is solved (it first subtracts `u⁰` from `u*` and then adds the current field, with a small `add_scaled_field_kernel!`), as in a sequential split; with one component the step is unchanged.
> Item 4: `adjust_saturation_profile!` takes an optional seventh argument `heat` (`redistributed_heat(state, energy)`, a `NamedTuple` of `internal_energy`, `temperature`, and `heat_capacity_water` for `SoilThermodynamics`, `nothing` otherwise), and every saturation change `Δs` of a cell changes its energy by `C_w T φ Δs` with the cell's own porosity and the temperature of the cell the water leaves (`compute_redistributed_heat!`), including the bottom clamp to the residual saturation.
> The hydrology `closure!` keeps its signature and passes `redistributed_heat(state, get_energy_balance(soil))`; the implicit closure calls the parts of the closure directly (via the new internal `compute_pressure_head!`) with `redistributed_heat(state, energy, iteration)`, which is `nothing` for non-final iterations, mirroring `excess_water_runoff`, so the energy is adjusted exactly once per step; `invclosure!` moves no heat.
> `cell_advection_timescale(state, grid, soil, timestepper)` returns `Inf` when `internal_energy` is implicit, mirroring `cell_diffusion_timescale`.
> The uniform-temperature tests set the heat capacity of air to zero: the bulk heat capacity includes air, so water replacing air raises the temperature by about `C_air/C_w` of the change of the water content (up to 1.4e-3 K after one day in the investigated columns, for both schemes), which no advection scheme of water heat alone can avoid; with it set to zero, the four columns stay at 5 °C to within 4e-13 K for forward Euler at 1 s and `IMEX` at 300 s to 3 h.
> The Stefan benchmark (`test/soil/soil_freeze_thaw_tests.jl`) implements the classical two-phase Neumann solution itself, without depending on CryoGrid.jl, starting from the Neumann profile at the time the front reaches 0.1 m and prescribing the Neumann surface heat flux (value conditions on `temperature` are not supported under implicit energy), with Richards hydrology in a saturated column (no flow, compared with the Neumann solution) and an unsaturated column with `Ks = 1e-7` m/s (compared with the explicit reference only).
> Cross-check with CryoGrid.jl (`src/Physics/Heat/analytic/stefan_analytic.jl`, after Hu et al., 1996, Eq. 13): with the roles of the phases exchanged (CryoGrid's benchmark thaws a frozen column, ours freezes a thawed one), the transcendental equation for λ, the front `X = 2λ √(α t)`, the `erf`/`erfc` temperature profiles, and the sign convention (depth positive downward) are identical; no discrepancy was found.
> Aligned with CryoGrid's benchmark (`test/Solvers/cglite_implicit_tests.jl`): the 1 cm tolerance on the front; not aligned: its daily time step (about eight steps over the nine days of this benchmark, with a front error of 6.6 cm), and its Dirichlet step to +1 °C at t = 0 on a frozen column at −1 °C (porosity 0.3) over five years, which would need value conditions on `temperature` under implicit energy and an unaffordable explicit reference.
> Known limitation found by the bounded-column test: when the explicit drainage flux at `tⁿ` overdraws a sandy column so strongly that it alternates between empty and saturated (1 m column at Δt = 3 h), a cell can receive far more water in one step than it holds; the water closure, which runs before the heat step, then moves the excess and its heat out of the cell before the heat step brings that water's heat in, so the cell's heat capacity at the start of the heat step becomes negative and the temperature leaves the initial range (a transient spike to 52 °C, energy still conserved); at Δt ≤ 1 h the column stays bounded, and the test uses 30 min and 1 h. Routing the closure's transfers through the implicit advective face fluxes would remove this, but needs the step's Δt in the closure; together with proposal (c) this is left for a separate change.
> The existing IMEX advection tests now use time steps of 2 and 5 times the explicit advective limit (larger steps are limited by the explicit drainage flux, proposal (c)), and the thermal front test compares `IMEX` with a fine explicit reference.

> **Revision 16, 2026-10-10. Approved for implementation: explicit increment in the cache, and a simpler solver interface.**
> Reviewer concern: the solver-level `implicit_step!` interface is hard to follow, and the re-anchoring of the explicit stage in the `SplitOperator` step of `Picard` (revision 15) is indirect.
> Today the block form takes eight positional arguments, `implicit_step!(solver, operator, state, model, rhs, explicit_state, Δt, iterations)`, four of which (`operator`, `rhs`, `explicit_state`, and `iterations`) are fields of `BackwardEulerCache` passed one by one.
> The per-variable form has the same name and argument order and differs only in whether the last arguments are `NamedTuple`s or single values; the explicit stage is called `explicit_state` in one form and `explicit_stage` in the other; and `iterations` is both the iteration count of `Picard` and the `NamedTuple` of `Iteration` labels.
> The re-anchoring exists because `BackwardEuler` stores `u* = uⁿ + Δt G_exp`, which the heat step's conservative update would apply on top of a stale `uⁿ` after the water closure has moved heat; the `SplitOperator` step therefore subtracts `u⁰` from the stage of every variable after the first before any solve, and adds the current field back just before its solve.
>
> 1. **Explicit increment in the cache.** `BackwardEuler`'s block `implicit_step!` stores the explicit increment `Δt G_exp = u* − uⁿ` of every implicit variable (copy `u*`, subtract `previous_state`) before resetting the prognostic fields to `uⁿ`.
>    `previous_state` stays, since it still provides `uⁿ` for the reset.
> 2. **Uniform stage formation.** On entry, the per-variable solver step forms its explicit stage as the current prognostic field plus the increment, i.e. `u*` for the variable solved first and `uⁿ + δ + Δt G_exp` for a later one, where `δ` is what the closures of the earlier variables added (e.g. the heat moved with the redistributed water).
>    The hooks still receive the stage `u*` of their variable, so no process code changes.
> 3. **`ImplicitSolverWorkspace`.** A new unexported struct `ImplicitSolverWorkspace{I <: Iteration, R, E}` in `src/solvers/implicit_solvers.jl` bundles the per-variable working data of an implicit step: `rhs`, `explicit_stage` (holding the increment until the step forms the stage), and the `iteration` label; it has an `Adapt` rule.
>    `BackwardEulerCache` holds `previous_state`, `operator`, and `workspaces`, a `NamedTuple` of `ImplicitSolverWorkspace`s keyed by the routed names, in place of its current separate `rhs`, `explicit_state`, and `iterations` fields (the latter two become the workspace fields `explicit_stage` and `iteration`).
> 4. **Solver-level signatures.** The two forms become
>    `implicit_step!(solver, operator, workspaces::NamedTuple, state, model, Δt)` (all routed variables, called once per step by `BackwardEuler`) and
>    `implicit_step!(solver, operator, workspace::ImplicitSolverWorkspace, state, model, Δt)` (one variable),
>    so the operator and its working data come first, followed by the state, the model, and `Δt`, and the two forms are distinguished by a named workspace type rather than by the shape of four arguments.
>    The solver depends on no timestepper type: `BackwardEuler` passes `cache.workspaces`, not the cache.
>    The fallbacks for unsupported solver and operator pairs keep their error messages.
> 5. **`Picard`.** The `SplitOperator` method becomes a plain loop over the components, calling the per-variable step with `operator.operators[name]` and `workspaces[name]`; `first_name`, `later_names`, both adjustments, and the docstring paragraph on re-anchoring are removed.
>    The per-variable step forms the stage (item 2) and then runs the iterations as now, with `Iteration` labels derived from `workspace.iteration`.
> 6. **Field kernel.** `add_scaled_field_kernel!` moves from `picard.jl` to a shared field utility with an unexported `add_scaled!(out, field, factor)` wrapper used by `BackwardEuler` and `Picard`; if an existing kernel of the explicit steppers has the same form, it is reused instead.
> 7. **Naming.** The explicit stage is called `explicit_stage` everywhere; the `NamedTuple` of `Iteration` labels disappears into the workspaces, so `iterations` only refers to the count of `Picard`.
>    The process hooks (`linearize!`, `implicit_update!`, `implicit_closure!`) and their model-level forwarding are unchanged.
> 8. **Tests and docs.** The IMEX cache-structure tests check `workspaces` and the `ImplicitSolverWorkspace` components instead of `rhs`, `explicit_state`, and `iterations`; no numerical test changes.
>    The behavior is verified with a before-and-after snapshot; for the variable solved first, `(u* − uⁿ) + uⁿ` need not equal `u*` bitwise, so differences at the level of rounding (about `1e-16` relative) are expected, as already for the later variables under revision 15.
>    The solver, extending, and time stepping docs describe the workspace and the new signatures.
>
> Reviewer decisions: (a) the name is `ImplicitSolverWorkspace`; (b) the `Iteration` label is kept in the workspace.
>
> Implementation notes (2026-10-10):
> `ImplicitSolverWorkspace{I <: Iteration, R, E}` (fields `rhs`, `explicit_stage`, `iteration`, with an `Adapt` rule) lives in `src/solvers/implicit_solvers.jl`, is not exported, and has `canonical = false` entries on the solvers and extending pages.
> `BackwardEulerCache{NF, US, Operator, Workspaces}` holds `previous_state`, `operator`, and `workspaces`; `initialize` builds each workspace from two `similar` copies of the prognostic field and the `Iteration` resolved by `resolve_implicit_target`.
> `BackwardEuler`'s step copies `u*` into `explicit_stage`, subtracts `previous_state` (`add_scaled!` with factor `-1`), and resets the field to `uⁿ`; the per-variable `Picard` step adds the current field (`add_scaled!` with factor `1`) before its iterations, and the `SplitOperator` step is a plain loop.
> The fallback of the per-variable step dispatches on `workspace::ImplicitSolverWorkspace` without a type parameter (with one it was ambiguous with the `Picard` method) and takes the variable name for its message from `varname(workspace.iteration)`.
> `add_scaled!(out, field, factor)` and its kernel `add_scaled_kernel!` live in `src/utils/kernel_utils.jl` and launch on `out.grid`; the explicit step kernel `explicit_step_xyz_kernel!` has the same arithmetic but takes an `AbstractTimeStepper` argument for dispatch, which the solver does not have, so it is not reused.
> The before-and-after snapshot (final states of a freezing heat column, a draining sandy column and a closed freezing wet column with both soil variables implicit, and `LandModel`s with rain without and with snow; all prognostic, closure, and flux fields) is bitwise identical, not only identical to rounding: where `u*` and `uⁿ` lie within a factor of two of each other, as in these runs, `u* - uⁿ` is exact (Sterbenz lemma) and adding `uⁿ` back recovers `u*` exactly; in general, differences at the level of rounding remain possible.
> No numerical test was changed; the IMEX cache-structure tests check `workspaces` and the `ImplicitSolverWorkspace` components.

> **Revision 17, 2026-10-10. Scope reduction: second-order IMEX deferred.**
> Reviewer request, to keep the changeset small: the IMEX trapezoidal scheme of Phase 3 (R4, `Heun` as explicit partner with per-stage implicit corrections, and the stage hook it needs) is deferred to a future PR and moved to Future work.
> The rest of Phase 3 stays in scope, in particular value boundary conditions on the closure variables under the implicit operator (sharing the boundary-diagonal code with `IMEXFluxBoundaryCondition`), together with the implicit surface coupling and the wizard diagnostics.
> `IMEX` therefore continues to accept only `ForwardEuler` as its explicit partner, and the second-order convergence test is dropped from the testing section.

> **Revision 18, 2026-10-10. Approved. Scope reduction: Phase 3 restricted to the wizard diagnostics.**
> Reviewer request: the implicit surface coupling (an `IMEXFluxBoundaryCondition` for the ground heat flux with `λ = ∂G/∂T_ground` from the skin temperature solve) and the value boundary conditions on the closure variables under the implicit operator are deferred to a future PR and moved to Future work.
> Value conditions on `temperature` and `pressure_head` therefore stay rejected at `initialize` when the variable is integrated implicitly, and the ground heat flux of a `LandModel` stays explicit at `tⁿ`; a rough estimate, `Δt ≲ 2 C Δz / λ` with `λ ≈ 2κ/Δz`, suggests that the latter limits the step to the order of 10 min for a 2 cm top cell, which remains to be measured.
> Phase 3 consists only of the wizard diagnostics: (1) document and test that the `TimeStepWizard` is not throttled by implicitly integrated terms (`cell_diffusion_timescale` and `cell_advection_timescale` are `Inf` for them), and (2) report the time scales of the terms that remain explicit, so that the wizard bounds `Δt` by them: the advective limit of explicit energy advection, and the limit at which explicit water outflow (boundary drainage and sinks such as evapotranspiration, evaluated at `tⁿ`) would remove more water than a cell holds.
> A convergence-based diagnostic for the Picard iteration stays future work.
>
> Implementation notes (2026-10-10):
> (1) The existing `Inf` diagnostics for implicit heat conduction, implicit Richards flow, and implicit heat advection are unchanged; new `Simulation` tests in `test/timestepping/adaptive_timestep.jl` show that an IMEX `SoilModel` and an IMEX `LandModel` reach `max_Δt` (about 118 and 14 times the explicit heat conduction limit), and the "Adaptive time stepping" docs gained an IMEX subsection with an example.
> (2a) The advective limit of explicit heat advection is verified under mixed routing with a test-local `AbstractIMEX` subtype that routes `internal_energy` explicitly; with a closed bottom it is the binding limit and the wizard step equals `cfl` times it.
> (2b) The explicit water outflow limit is the new `water_outflow_timescale` (dispatch on the hydrology, `Inf` fallback) with the kernel function `compute_water_outflow_timescale`, combined with the advective limit by `min` in the soil `cell_advection_timescale`.
> The per-cell time scale is `τ = φ (s − s_res) Δz / q_out`, with `q_out = −Δz G_θ` where the explicit volumetric water content tendency `G_θ` is negative, and `Inf` elsewhere.
> `G_θ` reuses `compute_volumetric_water_content_tendency` with the resolved `Timestepping` class of `saturation_water_ice` (boundary Darcy fluxes, evapotranspiration, and forcing under `Implicit()`; all Darcy fluxes under `Explicit()`), plus the flux boundary conditions on `saturation_water_ice` through the new kernel function `compute_saturation_boundary_flux` (`getbc` for `Flux` conditions, zero otherwise).
> The stored tendency is not used: the wizard runs after the step, when `compute_auxiliary!` has updated the auxiliaries but the tendencies are still those of `tⁿ` of the previous step, so the direct evaluation measures the outflow of the next step instead.
> The net (not gross) explicit tendency is used, since the explicit stage `s⁺ = sⁿ + Δt G_exp` overdraws a cell only if the net tendency is negative.
> The limit also applies under explicit hydrology, where it includes the interior Darcy fluxes, but the diffusive limit is then usually more restrictive.
> Both soil advective diagnostics fill the `pressure_head` halos first, since the wizard otherwise sees boundary Darcy fluxes computed from the halos of `tⁿ`.
> `cell_advection_timescale(state, grid, soil, ...)` now takes `constants` and an optional `surface_hydrology` (for the evapotranspiration sink) before the `timestepper`, like `compute_tendencies!`.
> In the draining sandy column (Δz = 5 cm, `Ks = 1e-4` m/s, infiltration `2e-5` m/s) the bottom cell drains in about 565 s, so the wizard (default `cfl = 0.2`) chooses 90 to 113 s, close to the explicit Richards limit of 97 s; fixed steps of 1800 s and longer overdraw the column, while wizard runs with `cfl` up to 1 stay bounded.
> The explicit boundary drainage is therefore the binding limit of such columns, which only an implicit treatment of the boundary Darcy fluxes would remove.

> **Revision 19, 2026-10-12. Approved. IMEX as the default timestepper of `SoilModel` and `LandModel`.**
> Reviewer request: the default timestepper of `SoilModel` and `LandModel` becomes `IMEX(ForwardEuler(NF; Δt = 1800), BackwardEuler(NF))`, i.e. backward Euler with the `Picard` solver for the implicitly routed soil variables and forward Euler for the rest, with a default time step of 30 min.
> The other models (`SnowModel`, `VegetationModel`, the surface models) keep `ForwardEuler` as their default.
> Tests and examples whose purpose is explicit time stepping pass `ForwardEuler` explicitly; the explicitly integrated components of a default `LandModel` (snow, vegetation, surface hydrology, and the surface energy balance) must remain stable at 30 min, which the implementation verifies.
>
> Implementation notes (2026-10-12):
> Mechanism: a two-argument `default_timestepper(::Type{<:AbstractModel}, NF)` falls back to `ForwardEuler(NF)`; `SoilModel` and `LandModel` specialize it to `IMEX(ForwardEuler(NF; Δt = 1800), BackwardEuler(NF))` and call it in their `@kwdef` defaults, so include order does not matter (committed in `cd4ee95b4`).
> The defaults of both models contain no value conditions on `temperature` or `pressure_head`; `SoilModel` defaults to `NoFlow` (only `internal_energy` implicit), and `LandModel` with vegetation defaults to `RichardsEq` (both soil variables implicit).
> Stability was checked with ERA5-Land point forcing (January 2024, 12 sites from Yakutia to the Congo, 30 layers with `Δz_min = 5 cm`, `Float32`) for 31 days, against IMEX at 60 s and 300 s and forward Euler at 10 s and 60 s.
> **Result: the explicit snow is not stable at 30 min, so the default is not yet verified.**
> Bare ground without snow: IMEX at 1800 s agrees with IMEX at 60 s (hourly RMS 0.02 to 0.14 K top soil, 0.1 to 0.6 K skin, 0.4 to 14 W/m² ground heat flux; monthly means within 0.06 K), so the explicit surface energy balance and ground heat flux are stable at 30 min.
> Bare ground with snow: at 1800 s the thin snowpack energy is unstable, with snow temperatures down to −104 °C at Yakutia (−46 °C at 60 s and 300 s), spurious melt at 0 °C under −23 °C air in western Siberia (SWE 2.6 mm instead of 41 mm after 31 days), hourly RMS snow temperature errors of 12 to 36 K, and ground heat flux errors of 44 to 110 W/m² at five cold sites; 300 s agrees with 60 s to within 0.08 K.
> Default `LandModel` with vegetation: every run (IMEX at 60, 300, 600, and 1800 s, forward Euler at 10 s) produces NaN in the vegetation at Yakutia after 139 h (air near −42 °C), and the vegetation carbon pool swings to ±10 to ±500 kg C/m² within hours at all steps, since the autotrophic respiration is about 4×10⁻⁴ kg C m⁻² s⁻¹ at 25 °C; this is independent of the time step, so the vegetation and canopy interception cannot be assessed until it is fixed.
> Forward Euler at 60 s fails after 3 h from explicit Richards flow near saturation, which is why IMEX at 60 s serves as the reference for the explicit components.
> Tests whose purpose is explicit stepping now pass `ForwardEuler`: value-condition tests (`boundary_conditions.jl`, the periodic and step heat diffusion tests, the explicit halves of the implicit-rejection tests), explicit-only checks (`cell_diffusion_timescale` and wizard tests in `adaptive_timestep.jl`, the forcing step in `soil_hydrology_tests.jl`, the immobile-water advection check, "run! SoilModel w/ ForwardEuler"), and `test/timestepping/imex.jl` gained a "Default timesteppers" test set.
> The default suite passes (9329 tests, no failures or errors), and the draft doc build shows no new warnings.
> Examples and doc pages with `PrescribedSurfaceTemperature` (quick start, Reactant page, `soil_heat_column`, `soil_heat_global`, `soil_heat_global_soilgrids`, `speedy_dry_land`, `differentiating_terrarium_reactant`) and the explicit wizard examples pass `ForwardEuler`; the time stepping and model pages state the new default.
> Not changed, for the reviewer: `test/reactant/setup.jl` (`:soil_heat_column`, `:soil_heat_column_stretched`, and `:soil_heat_global` now fail at `initialize` because of their value conditions; `:land_soil_snow` and `:land_default` now run IMEX `LandModel`s, which are untested under Reactant), `test/differentiability/soil_energy_diff.jl` and `snow_model_diff.jl` (read `get_timestepper(model).Δt`, which `IMEX` lacks), `soil_hydrology_diff.jl` and `prescribed_seb_diff.jl` (now IMEX under Enzyme), and the `:land`, `:land_no_vegetation`, and `:soil_heat` benchmark sweeps (`:soil_heat` now fails on its value condition).
> Follow-up (2026-10-12, reviewer request): the defaults are 1 h for `SoilModel` (`ForwardEuler(NF; Δt = 3600)`) and 15 min for `LandModel` (`Δt = 900`), set by the explicit snow.
> With the same 31-day ERA5-Land setup (bare ground with snow, IMEX at 60 s as reference), 15 min removed the 30 min failures (Yakutia minimum snow temperature −45.9 °C vs −45.8 °C, W-Siberia SWE 41.4 mm as in the reference) but a melting pack at NE-US still dropped to −92 °C for one hour (reference minimum −13.3 °C).
> The cause is the explicit Darcy meltwater outflow: near melt-out, a 900 s step drains more mass than the pack holds as liquid (SWE 20.5 → 3.7 mm in one hour against 19.5 → 12.1 mm in the reference), and since meltwater leaves with zero enthalpy, the drained ice leaves its latent deficit in the remaining pack.
> The wizard does not catch this: its snow diffusion timescale is about 2000 s at the conduction floor (2.5 cm), so it kept Δt = 900 s throughout, and it has no meltwater outflow timescale.
> At 10 min all snow sites track the reference: minimum snow temperatures within 2.4 K, final SWE within 0.3 mm, hourly RMS snow temperature 0.02–0.63 K and ground heat flux 0.2–5.4 W/m².

> **Revision 20, 2026-10-12. Implicit snow meltwater drainage (pending approval).**
> Reviewer request: integrate the snow meltwater outflow implicitly under IMEX, so that a step can never drain more water than the pack holds as liquid.
>
> *Motivation.*
> A meltwater outflow limit for the `TimeStepWizard` (`θ_liq W / M_r`) was implemented and reverted.
> The default saturated conductivity of `ConstantSnowHydraulics` is 20 m/h ([tarbotonSpatiallyDistributedEnergy1994](@cite)), so a fully melted pack drains its liquid within seconds: about 1 s for 6 mm of water (the floor at the minimum conduction thickness) and 4 s for 20 mm.
> In the 2-day ERA5-Land run, the wizard step averaged 2 s, with a minimum of 0.2 s.
> The explicit drainage is therefore stiff at any useful step, and only an implicit treatment removes the overshoot behind the NE-US spike of revision 19.
>
> *Formulation.*
> Drainage removes liquid water only, so the ice content `W_i = (1 − θ_liq) W` is unchanged by it, and the snow energy `Ū` is unchanged as well, since meltwater leaves at the zero-enthalpy reference (liquid water at 0 °C).
> The backward Euler step of the drainage, with snowfall, rain, sublimation, and the energy fluxes in the explicit stage (`W⁺`, `Ū⁺`), is a scalar equation per column for the liquid content `W_l = W − W_i`:
> `W_l = W_l⁺ − Δt M_r(θ_liq)`, with `θ_liq = W_l / (W_i + W_l)`.
> `W_i` follows from `W⁺` and `Ū⁺` through the closure (`θ_liq⁺` from the `FreeWater` relations, at fixed `Ū⁺`).
> With the liquid content at the capillary retention `W_c = θ_c W_i / (1 − θ_c)`, the excess saturation is `S* = (W_l − W_c) / W`, so the outflow `M_r = K_sat S*³` has the lagged-coefficient form `M_r = c (W_l − W_c)` with `c = K_sat S*² / W ≥ 0`.
> `M_r` increases monotonically with `W_l` and vanishes for `W_l ≤ W_c`, so the equation has a unique root in `[W_c, W_l⁺]` whenever `W_l⁺ > W_c`, and `Wⁿ⁺¹ = W⁺` otherwise.
> In a fully liquid pack (`Ū⁺ ≥ 0`, so `W_i = W_c = 0` and `S* = 1`), `M_r = K_sat` is constant until the pack is empty, and the root is `max(W_l⁺ − Δt K_sat, 0)`.
>
> *Solver.*
> The existing `Picard` solver, with `c` lagged at the latest iterate: each iteration solves `W_lᵐ⁺¹ = (W_l⁺ + Δt cᵐ W_c) / (1 + Δt cᵐ)`, a pointwise division.
> Every iterate is a convex combination of `W_l⁺` and `W_c`, so already after one iteration the step never drains ice and never drains below the capillary retention, at any `Δt`, and no safeguard or clamp is needed.
> In a fully liquid pack the iteration is `Wᵐ⁺¹ = W⁺ / (1 + Δt K_sat / Wᵐ)`, which converges to `W⁺ − Δt K_sat`, or decays toward zero when the pack empties within the step.
> Since `c` increases with `W_l`, the iterates alternate around the root; the iteration count of the solver (2 by default, shared with the soil) sets only the accuracy of the drainage, not its bounds.
>
> *Integration with the IMEX framework.*
> (1) `timestepping(::AbstractVariable{:snow_water_equivalent}, ::Union{LandModel, SnowModel}, ::AbstractIMEX) = Implicit()`; `snow_energy` stays explicit, and the defaults of `SnowModel` (`ForwardEuler`) are unchanged.
> (2) `compute_snow_water_tendency` receives the resolved `Timestepping` class of `snow_water_equivalent`, as the soil tendencies do, and omits `M_r` when it is `Implicit()`.
> (3) A new `DiagonalOperator <: AbstractLinearOperator` on the snow (`XY`) domain, holding the diagonal `1 + Δt cᵐ`, whose `solve!` divides the right-hand side by it pointwise; the snow's `implicit_operator` returns it.
> The snow implements the usual hooks: `linearize!` (diagonal and right-hand side `W_l⁺ + Δt cᵐ W_c` from the closure fields of the latest iterate), `implicit_update!` (`W = W_i + W_l`), and `implicit_closure!` (the snow closure at fixed `Ū⁺`, which keeps `W_i` unchanged).
> `implicit_operator(::LandModel, ...)` merges it with the soil components into one `SplitOperator`, and the `LandModel` hooks forward by the variable of the `Iteration` label to the snow or the soil; its position in the split does not matter, since the soil solves do not depend on it within the step (see (4)).
> (4) Water conservation with the soil: a new snow auxiliary `meltwater_flux` (m/s) holds the drainage of the last step, `(W⁺ − Wⁿ⁺¹) / Δt`, set by the implicit update.
> Under implicit routing, `snow_meltwater_flux` reads it, so the surface runoff and infiltration of the next step receive exactly the water that the snow lost, one step late.
> Under explicit routing, `compute_auxiliary!` sets it to `M_r(tⁿ)`, and the behavior is unchanged.
> The alternative, adding the drained water to `surface_excess_water` at the end of the step, has no lag but changes the routing (meltwater would always pond before infiltrating); the plan proposes the lag, pending the reviewer's decision.
> (5) The `SWE ≥ 0` clamp of the post-step hook (`enforce_snow_constraints!`) stays for explicit overshoot by sublimation, which is out of scope.
>
> *Expected effect on the default step.*
> Implicit drainage should remove the melt-out spike at 15 min.
> It does not make 30 min stable: the 30 min failures of revision 19 at Yakutia (−104 °C) occurred in a cold pack without melt, so they come from the explicit snow heat conduction, whose limit for a pack at the conduction floor is about 1000 s (`τ ≈ 2000 s`, forward Euler stable for `Δt < τ/2` with conduction to both the surface and the soil).
> The `LandModel` default therefore stays at 15 min, and an implicit snow heat conduction (per-column backward Euler of `snow_energy` with the surface and soil temperatures at `tⁿ`, conservative with the soil if the snow-soil flux is recomputed after the snow solve) is added to Future work.
>
> *Testing.*
> Unit tests of the snow implicit step: with many iterations the result satisfies the backward Euler equation to tolerance, and the default two iterations stay close to it; the ice content and `Ū` are unchanged; `θ_liq ≥ θ_c` after every iteration, including the first; a frozen pack is unchanged; a fully liquid pack drains `Δt K_sat` or tends to empty; Float32 type stability.
> `SnowModel` with IMEX: a melting pack at 900 s and 3600 s stays within physical temperature bounds, and water and energy are conserved over a melt-out (snow plus drained water, with `meltwater_flux` counted as in transit).
> `LandModel` with IMEX: water is conserved across a melt event (snow, soil, surface excess, runoff, and in-transit meltwater); the routing test of `imex.jl` gains `snow_water_equivalent`.
> The 31-day ERA5-Land check at 900 s: the NE-US spike is gone, and all snow sites track the 60 s reference.
> Enzyme and Reactant tests are run by CI only, unless their extensions or tests change.
>
> *Documentation.*
> `snow_mass.md` (implicit drainage under IMEX, with the formulation above), `time_stepping.md` (the list of implicitly routed variables, which currently says that snow variables always remain explicit), and this plan's Future work (implicit snow heat conduction).
>
> *Known limitations.*
> The one-step lag of the meltwater input to the soil; explicit stepping (`ForwardEuler`) keeps the overshoot, so explicit `LandModel` runs with melting snow need steps of the order of seconds near melt-out to be fully accurate; the snow heat conduction remains explicit.

## Problem description

Terrarium integrates every prognostic variable with fixed-step explicit schemes (`ForwardEuler`, `Heun`).
The `IMEX` type in `src/timesteppers/imex.jl` exists, but only as routing scaffolding: it assigns each prognostic *variable* to an explicit or an implicit sub-stepper, and no implicit sub-stepper exists.
The soil operators are the stiff part of the model:

- Heat conduction has a per-cell diffusive timescale `τ = Δz² C / κ`.
  For a 2 cm surface layer with `C ≈ 2×10⁶ J m⁻³ K⁻¹` and `κ ≈ 1 W m⁻¹ K⁻¹` this is roughly 800 s, which is the same order as the 300 s default step.
  Finer near-surface layers (needed to resolve the diurnal cycle in permafrost applications) make it quadratically worse.
- The Richardson–Richards equation has `τ = Δz² (∂θ/∂ψ) / K`, and `∂θ/∂ψ → 0` as the soil saturates, so the explicit step size collapses toward zero near saturation.
  This was documented as the motivating case in the adaptive-timestepping plan (`docs/dev/2026-08/2026-08-11_PLAN_01_adaptive_timestepping_cell_diffusion_timescale.md`), which made the timescale an explicit, testable diagnostic but did not remove the restriction.

The goal is an implicit-explicit stepping system in which the stiff vertical diffusion operators can be integrated implicitly while sources, boundary fluxes, vegetation, snow, and surface hydrology stay explicit, on CPU, CUDA, and Reactant, without giving up Enzyme differentiability.

## Background

### B0. Repository state at revision 2 (2026-10-08)

Changes since the base revision that bear on this plan:

- **Operator naming.** `ExplicitTwoPhaseHeatConduction` was renamed to `TwoPhaseHeatTransport` (`src/processes/thermodynamics/heat_conduction.jl`).
  The operator types are not modified by this plan; the explicit/implicit split is carried by the `Timestepping` trait passed into `compute_tendencies!` (R1).
- **Oceananigans 0.113.5** is pinned (`Project.toml` compat `0.113.5`, both Manifests).
  The `BatchedTridiagonalSolver` constructor, `solve!`, `get_coefficient`, `KrylovSolver`, `implicit_step!`, and `IMEXFluxBoundaryCondition` are all present with the signatures described in B2 (verified in the 0.113.1 through 0.113.6 depot copies).
  No compat work remains.
- **Grids.** `ModelIntegrator` and `StateVariables` no longer require an `AbstractLandGrid`; fields are allocated on whatever grid is given, and `variable_grid`/`ground_domain` are identity on a plain `AbstractGrid`.
  The tridiagonal scratch and coefficient buffers must therefore be allocated on `ground_domain(get_grid(model))`, which covers both cases.
- **Variable element types.** `Variable` gained an `eltype` (e.g. `SVector{3, NF}`), the explicit step kernels convert `Δt` with `eltype(eltype(tendency))`, and `test/timestepping/vector_eltype.jl` covers it.
  The implicit solvers are scalar only in this plan; the host-side constructor must reject routing a non-scalar variable to the implicit sub-stepper.
- **`Top()` semantics.** A `Top()` variable now defaults to `z = Face()` and resolves to `k = Nz + 1`; `Top(z = Center())` gives the uppermost cell.
  The XY explicit kernel indexes `k = 1`.
  Any implicit right-hand-side assembly that touches surface pools must follow these conventions.
- **Kernel fusion.** The fused-kernels plan (rev 7) abandoned soil tendency fusion after benchmarking and fuses auxiliaries only, so the implicit split of the soil tendency in R1 touches only the per-process `compute_tendencies!` launches, not a fused kernel.
- **Evapotranspiration sink.** `LandModel` now passes `surface_hydrology` into the soil `compute_tendencies!`, so the evapotranspiration sink is an ordinary explicit source term in the Richards tendency.
  It stays on the right-hand side in Path 2 and inside the residual in Path 1.
- **Surface energy balance.** The ground heat flux is now a first-class SEB sub-process with `Diagnosed` and `Prescribed` implementations (plan `2026-09-01_PLAN_prescribed_surface_energy_balance.md`, completed).
  The linearized surface flux (`λ = ∂G/∂T_ground`, deferred to a future PR by revision 18) would attach to the `Diagnosed` ground heat flux implementation.
- **Reactant.** `TerrariumReactantExt` now defines `convert_dt` for a `TracedRNumber` clock time, and the Reactant test registry gained time-varying input configurations.
  R7 (ticking the clock before a residual evaluation) is therefore supported inside the compiled loop.
- **Heun.** `HeunCache` allocates with `similar` rather than `deepcopy`; implicit caches should do the same.
- The `IMEX` routing tests now use plain `XY()` declarations on an `AbstractGrid`; the planned rewrite of those tests under R3 applies to the current versions.

### B1. The current Terrarium timestepping system

| File | Role |
|---|---|
| `src/timesteppers/abstract_timestepper.jl` | `AbstractTimeStepper{NF}`, `AbstractTimeStepperCache`, the `Timestepping` trait (`Explicit`/`Implicit`), `timestepping(var, model, ts)` (defaults to `Explicit()`), `initialize(ts, state, progvars, model)`, `get_cache`, and the `explicit_step!` kernels (`u += ∂u∂t Δt`) for XYZ, XY, and vertically sliced fields |
| `src/timesteppers/forward_euler.jl`, `heun.jl` | The two explicit schemes. Each `timestep!(integrator, ts, Δt, names)` calls `update_state!` itself, steps the fields in `names`, calls the model hook `timestep!(state, model, ts, Δt)`, then `closure!` |
| `src/timesteppers/imex.jl` | `AbstractIMEX`, `IMEX(explicit, implicit)`, `IMEXCache{classes}`; `timestep!` splits `prognostic_names` by resolved class (a `@generated` compile-time split) and calls each sub-stepper's `timestep!(integrator, ts, Δt, names)` in sequence, then ticks the clock once |
| `src/timesteppers/model_integrator.jl` | `ModelIntegrator <: Oceananigans.AbstractModel`; `timestep!(integrator, Δt)`, `run!`, `run_timesteps!` (host loop, overridden in `TerrariumReactantExt`), the Oceananigans `Simulation` glue |
| `src/timesteppers/cell_diffusion_timescale.jl`, `src/processes/soil/soil_diffusion_timescales.jl` | The `TimeStepWizard` diagnostics from the adaptive-timestepping plan |
| `src/solvers/` | Per-point nonlinear solvers used *inside* kernels: `ObjectiveFunction`, `FixedPointSolver`, `NewtonSolver{NF, iterations}` (fixed trip count, finite-difference derivative, Reactant-raisable), `RootSolver` (RootSolvers.jl). Used today by `ImplicitSkinTemperature` in `src/processes/surface/skin_temperature.jl` |

The step sequence of `update_state!` (`src/state_variables.jl`) is: reset tendencies, `update_inputs!` at the clock time, `compute_auxiliary!`, `compute_boundary_conditions!` (halo fills, the surface energy balance solve, and `compute_z_bcs!` which *adds* flux boundary conditions into the tendency fields), then `compute_tendencies!`.

Structural facts that constrain any implicit scheme:

1. **Prognostic variables are conserved quantities with closures.** Soil energy is stepped in internal energy `U` with the closure `U ↦ T` (`SoilEnergyTemperatureClosure`), and water in saturation with the closure `saturation ↦ pressure_head` (`SoilSaturationPressureClosure`).
   The diffusive operators act on the closure variables (`∂z(κ ∂z T)`, `∂z(K ∂z ψ)`), not on the prognostics.
   The `closure!` call happens *after* the explicit step (see `forward_euler.jl`), and the tendency kernels read the closure fields (`fields.temperature`, `fields.pressure_head`) computed in the previous `closure!`.
2. **Operators are expressed as flux divergences on the Oceananigans ground grid** via `∂zᵃᵃᶜ`, `∂zᵃᵃᶠ`, `ℑzᵃᵃᶠ` (`src/processes/thermodynamics/heat_conduction.jl`, `src/processes/soil/hydrology/soil_hydrology_rre.jl`).
   Hydraulic conductivity is upwinded as the minimum of the neighboring cell values (`darcy_flux`), which makes the Richards operator nonsmooth in the state.
3. **Boundary conditions.** Flux conditions enter the tendency through `compute_z_bcs!` and are therefore naturally explicit right-hand-side terms.
   Value and gradient conditions enter through halo fills of the closure fields.
   The soil top flux is the ground heat flux produced by the surface energy balance, which is itself a per-column Newton solve for skin temperature at each `compute_boundary_conditions!`.
4. **Discrete corrections exist and are applied post-step**: `adjust_saturation_profile!` (pushes oversaturated water upward and into `surface_excess_water`, called inside the hydrology `closure!`), and the snow model's `enforce_snow_constraints!` clamp in the model hook `timestep!(state, model, ts, Δt)` (`src/models/snow/snow_model.jl`, `land_model.jl`).
5. **Architectures.** CPU and CUDA run kernels eagerly through `launch!`.
   Under Reactant only `timestep!(integrator, timestepper, Δt)` is traced (inside a `@trace for` loop in `ext/TerrariumReactantExt/integrator.jl`), so every loop inside a step must have a trip count that is known at trace time, kernels must have no reachable throw path, and host-side reductions (norms for convergence tests) force synchronization.
   `NewtonSolver{NF, iterations}` was written precisely for this.
6. **Differentiability.** Enzyme reverse mode through `timestep!` is tested on CPU (`test/differentiability/*.jl`, `Duplicated(integrator, make_zero(integrator))`) and under Reactant (`test/reactant/autodiff.jl`, with `Reactant.Periodic` checkpointing).
   Enzyme is a *test* dependency only; `src/` never references it.

Limitations of the present `IMEX` scaffolding that the new design has to fix:

- It splits by **variable**, not by **term**.
  Real IMEX splits terms within one variable: the diffusive part of the soil energy equation is stiff, its boundary flux and source terms are not.
- Each sub-stepper calls `update_state!` on its own, so a two-sub-stepper `IMEX` evaluates all tendencies twice per step and the second sub-stepper sees a state that the first has already advanced (an uncontrolled Gauss–Seidel splitting).
- `is_adaptive` is declared but unused, and the clock is ticked only after both sub-steps, so no sub-stepper can evaluate inputs at `tⁿ⁺¹`.
- Multi-stage explicit schemes (`Heun`) have no hook where an implicit correction could be applied per stage.

### B2. What Oceananigans provides (investigation summary)

Investigated against the local checkout at `e8975f17d` (2026-09-18) and cross-checked against the released 0.111.0 pinned at drafting time and, at revision 2, against the now-pinned 0.113.5.

**Generic and directly reusable**

- `Oceananigans.Solvers.BatchedTridiagonalSolver` (`src/Solvers/batched_tridiagonal_solver.jl`).
  Constructed as `BatchedTridiagonalSolver(grid; lower_diagonal, diagonal, upper_diagonal, scratch, parameters, tridiagonal_direction = ZDirection())`.
  It allocates exactly one `Nx×Ny×Nz` scratch array. `solve!(ϕ, solver, rhs, args...)` launches one KernelAbstractions work item per `(i, j)` column and runs a serial Thomas sweep along `k` inside the kernel, in place (Oceananigans passes `ϕ === rhs`, which destroys the right-hand side).
  Coefficients are fetched through `get_coefficient(i, j, k, grid, coeff, p, direction, args...)`; 1D and 3D arrays work out of the box and **arbitrary coefficient functions are attached by defining `get_coefficient` on a singleton marker struct**, which is how `VerticallyImplicitDiffusionDiagonal` and friends work.
  A `abs(β) > 10 eps` guard skips the update when the system is not diagonally dominant.
  There are no CUDA-specific code paths; GPU support is entirely via `launch!`.
  The same pattern also underlies `ConjugateGradientPoissonSolver`'s `ColumnwiseTridiagonal*Diagonal` coefficients.
- `Oceananigans.Solvers.KrylovSolver` (`src/Solvers/krylov_solver.jl`, present in 0.111.0 and 0.113.5): a Krylov.jl wrapper (`:cg`, `:gmres`, `:bicgstab`, `:fgmres`, ...) over a single `AbstractField`, with matrix-free `linear_operator(y, x, args...)` and optional preconditioner callables.
  The `KrylovField` vector wrapper (kdot, knorm, kaxpy on `Field`s) is the template for a multi-field vector type.
  Krylov.jl reaches Terrarium only as a transitive dependency of Oceananigans.
- `IMEXFluxBoundaryCondition(Fₑ, λ)` (`src/BoundaryConditions/implicit_explicit_flux_boundary_condition.jl`): an affine flux `J = Fₑ + λ φ_boundary` whose linear coefficient `λ` is folded into the boundary cell diagonal by `boundary_flux_diagonal`.
  This is a ready-made linearized surface flux coupling.
  It is present in the pinned 0.113.5.
- The vertically implicit coefficient construction in `src/TurbulenceClosures/vertically_implicit_diffusion_solver.jl` (`ivd_upper_diagonal`, `ivd_lower_diagonal`, `ivd_diagonal`, `implicit_linear_coefficient`): off-diagonals are `−Δt κ_face / (Δz_center Δz_face)` masked to zero at `peripheral_node`s, and the diagonal is `1 − Δt L − (upper + lower)`, which makes the operator conservative and an M-matrix and gives a no-flux boundary for free.
  Explicit vertical fluxes are zeroed in the interior for closures with `VerticallyImplicitTimeDiscretization` and kept only at the two boundary faces (`abstract_scalar_diffusivity_closure.jl`), which is how double counting is avoided.
  These functions are short and tied to the closure argument convention (`closure, K, Val(id), clock, fields`), so they should be **copied and adapted**, not imported.
- `implicit_step!(field::Field, solver::BatchedTridiagonalSolver, ...)` is generic over `Field`, and `implicit_step!(field, ::Nothing, ...) = nothing` gives a free "no implicit solver" fallback.
- The CATKE sub-cycling loop (`time_step_catke_equation.jl`): recompute frozen coefficients and a linear sink coefficient, explicit sub-step, tridiagonal solve, repeat.
  It is the closest in-repo template for a nonlinear column (Richards, freeze–thaw) and confirms that Oceananigans offers **no Newton or Jacobian infrastructure**: all nonlinearity is handled by lagging coefficients.

**Not transferable as-is**

- The time stepper structs (`QuasiAdamsBashforth2TimeStepper`, `RungeKutta3TimeStepper`, `SplitRungeKuttaTimeStepper`) are generic, but every hook they call (`ab2_step!`, `rk_substep!`, `cache_previous_tendencies!`, `cache_current_fields!`) errors by default and is implemented per model under `src/Models/*`, with ocean-specific details (z-star scaling, free surface, CATKE skips).
  The "explicit update then `implicit_step!` on the same field with the same stage `Δt`" loop is written by each model, never provided generically.
  Terrarium's `ForwardEuler`/`Heun` already play the role of these structs; adopting Oceananigans' would mean replacing the `AbstractTimeStepper` hierarchy for no gain.
  What *is* worth borrowing is the `SplitRungeKuttaTimeStepper` stage structure (every stage is an Euler step from the cached state with `Δτ = Δt/β`, and the implicit solve uses that same `Δτ`), which is the cleanest host for a second-order IMEX scheme.
- The `Reactant` extension replaces AB2's `time_step!` because `Δt != clock.last_Δt` becomes a traced boolean; `BatchedTridiagonalSolver` has **no** Reactant specialization and is traced as a plain serial loop.
  The `Enzyme` extension only marks grid helpers inactive; **nothing in the implicit machinery is exercised under Enzyme or Reactant in Oceananigans' own tests**, and there is no adjoint rule for the tridiagonal solve.
  Its `ϕ === rhs` in-place use is a classic Enzyme aliasing hazard, so Terrarium should keep a separate right-hand-side field.
- `Δt` never lives in an Oceananigans timestepper; it flows `sim.Δt → time_step!(model, Δt) → implicit_step!(..., Δt) → get_coefficient(..., Δt, ...)`.
  Terrarium's `timestep!(integrator, ts, Δt, names)` already has this shape.
- NumericalEarth.jl's land components are hand-rolled explicit forward Euler kernels with clamping; they reuse only the `Simulation`/`Clock`/`AbstractModel` interface and no implicit machinery.

### B3. The stiff operators and how they linearize

**Heat conduction.** `∂U/∂t = ∂z(κ(T) ∂z T)` with `U = U(T)` given by the energy closure.
Define the apparent heat capacity `C_app = dU/dT = C + ρL θ dF/dT`, where `F(T)` is the liquid water fraction.
Two ways to make the diffusion implicit:

- *Solve in temperature with lagged apparent capacity* (Path 2): `C_app,k (T_k^{n+1} − T_k^n) / Δt = [∂z(κⁿ ∂z T^{n+1})]_k + G_k^{exp}`, a tridiagonal system in `T`.
  The energy is then updated **conservatively** from the implicit flux divergence, `U^{n+1} = U^n + Δt ([∂z(κⁿ ∂z T^{n+1})] + G^{exp})`, and `closure!` recovers the `T` consistent with `U^{n+1}`.
  When `C_app` varies inside the step the two temperatures differ; a small fixed number of Picard iterations (re-evaluate `C_app`, `κ` at the new iterate, re-solve) reduces the mismatch.
  The `FreeWater` freeze curve (Terrarium's default) has a Dirac `C_app` at 0 °C, so this form requires either a regularized capacity `C_app ≈ ρLθ / ΔT_reg` over a small interval (as several land models do) or a smooth freeze curve (`SFCC` types from FreezeCurves.jl).
  The energy-conserving Newton scheme of [dallamicoEnergyConservingFreezingSoil2011](@cite) (already in `references.bib`) is the reference for the smooth case.
- *Solve in energy with the closure inside the residual* (Path 1): `R(U) = U − Uⁿ − Δt f(T(U))`.
  The Jacobian `∂f/∂U = ∂f/∂T · ∂T/∂U` is well defined everywhere for `FreeWater` (`∂T/∂U = 0` in the phase-change interval, so the Jacobian reduces to the identity there), which is exactly why the enthalpy formulation is the robust choice for Newton.

**Richards' equation.** Mixed form `∂θ/∂t = ∂z(K(θ) ∂z Ψ)` with `Ψ = ψ_m(θ) + ψ_z + ψ_h`.
The standard mass-conservative implicit scheme is the modified Picard iteration of Celia, Bouloutas, and Zarba (1990): with `C = ∂θ/∂ψ` lagged at iterate `m`, `C^m (ψ^{m+1} − ψ^m) / Δt + (θ^m − θⁿ) / Δt = ∂z(K^m ∂z ψ^{m+1}) + G^{exp}`, tridiagonal in `ψ`, followed by the conservative update `θ^{n+1} = θⁿ + Δt (∂z(K^m ∂z ψ^{m+1}) + G^{exp})`.
The gravity and hydrostatic parts of `Ψ` are known and go to the right-hand side.
The saturated limit `C → 0` is harmless here because the diagonal keeps its `Δt K / Δz²` contributions as long as `K > 0`. [farthingNumericalSolutionRichards2017](@cite) and [kavetskiModelSmoothingStrategies2007](@cite) (both already cited) cover the smoothing and convergence issues of this iteration.

**Cross-couplings.** Energy and water are coupled through the freeze curve, the thermal properties, evapotranspiration sinks, and (from Phase 2) the advection of sensible heat by the water flux.
In both paths these couplings are treated by operator splitting between variables (evaluated at the lagged state) except in the fully coupled Newton option of Path 1.

**Energy advection (Phase 2).** The heat operator becomes `∂U/∂t = ∂z(κ ∂z T) − ∂z(ρ_w c_w T_face q)`, where `q` is the Darcy flux at the face (the existing `darcy_flux` kernel function of `RichardsEq`, zero for `NoFlow`) and `T_face` is the upwind cell temperature (`T_k` for downward flow across face `k+½`, `T_{k+1}` for upward).
The term is explicit and first-order upwind, so it belongs to `G^{exp}` under the implicit configuration and introduces an advective CFL limit `Δt ≲ Δz / |q|`, which for soil water fluxes is orders of magnitude weaker than the diffusive limit the implicit solve removes.
At the top face the infiltration flux carries heat at the top cell temperature `T_Nz` (upwind), and at the bottom face drainage carries heat at `T_1`; the surface energy balance is not consulted (see Known limitations).
Only the sensible heat of liquid water is advected, consistent with `U` being measured relative to liquid water at 0 °C, and ice is immobile.
Conservation: the advected energy flux is built from the same `q` that moves the water, so the column energy change equals the boundary conductive plus advective fluxes.

**Surface coupling.** The ground heat flux from the surface energy balance is a flux boundary condition and stays explicit in the first implementation.
`IMEXFluxBoundaryCondition` (with `λ = ∂G/∂T_ground` from the skin-temperature solve) is the later route to an implicitly coupled surface, and is standard practice in land surface models such as CLM and JSBACH.

## Approach 1: Jacobian-free Newton–Krylov (JFNK) with Enzyme JVPs

### Formulation

Backward Euler over the set `u` of prognostic fields assigned to the implicit solver (by default *all* of them, so that explicit terms, closures, and the surface energy balance are inside the solve):

```
R(u) = u − uⁿ − Δt f(u, tⁿ⁺¹) = 0,        f = tendencies produced by update_state!
J_k δ = −R(u_k),   u_{k+1} = u_k + δ,      J v = v − Δt (∂f/∂u) v
```

The Jacobian–vector product `(∂f/∂u) v` is computed without forming `J`:

- **Enzyme forward mode** (the requested path): `Enzyme.autodiff(Forward, update_state!, Duplicated(state, dstate), Const(model), Const(inputs))` with `dstate.prognostic := v` and the result read from `dstate.tendencies`.
  The shadow `dstate` is a second, preallocated `StateVariables` tree held in the stepper cache.
  Because `closure!` maps `U ↦ T` inside the residual, the shadow must also carry the closure fields, which `make_zero(state)` gives.
- **Finite-difference JVP** as a backend-independent fallback: `(f(u + εv) − f(u)) / ε`, one extra tendency evaluation, following the pattern already used in `NewtonSolver`.
  This is the standard JFNK approximation (Knoll and Keyes, 2004) and is what makes the method available on every backend before the Enzyme extension is proven.

The linear solve is a right-preconditioned restarted GMRES(m) (or BiCGSTAB); `J` is nonsymmetric because of the upwinded hydraulic conductivity, the closure Jacobians, and the multi-variable coupling.
Two implementation options for the Krylov iteration:

1. Krylov.jl through a Terrarium `StateVector` wrapper (a `NamedTuple` of `Field`s with `kdot`, `knorm`, `kaxpy!` summing over the tree), modeled on Oceananigans' `KrylovField`.
   This needs Krylov.jl as a **direct** dependency (it is currently transitive), and Krylov.jl's convergence loops are not Reactant-raisable.
2. A hand-written fixed-iteration GMRES(m) on `Field`s with KernelAbstractions kernels for the vector operations and `mapreduce` for dot products.
   More code, but no dependency change and raisable under Reactant (fixed trip counts, no host branching, as in `NewtonSolver`).

The plan proposes option 1 for CPU/CUDA if the dependency is approved, otherwise option 2 for everything; option 2 is required for Reactant in any case.

**Preconditioning is not optional.** For diffusion-dominated columns the unpreconditioned Krylov iteration count grows like `√(Δt/τ)`, which defeats the purpose.
The natural preconditioner is the per-column tridiagonal operator of Path 2 (block Jacobi over columns, exact for the lagged 1D diffusion), applied with `BatchedTridiagonalSolver`.
With it, 2–5 Krylov iterations per Newton step are expected.
This dependency is the main reason Path 2 should be built first.

### Assessment

Strengths:

- **Fully consistent nonlinear treatment**: phase change, saturation, evapotranspiration sinks, and the surface energy balance are solved together at `tⁿ⁺¹`, so there is no splitting error and no need for per-operator implicit code.
  Any process that provides `compute_tendencies!` is automatically eligible.
- Extends naturally to higher-order stiffly accurate schemes (trapezoidal, SDIRK) once the residual and JVP exist.
- The enthalpy formulation is the robust choice for `FreeWater` freeze–thaw.

Risks and costs:

- **Enzyme forward mode through `update_state!` is untested.** The suite tests reverse mode on CPU only.
  Forward mode through KernelAbstractions kernels on CUDA relies on KernelAbstractions' Enzyme extension and CUDA.jl's, which are less exercised than the CPU path.
  The `RootSolver` (RootSolvers.jl) inside the surface energy balance has data-dependent loops, which Enzyme handles but Reactant does not.
  A Phase 4 spike must settle this before the path is committed to.
- **Nested AD.** Differentiating a JFNK step in reverse mode for inverse modeling means reverse over forward.
  Enzyme LLVM supports this in principle, but it is a fragile combination; the correct long-term answer is a custom rule implementing the implicit-function-theorem adjoint (solve `Jᵀ λ = ∂L/∂u`), which belongs to the future Path 1 plan.
- **Reactant.** Convergence-tested Newton and Krylov loops cannot be raised.
  Fixed-iteration variants (`NewtonKrylov{NF, newton_iterations, krylov_iterations}`) are needed, as `NewtonSolver` already demonstrates, and every host-side norm inside the compiled program becomes a synchronization point.
  Enzyme forward mode inside a Reactant-compiled program uses Enzyme-MLIR, which is supported but unverified for this code.
- **Cost.** Per step, roughly `N_newton × (N_krylov × c_JVP + 1)` tendency evaluations with `c_JVP ≈ 2–3` for an Enzyme forward pass.
  With a good preconditioner and 2–3 Newton iterations this is 10–30 explicit-step equivalents per implicit step, so the step must be at least that much longer than the explicit limit to pay off.
  For Richards near saturation it always does; for heat conduction on coarse grids it may not.
- **Memory.** A full shadow state, `uⁿ` copies, the residual, and the Krylov basis (`m` state-sized vectors, `m ≈ 10–20`).
- **Nonsmooth residuals.** `adjust_saturation_profile!` and the snow clamps are discrete corrections; inside a residual they stall Newton.
  They must stay post-step (as now), and the upwinded `K` and `max(0, ·)` hydrostatic terms will still degrade Newton to linear convergence in places.
- **Dependency change.** Enzyme JVPs require Enzyme (or EnzymeCore) in `[weakdeps]` and a new `TerrariumEnzymeExt`.
  Per the repository rules this needs explicit approval.

### Implementation sketch

New files under `src/timesteppers/implicit/`:

- `newton_krylov.jl`: `NewtonKrylov{NF, JVP, Lin, Pre} <: AbstractImplicitSolver` with fields `jvp` (`FiniteDifferenceJVP()` or `EnzymeJVP()`), `linear_solver` (`GMRES(m; iterations)` settings), `preconditioner` (`nothing` or `TridiagonalPreconditioner()`), `newton_iterations`, and, for the host-only variant, tolerances.
  The Reactant-compatible variant carries all iteration counts as type parameters.
- `newton_krylov_cache.jl`: `NewtonKrylovCache` holding `u₀` and residual trees mirroring the state (same recursion as `HeunCache`), the Krylov basis, and the shadow state slot (filled by the Enzyme extension, `nothing` otherwise).
- `residual.jl`: `evaluate_residual!(residual, cache, integrator, u, Δt)`: write `u` into the prognostic fields, `closure!`, `update_state!` with the clock at `tⁿ⁺¹`, then `R = u − u₀ − Δt · tendencies` over the selected names via a generic axpy kernel.
- `jvp.jl`: `jvp!(out, ::FiniteDifferenceJVP, ...)` in base; `jvp!(out, ::EnzymeJVP, ...)` in `ext/TerrariumEnzymeExt/`.
- `state_vector.jl`: tree-wide `dot`, `norm`, `axpy!`, `copy!` over the selected prognostic names, built from `fastiterate` and `launch!`, allocation free.
- The tridiagonal preconditioner reuses Path 2's coefficient kernel functions and `BatchedTridiagonalSolver` unchanged.
- **Trait distinction (deferred from R1).** The Newton residual needs the full tendency including the interior diffusion term, whereas the Path 2 tendency omits it under `Implicit()`.
  The future plan must decide how `compute_tendencies!` learns which it is: a parameterized trait such as `Implicit{LinearlyImplicit}()` versus `Implicit{FullyImplicit}()` (preferred, since process methods then dispatch on one argument), or a query `omits_interior_diffusion(::AbstractImplicitSolver)` consulted when the model-level `compute_tendencies!` resolves the trait.

This sketch, the assessment above, and the spikes of Phase 4 are the inputs to the future Path 1 plan.
Nothing in this section is implemented in the present PR.

## Approach 2: Oceananigans-style linearly implicit vertical diffusion

### Formulation

Per implicit variable and per stage (IMEX Euler with `ForwardEuler` as the explicit partner):

1. `update_state!` once.
   Processes whose operator is marked implicit contribute **only their non-diffusive terms** to the tendency (sources, sinks, the two boundary-face fluxes via `compute_z_bcs!`), mirroring Oceananigans' zeroing of the interior explicit flux.
2. Explicit update for all variables with the existing `explicit_step!` kernels: `u* = uⁿ + Δt G^{exp}`.
   For implicitly stepped variables this is not a predictor in the predictor-corrector sense; `u*` is simply the right-hand side of the backward Euler system that step 3 solves.
   The scheme is a straight implicit solve with one tendency evaluation per step; no corrector pass follows.
   (The two-stage `Heun` pairing, deferred to a future PR by revision 17, is where a genuine predictor-corrector structure appears.)
3. For each implicit variable, `M ≥ 1` Picard iterations of: assemble lagged coefficients (`κ` or `K` at faces, `C_app` or `C = ∂θ/∂ψ` at centers), solve the tridiagonal system for the closure variable (`T` or `ψ`), update the prognostic conservatively from the implicit flux divergence, and re-evaluate the closure.
   `M = 1` is the linearly implicit scheme used by most land surface models; `M = 2–3` is the Celia modified Picard for Richards.
   The Picard loop is a fixed-point iteration on the nonlinear coefficients within one backward Euler step, not a multi-stage time scheme.
4. Model post-step hooks and `closure!` as today.

Coefficient assembly follows the Oceananigans `ivd_*` construction with the capacity on the diagonal:

```
lower_k = −Δt κ_{k−½} / (Δz^c_k Δz^f_{k−½})          (0 at the bottom boundary)
upper_k = −Δt κ_{k+½} / (Δz^c_k Δz^f_{k+½})          (0 at the top boundary)
diag_k  = C_app,k − lower_k − upper_k
rhs_k   = C_app,k Tⁿ_k + Δt G^{exp}_k
```

**Flux boundary conditions are the primary case and are required from Phase 1.** In Terrarium they are attached to the conserved prognostic (`SoilHeatFlux` and `GeothermalHeatFlux` on `internal_energy`, `InfiltrationFlux` on `saturation_water_ice`) and `compute_z_bcs!` adds them into the tendency during `update_state!`.
The implicit step takes that tendency as `G^{exp}`, so the ground heat flux enters the top row through the right-hand side while the masked off-diagonals keep the matrix itself flux-free, exactly as in Oceananigans.
The R1 split must therefore omit only the *interior* diffusive flux under `Implicit()` and leave the `compute_z_bcs!` contributions in place; the conservation tests check this by requiring the column energy and water change to equal the integrated boundary fluxes.
The boundary flux is evaluated at `tⁿ` (from the surface energy balance of that step) and held fixed over the step; a future PR adds the `λ T_top` linearization on top of this flux form for cases where the explicit surface coupling limits `Δt` (deferred by revision 18).
Value and gradient conditions are instead attached to the closure variables (`temperature`, `pressure_head`; e.g. `PrescribedSurfaceTemperature` and `PrescribedBottomTemperature` in `src/models/soil/soil_model_bcs.jl`) and reach the explicit operator through halo fills, which the implicit operator does not read.
They are handled as follows:

- A **gradient** condition on `T` at a boundary face is a known flux `−κ ∂T/∂z`, so it is added to the right-hand side exactly like a flux condition (Phase 1).
- A **value** condition `T_bc` at the top gives the boundary flux `κ (T_bc − T_Nz) / (Δz_Nz / 2)`, which is affine in the unknown `T_Nz`: the coefficient `κ / (Δz_Nz / 2)` is folded into `diag_Nz` and the `T_bc` part into the right-hand side (and symmetrically at the bottom).
  This is the same structure as `IMEXFluxBoundaryCondition`.
  Value conditions under the implicit operator are deferred to a future PR (revision 18); until then, a value condition on a closure variable routed to the implicit stepper is rejected at `initialize`.

### Reuse from Oceananigans

| Element | How |
|---|---|
| `BatchedTridiagonalSolver` | Used as-is, constructed on `ground_domain(grid)` with three Terrarium marker structs (`ImplicitDiffusionLowerDiagonal`, `ImplicitDiffusionDiagonal`, `ImplicitDiffusionUpperDiagonal`) and `Oceananigans.Solvers.get_coefficient` methods that call Terrarium kernel functions with `(fields, process, Δt, ...)` in `args`. Separate `rhs` field, never `ϕ === rhs` |
| `ivd_upper_diagonal`, `ivd_lower_diagonal`, `ivd_diagonal` | Copied and adapted (capacity on the diagonal, Terrarium argument convention); not imported |
| `implicit_step!(field, ::Nothing) = nothing` idiom | Adopted as `implicit_step!(…, ::Nothing, …) = nothing` for processes with no implicit operator |
| `IMEXFluxBoundaryCondition`, `boundary_flux_diagonal` | Future PR, deferred by revision 18 (available in the pinned 0.113.5) |
| `SplitRungeKuttaTimeStepper` stage pattern | Design template for the second-order IMEX (deferred to a future PR), not imported |
| `KrylovSolver` / `KrylovField` | Template for Path 1's multi-field Krylov vector |

### Assessment

Strengths:

- **Proven and cheap**: one Thomas sweep per column per implicit variable, `O(Nz)` work, no reductions, no data-dependent loops.
  Fits the land grid (independent columns) perfectly.
- **GPU and Reactant friendly**: fixed trip counts, no throw paths, no host synchronization; the Reactant raise of the serial `k` loop is verified in Phase 2 and has no structural blocker.
- **Enzyme friendly**: straight-line code per column, so reverse mode works without custom rules once the `ϕ === rhs` aliasing is avoided.
- **No new dependencies** for the core.
- Unconditionally stable for the linear diffusion; the step is then limited by accuracy and by the nonlinearity, not by `Δz²`.

Weaknesses:

- **Linearly implicit**: first order in time with lagged coefficients.
  Freeze–thaw and saturation nonlinearities are handled by Picard iterations, which converge linearly and can stall for the Dirac `FreeWater` capacity without regularization.
- **Per-process implicit code**: every process that wants implicit treatment must implement the coefficient kernel functions and the split of its tendency into explicit and implicit parts.
  Initially only soil heat conduction and Richards flow.
- **Only the vertical diffusion is implicit**; cross-couplings and the surface flux remain explicit (splitting error, and the surface energy balance can still limit `Δt` for thin top layers until the IMEX flux boundary condition is added).

## Common refactoring of the timestepping system

Both paths need the same changes to the core.
These are the substantive refactoring items.

- **R1. Term-level splitting by passing the `Timestepping` trait into `compute_tendencies!`.** The operator types are left unchanged.
  Instead, the resolved class of each prognostic variable, which the `IMEXCache{classes}` type parameter already holds, is handed to the process tendency methods: a type-stable helper `timestepping(state, var)` (or `timestepping(state, ::Val{name})`) reads it from `state.timestepper_cache` and returns `Explicit()` when the cache is not an `IMEXCache`.
  The model-level `compute_tendencies!(state, model::SoilModel)` and `compute_tendencies!(state, model::LandModel)` pass their timestepper to `SoilEnergyWaterCarbon`, which resolves the traits of `:internal_energy` and `:saturation_water_ice` with `timestepping(timestepper, state, Val(name))` (revision 11) and passes them to the soil energy and hydrology `compute_tendencies!`, which dispatch on them: under `Implicit()` the interior diffusive flux is omitted from the tendency (mirroring Oceananigans' vertically implicit closures) and only the boundary-face fluxes and sources remain.
  Process methods default the trait argument to `Explicit()`, so standalone use and the existing tests are unaffected.
  Only the trait is passed, never the timestepper itself, so kernels stay free of `AbstractTimeStepper` objects (the trait is `isbits`).
  New kernel function interface for implicit operators: `compute_implicit_diffusivity(i, j, k, grid, fields, proc, args...)` (face), `compute_implicit_capacity(i, j, k, grid, fields, proc, args...)` (center), and `compute_implicit_linear_coefficient` (for linear sinks, zero by default).
  Two clarifications on what "omit the interior diffusive flux" means:
    - In this plan `Implicit()` always means the linearly implicit tridiagonal treatment, so the interior diffusion term is omitted from the explicit tendency whenever the trait is `Implicit()`.
      A fully implicit Newton solver would instead need the full right-hand side; how the trait expresses that distinction is a Path 1 consideration (see Approach 1, Implementation sketch) and is not part of this PR.
    - Path 2 still evaluates the diffusive flux divergence once per Picard iteration, *after* the tridiagonal solve and at the implicit closure variable (`T^{n+1}` or `ψ^{m+1}`), to update the conserved prognostic.
      The existing flux-divergence kernel functions (`compute_energy_tendency`, `compute_volumetric_water_content_tendency`) are refactored so that `compute_tendencies!` and `implicit_step!` share them; the operator code is reused, it is only skipped inside `compute_tendencies!`.
- **R2. Routing has a single source of truth.** `timestepping(var, model, imex)` remains the only place a variable is assigned to the implicit sub-stepper.
  `SoilModel`/`LandModel` define it as `Implicit()` for `:internal_energy` and `:saturation_water_ice` whenever the model's timestepper is an `AbstractIMEX` with an implicit sub-stepper, and users override it per variable if they want, e.g., implicit heat but explicit water.
  The user therefore selects only the timestepper; no operator or process type changes hands.
- **R3. Single `update_state!` per stage, orchestrated by the IMEX.** `timestep!(integrator, ::AbstractIMEX, Δt)` performs `update_state!`, calls the explicit sub-stepper's stage for its names *without* re-evaluating tendencies, then `implicit_step!(integrator, implicit_ts, Δt, names)`.
  This requires splitting the explicit steppers' `timestep!` into `explicit_stage!` (the update only) and the outer driver that calls `update_state!`, hooks, and `closure!`.
  The mock-based tests in `test/timestepping/imex.jl` are updated accordingly.
- **R4. Stage hook for multi-stage explicit schemes.** Phase 1 supports only `ForwardEuler` as the explicit partner (IMEX Euler).
  Deferred to a future PR (revision 17): a per-stage `implicit_step!` call inside `Heun` under an IMEX (predictor stage with implicit correction, corrector with averaged explicit tendencies and a second implicit correction), following the Oceananigans "Euler from cached state with the stage `Δτ`" pattern.
- **R5. Caches.** `initialize(ts, state, progvars, model)` already receives the model, so implicit caches can allocate the tridiagonal scratch on `ground_domain(get_grid(model))`, a right-hand-side field and coefficient buffers per implicit variable, and, for Path 1, the shadow state and Krylov basis.
  All caches must be `Adapt`-able as `HeunCache` is, allocate with `similar` as `HeunCache` now does, work on a plain `AbstractGrid` as well as a `LandGrid`, and the host-side constructor must reject implicit routing of variables with non-scalar element types (B0).
- **R6. `Δt` plumbing.** `Δt` is passed down to the coefficient functions (via `BatchedTridiagonalSolver` `args`) rather than stored, matching Oceananigans and the existing `timestep!(integrator, ts, Δt, names)` signature.
- **R7. Clock semantics.** Path 2 evaluates explicit terms at `tⁿ` (Oceananigans convention).
  Path 1 evaluates the residual at `tⁿ⁺¹`, so the IMEX driver must be able to `tick!` before the residual evaluation and the traced Reactant clock must tolerate that ordering; the extension's `convert_dt` for traced times (B0) makes input updates at a traced `tⁿ⁺¹` possible.
  The single `tick!` in `timestep!(integrator, ts, ::Timestepping, Δt)` moves into the IMEX driver.
- **R8. Post-step hooks stay outside the solve.** `timestep!(state, model, ts, Δt)` (snow clamps) and `closure!` (including `adjust_saturation_profile!`) run after the implicit solve, never inside a residual.
- **R9. Wizard interplay.** `cell_diffusion_timescale` returns `Inf` for operators of class `Implicit()` so the `TimeStepWizard` no longer throttles `Δt` by the stiff limit; a later diagnostic may bound `Δt` by the Picard/Newton convergence instead. `is_adaptive` is removed (revision 9): adaptivity is provided by the Oceananigans `TimeStepWizard`, so the trait was redundant.
- **R10. Naming.** The Terrarium function is `implicit_step!`, defined as Terrarium's own function (Terrarium imports `Oceananigans.TimeSteppers` selectively and does not import its `implicit_step!`, so there is no method clash; the Oceananigans function keeps its own signature).

Proposed user-facing API (both paths behind one implicit stepper with a pluggable solver):

```julia
# Path 2 (default): lagged-coefficient tridiagonal solve, M Picard iterations
ts = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver = TridiagonalPicard(iterations = 1)))

# Path 1: fully coupled Newton–Krylov, tridiagonal preconditioner, Enzyme or finite-difference JVP
ts = IMEX(ForwardEuler(NF), BackwardEuler(NF; solver = NewtonKrylov(jvp = EnzymeJVP(), preconditioner = TridiagonalPreconditioner())))

# Operators and processes are unchanged; the IMEX routes soil energy and water to the implicit stepper (R2)
model = SoilModel(grid; timestepper = ts)

# Optional per-variable override: keep soil water explicit
Terrarium.timestepping(::AbstractVariable{:saturation_water_ice}, ::SoilModel, ::AbstractIMEX) = Explicit()
```

`BackwardEuler` declares `timestepping(::BackwardEuler) = Implicit()` and slots into the existing `IMEXCache` routing.
`AbstractImplicitSolver` is the new extension point, and `TridiagonalPicard` and `NewtonKrylov` are its two implementations.

## Recommendation

**Implement Path 2 in this PR. Defer Path 1 to a future plan and PR, with both eventually exposed as solvers of the same `BackwardEuler` stepper.**

1. Path 2 is low risk, adds no dependencies, runs on all three backends with no structural obstacle, differentiates in reverse mode without custom rules, and removes the dominant stiffness (vertical diffusion in the soil column).
   It is the right default for users.
2. Path 2 produces exactly the preconditioner that makes Path 1 practical.
   Without it, Path 1 would spend tens of Krylov iterations per Newton step on diffusion-dominated columns, so Path 1 cannot sensibly be built first.
3. Path 1's feasibility hinges on questions that cannot be answered on paper: Enzyme forward mode through `update_state!` on CUDA, Enzyme forward inside a Reactant program, and nested reverse over forward.
   The spikes of Phase 4 settle them once Path 2 exists to compare against, and their results feed the future Path 1 plan together with the Enzyme weak-dependency and Krylov.jl decisions.
   Path 1 then becomes the advanced option for fully coupled freeze–thaw, saturated infiltration, and implicitly coupled surface energy balance, and the natural host for higher-order stiff schemes.

## Phased implementation

Phases 1 through 4 are the scope of this PR.
Path 1 is deferred to a future plan.

**Phase 1: core refactor and implicit heat conduction (Path 2).**

- R1–R3, R5–R10.
  `BackwardEuler`, `AbstractImplicitSolver`, `TridiagonalPicard`, marker structs and `get_coefficient` methods, `implicit_step!`, the `timestepping(state, var)` helper.
- Implicit heat conduction: `compute_tendencies!` for `SoilThermodynamics` dispatching on the trait, coefficient kernel functions for `TwoPhaseHeatTransport`; explicit tendency reduced to boundary faces and sources; apparent heat capacity kernel function with a documented regularization parameter for `FreeWater` and the analytic `C_app` for `SFCC` curves; conservative energy update.
- Flux boundary conditions (`SoilHeatFlux`, `GeothermalHeatFlux`) verified to enter the implicit step through the right-hand side, with conservation tests against the integrated boundary fluxes.
- `SoilModel` routing; tests; docs.

**Phase 2: implicit Richards flow and the coupled land model.**

- `RichardsEq` implicit variant with Celia modified Picard (`iterations = 2` default), gravity and hydrostatic terms on the right-hand side, conservative saturation update, interaction with `adjust_saturation_profile!` (post-step) verified by mass balance.
- Energy advection by the water flux in `TwoPhaseHeatTransport` (revision 5): an explicit upwind advective heat flux kernel function `advective_heat_flux(i, j, k, grid, fields, energy, hydrology, constants)` at faces using `darcy_flux` and the upwind temperature, added to `compute_energy_tendency` for both explicit and implicit configurations (it is part of `G^{exp}` under `Implicit()`); zero for `NoFlow` hydrology by dispatch; boundary faces carry the infiltration and drainage fluxes at the adjacent cell temperature. `cell_advection_timescale` returns `min(Δz / |q|)` for the energy equation so the `TimeStepWizard` can honor the advective CFL.
- `LandModel` routing (snow, vegetation, surface hydrology stay explicit); GPU tests; Reactant registry configuration in `test/reactant/setup.jl` (the tridiagonal solve has fixed trip counts and no throw paths, so no structural obstacle is expected; compile time is measured here).

**Phase 3: wizard diagnostics.**

- (Deferred to a future PR by revision 17: R4, `Heun` as explicit partner with per-stage implicit corrections (IMEX trapezoidal).)
- (Deferred to a future PR by revision 18: `IMEXFluxBoundaryCondition` for the ground heat flux, and value boundary conditions on the closure variables under the implicit operator.)
- Wizard diagnostics for implicit runs: no throttling by implicit terms, and time scales of the remaining explicit terms (explicit energy advection, explicit water outflow that could overdraw a cell).

**Phase 4: feasibility spikes for Path 1 (no production code).** Run after Path 2 is complete so the results can be compared against a working implicit stepper; recorded in the future Path 1 plan.

- (a) Enzyme forward-mode JVP through `update_state!` for a `SoilModel` column on CPU; compare with a finite-difference JVP.
- (b) The same on CUDA.
- (c) Enzyme forward mode inside a Reactant-compiled program for the same JVP.
- (d) Enzyme reverse mode through the Phase 1 `implicit_step!` (tridiagonal solve with a separate `rhs` field), as a baseline for the nested-AD question.

Each phase ends with the full test suite, the Enzyme test set, and a draft doc build.

**Deferred to a future plan and PR (Path 1).** For the record, the work previously planned here:

- `NewtonKrylov` with `FiniteDifferenceJVP` in base, GMRES(m) (Krylov.jl if the dependency is approved, otherwise a hand-written fixed-iteration version), `TridiagonalPreconditioner` reusing the Phase 1–2 coefficients, state-tree vector operations, and the trait distinction described in the Approach 1 implementation sketch.
- `TerrariumEnzymeExt` with `EnzymeJVP` (weak dependency, to be approved then).
- CPU/CUDA tests: JVP agreement, Newton convergence rates, agreement with Path 2 in the linear limit.
- Fixed-iteration `NewtonKrylov` variant for Reactant, raise tests, Enzyme reverse tests, and an implicit-function-theorem adjoint rule for `implicit_step!` if nested AD proves fragile.

## Testing and verification

- **Linear consistency**: for constant `κ`, `C` on a small column, assemble the dense `I − Δt L` from finite differences of the explicit operator and check that one implicit step matches the dense solve to roundoff (both solvers).
- **Order and stability**: the existing `examples/extending/linear_heat_conduction.jl` analytic solution; first-order convergence in `Δt` for IMEX Euler; stable and accurate runs with `Δt` 100× the explicit limit.
- **Conservation**: total column energy and water change equals the integrated boundary fluxes to roundoff, for every solver, with and without Picard iterations, and with `adjust_saturation_profile!` active.
- **Freeze–thaw**: a Stefan-type freezing column compared against a fine-step explicit reference with `FreeWater` (regularized) and an `SFCC` curve; the Newton path must show quadratic convergence on the smooth curve.
- **Energy advection**: a uniform-temperature column with steady downward flow must keep `T` uniform (advection of a constant); a warm infiltration pulse into a cold column must conserve total energy to roundoff and move the thermal front at the thermal advection velocity `C_w q / C` (revision 7; not the pore velocity `q/θ`); a coupled freeze–thaw infiltration case that is unstable without advection must run stably with it.
- **Richards**: the Celia et al. (1990) infiltration test against a fine explicit reference, including a near-saturated column where explicit stepping is impractical.
- **Routing**: `test/timestepping/imex.jl` updated for the single-`update_state!` orchestration; since revision 15 it also holds the routing and implicit cache and operator structure tests of the soil variables.
- **Test layout (revision 15)**: the physics tests of the implicit soil processes live with the process tests, which `test/soil/soil_model_tests.jl` includes: implicit heat conduction and heat advection in `test/soil/soil_energy_tests.jl`, the implicit Richards equation in `test/soil/soil_hydrology_tests.jl`, and the Stefan benchmark in `test/soil/soil_freeze_thaw_tests.jl`; the `TridiagonalOperator` with upwind advection is tested in `test/solvers/test_solvers.jl`.
- **JVP**: Enzyme versus finite-difference JVP agreement on `SoilModel` and `LandModel` (Phase 4).
- **Differentiability**: `test/differentiability/` gains implicit `SoilModel` cases for both solvers; gradients cross-checked with FiniteDifferences as in the existing tests.
- **Reactant**: `test/reactant/correctness.jl` gains implicit configurations; compiled results match the CPU run.
- **GPU**: implicit configurations in the CUDA test set.
- Benchmarks are added to `benchmark/model_configurations.jl` only when explicitly requested.

## Documentation changes

- `docs/src/running/time_stepping.md`: new "Implicit-explicit time stepping" section (formulation, choosing operators and solvers, Picard iterations, the `FreeWater` regularization caveat, cost guidance, Reactant and AD notes); update the "planned" warning in the adaptive section.
- `docs/src/extending/implementing_processes.md`: the implicit operator interface (R1) with the kernel-function signatures.
- `docs/src/solvers/solvers.md`: `AbstractImplicitSolver`, `TridiagonalPicard`, `NewtonKrylov`.
- Process pages for soil energy and soil hydrology: implementation sections for the implicit operators with non-canonical docstrings and kernel-function lists.
- `docs/src/references.bib`: add Celia, Bouloutas, and Zarba (1990, Water Resources Research 26, 1483–1496); Knoll and Keyes (2004, Journal of Computational Physics 193, 357–397); Ascher, Ruuth, and Spiteri (1997, Applied Numerical Mathematics 25, 151–167) for the IMEX Runge–Kutta background.
  Dall'Amico et al. (2011), Farthing and Ogden (2017), and Kavetski and Kuczera (2007) are already present.

## Known limitations

- Path 2 is first order with lagged coefficients; Picard iterations reduce but do not remove the nonlinear error within a step.
  The Dirac `FreeWater` capacity requires regularization in the temperature form.
- Energy advection is explicit and first-order upwind, so it is diffusive and imposes an advective CFL limit `Δt ≲ C Δz / (C_w |q|)`, which for highly conductive soils can be the binding limit of an implicit run; infiltrating water enters at the top cell temperature rather than the skin or rain temperature, so rain heat is not exchanged with the surface energy balance, and water removed by evapotranspiration or routed to the surface excess pool carries no heat.
- The implicit Richards step lags the hydrostatic head and the water table within a step and bounds the moisture capacity from below (`minimum_moisture_capacity`); with a fixed number of Picard iterations the result is not converged when a large step saturates part of the column.
- Only vertical diffusion in the soil is implicit in Phases 1–3.
  Snow, vegetation, surface hydrology, and the energy–water cross-couplings remain explicit, and the surface energy balance stays an explicit flux (implicit surface coupling deferred by revision 18).
- Value conditions on the closure variables (`temperature`, `pressure_head`), such as `PrescribedSurfaceTemperature`, are not supported by the implicit operator (deferred by revision 18); flux conditions on the conserved prognostics and gradient conditions on the closure variables are supported.
- Path 1 on Reactant requires fixed iteration counts and is not adaptive; Path 1's Enzyme JVP requires a new weak dependency; nested AD through Path 1 is not guaranteed until the adjoint rule from the future Path 1 plan exists.
- Adaptive `Δt` under Reactant remains out of scope, as in the adaptive-timestepping plan.

## Future work

- Implicit-function-theorem adjoints for `implicit_step!` (efficient inverse modeling through implicit steps).
- Implicit surface coupling: an `IMEXFluxBoundaryCondition` for the ground heat flux with `λ = ∂G/∂T_ground` from the skin temperature solve, attached to the `Diagnosed` ground heat flux sub-process of the SEB, deferred from Phase 3 by revision 18; first measure whether the explicit ground heat flux limits `Δt` of a `LandModel`.
- Value boundary conditions on the closure variables (`temperature`, `pressure_head`) under the implicit operator: the boundary coefficient on the diagonal and the boundary value in the right-hand side, with the boundary flux in the conservative update; deferred from Phase 3 by revision 18. The explicit boundary tendencies must skip faces whose flux the operator carries, preferably by dispatch on the boundary condition type rather than by flags threaded through the tendency kernels.
- Second-order IMEX (R4): `Heun` as explicit partner with per-stage implicit corrections (IMEX trapezoidal), deferred from Phase 3 by revision 17, which also introduces the stage hook.
- Stiffly accurate higher-order IMEX Runge–Kutta pairs (ARS(2,2,2), ARS(4,4,3)) on top of the stage hook.
- Implicit snow heat conduction (revision 20): per-column backward Euler of `snow_energy` with the surface and soil temperatures at `tⁿ`, conservative with the soil if the snow-soil flux is recomputed after the snow solve; needed for `LandModel` steps beyond about 15 min, since the explicit conduction of a thin pack is limited to about 1000 s.
- Implicit treatment of the lateral/2D couplings if they are ever added.
- A convergence-based step controller for the implicit solvers to replace the diffusive CFL wizard.

### Oceananigans advection operators for soil energy advection

**Verdict: partially feasible.**
The explicit interior-face fluxes could reuse Oceananigans' reconstruction schemes, but the implicit operator and the boundary faces should stay our own.
Source references are to Oceananigans 0.113.5 (Terrarium's Manifest), whose cited `src/Advection` files match the local v0.113.0 checkout; `test/` references are to that checkout.

Relevant operators and evidence:
- `advective_tracer_flux_z(i, j, k, grid, scheme, td, W, c)` (`src/Advection/upwind_biased_advective_fluxes.jl:115`, `centered_advective_fluxes.jl:37`) returns `Azᶜᶜᶠ * W[i, j, k] * ĉ`, where `ĉ` is reconstructed upwind of `bias(W)` by `_biased_interpolate_zᵃᵃᶠ`; `div_Uc` (`src/Advection/tracer_advection_operators.jl:31`) wraps it for a `(u, v, w)` NamedTuple that accepts `ZeroField`s for `u` and `v`.
- First-order upwind is `UpwindBiased(order = 1)`; higher orders are `UpwindBiased(order = 3, 5, …)` and `WENO(order = 5, …)`, and a `WENO` must first pass through the internal `materialize_advection(scheme, grid)` (`src/Advection/materialize_advection.jl`), otherwise its weight division throws a `MethodError`.
- A check on a `ColumnGrid` (`Periodic, Flat, Bounded`, halo 2×0×3) with `W = water_flux` and `c = temperature` reproduced `advective_heat_flux / C_w` exactly at all interior faces for `UpwindBiased(order = 1)`, and `WENO` up to order 9 ran with the default z halo of 3, because Bounded z switches to buffer schemes near the boundaries (`src/Advection/topologically_conditional_interpolation.jl:120`).
- Since the flux is linear in `W` and does not involve the bulk capacity `C`, no scaled "velocity" is needed: pass `water_flux` as `W`, multiply by the constant `C_w`, and the `internal_energy` update stays conservative for variable `C`.
- Oceananigans does have implicit vertical advection: `AdaptiveImplicitVerticalAdvection` (`src/Utils/time_discretizations.jl:36`, after Shchepetkin 2015) splits `w = wᵉ + wⁱ` by the local Courant number, advects `wᵉ` explicitly with the high-order scheme (`src/Advection/adaptive_implicit_vertical_advection.jl:59`), and puts `wⁱ` as first-order upwind into the vertically implicit diffusion tridiagonal (`src/Advection/implicit_vertical_advection.jl:102–150`, `src/TurbulenceClosures/vertically_implicit_diffusion_solver.jl:235`).
- With `density = nothing`, those coefficients equal our `compute_implicit_advection_*` in `src/solvers/linear_operators.jl`, but they are tied to a scheme carrying a host-mutated `Ref` `Δt` and to Oceananigans' solver layout.

Required adaptations:
- Keep our boundary faces: Oceananigans reads the boundary-face upwind value from the temperature halo, not the inflow temperature, so only faces `k = 2, …, Nz` would come from Oceananigans, with our own divergence instead of `div_Uc`, and `compute_boundary_advective_energy_tendency` and `compute_boundary_advected_heat!` unchanged.
- Restrict higher-order schemes to SSP stepping (`Heun`) at a reduced Courant number, since `ForwardEuler` is unstable for `WENO`, for `UpwindBiased(order ≥ 3)`, and for `Centered` without enough conduction.
- Pass `minimum_buffer_upwind_order = 1` or accept that, by default, `WENO` falls back to `Centered(order = 2)` at the faces next to the boundaries (`src/Advection/weno_reconstruction.jl:135–137`), exactly where infiltration fronts enter.

Blockers and risks:
- Only `Centered`, `UpwindBiased`, `WENO`, and `AdaptiveImplicitVerticalAdvection` are exported at top level; `advective_tracer_flux_z` is exported only from `Oceananigans.Advection`, and `materialize_advection`, the `td` argument, and the implicit coefficient functions are internals that may change between minor releases.
- Oceananigans' Reactant tests mark `raise = true` as broken for `WENO` on Bounded topologies (`test/test_reactant_correctness.jl:187`, `:261`), which is the compile mode of Terrarium's reverse-mode path, and `ReactantState` computes the `WENO` weights with `Float32` division (`ext/OceananigansReactantExt/Models.jl:48`).
- Enzyme differentiates `WENO` tracer advection in `test/test_enzyme.jl:191`, so Enzyme alone is not a blocker, but the schemes are not monotone, so temperature can overshoot near freezing fronts.
- The reconstruction coefficients assume uniform spacing (`src/Advection/upwind_biased_reconstruction.jl:26`), so the formal order drops on our exponentially stretched soil grids.
- The benefit is modest: the numerical diffusion of first-order upwind relative to conduction is the cell Péclet number over two, `C_w |q| Δz / (2λ)`, which is about 0.01 for drainage (`q ≈ 10⁻⁷` m/s) and only exceeds 1 during strong infiltration (`q ≳ 10⁻⁵` m/s).

**Recommended path:** keep our own first-order upwind as the default for both explicit and implicit stepping.
If sharper fronts are ever needed, add an explicit-path option with an `advection` scheme on `SoilThermodynamics` (default `UpwindBiased(order = 1)`) that evaluates interior fluxes as `C_w * advective_tracer_flux_z(…, ExplicitTimeDiscretization(), water_flux, temperature)`, then `WENO`, gated on a Reactant `raise = true` test and an Enzyme gradient test.
For implicit and IMEX stepping, reimplement the Shchepetkin split in our own `TridiagonalOperator` (explicit high-order flux with `wᵉ`, implicit first-order upwind with `wⁱ`) rather than reusing Oceananigans' coefficient functions, which keeps the M-matrix property.

## Reviewer decisions

1. **Approved:** scope is Path 2 and the core refactoring in this PR (Phases 1–3), spikes in Phase 4, Path 1 deferred to a future plan.
2. **Decided:** `FreeWater` under the implicit heat operator uses a regularized apparent heat capacity `C_app ≈ C + ρLθ / ΔT_reg` over a phase-change interval of width `ΔT_reg`, exposed as a documented parameter of the implicit solver; `SFCC` curves use their analytic `C_app`.
3. **Approved:** type names `BackwardEuler`, `AbstractImplicitSolver`, `TridiagonalPicard`.

Deferred with Path 1: Enzyme (or EnzymeCore) as a weak dependency with `TerrariumEnzymeExt`, and Krylov.jl as a direct dependency versus a hand-written fixed-iteration GMRES(m).
