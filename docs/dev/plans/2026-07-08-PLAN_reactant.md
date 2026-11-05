# Plan: Reactant-compatible Terrarium

> ## Session 4 (2026-07-10) — array-`z` / stretched spacing unblocked (upstream fixed)
>
> The deferred "non-range `z` / `ExponentialSpacing` under Reactant" TODO (below) is **DONE**.
> - **Upstream fix confirmed.** Oceananigans `main` (post-0.110.8-tag) adapts the `RectilinearGrid`
>   extents in its `Adapt.adapt_structure` rule (`src/Grids/rectilinear_grid.jl`:
>   `Adapt.adapt(to, grid.Lx)` etc., previously passed through unadapted) — exactly the surviving
>   traced-scalar candidate the issue diagnosis flagged. Verified with the Case A/B reproduction:
>   a `RectilinearGrid` with `Vector`/range `z` now traces through the halo kernel and time-steps
>   under Reactant (Reactant 0.2.270). The registry-tagged 0.110.8 does **not** have the fix.
> - **Terrarium change (minimal, per request).** Reverted the z-coordinate workaround to `main`:
>   removed `z_coordinates` (both the generic and the `UniformSpacing` tuple methods) and restored
>   the inlined `vcat(-reverse(cumsum(...)), 0)` `Vector` form in `ColumnGrid`/`ColumnRingGrid`.
>   `git diff main` for those three files is now **empty**. Updated the stale
>   `AGENTS.md` Reactant bullet and the `test/reactant/setup.jl` comments.
> - **Coverage.** Added a `:soil_heat_column_stretched` (`ExponentialSpacing`) config to
>   `test/reactant/`; the full suite is green (**45/45** correctness incl. the new stretched
>   config + **6/6** autodiff) against Oceananigans `main`.
> - **Env pin.** `test/reactant/Project.toml` initially gained `[sources] Oceananigans = {rev =
>   "main"}` while the fix was unreleased. **Update (2026-07-13):** Oceananigans **0.110.9** was
>   released with the fix; the `[sources]` pin was dropped and `[compat]` bumped to
>   `Oceananigans = "0.110.9"`. Suite re-verified green (45/45 + 6/6) on the tagged release.
>
> **Also this session — build-on-CPU initialization removed (simplification).** Investigated whether
> the CPU-twin build/transfer in `transfer.jl` was still needed, or whether `@jit initialize!` would
> work. Findings (empirical, at Oceananigans main / Reactant 0.2.270):
> - `@jit initialize!` does **not** work: the user field initializers call `set!(field, function)`,
>   whose Oceananigans device impl (`set_to_function!` on a `ReactantField`) detours through the CPU
>   — inherently eager, un-traceable (fails `getindex(::OffsetVector{TracedRNumber}, ::UnitRange)`
>   method ambiguity inside a trace).
> - The Phase-0 premise ("eager KA launches fail on ReactantBackend outside a compiled region") is
>   **stale**: eager `set!(field,f)`, `fill_halo_regions!`, and the full `initialize!(integrator)`
>   (incl. `compute_auxiliary!`'s closure kernels) all run **eagerly** on a device-allocated state now.
> - So the CPU **twin model** is unnecessary. Removed `rebuild_model`/`to_device`/
>   `copy_state_data!`/`_copy_group!`/`_copy_clock!`/`_scalar` and the now-dead ext imports. Verified:
>   device-vs-CPU initial state **bit-identical** (`max_abs_diff = 0.0`) for column + global configs.
> - **Final design (the `default_clock` hook the plan anticipated, §2.1/Phase B).** The ext no longer
>   specializes `initialize` at all. `src/timesteppers/model_integrator.jl` gains
>   `default_clock(model) = Clock(time = zero(NF))`, and `initialize`'s `clock` default is now
>   `default_clock(model)`. `TerrariumReactantExt` overrides just
>   `Terrarium.default_clock(::ReactantModel) = Clock(get_field_grid(get_grid(model)))` (the traced
>   `ConcreteRNumber` clock), so the generic `Terrarium.initialize` transparently produces a device
>   integrator. `transfer.jl` is now a single one-line method (no `@invoke`, no recursion hazard).
>   CPU regression (timestepping + state_variables): 42/42; full Reactant suite still **45/45 + 6/6**.
>   AGENTS.md Reactant-compat design paragraph updated to "eager device initialization".
>
> ## Session 3 plan (2026-07-06) — hybrid ML example
>
> Goal: a **technical** demonstration of hybrid (NN-in-model) land modeling with Terrarium +
> Reactant + Enzyme, based on `examples/extending/simple_snow_ddm.jl`. New self-contained folder
> `examples/hybrid_models/` with its **own environment**.
>
> - **Target physics.** The degree-day melt is `M(T) = max(0, k·(T − T_melt))` (a ReLU), so the
>   snow tendency is `dS/dt = P − M(T)`. This is a clean 1-D regression target for the NN.
> - **Data.** Generate `(T, M(T))` pairs from `DegreeDaySnow` over a temperature range (offline,
>   on CPU).
> - **`NeuralSnowMelt` process** (alternative to `DegreeDaySnow`): holds a small Lux MLP + params +
>   states + input/output normalization; `compute_tendencies!` launches a KA kernel that applies
>   the NN **per grid point** (`dS/dt = P − M_nn(T)`). Per the request, the NN runs *inside* the
>   kernel (a first test — the BjerknesWorkshop notes this is the hard case vs. batched global
>   eval; we accept it as a demo).
> - **Training.** Reactant + Enzyme: loss = MSE(`M_nn(T)`, `M(T)`) over the data; differentiate
>   w.r.t. NN params with `Enzyme.autodiff(set_strong_zero(ReverseWithPrimal), …, Duplicated(ps,
>   dps), …)` compiled with `@compile raise=true raise_first=true sync=true`; step with Optimisers.
> - **De-risk first (spike):** can a tiny Lux MLP be applied inside a Terrarium KA kernel compiled
>   by Reactant (`raise=true`)? If `Lux.apply` in-kernel does not trace, fall back to a manual MLP
>   forward (extract weight arrays, do the matmuls explicitly) — still trained via Lux/Enzyme.
> - **Verify:** training reduces the loss and the trained hybrid model runs and roughly matches
>   `DegreeDaySnow`. (Example is not part of CI; heavy Lux/Reactant/Enzyme/CUDA stack.)
>
> Findings recorded inline as executed (below the Session-2 block).
>
> ### Session-3 findings (COMPLETE — `examples/hybrid_models/neural_snow_melt.jl`, own env)
> - **NN-in-kernel:** `Lux.apply` cannot be compiled inside a KA kernel (it pulls in string ops —
>   `print_to_string`/`utf8proc_*` — with no device lowering; the BjerknesWorkshop noted this too).
>   The kernel instead evaluates the MLP's forward pass **explicitly** from its weight arrays (a
>   scalar unroll over the hidden units). This compiles for CPU/GPU/Reactant. Also required **3-D
>   field indexing** (`T[i,j,1]`), not 2-D — 2-D indexing throws `jl_f_throw_methoderror` under
>   `CuTracedArray` (AGENTS.md pitfall #10).
> - **Offline training (Reactant + Enzyme):** hand-rolled `Enzyme.autodiff` + `Optimisers.update!`
>   inside `@compile` fails — Adam's `Float32` step-count `Leaf` field can't hold a `TracedRNumber`.
>   Use Lux's Reactant path instead: `Lux.Training.single_train_step!(AutoEnzyme(), loss, data,
>   tstate)` with params on `MLDataDevices.reactant_device()` (it wraps the optimizer as a
>   `ReactantCompatibleOptimisers.ReactantOptimiser`). Fits `M(T)=max(0,k(T−T_melt))`: loss
>   0.69→2e-5, melt error 2.9e-8.
> - **Online fine-tuning (differentiable simulation):** a `single_train_step!` with a **custom
>   rollout loss** that steps the snow dynamics with the NN melt and matches the analytic
>   trajectory. Gotcha: a `TrainState` caches its compiled step for **one** loss — reusing the
>   offline tstate with a different loss errors ("no method for this Thunk"); build a **fresh
>   TrainState** from the trained params for the new loss. Rollout loss 1.4e-5→1.2e-13; it
>   *improves* the 30-day snow match (0.0195 m → 9e-7 m) because the online objective targets the
>   trajectory directly.
> - **Running the hybrid model:** on **CPU** (both `DegreeDaySnow` and `NeuralSnowMelt` SnowModels,
>   input-driven `run!`), since Terrarium's input sources are not yet traced under Reactant; the
>   NN process kernel is device-agnostic and compiles under Reactant (that is how it is trained).
>   `InputSource(::ColumnRingGrid, RingGrids.Field(vec, rings); name, units)` — needs a
>   `RingGrids.Field`, not a plain vector.
> - Env: `examples/hybrid_models/{Project.toml, neural_snow_melt.jl}` (Manifest gitignored),
>   pinned via the shared Reactant 0.2.270. Example not part of CI.
> - **Online fine-tuning THROUGH THE FULL MODEL (revised per request).** The final example
>   differentiates through the actual `SnowModel(NeuralSnowMelt)` (not a hand-written rollout):
>   `Enzyme.autodiff(ReverseWithPrimal, loss, Active, Duplicated(integrator, dintegrator), …)`
>   over `run_timesteps!`, compiled with `@compile raise=true …` — the full input-driven model
>   runs and differentiates under Reactant. Key gotchas found:
>   - The full `SnowModel(NeuralSnowMelt)` **runs under Reactant** (static `FieldInputSource`
>     inputs + ext transfer are fine). The process stores the Lux **`ps` NamedTuple** in a single
>     type parameter — the NamedTuple type already captures each array's distinct
>     (shape-encoded) `CuTracedArray` type, so no per-field type params are needed. (Storing the
>     weights as separate fields also works, but then each needs its own type param, since
>     `W1` (H×1) and `W2` (1×H) are different `CuTracedArray` types.) The kernel reads
>     `p.ps.layer_1.weight` etc. and does the forward by hand.
>   - `Enzyme.make_zero!` **inside** the compiled function fails ("cannot set SnowModel …" —
>     can't reconstruct the immutable model); `make_zero` the shadow **outside** each step instead
>     (matches the soil `autodiff.jl` pattern).
>   - Rolling melt through `Nt·Δt ≈ 10⁶ s` amplifies its gradient enormously → plain SGD diverges
>     identically at any lr (saturates the melt clip in ~1 step). An **adaptive optimizer**
>     (eager host-side `Optimisers.Adam` over the device weight arrays — eager, so no traced-scalar
>     issue) is robust. Result: snow loss 1.6e-5 → ~1e-8, match 0.0195 m → 0.0116 m.
>   - Kept the **hand-rolled MLP forward** in the kernel (`Lux.apply` compiles in a CPU kernel but
>     not a Reactant device kernel, and Enzyme through a CPU `Lux.apply`-in-kernel model also
>     errored). Lux is used for the architecture, init, and offline training only.

> ## Session 2 plan (2026-07-06) — revisions + autodiff
>
> Base state: the correctness suite is green (30/30) and committed on `mg/reactant-compat`.
> Two work streams; **correctness tests must stay green after each step**
> (`julia --project=test/reactant test/reactant/runtests.jl`).
>
> ### Stream A — revise the existing implementation
> A1. **100 steps.** Bump `NSTEPS` 10 → 100 in `test/reactant/runtests.jl`; run the suite.
>     If accumulated round-off breaks the current `RTOL=1e-3`/`ATOL=1e-6`, record and loosen
>     minimally (document the chosen tolerance).
> A2. **`@trace` vs `ifelse` check.** Confirm the in-kernel `ifelse`es need no `@trace`.
>     Reasoning to verify empirically: kernels are compiled through the KA + `raise=true`
>     (GPUCompiler→StableHLO) path, where Julia-level `@trace` does not apply and `ifelse`
>     lowers to LLVM `select` → `stablehlo.select`. The passing 100-step correctness comparison
>     against CPU is the evidence (a mis-lowered select would diverge numerically). Document the
>     conclusion; no code change expected.
> A3. **Remove the compile cache.** Delete `COMPILED_STEPS` and `compiled_step` from
>     `ext/TerrariumReactantExt/integrator.jl`. Follow Oceananigans' `run_timesteps!` pattern:
>     - add `run_timesteps!(integrator, Δt, Nt)` using `@trace for _ in 1:Nt; step_core!(…); end`
>       (one compiled StableHLO while-loop for the whole run, instead of a host loop over a
>       compiled single step);
>     - `run!(integrator::ReactantIntegrator; steps, period, Δt, compiled_run! = nothing)` —
>       if `compiled_run!` is `nothing`, compile it here
>       (`@compile raise=true raise_first=true sync=true run_timesteps!(integrator, Δt, nsteps)`),
>       then call it; return the integrator. Lets callers reuse a compiled function across runs.
>     - `timestep!(integrator::ReactantIntegrator, Δt)` becomes a thin `run!(…; steps=1)` (single
>       code path, no hidden cache).
>     Keep `step_core!` (raw step: `timestep!(integrator, timestepper, Δt)` + `compute_auxiliary!`).
>     Re-run the suite.
>
> ### Stream B — autodiff via Reactant + Enzyme
> Pattern from Oceananigans `test/test_reactant_hydrostatic_free_surface_models.jl`:
> a scalar `loss(integrator, Δt, nsteps)` whose time loop is
> `@trace checkpointing=true track_numbers=false for …` calling the **raw** step
> (`timestep!(integrator, get_timestepper(model), Δt)` + `compute_auxiliary!`; public API — NOT
> the auto-compiling `run!`, which would nest `@compile`); a `grad_loss` wrapping
> `Enzyme.autodiff(set_strong_zero(ReverseWithPrimal), loss, Active, Duplicated(integrator,
> dintegrator), Const(Δt), Const(nsteps))`; both compiled with
> `Reactant.@compile raise=true raise_first=true sync=true`. Sensitivity of a scalar loss
> (mean-square final temperature) w.r.t. the initial prognostic `internal_energy`
> (`interior(dintegrator.state.internal_energy)`).
> B1. Add `Enzyme` to `test/reactant/Project.toml`; instantiate.
> B2. `test/reactant/autodiff.jl` (included from `runtests.jl`): compile `grad_loss` for a small
>     `:soil_heat_column` integrator; assert loss `> 0` & finite, gradient nonzero & finite.
> B3. `examples/autodiff/differentiating_terrarium_reactant.jl` — analogous to
>     `differentiating_terrarium.jl` but with `ReactantState()` + `@trace checkpointing=true`
>     instead of Checkpointing.jl's `Revolve`. Not executed in docs builds (Enzyme/Reactant
>     compile time), matching the existing autodiff example.
> B4. Full suite green (correctness + autodiff).
>
> Risk: Terrarium's soil-energy kernels use `sqrt`/`^` (InverseQuadratic conductivity) and
> freeze-curve `ifelse`; Enzyme-through-Reactant should handle these but is unverified — B2 is
> the check. Diagnose if the adjoint fails/NaNs; do not weaken the forward path to make AD pass.
>
> ### Session-2 findings (as executed)
> - **A1 done.** `NSTEPS = 100`. Both configs pass at 100 steps within the existing
>   `RTOL=1e-3`/`ATOL=1e-6` (prognostic `internal_energy` max rel diff ~4.7e-7); no tolerance
>   change needed. Verified 30/30 on the pinned 0.110.7/0.2.270 env.
> - **A2 done — no `@trace` needed for the in-kernel `ifelse`es (confirmed).** `@trace` is a
>   Reactant *frontend* macro (Julia-level tracing via `call_with_reactant`); it does not apply
>   inside kernels, which are compiled through the KernelAbstractions + `raise=true` path
>   (GPUCompiler → LLVM → StableHLO). There, `ifelse` is a Julia builtin that lowers to LLVM
>   `select` → `stablehlo.select` — exactly the branchless data-select we want; putting `@trace`
>   there would be inapplicable. Evidence it lowers correctly: the 100-step CPU-vs-Reactant
>   numerical match (a mis-lowered select would diverge). No code change. (`@trace` *is* used —
>   correctly — for the Julia-level time-loop in `run_timesteps!`; see A3.)
> - **Time-dependent continuous BCs ARE supported (earlier "unsupported" conclusion was wrong).**
>   The committed `:soil_heat_global` BC is `amplitude*sin(ω*t)*cos(x)`. It passed 30/30 locally
>   on 2026-07-03 (see `scratchpad/suite_final.log`) and passes on remote CI with
>   `raise=true` — so the time-dependent BC is fine and the doc/BC "fix" was reverted
>   (`setup.jl` restored to the time-dependent BC; the false doc limitation removed).
>   - **Local-only failure (open, non-blocking):** on this machine the global config now fails to
>     compile (`InvalidIRError` in `gpu__fill_bottom_and_top_halo!`; a traced `Reactant.Ops.multiply`
>     of the clock time appears in device IR). It reproduces with and without `raise=true`. But the
>     package builds are **identical** to the Jul-3 passing run (Oceananigans XnMRS 0.110.7,
>     Reactant N8Jgp 0.2.270, `Reactant_jll` 0.0.391+1; no `LocalPreferences.toml`, no XLA env),
>     and remote passes — so this is a **stale local artifact** from the version drift/re-pin churn
>     (env had drifted to 0.110.5/0.2.269 and was re-pinned to 0.110.7/0.2.270), not a code/version
>     limitation. Fix attempt: fresh Manifest resolve (mirroring remote). `:soil_heat_column`
>     it; the global config is covered by remote CI.
>   - **RESOLVED:** a fresh Manifest resolve (deleting `test/reactant/Manifest.toml`, then
>     `Pkg.develop(".") + instantiate`, mirroring remote) fixed the local failure — it pulled
>     Oceananigans **0.110.8** (newly released; compat `>=0.110.7` allows it) with a consistent
>     transitive set, and the full suite (time-dependent BC included) now passes 30/30 locally.
>     Confirms it was a stale/mixed local Manifest from the drift/re-pin churn, never a code or
>     fundamental limitation. Compat pin kept at `Oceananigans = "0.110.7"` (allows 0.110.8).
> - **A3 done (run cache removed).** `COMPILED_STEPS`/`compiled_step` deleted. Added
>   `run_timesteps!(integrator, Δt, Nt, checkpointing=false)` with `@trace for` (single compiled
>   `stablehlo.while`); `run!(integrator; steps, period, Δt, checkpointing=false, compiled_run!=nothing)`
>   compiles it on demand and accepts a reusable compiled function; `timestep!` = `run!(; steps=1)`.
> - **`run_timesteps!` is now a Terrarium function (src), extended in the ext (per request).**
>   Generic host-loop method in `model_integrator.jl` (Reactant-free, ignores `checkpointing`),
>   exported; `TerrariumReactantExt` adds the `ReactantIntegrator` method (`@trace`). Tests/example
>   call the public `Terrarium.run_timesteps!` (no `Base.get_extension`). CPU method verified
>   (iter 0→5); Reactant method verified via the suite.
> - **Checkpointing is an optional argument (per request).** `checkpointing` kwarg on `run!` and
>   4th positional on `run_timesteps!`, forwarded to `@trace checkpointing=…`: `false` (default) or
>   a scheme (`Reactant.Periodic(n)`/`Binomial(n)`). No effect on forward runs; bounds reverse-pass
>   memory under AD. `true` (auto-`isqrt`) only works as a `@trace` literal, so the runtime knob
>   takes `false` or an explicit scheme — documented.
> - **B (autodiff) done.** `Enzyme` added to `test/reactant/Project.toml`.
>   `test/reactant/autodiff.jl`: compiles `Enzyme.autodiff(ReverseWithPrimal, …, Duplicated(integrator,
>   dintegrator), …)` over `run_timesteps!`; asserts ∂loss/∂U₀ finite + nonzero, and that the
>   gradient is **invariant** to the checkpointing scheme (`false` vs `Periodic(2)`). Passes 6/6
>   (`loss=0.795`, `max|∂loss/∂U₀|≈1.9e-7`). Example
>   `examples/autodiff/differentiating_terrarium_reactant.jl` mirrors the CPU autodiff example using
>   `ReactantState` + `run_timesteps!` with a `Periodic` checkpointing scheme (not executed in docs
>   builds, like the CPU one).
> - **Final state: full suite 30/30 correctness + 6/6 autodiff green on Oceananigans 0.110.8 /
>   Reactant 0.2.270.** All Session-2 asks (100 steps, `@trace`/`ifelse` check, cache removal +
>   `compiled_run!`, `run_timesteps!` in src, checkpointing arg, autodiff + example) complete.

> **Execution status (living section — updated as work proceeds).**
> Branch: `mg/reactant-compat` (Session-1 work committed; see git log).
>
> **Done & verified**
> - **PR2 (`ifelse` rewrites):** rewrote the value-dependent conditionals in
>   `soil_energy_closures.jl` (`liquid_water_fraction`, `energy_to_temperature`),
>   `kernel_utils.jl` (`findfirst_z`), and `canopy_interception.jl` (saturation fraction) to
>   branchless `ifelse`. Verified behavior-preserving: existing soil-energy + canopy tests pass
>   (38/38). Deferred with markers: RRE double-write (`soil_hydrology_rre.jl`), the
>   type-unstable `Ice()/Liquid()` phase ternary, and the fixed-point `while` loop.
> - **PR1 (test infra):** `test/reactant/` created with its own `Project.toml`
>   (Reactant 0.2.270, Oceananigans 0.110.7), `runtests.jl`, `setup.jl` (model registry:
>   `:soil_heat_column`, `:soil_heat_global`), and `correctness.jl` (state sync, per-field
>   comparison, `test_model`). Instantiated successfully.
> - **Enabling change:** `ReactantState` re-exported from `Terrarium`; `Reactant` added to
>   `[weakdeps]`/`[extensions]`/`[compat]` in the root `Project.toml`.
> - **Phase 0 spike — findings (the important part):**
>   1. Model/grid/state **construction on `ReactantState` works**; fields become
>      `ConcretePJRTArray`-backed.
>   2. **Eager KA kernel launches fail** on `ReactantBackend` (bare `launch!` outside a compile
>      context) — so `initialize!`'s eager `fill_halo_regions!` cannot run on the device.
>      → Design decision: **build + initialize on CPU, then transfer to device**; only
>      `timestep!` is compiled.
>   3. Transfer primitive: `Oceananigans.on_architecture(ReactantState(), x)` (NOT
>      `adapt(array_type, x)` — that strips the `Field` wrapper and sets grid arch to
>      `Nothing`). Verified it preserves `Field`s and gives concrete device arrays.
>   4. The land-grid `on_architecture` must transfer the **inner** `RectilinearGrid` (which
>      keeps `arch = ReactantState`) and rewrap — the generic land-grid fallback uses `adapt`
>      and yields a `Nothing`-arch grid whose field allocation calls `zeros(Nothing, …)`.
>   5. Rebuilding a model with a swapped grid must go through the type's `wrapper` (UnionAll)
>      constructor to **re-infer type parameters** — `setproperties` keeps the old concrete
>      params and errors trying to `convert` the device grid to the CPU grid type.
>   6. With the CPU-build→device-transfer path, a full reactant integrator constructs and syncs,
>      and **`@compile timestep!` enters tracing** (fields become `TracedRNumber`).
>   7. `TerrariumReactantExt` fully drafted: `on_architecture` grid transfer (grids.jl),
>      build-on-CPU→transfer `initialize` (transfer.jl), compiled+cached `timestep!`/`run!`
>      (integrator.jl). **Initialization correctness passes** for `:soil_heat_column` (initial
>      CPU vs device states match, 7/7) — so the whole construct→transfer→sync pipeline is
>      correct. `CUDA` must be loaded in the test env (KA↔Reactant glue); added to
>      `test/reactant/Project.toml`.
>
> **CORRECTED DIAGNOSIS (supersedes the earlier "hard upstream blocker" framing).**
> After studying Oceananigans' own Reactant tests (`Oceananigans/test/reactant_test_utils.jl`,
> `test_reactant.jl`, `test_reactant_single_column_models.jl`) and **running their patterns in our
> exact env**, the version combo (Oceananigans 0.110.7 + Reactant 0.2.270 + CUDA loaded) is **not
> broken**: native `HydrostaticFreeSurfaceModel`s on `RectilinearGrid` compile and time-step
> correctly. So the blocker is **Terrarium-side and specific**, layered as follows.
>
> **What the working Oceananigans setups do** (adopt these):
> - Build the model **natively on the device** (`GridType(ReactantState(); …)`), set ICs, then
>   `@jit update_state!` and `@compile sync=true first_time_step!/time_step!`; the single-column
>   test uses `@compile raise=true raise_first=true sync=true` around a
>   `@trace track_numbers=false for _ in 1:Nt; time_step!(model, Δt); end` loop
>   (`run_timesteps!`), and drives first-vs-subsequent steps via a host loop (`r_run!`).
> - Correctness is checked by building the *same* model on CPU and on device, `set!`-ing the same
>   ICs, running both, and comparing `interior(...)` with `≈` (our `test/reactant/` mirrors this).
> - Key knobs: `Reactant.set_default_backend("cpu"/"gpu")`, `@compile … sync=true`,
>   `raise=true raise_first=true`, `@trace … for` for the step loop, `ConcreteRNumber` step counts.
>
> **Root cause #1 — array vertical coordinates (SOLVED for the uniform case).**
> A `RectilinearGrid` whose vertical coordinates are an explicit **array** (Terrarium's `ColumnGrid`
> always builds `z` via `vcat(...)` → a `Vector`, for *all* spacings incl. uniform) fails Reactant
> kernel tracing: the coordinate arrays are not converted to `CuTracedArray` and
> `_check_no_traced_in_kernel_arg` rejects the grid argument to the halo kernel.
> - Isolated with **bare Oceananigans**: identical `HydrostaticFreeSurfaceModel` compiles with a
>   regular-range `z=(-200,0)` (StepRangeLen coords) but **fails** with an explicit `z`-vector.
>   Matches the ext's own note that `RectilinearGrid` "retains StepRangeLen coordinates which
>   sidestep the CuTracedArray VX type constraint" (i.e. array-coord RG is unsupported upstream).
> - **Fix:** have `ColumnGrid`/`ColumnRingGrid` pass a **range** `z` to `RectilinearGrid` when the
>   spacing is uniform. Confirmed this removes the grid-argument error end-to-end. Non-uniform
>   spacings (`ExponentialSpacing`, `PrescribedSpacing`) inherently need array `z` → remain blocked
>   on an upstream Oceananigans fix (report: array-coord `RectilinearGrid` under Reactant).
>
> **Root cause #2 — Terrarium kernels emit `llvm.intr.trap` (CURRENT FRONTIER).**
> With a range-`z` grid, tracing gets past the grid/halo stage and now fails while *raising* the
> Terrarium compute kernels to StableHLO:
> `error: cannot raise op to stablehlo "llvm.intr.trap"`, on
> `gpu_energy_to_temperature_kernel!` and `gpu_compute_tendencies_kernel!`. A `trap` comes from a
> residual bounds check / error path / possibly-throwing op in the kernel call graph (Oceananigans'
> own kernels raise cleanly, so Terrarium's have an eliminable trap source — candidates: bounds
> checks on the `ground_temperature` `@view`, a `convert`/division that can throw, or a missing
> `@inbounds`/`@propagate_inbounds` on a callee). Next step: bisect which op in
> `energy_to_temperature!`/`compute_energy_tendency` emits the trap and remove it.
>
> **Status of dependency facts (still true):** our stack forces Reactant `≥0.2.243`
> (RingGrids ≥0.1.5); viable window `[0.2.243, 0.2.270]`; 0.2.243 segfaulted on the *isolated*
> `fill_halo_regions!` probe (a non-representative pattern — ignore), 0.2.270 is what we use.
>
> **Decision (user): report upstream + land safe pieces — done, and refined:**
> - `test/reactant/` suite is CI-safe: initialization comparison passes; device time-stepping is
>   wrapped `@test_broken` (flips to "Unexpectedly Pass" once #2 is fixed).
> - `REACTANT_upstream_issue.md` retargeted to the *precise* upstream gap (array-coord
>   `RectilinearGrid` under Reactant) — only relevant for non-uniform spacings.
> - Ready to merge: PR2 `ifelse` refactors (verified), `test/reactant/` infra,
>   `TerrariumReactantExt` (construction/init verified).
>
> **MILESTONE (2026-07-03): full correctness suite GREEN — 30/30.**
> `julia --project=test/reactant test/reactant/runtests.jl` passes for both registry configs
> (`:soil_heat_column` and `:soil_heat_global` on `ColumnRingGrid` with a function surface BC):
> initialization + 10 compiled steps + clock, CPU vs Reactant, diffs at Float32 rounding level
> (max rel ~1e-7). The user API is exactly the CPU one — only the architecture differs.
>
> **Immediate next steps — all DONE:**
> 1. Uniform-spacing `z` fix: `z_coordinates(NF, spacing)` in `vertical_discretization.jl` —
>    `Vector` fallback, **2-tuple `(-depth, 0)`** specialization for `UniformSpacing`; both grid
>    constructors use it. Registry switched to `UniformSpacing`; `@test_broken` → `@test`.
>    ⚠️ Nuance: passing an explicit `AbstractRange` z is NOT enough — Oceananigans materializes
>    it into coordinate arrays (→ untraceable, root cause #1 again); only the **tuple** form
>    yields `StepRangeLen` coordinates. CPU regression: 305/305 green with the tuple form.
> 2. Root cause #2 (`llvm.intr.trap`): **SOLVED** — see "PHASE A EXECUTED" below.
> 3. Compiled `run!`/`timestep!` wired through the ext; correctness `@test`s re-enabled.
>
> **CI: `.github/workflows/Reactant_CI.yml` added.** Runs the `test/reactant` suite on
> ubuntu / Julia 1.12 for pushes + PRs to `main` (plus `workflow_dispatch`); skippable via
> `skip-ci` / `skip-reactant-ci` PR labels; 90-min timeout (Reactant compile is minutes-scale,
> heavily amortized by `julia-actions/cache`). Steps mirror SpeedyWeather's Reactant CI:
> `Pkg.develop(path=".") + instantiate` into `test/reactant`, then run `runtests.jl`. Both steps
> validated locally verbatim (fresh instantiate OK; suite 30/30). `test/reactant/Manifest.toml`
> is already covered by the repo's global `Manifest*.toml` gitignore rule.
>
> **Two additional gotchas found while wiring the suite (document in the docs phase):**
> - `using CUDA` is required in any Reactant driver script/env (Reactant's KernelAbstractions
>   integration needs it even on CPU; without it, kernel launches fail with
>   `MethodError: ka_with_reactant(::Nothing, …)`).
> - Boundary-condition closures must capture only **isbits** values: capturing `NF` (a
>   `Type{Float32}`) makes the BC function a non-bitstype kernel argument. Hoist all
>   constants (`amplitude = NF(5)` etc.) out of the closure.
>
> *(2026-07-03: `REACTANT_upstream_issue.md` reviewed & upgraded for the independent
> Oceananigans session — every claim re-verified against bare Oceananigans, incl. a new
> constructor-level probe showing `AbstractRange` z inputs are also materialized (only endpoint
> tuples stay `StepRangeLen`); misleading "missing @adapt_structure" framing replaced with
> verified pointers (both adapt rules exist; unadapted scalar fields are a candidate survivor);
> appendix added with env recipe, red-test suggestion, and fix-location shortlist.)*
>
> **TODO (deferred, tracked): non-range `z` / `ExponentialSpacing` under Reactant.**
> Stretched vertical grids (`ExponentialSpacing`, `PrescribedSpacing`) require array-valued
> vertical coordinates, which currently fail Reactant kernel tracing inside Oceananigans
> (coordinate arrays not adapted to `CuTracedArray`; isolated with bare Oceananigans — see
> `REACTANT_upstream_issue.md`). This is deliberately **out of scope for the current push** and
> will be tackled separately:
> - file the upstream issue (`REACTANT_upstream_issue.md`) against Oceananigans.jl and track it;
> - when picking it up: check whether newer Oceananigans/Reactant releases adapt
>   `StaticVerticalDiscretization` arrays for kernel launch (the ext already does this for
>   `LatitudeLongitudeGrid` — the fix pattern exists upstream);
> - possible interim option if urgently needed: approximate stretched grids by conformal
>   remapping on a uniform grid, or carry a local ext patch adapting the RG `z` arrays
>   (type-piracy, last resort);
> - once fixed, add an `ExponentialSpacing` config to the `test/reactant` registry so both
>   coordinate representations stay covered.

---

## Root cause #2 investigation plan — `llvm.intr.trap` in soil-energy kernels

**Symptom.** With a range-`z` grid, `@compile raise=true …` of the Terrarium step fails in the
MLIR "raise to StableHLO" pass: `error: cannot raise op to stablehlo "llvm.intr.trap"`, with the
failing functions named as `gpu_energy_to_temperature_kernel!` and
`gpu_compute_tendencies_kernel!` (the soil-energy closure and tendency kernels).

**Why traps appear (mechanism).** Under `raise=true`, ReactantCUDAExt compiles each
KernelAbstractions kernel through the CUDA GPUCompiler pipeline to LLVM IR, then EnzymeJAX raises
that IR to StableHLO. GPUCompiler lowers *every* Julia `throw` path — `@assert`, `BoundsError`
from un-elided bounds checks, `InexactError` from `convert`/`round(Int, …)`, `DivideError` from
integer `div`/`mod`, `error(...)` guard methods — to `llvm.intr.trap`. On a real GPU the trap is
harmless unless triggered; but StableHLO has no trap, so **any reachable throw path in the kernel
call graph, even one that never fires, hard-fails the raise**. (This is also why the kernels work
on CUDA today.) Consequently the search is for *reachable throw sites*, not for actual errors.

> **PHASE A EXECUTED — ROOT CAUSE #2 SOLVED (2026-07-03).**
> - Neutralizing the `SoilVolume`/`MineralOrganic` inner-constructor asserts eliminated the trap:
>   **first fully working Reactant-compiled `timestep!`** on the range-`z` uniform `ColumnGrid` —
>   4 compiled steps via the ext's cached `run!`, traced clock advanced to 4, CPU-vs-Reactant
>   `internal_energy` max abs diff 0.0625 on O(10⁶) values (rel ~1e-8), `isapprox = true`.
> - An intermediate attempt to keep the asserts behind `@boundscheck` + `@propagate_inbounds`
>   FAILED, with an instructive reason (from the MLIR debug locations): the tendency-kernel call
>   chain passes through **Oceananigans' operators** (`∂zᵃᵃᶜ`, `δzᵃᵃᶜ`, `ℑzᵃᵃᶠ`), which are
>   `@inline` but not `@propagate_inbounds` — the inbounds context is lost there, so the
>   `@boundscheck` block survives and traps. Rule of thumb: **@boundscheck-gating cannot protect
>   anything reached through Oceananigans operator functors.**
> - **Final fix (implemented):** `SoilVolume` and `MineralOrganic` now have raw positional inner
>   constructors (kernel path, no validation) and hand-written validating **keyword** constructors
>   (host path; `@kwdef` replaced since Julia ≥1.12 forbids overwriting its generated method).
>   All existing `@test_throws AssertionError` semantics preserved.
> - **Bonus bug found & fixed by the widened test run:** the PR2 `findfirst_z` rewrite returned
>   `ifelse(idx > 0, z_nodes[idx], z_nodes[n])` — `ifelse` evaluates *both* branches, so the
>   not-found case (`idx = -1`) read `z_nodes[-1]` (BoundsError on CPU, silent OOB under
>   `@inbounds` on GPU). Fixed by selecting the *index*, not the nodes:
>   `z_nodes[ifelse(idx > 0, idx, n)]`. Lesson for all ifelse rewrites: never index in an
>   unselected branch.
> - Verification: Reactant drive passes (above) AND 288/288 CPU tests (composition incl. assert
>   throws, energy, full soil model incl. hydrology water-table path).

### Phase A — confirm the prime suspect (static audit already done)

A grep of the failing kernels' call graph found in-kernel `@assert`s — which already violate the
AGENTS.md rule "No error messages … inside kernels":

| Suspect | Where | Reached from |
|---|---|---|
| 3× `@assert` in `SoilVolume` inner constructor | `src/processes/soil/stratigraphy/soil_composition.jl:26-28` | `SoilVolume(por, sat, liq, solid)` built **per grid point** in `energy_to_temperature!`, `temperature_to_energy!`, and via `soil_composition(...)` in the thermal-conductivity chain of the tendency kernel — matches **exactly** the two failing kernels |
| 1× `@assert` in `MineralOrganic` inner constructor | `soil_composition.jl:87` | `soil_matrix(i,j,k,…)` constructs the solid-phase struct per point in the same kernels |
| 4× `@assert` in `SoilTexture` constructor | `soil_texture.jl:18-21` | likely host-only (model setup) — verify whether `soil_matrix` reconstructs textures per point |

**Action A1 (decisive, ~1 h):** locally neutralize the `SoilVolume` + `MineralOrganic` asserts and
recompile the two failing targets. Expected outcomes:
- raise succeeds → root cause confirmed; go to Phase C (fix properly).
- raise fails with a *different* trap location → repeat: the audit list above is ordered; continue
  down it (then Phase B for anything non-obvious).

**Environment guard (from Oceananigans' own unit tests):** assert `Base.JLOptions().check_bounds == 0`
in the reactant test harness — running with `--check-bounds=yes` would materialize *every* bounds
check as a trap and make raising impossible regardless.

### Phase B — systematic localization (only if Phase A is insufficient)

1. **Tight iteration harness** (`scratchpad/trap_bisect.jl`): compile each kernel family
   *separately* against the range-`z` reactant integrator — `closure!` (energy→temperature),
   `invclosure!` (temperature→energy), `compute_tendencies!` — so each attempt is ~30 s, not a
   full `timestep!` compile.
2. **Read the IR instead of guessing:** dump the pre-raise module (`@code_hlo` at the stage the
   installed Reactant supports, e.g. `optimize=:before_raise`/`optimize=false`) and grep for
   `llvm.intr.trap`. GPUCompiler's exception lowering leaves `gpu_report_exception` calls and
   string constants naming the Julia exception type and source function — read the enclosing
   `func.func` to get the culprit directly.
3. **Body bisection protocol** (when IR reading is inconclusive): replace the kernel body with a
   trivial copy (`out[...] = in[...]`); if that raises, restore sub-expressions in dependency
   order — `SoilVolume`/`soil_matrix` construction → `porosity`/`saturation_water_ice` lookups →
   `compute_heat_capacity` → closure math (`liquid_water_fraction`, `energy_to_temperature`) →
   `∂zᵃᵃᶜ`/`ℑzᵃᵃᶠ` operator chain → `InverseQuadratic` conductivity (`sqrt`, `^`) — until the trap
   reappears. If the trivial body *still* traps: bisect the **kernel arguments** instead (drop
   fields from the NamedTuple, starting with the `ground_temperature` SubArray view), and test the
   launch path (Terrarium's `launch!` passes the `ColumnGrid` wrapper into kernels — try an
   Oceananigans-style launch with the raw field grid to rule the wrapper in/out).
4. **Secondary suspects** (only reachable-throw candidates the audit flagged): `sqrt`/`^` domain
   branches in `InverseQuadratic` (`soil_thermal_properties.jl:111,174`) — the CUDA device
   overrides normally replace these with intrinsic versions, so they are *expected* to be clean,
   but the bisection order above covers them.

### Phase C — fix patterns and hardening

- **In-kernel asserts:** keep validation for host-side construction but remove it from the
  per-point hot path. Options, in order of preference: (i) delete the asserts from the inner
  constructors (AGENTS.md already forbids error paths in kernels; validation belongs to the
  user-facing keyword constructors), or (ii) split into a raw positional constructor used by
  kernels and a validating `kwdef`/keyword path used at model setup. Verify with the existing CPU
  test suite (soil tests assert physical bounds indirectly).
- **Repo sweep:** grep all kernel call graphs for `@assert`/`throw`/`error(`/`round(Int`/integer
  `div`/`mod` and fix what is on the SoilModel path now; list the rest in this plan for later
  phases (hydrology/vegetation/surface energy).
- **Regression guards:**
  - `check_bounds` guard — **DONE**: `test/reactant/runtests.jl` now fails fast (~1.5 s, before
    any package loading) with a readable error when run with `--check-bounds=yes` (which e.g.
    `Pkg.test` forces by default), since forced bounds checks make every kernel un-raisable.
    Verified in both modes.
  - "Raise-ability" unit test — **OPTIONAL, deferred by decision (2026-07-03)**: `@compile
    raise=true` each Terrarium kernel family standalone (analogous to Oceananigans'
    `test_reactant_unit.jl`), so a future in-kernel assert/throw is caught at kernel granularity
    with the culprit named, instead of as an opaque full-`timestep!` MLIR error. Not needed while
    the end-to-end suite covers the active kernel set; revisit when enabling further process
    families (hydrology RRE, vegetation, surface energy) under Reactant.
- **AGENTS.md — DONE (2026-07-06).** Kernels rules tightened (no reachable throw path →
  `llvm.intr.trap`, unraisable; `ifelse` evaluates both branches so select the index, not the
  loads) and a new **"### Reactant compatibility"** subsection added (build-on-CPU→device model,
  `using CUDA` requirement, `isbits`-only kernel closures, uniform-`z` requirement, reverse-mode-AD
  recipe + `checkpointing`, no `--check-bounds=yes`, and the stale-Manifest debugging tip).

### Phase D — fallback if a trap lives in third-party code

If after Phase B a trap remains inside Base/KernelAbstractions/Oceananigans code we cannot edit:
extract the dumped `func.func`, file it against Reactant.jl/EnzymeJAX (they have handled
trap-stripping for common patterns before), and check for a Reactant pass/knob that strips
provably-dead traps. Track versions; do not fork.

**Exit criterion.** `@compile raise=true raise_first=true sync=true step_core!(integrator, Δt)`
compiles for the range-`z` `:soil_heat_column` config, steps run, and the correctness comparison
against CPU passes — at which point the `@test_broken` markers in `test/reactant/correctness.jl`
flip back to `@test`.

---

**Goal:** run Terrarium models through [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
(tracing → MLIR/StableHLO → XLA) with the architecture as the *only* user-facing knob:

```julia
grid = ColumnRingGrid(ReactantState(), Float32, ExponentialSpacing(N = 30), rings, mask)
model = SoilModel(grid)
integrator = initialize(model; boundary_conditions, initializers)
run!(integrator, period = Hour(12), Δt = 600.0)   # transparently compiled + executed by XLA
```

Everything else — `Reactant.to_rarray`, `@compile`, traced clocks, the compiled step cache —
stays hidden inside a package extension, following the Oceananigans and SpeedyWeather precedents.

---

## 1. Background: how the two reference implementations work

### 1.1 Reactant in one paragraph

Reactant *traces* a Julia function call with `TracedRArray` arguments into an MLIR module and
compiles it with XLA (`@compile f(args...)` returns a compiled callable; `@jit` compiles + runs).
Data lives in `ConcreteRArray`s (device buffers); scalars are compile-time constants unless
converted with `to_rarray(x; track_numbers = true)` → `ConcreteRNumber`. During tracing, ordinary
Julia control flow is *executed*, so:

- `if`/`&&`/`? :` on a **traced value** throws (`TracedRNumber{Bool}` cannot be `Bool`-converted) →
  must become `ifelse` (both branches evaluated, same result type) or `@trace if` (compiled to
  `stablehlo.if`).
- loops with static bounds are **unrolled** at trace time; `@trace for` compiles a real
  `stablehlo.while` loop instead (supports `checkpointing = ...` for AD).
- `while` loops with value-dependent exit conditions cannot be traced as-is.
- Type instabilities in the traced call graph fail or silently freeze the traced branch.

`ReactantCore` is a tiny package providing `@trace` (no-op without Reactant loaded) — Oceananigans
carries it as a *hard* dependency and keeps `Reactant` itself as a weakdep.

### 1.2 Oceananigans (v0.110.5, already in our Manifest)

This is the big win for Terrarium: **most of the machinery we need already exists** in the
Oceananigans we build on.

- `ReactantState <: AbstractSerialArchitecture` is defined and exported by the *main* package
  (`Oceananigans.Architectures`); it works everywhere an architecture is accepted.
- `OceananigansReactantExt` (activated by loading `Reactant`) provides:
  - `architecture`/`array_type`/`on_architecture` for Reactant arrays
    (`on_architecture(::ReactantState, ::Array) = Reactant.to_rarray(a)`),
  - grid construction/transfer for `RectilinearGrid` etc. (incl. `Reactant.traced_type_inner`
    rules and `ConstructionBase.constructorof` so grids can be re-materialized with traced
    type parameters),
  - `Field` support: `set_to_function!`/`set_to_field!` take a CPU detour (build the field on a
    CPU grid, then `copyto!` into the Reactant field) because some KA kernels don't trace yet,
  - `Clock` handling: `Clock(grid::ReactantGrid)` stores `time`/`iteration`/`last_Δt` as
    `ConcreteRNumber`s; `tick!`/`tick_time!` overloads for `Clock{<:TracedRNumber}` mutate
    `.mlir_data` in place,
  - time stepping: `first_time_step!`, `time_step_for!(sim, N) = @trace for _ in 1:N ...`, and
    `run!(::ReactantSimulation) = error(...)` (Oceananigans deliberately does **not** make its
    high-level `run!` work — a lesson for us: our own `run!` must be overridden in the ext),
  - KernelAbstractions kernels trace via Reactant's `ReactantKernelAbstractionsExt`
    (`launch!` works, with known gaps, e.g. `interpolate!`'s kernel — Reactant.jl#2364).

### 1.3 SpeedyWeather (PR [#970](https://github.com/SpeedyWeather/SpeedyWeather.jl/pull/970) merged 2026-03, PR [#985](https://github.com/SpeedyWeather/SpeedyWeather.jl/pull/985) follow-up)

SpeedyWeather had to build its own architecture type (`ReactantDevice{D}` in
`SpeedyWeatherInternals.Architectures`) because it doesn't sit on Oceananigans. What we take from
it is the *workflow*, not the plumbing:

- **`SpeedyWeatherReactantExt`**: `time_stepping!(sim, r_time_step! = nothing)` — compiles
  `time_step!` on the fly if no precompiled function is passed, then runs a plain host loop
  calling the compiled step (`@trace for` is commented out pending upstream fixes). A
  Reactant-aware `Clock` constructor uses `to_rarray(...; track_numbers = true)`.
- **`@maybe_jit arch expr`**: expands to `_jit(arch, f, args...; kwargs...)`; the default `_jit`
  just calls `f`, the ext overloads it for `ReactantDevice` to `@jit`, and it detects nested
  compile contexts (`within_compile()`) so an inner `@maybe_jit` inside an outer trace is a
  no-op. This is how "minimal user contact" is achieved.
- **`test/reactant/` correctness suite** (own `Project.toml`, separate CI workflow
  `CI_Reactant_Correctness.yml`, *not* part of the default test run):
  - `setup.jl`: `create_cpu_model(ModelType)` / `create_reactant_model(ModelType)` — same
    configuration, different architecture;
  - `sync_variables!(sim_cpu, sim_reactant)`: recursive copy of all variables Reactant → CPU so
    both start from bit-identical state;
  - `compare_arrays(nt_cpu, nt_reactant; rtol, atol)`: walks NamedTuples, reports
    max/mean abs/rel differences per variable + `isapprox` verdict;
  - `test_tendencies!` (one step, compare tendencies) and `test_time_stepping!`
    (N steps, compare prognostic + diagnostic variables and the clock);
  - `test_model(ModelType; trunc, nsteps, rtol, atol)` orchestrates the above per model type —
    this is the "flexible over models" pattern we're asked to reproduce;
  - tolerances `RTOL = 1e-3`, `ATOL = 1e-8` at Float32 (XLA reorders floating-point math, so
    bitwise equality is not achievable).
- PR #985 shows the long tail: dozens of small `if → ifelse` rewrites and constructor
  parametrizations across parameterizations. Expect the same shape of work in Terrarium's
  process library.

### 1.4 Design conclusion

Terrarium re-exports Oceananigans architectures and builds all state on Oceananigans `Field`s,
grids, `launch!`, and `Clock`. So unlike SpeedyWeather we do **not** define our own architecture
type: we adopt **`Oceananigans.ReactantState`** directly (already re-exportable through our
existing `export CPU, GPU` line) and inherit `OceananigansReactantExt` wholesale. Our own
`TerrariumReactantExt` only has to cover what Terrarium adds on top: the land grids, the
`ModelIntegrator`/`run!` loop, the clock default, and trace-safety of our kernels.

---

## 2. Current-state audit of Terrarium

What the survey of `src/` found, mapped to the three work categories from the task.

### 2.1 Type parametricity — mostly already there

Good news: the core containers are fully parametric and concretely typed already
(`StateVariables{NF, names..., Fields..., Cache, ClockType}`, `ModelIntegrator{NF, Arch, Grid,
TimeStepper, Model, StateVars, ClockType, ...}`, `SoilModel{NF, GridType, Soil, Initializer,
Timestepper}`, `ColumnGrid`/`ColumnRingGrid{NF, Arch, ...}`, all process structs). Remaining
items:

| Item | Location | Action |
|---|---|---|
| `StateVariables` `constructorof` returns a closure | `src/state_variables.jl:69` | Verify Reactant's `make_tracer` can rebuild it; likely needs an explicit `Reactant.traced_type_inner` rule in the ext (pattern: OceananigansReactantExt's rules for `RectilinearGrid` and friends) since `NF` and the name tuples are type parameters |
| `ColumnGrid` / `ColumnRingGrid` | `src/grids/` | May need `traced_type_inner` + `make_tracer_via_immutable_constructor` rules in the ext (the wrapped `RectilinearGrid` is already covered upstream) |
| `Clock` created as `Clock(time = zero(NF))` | `src/state_variables.jl`, `src/timesteppers/model_integrator.jl:200` | Introduce an architecture-dispatched `default_clock(grid)` so the ext can return the `ConcreteRNumber`-backed clock (§3, Phase B) |
| `on_architecture(arch, grid::AbstractLandGrid) = adapt(array_type(arch), grid)` | `src/grids/grids.jl:25` | `adapt`-based transfer won't produce `ConcreteRArray`s; route through `on_architecture` recursion instead (upstream ext pattern) |

Process parameter structs (e.g. `SoilThermalProperties{NF}`) hold plain `NF` scalars. Under
tracing these become **compile-time constants** — fine for forward runs (changing a parameter
merely recompiles); revisit with `track_numbers = true` when we want parameter
gradients/estimation through Reactant (out of scope here, see §6).

### 2.2 Runtime conditionals and ternaries — enumerated

Only 13 ternaries exist in `src/`; most branch on *types* (`isnothing(...)` on process fields),
which is resolved at trace time and fine. The value-dependent ones, plus `if`/`while` blocks that
matter, in priority order:

**Blocking for phase-1 target (SoilModel, heat conduction only):**

1. `src/processes/soil/energy/soil_energy_closures.jl:135-144` — `liquid_water_fraction(::FreeWater, ...)`:
   `if U >= zero(U) ... else ...` → single `ifelse` expression (both branches are same-type
   scalars; the `else` branch is already branchless).
2. `src/processes/soil/energy/soil_energy_closures.jl:150-163` — `energy_to_temperature(::FreeWater, ...)`:
   `if/elseif/else` → nested `ifelse` (the branchless one-liner is already sketched in a comment).
3. `src/timesteppers/abstract_timestepper.jl:147-167` — `explicit_step!` filtering
   `if name ∈ names`: symbols/tuples only ⇒ resolved at trace time, **no change needed**
   (same for the equivalent branches in `heun.jl`).
4. `src/diagnostics/debugging.jl` — `debugsite!` branches on the global `DEBUG::Bool`: not traced,
   fine when `false`; `checkfinite!` (`any(!isfinite, ...) && error(...)`) **cannot** run under
   tracing. Action: document that debug mode is unsupported with `ReactantState`, and short-circuit
   `debugsite!`/`checkfinite!` for Reactant fields in the ext so an enabled debug flag degrades
   gracefully instead of throwing mid-trace.

**Needed for later phases (hydrology / surface energy / vegetation):**

5. `src/processes/soil/hydrology/soil_hydrology_rre.jl:162-164` — `if k <= 1 ... elseif k >= Nz`:
   branches on the KA index, which is traced under Reactant's KA path → rewrite with `ifelse`.
6. `src/utils/kernel_utils.jl:7-16` — `findfirst_z`: `if idx < 0 && cond(...)` inside the loop and
   a trailing ternary → branchless rewrite (`idx = ifelse((idx < 0) & cond, k, idx)`, return via
   `ifelse`).
7. `src/processes/physics_utils.jl:42` — `phase = T <= zero(T) ? Ice() : Liquid()`: branches on a
   value *and* returns different **types** — `ifelse` can't fix this. Restructure: evaluate the
   Thermodynamics call for both phases and `ifelse`-blend the scalars, or use a phase-free
   formulation. (Used by surface energy balance, not by the phase-1 soil model.)
8. `src/processes/surface_hydrology/canopy_interception/canopy_interception.jl:92` —
   `w_can_max > 0 ? ... : zero(NF)` → `ifelse` (guard the division with `safediv`).
9. `src/solvers/fixed_point.jl:47-55` — `while abs(x₁-x₀) > tol && iter <= max_iterations` inside
   a kernel (used by `skin_temperature.jl`), similarly `src/solvers/root_solvers.jl` /
   RootSolvers.jl: value-dependent `while` is the hardest category. Options, in order of
   preference: (a) fixed-iteration `for` loop with a converged mask
   (`x = ifelse(converged, x, update)`) — trace-unrollable and branch-free; (b) `@trace while`
   once upstream support is solid; (c) exclude models using implicit skin temperature from
   Reactant support initially. Decide when we reach the LandModel phase.

**Host-side control flow (fine as-is):** `run!`'s `for _ in 1:steps` loop, `get_steps`
(`Dates.Period` arithmetic), `initialize` (documented as type-unstable, runs once outside the
trace), `update_inputs!` scope matching (`if matches_scope(...)` — static), `Base.getproperty`
name dispatch on `StateVariables` (symbol comparisons, resolved at trace time).

### 2.3 Structural pieces that need ext support

- **`run!(integrator::ModelIntegrator; ...)`** (`model_integrator.jl:88`) and
  **`timestep!(integrator, Δt)`**: the ext must intercept these for Reactant architectures,
  compile the step once, and drive a host loop (SpeedyWeather pattern; Oceananigans errors out of
  its `run!` instead — we want ours to *work*).
- **Clock**: `tick!` on a plain `Clock{Float64}` mutates a plain float — invisible to the compiled
  program. With the `ConcreteRNumber` clock from `OceananigansReactantExt` + its traced `tick!`,
  time advances *inside* the compiled step. Needs the `default_clock(grid)` hook (§2.1).
- **`ColumnRingGrid` construction**: builds a `RectilinearGrid(arch, ...)` (covered upstream) but
  also calls `on_architecture(arch, rings)` / `(arch, mask)`. The rings/mask are only used for
  CPU-side pre/post-processing (mask indexing in `RingGrids.Field` conversion, `sum(mask)` at
  construction) — keep them on the **CPU** under `ReactantState` (mirrors how the land-sea mask is
  deliberately kept on CPU in the GPU example). RingGrids ≥ the version with
  `RingGridsReactantExt` (from SpeedyWeather PR #970) would also allow device transfer, but we
  don't need it in kernels.
- **Output/diagnostics path** (`RingGrids.Field(arch, field, grid)`, plotting, `JLD2Writer`): all
  logical-mask indexing — must materialize with `Array(...)`/`on_architecture(CPU(), ...)` first.
  Not needed for correctness tests; document as a known limitation initially.
- **Boundary conditions**: `PrescribedSurfaceTemperature(:T_ub, f)` with `f(x, t)` a smooth
  function traces through `getbc`/`compute_z_bcs!` like any Oceananigans BC. The
  `soil_heat_global.jl` BC additionally does `lon_device[round(Int, x)]` — a traced integer gather;
  should lower to `stablehlo.gather` but is a known risk (§5). The correctness test starts with the
  purely functional variant.

---

## 3. Implementation plan

Phased so that every phase lands as one focused PR with green CI (ColPrac / AGENTS.md rule 12).

### Phase 0 — Spike (no PR, throwaway branch)

Timebox: a day. In a scratch environment with `Reactant` + current Terrarium:

1. `grid = ColumnGrid(ReactantState(), Float32, ExponentialSpacing(N = 10))` → catalog what breaks
   in grid construction and `StateVariables` allocation.
2. `@code_hlo` / `@jit timestep!(integrator, Δt)` on the phase-1 model with items 2.2.1–2.2.2
   patched locally → catalog tracing failures (this tells us whether our KA kernels trace at all,
   the single biggest unknown, cf. Reactant.jl#2364).
3. Record findings in `PROGRESS_reactant.md`; adjust the phase ordering below if the spike
   surfaces a structural blocker (e.g. `launch!` on our wrapped grids not tracing).

**Exit criterion:** we know the exact error list between us and a compiled `timestep!`.

### Phase A — Correctness-test infrastructure (test-first, as requested)

New directory `test/reactant/` with its **own Project.toml** (Reactant must not enter the root
`[deps]`; the main `Pkg.test()` run stays Reactant-free):

```
test/reactant/
├── Project.toml        # Terrarium (dev'd), Reactant, Oceananigans, RingGrids, Test, Statistics
├── runtests.jl         # config: NF, NSTEPS, RTOL/ATOL; includes the files below
├── setup.jl            # model registry: build (model, bcs, initializers, Δt) per ModelType & arch
└── correctness.jl      # generic comparison machinery + test_model entry point
```

Design (generalizing `SpeedyWeather/test/reactant/`, made flexible across Terrarium models):

- **Model registry** — one method per tested configuration, keyed by a symbol or config type so
  new models are added by adding a method, nothing else:

  ```julia
  # returns everything `initialize` needs; arch is the only degree of freedom
  function build_model(::Val{:soil_heat_global}, arch, NF)
      rings = RingGrids.FullGaussianGrid(...)   # small synthetic grid, e.g. N24
      mask  = <deterministic synthetic land mask>          # no NetCDF inputs in CI
      grid  = ColumnRingGrid(arch, NF, ExponentialSpacing(N = 20), rings, mask)
      model = SoilModel(grid)
      bcs   = PrescribedSurfaceTemperature(:T_ub, (x, t) -> ...)  # smooth sinusoidal fn of x, t
      inits = (temperature = (x, z) -> ...,)
      return (; model, boundary_conditions = bcs, initializers = inits, Δt = NF(600))
  end
  ```

  First registered configs: `:soil_heat_column` (single `ColumnGrid` column — the minimal
  debugging target) and `:soil_heat_global` (the requested `ColumnRingGrid` setup from
  `examples/simulations/soil_heat_global.jl`, with the file-based ERA5 mask replaced by a
  synthetic one and the array-lookup BC replaced by a pure function of `(x, t)` initially).
  Later: `:soil_energy_water` (hydrology on), `:land_column` (full LandModel).

- **State sync** — `sync_state!(integrator_cpu, integrator_reactant)`: copy every prognostic,
  auxiliary, and input `Field` (recursing into namespaces, reusing the traversal already in
  `Base.copyto!(::StateVariables, ::StateVariables)`) via `copyto!(interior(dst),
  Array(interior(src)))`, plus clock fields. Ensures both integrators step from identical state
  regardless of who was initialized first.

- **Comparison** — `compare_states(cpu, reactant; rtol, atol)` walks the three variable groups +
  namespaces and returns `Dict{Symbol, NamedTuple}` of
  `(max_abs_diff, mean_abs_diff, max_rel_diff, mean_rel_diff, matches)` exactly like SpeedyWeather's
  `compare_arrays`, printed in the test log for diagnosis.

- **Tests per model** (mirroring `test_tendencies!` / `test_time_stepping!`):
  1. *Initialization*: `initialize(model, ...)` on both architectures → compare initial state.
  2. *Tendencies*: sync, one `update_state!` (compiled on the Reactant side) → compare
     `state.tendencies`.
  3. *Time stepping*: sync, `NSTEPS` steps (compiled `timestep!` in a host loop vs plain CPU
     `run!`) → compare prognostic + auxiliary variables and clock time/iteration.
  - Tolerances as constants in `runtests.jl` (start `RTOL = 1e-3`, `ATOL = 1e-8` at Float32,
    tighten empirically — pure heat conduction should be far less noisy than a spectral
    atmosphere).

- Until Phase C lands, the Reactant side can be exercised with an explicit
  `r_step! = @compile timestep!(integrator, Δt)` inside the test helper — i.e. the tests are
  written **against the target user API but with a temporary local compile shim**, so this phase
  can merge before the ext exists (marked `@test_broken`/skipped in CI until Phase C).

### Phase B — `TerrariumReactantExt`: construction path (everything outside the trace)

- `Project.toml`: add `Reactant` to `[weakdeps]` + `TerrariumReactantExt = "Reactant"` to
  `[extensions]` (+ `[compat] Reactant = "0.2.x"` matching Oceananigans' bound). *No change to
  `[deps]`* — `ReactantCore` (for `@trace` in `src/`) is deliberately deferred until something in
  `src/` actually needs it; so far all traced-loop code lives in the ext where full Reactant is
  available (AGENTS.md rule 13).
- `ext/TerrariumReactantExt/` skeleton (submodule files mirroring the upstream ext layout):
  - `architectures.jl` — re-export/plumbing; `const ReactantLandGrid{NF} =
    AbstractLandGrid{NF, <:ReactantState}` etc.
  - `grids.jl` — `on_architecture(::ReactantState, ::ColumnGrid/::ColumnRingGrid)` (transfer the
    wrapped `RectilinearGrid` via upstream ext; keep `rings`/`mask` on CPU); constructor path so
    `ColumnRingGrid(ReactantState(), NF, spacing, rings, mask)` builds on CPU and transfers
    (upstream `LatitudeLongitudeGrid(::ReactantState)` pattern); `traced_type_inner` /
    `make_tracer` rules if the spike shows they're needed.
  - `state_variables.jl` — `default_clock(grid::ReactantLandGrid)` returning the
    `ConcreteRNumber`-backed clock via `Oceananigans` ext's `Clock(grid)`; any `set!`/initializer
    fixes that don't already come from upstream `set_to_function!`.
  - `diagnostics.jl` — no-op `debugsite!`/`checkfinite!` for Reactant fields.
- `src/` hook: replace the two hardcoded `Clock(time = zero(NF))` defaults with
  `default_clock(grid)` (default implementation preserves current behavior exactly).
- **Milestone/tests:** phase-A "Initialization" tests pass — model + state construct on
  `ReactantState`, fields are `ConcreteRArray`-backed, initializers applied, initial states match
  CPU.

### Phase C — Traced `timestep!` + hidden compilation (the core)

- `src/` kernel-safety edits (each a tiny, behavior-preserving diff, testable by the *existing*
  CPU suite):
  1. `ifelse` rewrites 2.2.1, 2.2.2 (soil energy closures) — required for phase-1 model;
  2. 2.2.5 (RRE boundary indices), 2.2.6 (`findfirst_z`), 2.2.8 (canopy interception) — same
     pattern, do them in the same sweep since AGENTS.md already mandates `ifelse` in kernels and
     these are pre-existing violations;
  3. leave 2.2.7 (Thermodynamics phase) and 2.2.9 (fixed-point `while`) for the LandModel phase,
     with `# TODO(reactant)` markers.
- `ext/TerrariumReactantExt/integrator.jl`:
  - `timestep!(integrator::ReactantIntegrator, Δt; finalize)` — look up / build the compiled step:

    ```julia
    r_step! = get!(COMPILED_STEPS, cache_key(integrator, Δt)) do
        @compile timestep_core!(integrator, Δt)
    end
    r_step!(integrator, Δt)
    ```

    where `timestep_core!` is the existing `timestep!(integrator, timestepper, Δt)` +
    `compute_auxiliary!` finalization, and `COMPILED_STEPS` is an ext-level `IdDict` keyed by
    (integrator identity, Δt value) — Δt is a trace constant, so a new Δt recompiles (documented;
    revisit with `ConcreteRNumber` Δt if it becomes annoying).
  - `run!(integrator::ReactantIntegrator; steps, period, Δt)` — compute `steps` host-side
    (unchanged `get_steps`), compile once, host loop `for _ in 1:steps r_step!(...) end`
    (SpeedyWeather's proven pattern; upgrade to `@trace for` once upstream is reliable — keep the
    switch in one function).
  - `initialize!(integrator::ReactantIntegrator)` — ensure the init sequence (`reset!`,
    `fill_halo_regions!`, user initializers, model initializer incl. the `invclosure!` kernel)
    runs correctly; individual kernel launches here may be `@jit`-wrapped or run through a
    one-shot compile — init is once-per-simulation, performance-irrelevant.
- **Milestone/tests:** all Phase-A correctness tests green for `:soil_heat_column` and
  `:soil_heat_global`; user-facing API identical to CPU.

### Phase D — CI, docs, examples

- `.github/workflows/CI_Reactant_Correctness.yml` (adapted from SpeedyWeather's): CPU-only runner,
  Julia 1.11, `julia --project=test/reactant test/reactant/runtests.jl`; path-filtered or
  label-triggered if runtime is heavy (Reactant compilation is minutes-scale).
- `docs/src/reactant.md`: usage (change one line — the architecture), what happens under the hood,
  limitations (no debug mode, recompile on Δt change, output writers need CPU materialization,
  unsupported components list).
- Extend `examples/simulations/soil_heat_global.jl` with a note or add a small
  `soil_heat_global_reactant.jl` variant (optional, could be deferred).
- AGENTS.md: add Reactant to the critical-rules section (no `while` on state values in kernels;
  new value-conditionals must be `ifelse`; run `test/reactant` when touching kernels).

### Phase E — Outlook (separate planning once C is green)

1. **More models in the registry**: SoilModel with active hydrology (needs C-2 rewrites +
   Richards-equation closures audit), `LandModel` (blocked on 2.2.7 + 2.2.9), vegetation.
2. **Inputs**: `FieldTimeSeriesInputSource` under Reactant (time interpolation = traced gather;
   or update inputs host-side between compiled steps — likely the pragmatic first cut).
3. **Differentiation through Reactant** (`Enzyme.autodiff` on the compiled step — SpeedyWeather's
   `differentation.jl` analog), connecting to the existing `test/differentiability` goals.
4. **GPU/TPU via Reactant** (`Reactant.set_default_backend("gpu")`) — should be free once CPU
   correctness holds; add a GPU CI job like SpeedyWeather's `test/GPU/reactant.jl`.
5. **Performance benchmarking** vs. plain CPU/CUDA paths; `@trace for` step loops with
   checkpointing.
6. **Parameter tracking** (`track_numbers`) for parameter estimation without recompilation.

---

## 4. Deliverables / PR breakdown

| PR | Contents | Merge gate |
|---|---|---|
| 1 | Phase A: `test/reactant/` infra (registry, sync, compare, test_model) with local compile shim | Runs locally; skipped/`test_broken` in default CI |
| 2 | Phase C-1/2 `ifelse` rewrites in `src/` (no Reactant anywhere) | Existing full test suite green (pure refactor) |
| 3 | Phase B ext skeleton + grids + clock + Project.toml weakdep | Phase-A init tests green |
| 4 | Phase C integrator: compiled `timestep!`/`run!` + cache | Full `test/reactant` green |
| 5 | Phase D: CI workflow + docs page + AGENTS.md | Docs build; CI job green |

(3 and 4 may merge as one if the ext stays small.)

---

## 5. Risks & open questions (please review)

1. **Do our KA kernels trace?** Biggest unknown; Oceananigans' own kernels do, but e.g.
   `interpolate!` doesn't (Reactant.jl#2364). Our kernels are simple stencils but pass NamedTuples
   of `Field`s/views (`ground_temperature` is a `@view field[:, :, Nz]`). → Phase 0 spike answers
   this before we commit to the phasing.
2. **`ContinuousBoundaryFunction` tracing** (`getbc` evaluating a Julia closure per boundary
   point): expected to work for pure functions; the `round(Int, x)`-indexed lon/lat lookup from
   `soil_heat_global.jl` is riskier (traced gather). Fallback: `Field`-valued BCs updated
   host-side between steps.
3. **Version churn**: Reactant moves fast (Oceananigans pins `0.2.236`); our Oceananigans compat
   is `0.100–0.110` — the Reactant path should be gated on recent Oceananigans (0.110+) and we
   should expect occasional upstream breakage. The separate test env isolates this from core CI.
4. **Compiled-step cache** keyed by integrator identity: a user mutating the model between calls
   (e.g. `initialize(integrator, params)`) gets a *new* integrator → new compile — correct but
   potentially surprising (silent multi-second pause). Mitigation: `@info` on compile, like
   SpeedyWeather.
5. **Δt as trace constant**: `run!(...; Δt = 600.0)` then `Δt = 900.0` recompiles. Acceptable?
   (Alternative: promote Δt to `ConcreteRNumber` from the start — slightly more ext code, avoids
   recompiles, matches how Oceananigans handles `last_Δt`.)
6. **Scope of "minimal user contact"**: is `run!`/`timestep!` coverage enough for v1, or must
   `Simulation(integrator)` + output writers also work under Reactant? (Oceananigans currently
   *errors* on `run!(::ReactantSimulation)`, so full `Simulation` support means going beyond
   upstream; I propose: v1 = `run!`/`timestep!` on the integrator, `Simulation` documented as
   unsupported.)
7. **Where the correctness tests run**: separate `test/reactant` env + dedicated CI workflow
   (SpeedyWeather model) vs. a `test_args=["reactant"]` group like the Enzyme tests. Plan assumes
   the former (keeps Reactant out of `test/Project.toml`); flag if you prefer the latter.
