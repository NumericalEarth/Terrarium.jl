const USAGE = """
Native step time of one model configuration and setting, with repetitions.

Usage: julia --project=. native_timing.jl [key=value ...]

Times `nsteps` consecutive steps `reps` times after a warm-up and appends one Markdown table row: the
median ms/step with its min and max, and the number of prognostic values that are subnormal at the end
(subnormal arithmetic can be much slower on the CPU). Run it once per setting to compare settings, for
example the two skin temperature solvers, Float32 against Float64, or flushing subnormals to zero.

Options:
  config=land            configuration (see ../model_configurations.jl)
  arch=cpu               `cpu` or `gpu`
  nlat=4                 grid size (`nlat_half`)
  nz=30                  number of soil layers
  float=Float32          `Float32` or `Float64`
  solver=default         skin temperature solver: `default`, `newton`, or `root` (see `solver_kwargs`)
  ftz=false              flush subnormals to zero (`set_zero_subnormals`; CPU only)
  nsteps=0               steps per repetition; 0 picks a number that takes similar time at every size
  reps=5                 repetitions
  label=                 free-text label for the first column
  header=false           print the table header first
  out=                   file to append the row to (default: standard output only)

Run each case in its own process, pinned to a core (`taskset -c <core>`), so that cases do not share a
compilation history (see `compile_order.jl`) or disturb each other.
"""
include(joinpath(@__DIR__, "common.jl"))

const OPTIONS = parse_options(
    (
        config = :land, arch = "cpu", nlat = 4, nz = 30, float = "Float32", solver = :default, ftz = false,
        nsteps = 0, reps = 5, label = "", header = false, out = "",
    ),
    USAGE
)

function count_subnormals(integrator)
    n = 0
    total = 0
    for f in values(Terrarium.prognostic_fields(integrator.state, integrator.model))
        f isa Oceananigans.Fields.AbstractField || continue
        a = Array(interior(f))
        n += count(issubnormal, a)
        total += length(a)
    end
    return n, total
end

const ARCH = architecture(OPTIONS.arch)
const NF = float_type(OPTIONS.float)
nsteps = OPTIONS.nsteps > 0 ? OPTIONS.nsteps : default_nsteps(OPTIONS.nlat, OPTIONS.nz; hi = 200)

with_output(OPTIONS.out) do io
    if OPTIONS.header
        say(io, environment_summary())
        say(io, "| label | config | arch | columns | layers | float | solver | FTZ | steps/rep | reps | ms/step median (min, max) | subnormal prognostic values |")
        say(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    end
    set_zero_subnormals(OPTIONS.ftz)
    integrator, Δt = setup(OPTIONS.config, ARCH, NF; nlat_half = OPTIONS.nlat, nz = OPTIONS.nz, model_kwargs = solver_kwargs(OPTIONS.solver, NF))
    t = time_steps(integrator, Δt, ARCH; nsteps, reps = OPTIONS.reps)
    nsub, total = count_subnormals(integrator)
    set_zero_subnormals(false)
    say(
        io, @sprintf(
            "| %s | %s | %s | %d | %d | %s | %s | %s | %d | %d | %s | %d/%d |",
            OPTIONS.label, OPTIONS.config, OPTIONS.arch, ncolumns(OPTIONS.nlat), OPTIONS.nz, NF, OPTIONS.solver, OPTIONS.ftz,
            nsteps, OPTIONS.reps, fmt_times(t), nsub, total
        )
    )
end
