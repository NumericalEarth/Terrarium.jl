const USAGE = """
Native step times and host allocations per step of several configurations, for before/after comparisons.

Usage: julia --project=. step_times.jl [key=value ...]

Builds a throwaway model first, so that the first-model slowdown (see `compile_order.jl`) does not fall on
the first configuration measured, then appends one Markdown table row per configuration and grid size: the
median ms/step of `reps` repetitions with min and max, and the allocations of one step. To compare two
versions of the code, run this once on each (for example from two worktrees), on the same machine, with
different labels, and with separate output files if the runs are concurrent.

Options:
  arch=cpu                                    `cpu` or `gpu`
  configs=land,land_no_vegetation,soil_heat   configurations (see ../model_configurations.jl)
  nlat=4,24                                   grid sizes (`nlat_half`)
  nsteps=0                                    steps per repetition; 0 picks a number that takes similar
                                              time at every size (at most 200)
  reps=5                                      repetitions
  label=                                      free-text label for the first column
  header=false                                print the table header first
  out=                                        file to append the rows to (default: standard output only)
"""
include(joinpath(@__DIR__, "common.jl"))

const OPTIONS = parse_options(
    (
        arch = "cpu", configs = [:land, :land_no_vegetation, :soil_heat], nlat = [4, 24], nsteps = 0, reps = 5,
        label = "", header = false, out = "",
    ),
    USAGE
)
const ARCH = architecture(OPTIONS.arch)

absorb_first_model_slowdown(ARCH)

with_output(OPTIONS.out) do io
    if OPTIONS.header
        say(io, environment_summary())
        say(io, "| label | arch | config | columns | steps/rep | ms/step median (min, max) | allocations per step |")
        say(io, "| --- | --- | --- | --- | --- | --- | --- |")
    end
    for config in OPTIONS.configs, nlat in OPTIONS.nlat
        integrator, Δt = setup(config, ARCH; nlat_half = nlat)
        nsteps = OPTIONS.nsteps > 0 ? OPTIONS.nsteps : default_nsteps(nlat, 30; hi = 200)
        allocs = @allocations timestep!(integrator, Δt; finalize = false)
        t = time_steps(integrator, Δt, ARCH; nsteps, reps = OPTIONS.reps)
        say(io, @sprintf("| %s | %s | %s | %d | %d | %s | %d |", OPTIONS.label, OPTIONS.arch, config, ncolumns(nlat), nsteps, fmt_times(t), allocs))
    end
end
