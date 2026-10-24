const USAGE = """
Step time of one model configuration compiled with Reactant, with compile time and repetitions.

Usage: julia --project=. reactant_timing.jl [key=value ...]

Compiles `run_timesteps!` for `nsteps` steps, runs it once as a warm-up, then times `reps` runs from a
re-initialized state, and appends one Markdown table row: compile time, median ms/step with min and max,
the number of OS threads of the process (XLA starts its own thread pool), the CPU affinity, and whether
all prognostic values are finite at the end.

Options:
  config=soil_heat     configuration (see ../model_configurations.jl)
  backend=cpu          Reactant backend: `cpu` or `gpu`
  nlat=4               grid size (`nlat_half`)
  nz=30                number of soil layers
  nsteps=0             steps per run; 0 picks a number that takes similar time at every size
  reps=5               repetitions
  label=               free-text label for the first column
  header=false         print the table header first
  out=                 file to append the row to (default: standard output only)

To restrict XLA's CPU backend to one core, run under `taskset -c <core>`. To inspect the compiled program,
set `XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text"`. Compiling a `land` configuration takes
about 20 minutes on the CPU backend.
"""
using CUDA  # provides the KernelAbstractions glue for Reactant, also on the CPU
using Reactant
include(joinpath(@__DIR__, "common.jl"))

const OPTIONS = parse_options(
    (config = :soil_heat, backend = "cpu", nlat = 4, nz = 30, nsteps = 0, reps = 5, label = "", header = false, out = ""),
    USAGE
)
Reactant.set_default_backend(OPTIONS.backend)

"Number of OS threads of this process (Linux only)."
function os_threads()
    try
        return parse(Int, split(only(filter(startswith("Threads:"), readlines("/proc/self/status"))))[2])
    catch
        return -1
    end
end

cfg = build_model(OPTIONS.config, ReactantState(), Float32; nlat_half = OPTIONS.nlat, nz = OPTIONS.nz)
integrator = initialize(cfg.model; boundary_conditions = cfg.boundary_conditions, initializers = cfg.initializers)
Δt = Float32(cfg.Δt)
nsteps = OPTIONS.nsteps > 0 ? OPTIONS.nsteps : default_nsteps(OPTIONS.nlat, OPTIONS.nz)

compile_time = @elapsed compiled = @compile raise = true raise_first = true sync = true Terrarium.run_timesteps!(integrator, Δt, nsteps, false)
compiled(integrator, Δt, nsteps, false)
times = Float64[]
for _ in 1:OPTIONS.reps
    Terrarium.initialize!(integrator)
    t = @elapsed compiled(integrator, Δt, nsteps, false)
    push!(times, 1.0e3 * t / nsteps)
end
finite = all(values(Terrarium.prognostic_fields(integrator.state, integrator.model))) do f
    !(f isa Oceananigans.Fields.AbstractField) || all(isfinite, Array(parent(f)))
end

with_output(OPTIONS.out) do io
    if OPTIONS.header
        say(io, environment_summary())
        say(io, "| label | config | backend | columns | steps/run | reps | compile time | ms/step median (min, max) | OS threads | all finite |")
        say(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    end
    say(io, @sprintf("| %s | %s | %s | %d | %d | %d | %.1f s | %s | %d | %s |",
        OPTIONS.label, OPTIONS.config, OPTIONS.backend, ncolumns(OPTIONS.nlat), nsteps, OPTIONS.reps, compile_time,
        fmt_times((median = median(times), min = minimum(times), max = maximum(times))), os_threads(), finite))
end
