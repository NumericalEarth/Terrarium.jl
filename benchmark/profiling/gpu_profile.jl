const USAGE = """
Native GPU profile: PTX statistics per kernel, step times across grid sizes, and CUDA profiles.

Usage: julia --project=. gpu_profile.jl [key=value ...]

Sections (select with `sections=`):
  ptx       PTX of every kernel compiled while building and stepping each configuration, with counts of
            Float64 arithmetic, Float32↔Float64 conversions, local-memory loads and stores, and `pow` calls;
            writes `<out>_ptx.md` and the full PTX to `<out>_<config>.ptx`
  timings   ms/step (median of five repetitions, with min and max) at each grid size; `<out>_timings.md`
  profiles  `CUDA.@profile` summaries (host API time against device time) at the `profile_nlat` sizes,
            and a trace of one step of `trace_config` with register counts; `<out>_profiles.txt`

Options:
  configs=soil_heat,land_no_vegetation,land   configurations (see ../model_configurations.jl)
  sections=ptx,timings,profiles               sections to run
  ptx_nlat=4                                  grid size (`nlat_half`) at which kernels are compiled for `ptx`
  nlat=4,8,16,24,32,72,144                    grid sizes for `timings`
  profile_nlat=4,144                          grid sizes for `profiles`
  trace_config=land                           configuration traced in `profiles`
  trace_nlat=144                              grid size of the trace
  out=gpu_profile                             prefix of the output files
"""
using CUDA
include(joinpath(@__DIR__, "common.jl"))

const OPTIONS = parse_options(
    (
        configs = [:soil_heat, :land_no_vegetation, :land], sections = [:ptx, :timings, :profiles], ptx_nlat = 4,
        nlat = [4, 8, 16, 24, 32, 72, 144], profile_nlat = [4, 144], trace_config = :land, trace_nlat = 144, out = "gpu_profile",
    ),
    USAGE
)
const ARCH = GPU()
const OUT = OPTIONS.out

const F64_ARITH = r"\b(add|sub|mul|div|fma|neg|abs|min|max|sqrt|rcp|rsqrt|setp\.[a-z]+|ex2|lg2)(\.[a-z]+)*\.f64\b"
const CVT_F64 = r"\bcvt(\.[a-z]+)*\.f64\.f32\b|\bcvt(\.[a-z]+)*\.f32\.f64\b"
const LOCAL = r"\b(ld|st)\.local\b"
const POW = r"__nv_pow|__nv_powf"
count_matches(rx, s) = length(collect(eachmatch(rx, s)))

function ptx_stats(ptx)
    rows = []
    for part in split(ptx, ".entry ")[2:end]
        name = String(match(r"^\s*(\S+?)\(", part).captures[1])
        short = something(match(r"(cpu_|gpu_)?[A-Za-z_]+?_kernel[^\d]*|_fill_[a-z_]+_halo|_compute_z_bcs|gpu_[a-z_]+", name), (; match = name)).match
        push!(rows, (
            short = short, f64 = count_matches(F64_ARITH, part), cvt = count_matches(CVT_F64, part),
            localops = count_matches(LOCAL, part), pow = count_matches(POW, part), lines = count('\n', part),
        ))
    end
    return rows
end

if :ptx in OPTIONS.sections
    open(OUT * "_ptx.md", "w") do io
        say(io, environment_summary())
        say(io, "| config | kernel | PTX lines | f64 arith | f32↔f64 cvt | local ld/st | pow calls |")
        say(io, "| --- | --- | --- | --- | --- | --- | --- |")
        for config in OPTIONS.configs
            buf = IOBuffer()
            CUDA.@device_code_ptx io = buf setup(config, ARCH; nlat_half = OPTIONS.ptx_nlat, warmup = 1)
            ptx = String(take!(buf))
            write(OUT * "_$(config).ptx", ptx)
            for r in sort(ptx_stats(ptx); by = r -> -(r.f64 + r.cvt + r.localops))
                say(io, "| $config | `$(r.short)` | $(r.lines) | $(r.f64) | $(r.cvt) | $(r.localops) | $(r.pow) |")
            end
        end
    end
end

if :timings in OPTIONS.sections
    open(OUT * "_timings.md", "w") do io
        say(io, environment_summary())
        say(io, "| config | columns | steps/rep | reps | ms/step median (min, max) |")
        say(io, "| --- | --- | --- | --- | --- |")
        absorb_first_model_slowdown(ARCH)
        for config in OPTIONS.configs, nlat in OPTIONS.nlat
            integrator, Δt = setup(config, ARCH; nlat_half = nlat)
            nsteps = default_nsteps(nlat, 30; hi = 200)
            t = time_steps(integrator, Δt, ARCH; nsteps, reps = 5)
            say(io, @sprintf("| %s | %d | %d | 5 | %s |", config, ncolumns(nlat), nsteps, fmt_times(t)))
        end
    end
end

if :profiles in OPTIONS.sections
    open(OUT * "_profiles.txt", "w") do io
        ctx = IOContext(io, :displaysize => (5000, 400), :limit => false)
        println(ctx, environment_summary())
        for config in OPTIONS.configs, nlat in OPTIONS.profile_nlat
            integrator, Δt = setup(config, ARCH; nlat_half = nlat)
            wall = @elapsed begin
                for _ in 1:5
                    timestep!(integrator, Δt; finalize = false)
                end
                CUDA.synchronize()
            end
            res = CUDA.@profile begin
                for _ in 1:5
                    timestep!(integrator, Δt; finalize = false)
                end
                CUDA.synchronize()
            end
            println(ctx, "\n\n================ $config, $(ncolumns(nlat)) columns, 5 steps (unprofiled wall time $(round(1.0e3 * wall / 5; digits = 3)) ms/step) ================")
            show(ctx, res)
        end
        integrator, Δt = setup(OPTIONS.trace_config, ARCH; nlat_half = OPTIONS.trace_nlat)
        res = CUDA.@profile trace = true begin
            timestep!(integrator, Δt; finalize = false)
            CUDA.synchronize()
        end
        println(ctx, "\n\n================ TRACE: $(OPTIONS.trace_config), $(ncolumns(OPTIONS.trace_nlat)) columns, 1 step ================")
        show(ctx, res)
    end
    say(stdout, "profiles written to $(OUT)_profiles.txt")
end
