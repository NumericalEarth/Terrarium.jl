# D2 (GPU host/device split, registers, spills), D6 on GPU (f64 in PTX), D7 on GPU (variance).
# usage: julia --project=. d2_gpu_profile.jl <outprefix>
using CUDA
include(joinpath(@__DIR__, "common.jl"))
const OUT = ARGS[1]
const ARCH = GPU()

# --- 1. PTX of every kernel compiled while building and stepping each configuration ---------------
const F64_ARITH = r"\b(add|sub|mul|div|fma|neg|abs|min|max|sqrt|rcp|rsqrt|setp\.[a-z]+|ex2|lg2)(\.[a-z]+)*\.f64\b"
const CVT_F64 = r"\bcvt(\.[a-z]+)*\.f64\.f32\b|\bcvt(\.[a-z]+)*\.f32\.f64\b"
const LOCAL = r"\b(ld|st)\.local\b"
const POW = r"__nv_pow|__nv_powf"
function ptx_stats(ptx)
    rows = []
    for part in split(ptx, ".entry ")[2:end]
        name = String(match(r"^\s*(\S+?)\(", part).captures[1])
        short = something(match(r"(cpu_|gpu_)?[A-Za-z_]+?_kernel[^\d]*|_fill_[a-z_]+_halo|_compute_z_bcs|gpu_[a-z_]+", name), (; match = name)).match
        push!(rows, (short = short, name = name,
            f64 = length(collect(eachmatch(F64_ARITH, part))),
            cvt = length(collect(eachmatch(CVT_F64, part))),
            localops = length(collect(eachmatch(LOCAL, part))),
            localdecl = occursin(r"\.local\s", part),
            pow = length(collect(eachmatch(POW, part))),
            lines = count('\n', part)))
    end
    return rows
end

open(OUT * "_ptx.md", "w") do io
    say(io, "| config | kernel | PTX lines | f64 arith | f32↔f64 cvt | local ld/st | pow calls |")
    say(io, "| --- | --- | --- | --- | --- | --- | --- |")
    for config in (:soil_heat, :land_no_vegetation, :land)
        buf = IOBuffer()
        CUDA.@device_code_ptx io = buf begin
            integrator, Δt = setup(config, ARCH; nlat_half = 4, warmup = 1)
        end
        ptx = String(take!(buf))
        write(OUT * "_$(config).ptx", ptx)
        for r in sort(ptx_stats(ptx); by = r -> -(r.f64 + r.cvt + r.localops))
            say(io, "| $config | `$(r.short)` | $(r.lines) | $(r.f64) | $(r.cvt) | $(r.localops) | $(r.pow) |")
        end
    end
end

# --- 2. timings across resolutions, with repeats -----------------------------------------------
open(OUT * "_timings.md", "w") do io
    say(io, "| config | columns | steps/rep | reps | ms/step median (min, max) |")
    say(io, "| --- | --- | --- | --- | --- |")
    for config in (:soil_heat, :land_no_vegetation, :land), nlat in (4, 8, 16, 24, 32, 72, 144)
        integrator, Δt = setup(config, ARCH; nlat_half = nlat)
        nsteps = clamp(round(Int, 3.0e7 / (ncolumns(nlat) * 30)), 20, 200)
        t = time_steps(integrator, Δt, ARCH; nsteps, reps = 5)
        say(io, @sprintf("| %s | %d | %d | 5 | %s |", config, ncolumns(nlat), nsteps, fmt_times(t)))
    end
end

# --- 3. profiles: host API time vs device time, and the kernel trace with registers --------------
open(OUT * "_profiles.txt", "w") do io
    ctx = IOContext(io, :displaysize => (5000, 400), :limit => false)
    for config in (:soil_heat, :land), nlat in (4, 144)
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
        println(ctx, "\n\n================ $config, $(ncolumns(nlat)) columns, 5 steps (unprofiled wall time $(round(1e3 * wall / 5; digits = 3)) ms/step) ================")
        show(ctx, res)
    end
    integrator, Δt = setup(:land, ARCH; nlat_half = 144)
    res = CUDA.@profile trace = true begin
        timestep!(integrator, Δt; finalize = false)
        CUDA.synchronize()
    end
    println(ctx, "\n\n================ TRACE: land, $(ncolumns(144)) columns, 1 step ================")
    show(ctx, res)
end
say(stdout, "d2 done")
