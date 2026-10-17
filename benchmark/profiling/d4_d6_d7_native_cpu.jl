# D4 (native solver parity), D6 (Float32 anomaly at 4608 columns), D7 (variance), native CPU.
# usage: julia --project=. d4_d6_d7_native_cpu.jl <outfile>
include(joinpath(@__DIR__, "common.jl"))
const OUT = ARGS[1]

function count_subnormals(integrator)
    n = 0
    total = 0
    for f in values(Terrarium.prognostic_fields(integrator.state, integrator.model))
        f isa Oceananigans.Fields.AbstractField || continue
        a = interior(f)
        n += count(issubnormal, a)
        total += length(a)
    end
    return n, total
end

function case(io, label, config; nlat_half, nz = 30, NF = Float32, model_kwargs = (;), nsteps, reps, ftz = false)
    set_zero_subnormals(ftz)
    integrator, Δt = setup(config, CPU(), NF; nlat_half, nz, model_kwargs)
    t = time_steps(integrator, Δt, CPU(); nsteps, reps)
    sub, tot = count_subnormals(integrator)
    set_zero_subnormals(false)
    say(io, @sprintf("| %s | %s | %d | %d | %s | %s | %d | %s | %d/%d |", label, config, ncolumns(nlat_half), nz, NF, ftz, reps, fmt_times(t), sub, tot))
    return t
end

open(OUT, "w") do io
    say(io, "| case | config | columns | L | NF | FTZ | reps | ms/step median (min, max) | subnormal prognostic values |")
    say(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    # D4: RootSolver (native default) vs NewtonSolver(5) (Reactant benchmark setting)
    for (nlat, nsteps, reps) in ((4, 200, 5), (24, 20, 5), (48, 6, 3))
        case(io, "D4 root", :land; nlat_half = nlat, nsteps, reps, model_kwargs = root_solver_kwargs(Float32))
        case(io, "D4 newton", :land; nlat_half = nlat, nsteps, reps, model_kwargs = newton_kwargs(Float32))
    end
    # D6/D7: the 4608-column Float32 anomaly and its neighbors
    case(io, "D6 FTZ", :land; nlat_half = 24, nsteps = 20, reps = 5, ftz = true)
    case(io, "D6 Float64", :land; nlat_half = 24, NF = Float64, nsteps = 20, reps = 5)
    case(io, "D7 neighbor", :land; nlat_half = 16, nsteps = 40, reps = 3)
    case(io, "D7 neighbor", :land; nlat_half = 32, nsteps = 10, reps = 3)
    case(io, "D7 layers", :land; nlat_half = 24, nz = 20, nsteps = 20, reps = 3)
    case(io, "D7 layers", :land; nlat_half = 24, nz = 60, nsteps = 10, reps = 3)
    case(io, "D7 variance", :land_no_vegetation; nlat_half = 24, nsteps = 20, reps = 5)
    case(io, "D7 variance", :soil_heat; nlat_half = 24, nsteps = 40, reps = 5)
end
