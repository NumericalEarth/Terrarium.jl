# D6 (and H9): static inspection of every CPU kernel launched in one step: Float64 arithmetic,
# allocations, dynamic dispatch, and pow calls in the LLVM IR; plus JET's optimization report.
# usage: julia --project=. d6_kernel_ir_audit.jl <outfile>
import KernelAbstractions as KA
const CAPTURE = Ref(false)
const CAPTURED = Dict{String, Any}()
# Wrap KA's CPU launch operator before anything is compiled, so no world-age issues arise.
function (obj::KA.Kernel{KA.CPU})(args...; ndrange = nothing, workgroupsize = nothing)
    if CAPTURE[]
        name = string(nameof(obj.f))
        key = name * "#" * string(hash(map(typeof, args)))
        haskey(CAPTURED, key) || (CAPTURED[key] = (name, obj, args, ndrange, workgroupsize))
    end
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    if length(KA.blocks(iterspace)) == 0
        return nothing
    end
    KA.__run(obj, ndrange, iterspace, args, dynamic, obj.backend.static)
    return nothing
end

include(joinpath(@__DIR__, "common.jl"))
using JET
using Oceananigans.TimeSteppers: update_state!
const OUT = ARGS[1]

const PATTERNS = (
    f64 = r"\b(fadd|fsub|fmul|fdiv|fcmp|fneg|fpext|sitofp|uitofp)\b[^\n]*\bdouble\b|call [^\n]*@llvm\.[a-z]+\.f64",
    pow = r"call [^\n]*(@j_pow|@julia_pow|@llvm\.pow|powf|@pow)|call [^\n]*@j_\^",
    transcendental = r"call [^\n]*(@j_exp|@j_log|@julia_exp|@julia_log|@llvm\.exp|@llvm\.log|expf|logf|@j_exp_impl|@j_log_impl)",
    alloc = r"(ijl_gc_small_alloc|ijl_gc_pool_alloc|ijl_gc_big_alloc|julia\.gc_alloc_obj|ijl_box_|jl_box_)",
    dispatch = r"(ijl_apply_generic|ijl_invoke|ijl_f_|jl_f_|ijl_get_nth_field_checked)",
    throws = r"(ijl_throw|ijl_bounds_error|throw_boundserror|ijl_type_error)",
)
count_matches(ir, rx) = length(collect(eachmatch(rx, ir)))

open(OUT, "w") do io
    for config in (:soil_heat, :land_no_vegetation, :land)
        integrator, Δt = setup(config, CPU(); nlat_half = 2)
        empty!(CAPTURED)
        CAPTURE[] = true
        timestep!(integrator, Δt; finalize = false)
        CAPTURE[] = false
        say(io, "\n### $config: $(length(CAPTURED)) distinct kernel instances")
        say(io, "| kernel | IR lines | f64 ops | pow calls | exp/log calls | alloc sites | dynamic calls | throw sites |")
        say(io, "| --- | --- | --- | --- | --- | --- | --- | --- |")
        rows = []
        for (key, (name, obj, args, ndrange, workgroupsize)) in CAPTURED
            buf = IOBuffer()
            try
                KA.ka_code_llvm(buf, obj, map(typeof, args); ndrange, workgroupsize, debuginfo = :none)
            catch err
                say(io, "| $name | failed: $(sprint(showerror, err)[1:min(end, 80)]) | | | | | | |")
                continue
            end
            ir = String(take!(buf))
            push!(rows, (name, count('\n', ir), map(rx -> count_matches(ir, rx), PATTERNS)))
            if name == "cpu_energy_to_temperature_kernel!" || occursin("solve_surface_energy_balance", name)
                write(OUT * "." * name * ".ll", ir)
            end
        end
        for (name, nlines, c) in sort(rows; by = r -> -r[3].alloc)
            say(io, "| $name | $nlines | $(c.f64) | $(c.pow) | $(c.transcendental) | $(c.alloc) | $(c.dispatch) | $(c.throws) |")
        end
        # JET optimization analysis of the host-side step (runtime dispatch etc.)
        state, model, inputs = integrator.state, integrator.model, integrator.inputs
        rep = JET.report_opt(update_state!, (typeof(state), typeof(model), typeof(inputs)); target_modules = (Terrarium,))
        reports = JET.get_reports(rep)
        say(io, "\nJET report_opt on update_state! for $config (target_modules = Terrarium): $(length(reports)) reports")
        open(OUT * ".jet_$(config).txt", "w") do jio
            show(IOContext(jio, :limit => true, :displaysize => (2000, 300)), rep)
        end
    end
end
