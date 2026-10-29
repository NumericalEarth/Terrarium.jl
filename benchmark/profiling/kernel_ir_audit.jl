const USAGE = """
Static audit of the LLVM IR of every native CPU kernel launched in one time step.

Usage: julia --project=. kernel_ir_audit.jl [key=value ...]

Captures every distinct kernel instance launched during one step of each configuration and counts, in its
LLVM IR: Float64 operations, `pow` calls, `exp`/`log` calls, allocation sites, dynamic calls, and throw
sites. Also runs JET's `report_opt` on `update_state!` (host-side runtime dispatch) and writes its report to
`<out>.jet_<config>.txt`.

Options:
  configs=soil_heat,land_no_vegetation,land   configurations (see ../model_configurations.jl)
  nlat=2                                      grid size (`nlat_half`)
  dump=                                       comma-separated substrings of kernel names whose IR is
                                              written to `<out>.<kernel>.ll`
  jet=true                                    run JET's `report_opt`
  out=kernel_ir_audit                         prefix of the output files; the table goes to `<out>.md`

The IR is that of a fully specialized kernel instance (`ka_code_llvm`), so it does not show kernels that
lose specialization at run time; `kernel_allocations.jl` finds those.
"""
import KernelAbstractions as KA

## Record every CPU kernel launch while `CAPTURE[]` is set. This wraps KernelAbstractions' CPU launch
## operator before anything is compiled, so that no world-age issues arise.
const CAPTURE = Ref(false)
const CAPTURED = Dict{String, Any}()
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

const OPTIONS = parse_options(
    (configs = [:soil_heat, :land_no_vegetation, :land], nlat = 2, dump = String[], jet = true, out = "kernel_ir_audit"),
    USAGE
)
const OUT = OPTIONS.out

const PATTERNS = (
    f64 = r"\b(fadd|fsub|fmul|fdiv|fcmp|fneg|fpext|sitofp|uitofp)\b[^\n]*\bdouble\b|call [^\n]*@llvm\.[a-z]+\.f64",
    pow = r"call [^\n]*(@j_pow|@julia_pow|@llvm\.pow|powf|@pow)|call [^\n]*@j_\^",
    transcendental = r"call [^\n]*(@j_exp|@j_log|@julia_exp|@julia_log|@llvm\.exp|@llvm\.log|expf|logf|@j_exp_impl|@j_log_impl)",
    alloc = r"(ijl_gc_small_alloc|ijl_gc_pool_alloc|ijl_gc_big_alloc|julia\.gc_alloc_obj|ijl_box_|jl_box_)",
    dispatch = r"(ijl_apply_generic|ijl_invoke|ijl_f_|jl_f_|ijl_get_nth_field_checked)",
    throws = r"(ijl_throw|ijl_bounds_error|throw_boundserror|ijl_type_error)",
)
count_matches(ir, rx) = length(collect(eachmatch(rx, ir)))

open(OUT * ".md", "w") do io
    say(io, environment_summary())
    for config in OPTIONS.configs
        integrator, Δt = setup(config, CPU(); nlat_half = OPTIONS.nlat)
        empty!(CAPTURED)
        CAPTURE[] = true
        timestep!(integrator, Δt; finalize = false)
        CAPTURE[] = false
        say(io, "\n### $config: $(length(CAPTURED)) distinct kernel instances")
        say(io, "| kernel | IR lines | f64 ops | pow calls | exp/log calls | alloc sites | dynamic calls | throw sites |")
        say(io, "| --- | --- | --- | --- | --- | --- | --- | --- |")
        rows = []
        for (name, obj, args, ndrange, workgroupsize) in values(CAPTURED)
            buf = IOBuffer()
            try
                KA.ka_code_llvm(buf, obj, map(typeof, args); ndrange, workgroupsize, debuginfo = :none)
            catch err
                say(io, "| $name | failed: $(first(sprint(showerror, err), 80)) | | | | | | |")
                continue
            end
            ir = String(take!(buf))
            push!(rows, (name, count('\n', ir), map(rx -> count_matches(ir, rx), PATTERNS)))
            if any(s -> occursin(s, name), OPTIONS.dump)
                write(OUT * "." * name * ".ll", ir)
            end
        end
        for (name, nlines, c) in sort(rows; by = r -> -r[3].alloc)
            say(io, "| $name | $nlines | $(c.f64) | $(c.pow) | $(c.transcendental) | $(c.alloc) | $(c.dispatch) | $(c.throws) |")
        end
        if OPTIONS.jet
            state, model, inputs = integrator.state, integrator.model, integrator.inputs
            report = JET.report_opt(update_state!, (typeof(state), typeof(model), typeof(inputs)); target_modules = (Terrarium,))
            say(io, "\nJET report_opt on update_state! for $config (target_modules = Terrarium): $(length(JET.get_reports(report))) reports")
            open(OUT * ".jet_$(config).txt", "w") do jio
                show(IOContext(jio, :limit => true, :displaysize => (2000, 300)), report)
            end
        end
    end
end
