# Generic CPU-vs-Reactant correctness machinery.
#
# Build the *same* model on two
# architectures, sync one state onto the other so both start identical, then advance both
# and compare every prognostic/auxiliary field within a tolerance (XLA reorders
# floating-point math, so exact equality is not expected).

using Statistics: mean

import Oceananigans
using Oceananigans: interior, architecture, on_architecture
using Oceananigans.Architectures: CPU
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField

const ReactantExt = Base.get_extension(Terrarium, :TerrariumReactantExt)

# --- scalar/array materialization to plain CPU ------------------------------------------
#
# Field/array data crosses architectures via `on_architecture(CPU(), …)` — Terrarium's canonical
# transfer primitive (AGENTS.md: never manual `Array()`/`CuArray()`). Clock scalars are the one
# case it does not cover: a traced `ConcreteRNumber` is unwrapped with `Reactant.to_number`.

_scalar(x::Number) = x
_scalar(x) = Reactant.to_number(x)   # ConcreteRNumber -> Julia number

_cpu_array(f::AbstractField) = Array(interior(on_architecture(CPU(), f)))
_cpu_array(a::AbstractArray) = Array(on_architecture(CPU(), a))
_cpu_array(::Any) = nothing          # skip non-array auxiliary entries (e.g. scalar refs)

# --- state traversal --------------------------------------------------------------------

# Collect all comparable field arrays (prognostic + auxiliary), recursing into namespaces,
# keyed by a dotted path so names stay unambiguous across namespaces.
function collect_state_arrays(state, prefix = "")
    out = Dict{String, Array}()
    _collect_group!(out, getfield(state, :prognostic), string(prefix, "prognostic."))
    _collect_group!(out, getfield(state, :auxiliary), string(prefix, "auxiliary."))
    for nsname in Terrarium.namespace_names(state)
        ns = getproperty(getfield(state, :namespaces), nsname)
        merge!(out, collect_state_arrays(ns, string(prefix, nsname, ".")))
    end
    return out
end

function _collect_group!(out, nt::NamedTuple, prefix)
    for name in keys(nt)
        arr = _cpu_array(nt[name])
        arr === nothing || (out[string(prefix, name)] = arr)
    end
    return out
end

# --- syncing src state onto dst (field data + clock), pairing by name -------------------

function sync_state!(dst, src)
    for group in (:prognostic, :tendencies, :auxiliary, :inputs)
        _sync_group!(getfield(dst, group), getfield(src, group))
    end
    for nsname in Terrarium.namespace_names(dst)
        sync_state!(
            getproperty(getfield(dst, :namespaces), nsname),
            getproperty(getfield(src, :namespaces), nsname)
        )
    end
    _sync_clock!(getfield(dst, :clock), getfield(src, :clock))
    return dst
end

function _sync_group!(dst_nt::NamedTuple, src_nt::NamedTuple)
    for name in keys(dst_nt)
        dst = dst_nt[name]
        # Only mutable Fields are synced directly; views (e.g. ground_temperature) track their
        # parent field and update automatically once the parent is synced, and lazy
        # `AbstractOperation`s (e.g. `kernel(...)`-backed auxiliaries such as `phenology_factor`)
        # recompute from their inputs, so they follow once those are synced.
        dst isa AbstractField && !(dst isa AbstractOperation) || continue
        # Move the source field onto the destination's architecture via `on_architecture`, then
        # copy interiors. `dst` is always a CPU field here (we sync Reactant → CPU).
        copyto!(interior(dst), interior(on_architecture(CPU(), src_nt[name])))
    end
    return dst_nt
end

function _sync_clock!(dst_clock, src_clock)
    dst_clock.time = convert(typeof(_scalar(dst_clock.time)), _scalar(src_clock.time))
    dst_clock.iteration = convert(typeof(_scalar(dst_clock.iteration)), _scalar(src_clock.iteration))
    return dst_clock
end

# --- comparison -------------------------------------------------------------------------

function compare_states(cpu_state, reactant_state; rtol, atol)
    a = collect_state_arrays(cpu_state)
    b = collect_state_arrays(reactant_state)
    results = Dict{String, NamedTuple}()
    for key in keys(a)
        haskey(b, key) || continue
        x, y = a[key], b[key]
        absdiff = abs.(x .- y)
        reldiff = absdiff ./ max.(abs.(x), abs.(y), eps(real(eltype(x))))
        results[key] = (
            max_abs_diff = maximum(absdiff),
            mean_abs_diff = mean(absdiff),
            max_rel_diff = maximum(reldiff),
            matches = isapprox(x, y; rtol, atol),
        )
    end
    return results
end

function report(results; label = "")
    isempty(label) || println("\n$label")
    for key in sort!(collect(keys(results)))
        r = results[key]
        flag = r.matches ? "✓" : "✗"
        println("  $flag $key   max_abs=$(r.max_abs_diff)  max_rel=$(r.max_rel_diff)")
    end
    return results
end

# --- top-level per-model test -----------------------------------------------------------
#
# Advancement uses `Terrarium.run!` directly and identically on both architectures — the only
# difference is the grid: `TerrariumReactantExt` overrides `run!` for `ReactantState` to compile
# the step once and drive a host loop; on CPU it is the plain loop.

"""
    test_model(config; nsteps, rtol, atol)

Build `config` on CPU and on `ReactantState`, sync Reactant→CPU so both start identical,
advance both by `nsteps`, and compare all fields. `config` is a `Val`-tag consumed by
`build_model` in `setup.jl`.
"""
function test_model(config::Symbol; nsteps = NSTEPS, rtol = RTOL, atol = ATOL, NF = DEFAULT_NF)
    name = string(config)
    println("\n" * "="^70)
    println("$name: CPU vs Reactant")
    println("="^70)

    cpu = build_integrator(Val(config), CPU(), NF)
    rea = build_integrator(Val(config), ReactantState(), NF)

    @testset "$name" begin
        # Regression guard: Test that the Reactant model and grid types are correct
        @testset "Reactant model and grid types" begin
            @test typeof(Terrarium.get_grid(rea.model)) <: ReactantExt.ReactantLandGrid
            @test typeof(rea.model) <: ReactantExt.ReactantModel
            @test typeof(rea) <: ReactantExt.ReactantIntegrator
        end

        @testset "initialization" begin
            # The device clock must be `ConcreteRNumber`-backed, so that it's traced correctly
            rea_clock = getfield(rea.state, :clock)
            @test rea_clock.time isa Reactant.ConcreteRNumber
            @test rea_clock.iteration isa Reactant.ConcreteRNumber

            init = compare_states(cpu.state, rea.state; rtol, atol)
            report(init; label = "Initial state diffs:")
            for (_, r) in init
                @test r.matches
            end
        end

        # start both from the (freshly initialized) Reactant state
        sync_state!(cpu.state, rea.state)

        Δt = cpu_dt(Val(config), NF)
        Terrarium.run!(cpu; steps = nsteps, Δt)
        Terrarium.run!(rea; steps = nsteps, Δt)

        @testset "after $nsteps steps" begin
            stepped = compare_states(cpu.state, rea.state; rtol, atol)
            report(stepped; label = "Stepped state diffs:")
            for (_, r) in stepped
                @test r.matches
            end
            @test _scalar(getfield(cpu.state, :clock).iteration) ==
                _scalar(getfield(rea.state, :clock).iteration)
        end
    end
    return nothing
end
