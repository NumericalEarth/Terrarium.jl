const USAGE = """
Allocation count and allocation sites of one phase of a native CPU time step.

Usage: julia --project=. allocation_sites.jl [key=value ...]

Reports the allocations of one call of the chosen phase at each grid size and, at the first grid size,
the most frequent allocation sites (stack traces) and allocated types, from `Profile.Allocs`.

Options:
  config=soil_heat     configuration (see ../model_configurations.jl)
  phase=closure        `closure`, `compute_auxiliary`, `compute_boundary_conditions`, `compute_tendencies`,
                       or `timestep` (a whole step)
  nlat=2,4             grid sizes (`nlat_half`); allocations that grow with the size come from kernels
  top=6                number of sites and types listed
  depth=6              stack frames shown per site
  out=                 file to append the results to (default: standard output only)
"""
include(joinpath(@__DIR__, "common.jl"))
using Profile

const OPTIONS = parse_options((config = :soil_heat, phase = :closure, nlat = [2, 4], top = 6, depth = 6, out = ""), USAGE)

function phase_function(phase, integrator, Δt)
    state, model = integrator.state, integrator.model
    phase == :closure && return () -> Terrarium.closure!(state, model)
    phase == :compute_auxiliary && return () -> Terrarium.compute_auxiliary!(state, model)
    phase == :compute_boundary_conditions && return () -> Terrarium.compute_boundary_conditions!(state, model)
    phase == :compute_tendencies && return () -> Terrarium.compute_tendencies!(state, model)
    phase == :timestep && return () -> timestep!(integrator, Δt; finalize = false)
    return error("unknown phase `$phase`; see `--help`")
end

with_output(OPTIONS.out) do io
    say(io, environment_summary())
    for (i, nlat) in enumerate(OPTIONS.nlat)
        integrator, Δt = setup(OPTIONS.config, CPU(); nlat_half = nlat)
        f = phase_function(OPTIONS.phase, integrator, Δt)
        f()
        bytes = @allocated f()
        n = @allocations f()
        say(io, "$(OPTIONS.config) $(OPTIONS.phase) at $(ncolumns(nlat)) columns: $n allocations, $bytes bytes")
        i == 1 || continue
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate = 1 f()
        allocs = Profile.Allocs.fetch().allocs
        sites = Dict{String, Int}()
        types = Dict{String, Int}()
        for a in allocs
            frames = [fr for fr in a.stacktrace if !occursin("Profile", string(fr.file)) && !occursin("boot.jl", string(fr.file))]
            site = join(["$(fr.func)@$(basename(string(fr.file))):$(fr.line)" for fr in first(frames, OPTIONS.depth)], " <- ")
            sites[site] = get(sites, site, 0) + 1
            type = first(string(a.type), 120)
            types[type] = get(types, type, 0) + 1
        end
        for (site, count) in first(sort(collect(sites); by = x -> -x[2]), OPTIONS.top)
            say(io, "  $count × $site")
        end
        for (type, count) in first(sort(collect(types); by = x -> -x[2]), OPTIONS.top)
            say(io, "  type $count × $type")
        end
    end
end
