# Allocation count and allocation sites of `closure!` for `soil_heat` on CPU.
# usage: julia --project=. closure_allocations.jl
include(joinpath(@__DIR__, "common.jl"))
using Profile
for nlat in (2, 4)
    integrator, Δt = setup(:soil_heat, CPU(); nlat_half = nlat)
    state, model = integrator.state, integrator.model
    Terrarium.closure!(state, model)
    a = @allocated Terrarium.closure!(state, model)
    n = @allocations Terrarium.closure!(state, model)
    println("soil_heat closure! at $(ncolumns(nlat)) columns: $n allocations, $a bytes")
    println("hydrology closure type: ", typeof(Terrarium.get_closure(model.soil.hydrology)), " hydrology: ", nameof(typeof(model.soil.hydrology)))
    if nlat == 2
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate = 1 Terrarium.closure!(state, model)
        allocs = Profile.Allocs.fetch().allocs
        counts = Dict{String, Int}()
        for al in allocs
            frames = [fr for fr in al.stacktrace if !occursin("Profile", string(fr.file)) && !occursin("boot.jl", string(fr.file))]
            key = join(["$(fr.func)@$(basename(string(fr.file))):$(fr.line)" for fr in frames[1:min(end, 6)]], " <- ")
            counts[key] = get(counts, key, 0) + 1
        end
        for (k, c) in first(sort(collect(counts); by = x -> -x[2]), 6)
            println("  $c × $k")
        end
        tc = Dict{String, Int}()
        for al in allocs
            t = string(al.type); t = t[1:min(end, 120)]
            tc[t] = get(tc, t, 0) + 1
        end
        for (k, c) in first(sort(collect(tc); by = x -> -x[2]), 6)
            println("  type $c × $k")
        end
    end
end
