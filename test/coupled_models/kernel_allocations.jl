using Terrarium
using Test

# Kernels must not allocate. A kernel whose trailing arguments are a pass-through `args...` is not
# specialized on them by Julia, which then boxes every argument for every cell; on the CPU this shows up
# as allocations proportional to the number of grid cells. Host-side allocations do not depend on the
# grid size, so comparing two grid sizes isolates allocations inside kernels.

function step_allocations(f, model, ncolumns)
    grid = ColumnGrid(CPU(), Float32, ExponentialSpacing(Δz_min = 0.05f0, Δz_max = 1.0f0, N = 10), ncolumns)
    integrator = initialize(model(grid))
    state, m = integrator.state, integrator.model
    f(state, m)
    return @allocations f(state, m)
end

@testset "Kernel allocations do not scale with the grid size" begin
    soil_model(grid) = SoilModel(grid)
    ## Richards hydrology, so that the soil hydraulics kernel is covered as well
    function land_model(grid)
        NF = eltype(grid)
        soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
        return LandModel(grid; soil, vegetation = nothing, snow = SingleLayerSnow(NF))
    end
    for (name, model) in (("SoilModel", soil_model), ("LandModel without vegetation", land_model))
        for (phase, f) in (("closure!", Terrarium.closure!), ("compute_auxiliary!", Terrarium.compute_auxiliary!))
            @testset "$name, $phase" begin
                @test step_allocations(f, model, 8) == step_allocations(f, model, 32)
            end
        end
    end
end
