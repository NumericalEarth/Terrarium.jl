# Does `sqrt` (instead of `x^(1/2)`) in `InverseQuadratic` still break Reactant when the soil
# conductivities are traced device scalars (promote-after-init, as in examples/autodiff/global_sensitivity.jl)?
# usage: julia --project=. reactant_sqrt.jl <pow|sqrt> <outfile>
using CUDA
using Reactant
using Enzyme
Reactant.set_default_backend("cpu")
include(joinpath(@__DIR__, "common.jl"))
const MODE, OUT = ARGS[1], ARGS[2]

if MODE == "sqrt"
    @eval (f::Terrarium.InverseQuadratic)(xs, weights) = sum(Terrarium.fastmap((x, w) -> sqrt(x) * w, xs, weights))^2
end

ext = Base.get_extension(Reactant, :ReactantCUDAExt)
traced_scalar = ext === nothing ? nothing : ext.CuTracedRNumber{Float32, 1}
has_sqrt = traced_scalar === nothing ? "n/a (ReactantCUDAExt not loaded)" : string(hasmethod(sqrt, Tuple{traced_scalar}))

grid = benchmark_grid(ReactantState(), Float32; nlat_half = 4, nz = 30)
model_with(conductivities) = SoilModel(grid;
    soil = SoilEnergyWaterCarbon(Float32;
        energy = SoilThermodynamics(Float32; thermal_properties = SoilThermalProperties(Float32; conductivities))))
cfg = build_model(:soil_heat, CPU(), Float32; nlat_half = 4, nz = 30)  # reuse the forcing and initializers only
integrator0 = initialize(model_with(SoilThermalConductivities(Float32)); boundary_conditions = cfg.boundary_conditions, initializers = cfg.initializers)
promoted = Reactant.to_rarray(SoilThermalConductivities(Float32); track_numbers = true)
integrator = ModelIntegrator(integrator0.clock, model_with(promoted), integrator0.inputs, integrator0.state, integrator0.initializers)
Δt, nsteps = Float32(600), 10

function mean_temperature(integrator, Δt, nsteps)
    Terrarium.run_timesteps!(integrator, Δt, nsteps, false)
    T = interior(integrator.state.temperature)
    return sum(T) / length(T)
end
function grad!(integrator, dintegrator, Δt, nsteps)
    _, value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), mean_temperature, Enzyme.Active,
        Enzyme.Duplicated(integrator, dintegrator), Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return value
end

function attempt(f)
    try
        return "ok: " * string(f())
    catch err
        msg = sprint(showerror, err)
        return "FAILED: " * replace(first(msg, 300), '\n' => ' ')
    end
end

fwd = attempt() do
    compiled = @compile raise = true raise_first = true sync = true mean_temperature(integrator, Δt, nsteps)
    Reactant.to_number(compiled(integrator, Δt, nsteps))
end
dintegrator = Enzyme.make_zero(integrator)
rev = attempt() do
    compiled = @compile raise = true raise_first = true sync = true grad!(integrator, dintegrator, Δt, nsteps)
    compiled(integrator, dintegrator, Δt, nsteps)
    (∂T_∂κ_mineral = Reactant.to_number(dintegrator.model.soil.energy.thermal_properties.conductivities.mineral),
     ∂T_∂κ_water = Reactant.to_number(dintegrator.model.soil.energy.thermal_properties.conductivities.water))
end
open(OUT, "a") do io
    say(io, "| $MODE | $has_sqrt | $fwd | $rev |")
end
