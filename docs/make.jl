using ArgParse
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Literate

using Terrarium

const EXAMPLES_DIR = joinpath(dirname(@__DIR__), "examples")
const EXAMPLES_OUTDIR = joinpath(@__DIR__, "src", "examples")
const EXAMPLES_OUTDIR_RELATIVE = "examples"

s = ArgParseSettings()
@add_arg_table! s begin
    "--local", "-l"
    action = :store_true
    help = "Local docs build mode"
    "--draft", "-d"
    action = :store_true
    help = "Whether to build docs in draft mode, i.e. skipping execution of examples and doctests"
    "--skip-examples", "-e"
    action = :store_true
    help = "Like --draft but applies only to example scripts"
    "--check-links", "-c"
    action = :store_true
    help = "Check that all external links are functional (`linkcheck=true` in `makedocs`)"
    "--debug"
    action = :store_true
    help = "Enable Documenter.jl debug logging for more detailed output from examples and doctests"
end
parsed_args = parse_args(ARGS, s)

IS_LOCAL = parsed_args["local"] || parse(Bool, get(ENV, "LOCALDOCS", "false"))
IS_DRAFT = parsed_args["draft"] || parse(Bool, get(ENV, "DRAFTDOCS", "false"))
SKIP_EXAMPLES = parsed_args["skip-examples"] || parse(Bool, get(ENV, "SKIP_EXAMPLES", "false"))
CHECK_LINKS = parsed_args["check-links"] || parse(Bool, get(ENV, "CHECK_LINKS", "false"))
BUILD_EXAMPLE_DOCS = !IS_DRAFT && !SKIP_EXAMPLES
if haskey(ENV, "GITHUB_ACTIONS") || parsed_args["debug"]
    ENV["JULIA_DEBUG"] = "Documenter"
end

# Literate scripts to be built and included in the docs
# Each entry should be a Pair: page_title => script_filename
running_scripts = [
    "Soil column heat conduction" => "soil_heat_column.jl",
    "Global soil heat conduction" => "soil_heat_global.jl",
]
extending_scripts = [
    "Simple exponential growth model" => "linear_ode_exp_growth.jl",
    "Degree-day snow melt model" => "simple_snow_ddm.jl",
    "Linear heat conduction" => "linear_heat_conduction.jl",
]

"""
Convert Literate.jl scripts in `indir` to markdown pages in `outdir`.
The differentiation example is never executed during docs builds (Enzyme compile is
too slow); the model interface example is executed unless IS_DRAFT is set.
"""
function build_literate_pages!(outdir, indir, scripts; kwargs...)
    mkpath(outdir)
    for (_, filename) in scripts
        ## the differentiation notebook is never auto-executed (Enzyme compile time)
        should_execute = BUILD_EXAMPLE_DOCS && filename != "differentiating_terrarium.jl"
        literate_kwargs = Dict{Symbol, Any}(
            :execute => should_execute,
            :documenter => true,
            :flavor => Literate.DocumenterFlavor(),
        )
        ## For non-executed scripts, use plain julia code fences so Documenter
        ## does not attempt to run them as @example blocks.
        if !should_execute
            literate_kwargs[:codefence] = "```julia" => "```"
        end
        merge!(literate_kwargs, Dict(kwargs))
        Literate.markdown(
            joinpath(indir, filename),
            outdir;
            literate_kwargs...,
        )
    end
    return nothing
end

# We'll append the following postamble to the literate examples, to include
# information about the computing environment used to run them.
example_postamble = """

# ---

# ### Julia version and environment information
#
# This example was executed with the following version of Julia:

using InteractiveUtils: versioninfo
versioninfo()

# These were the top-level packages installed in the environment:

import Pkg
Pkg.status()
"""

# Pages vector for makedocs
running_example_docpages = Pair{String, String}[]
extending_example_docpages = Pair{String, String}[]

# Temporary solution: copy input files to src
@info "Copying input files to $(EXAMPLES_OUTDIR)"
mkpath(EXAMPLES_OUTDIR)
cp("inputs", joinpath(EXAMPLES_OUTDIR, "inputs"), force = true)

# Build example pages with Literate.jl
build_literate_pages!(
    EXAMPLES_OUTDIR,
    joinpath(EXAMPLES_DIR, "simulations"),
    running_scripts;
    preprocess = content -> content * example_postamble,
)
build_literate_pages!(
    EXAMPLES_OUTDIR,
    joinpath(EXAMPLES_DIR, "extending"),
    extending_scripts;
    preprocess = content -> content * example_postamble,

)

# Add example pages to lists
for (title, filename) in running_scripts
    mdfile = replace(filename, ".jl" => ".md")
    push!(running_example_docpages, "Example: $title" => joinpath(EXAMPLES_OUTDIR_RELATIVE, mdfile))
end
for (title, filename) in extending_scripts
    mdfile = replace(filename, ".jl" => ".md")
    push!(extending_example_docpages, "Example: $title" => joinpath(EXAMPLES_OUTDIR_RELATIVE, mdfile))
end

# Create bibliography
bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style = :numeric
)

# Add documentation interlinking
links = InterLinks(
    "Oceananigans" => "https://clima.github.io/OceananigansDocumentation/stable/",
    "KernelAbstractions" => "https://juliagpu.github.io/KernelAbstractions.jl/stable/",
    "SpeedyWeather" => "https://speedyweather.github.io/SpeedyWeatherDocumentation/stable/",
    "FreezeCurves" => "https://cryogrid.github.io/FreezeCurves.jl/stable/",
    "Thermodynamics" => "https://clima.github.io/Thermodynamics.jl/stable/",
)


makedocs(
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        ansicolor = true,
        collapselevel = 1,
        repolink = "https://github.com/NumericalEarth/Terrarium.jl",
        canonical = "https://numericalearth.github.io/Terrarium.jl",
        assets = ["assets/citations.css"],
        size_threshold_warn = 500 * 1024, # 500 KiB
        size_threshold = 3 * 1024^2, # 3 MiB
        mathengine = Documenter.MathJax3(),
    ),
    sitename = "Terrarium.jl",
    authors = "Brian Groenke, Maximilian Gelbrecht, Maha Badri, and Contributors",
    modules = [Terrarium],
    plugins = [bib, links],
    pages = [
        "Home" => "index.md",
        "Introduction" => [
            "Basic concepts" => "introduction/basic_concepts.md",
            "Numerical core" => "introduction/numerical_core.md",
            "Mathematical formulation" => "introduction/mathematical_formulation.md",
        ],
        "Running Terrarium" => [
            "Configuring models" => "running/configuring.md",
            "Initialization" => "running/initialization.md",
            "Time stepping" => "running/time_stepping.md",
            "Input sources" => "running/input_sources.md",
            running_example_docpages...,
        ],
        "Extending Terrarium" => [
            "Core interfaces" => "extending/core_interfaces.md",
            "State variables" => "extending/state_variables.md",
            "Implementing processes" => "extending/implementing_processes.md",
            "Coupling processes" => "extending/coupling_processes.md",
            extending_example_docpages...,
        ],
        "Models" => [
            "Land model" => "models/land_model.md",
            "Soil model" => "models/soil_model.md",
            "Vegetation model" => "models/vegetation_model.md",
        ],
        "Processes" => [
            "Soil" => [
                "Overview" => "processes/soil/soil.md",
                "Stratigraphy" => "processes/soil/soil_stratigraphy.md",
                "Hydrology" => "processes/soil/soil_hydrology.md",
                "Energy balance" => "processes/soil/soil_energy.md",
                "Biogeochemistry" => "processes/soil/soil_biogeochemistry.md",
            ],
            "Vegetation" => [
                "Overview" => "processes/vegetation/vegetation.md",
                "Photosynthesis" => "processes/vegetation/photosynthesis.md",
                "Stomatal conductance" => "processes/vegetation/stomatal_conductance.md",
                "Plant available water" => "processes/vegetation/plant_available_water.md",
                "Autotrophic respiration" => "processes/vegetation/autotrophic_respiration.md",
                "Carbon dynamics" => "processes/vegetation/carbon_dynamics.md",
                "Vegetation dynamics" => "processes/vegetation/vegetation_dynamics.md",
                "Phenology" => "processes/vegetation/phenology.md",
                "Root distribution" => "processes/vegetation/root_distribution.md",
            ],
            "Surface hydrology" => [
                "Overview" => "processes/surface_hydrology/surface_hydrology.md",
                "Canopy interception" => "processes/surface_hydrology/canopy_interception.md",
                "Evapotranspiration" => "processes/surface_hydrology/evapotranspiration.md",
                "Surface runoff" => "processes/surface_hydrology/surface_runoff.md",
            ],
            "Surface energy balance" => [
                "Overview" => "processes/surface_energy/surface_energy_balance.md",
                "Radiative fluxes" => "processes/surface_energy/radiative_fluxes.md",
                "Turbulent fluxes" => "processes/surface_energy/turbulent_fluxes.md",
                "Skin temperature" => "processes/surface_energy/skin_temperature.md",
                "Albedo and emissivity" => "processes/surface_energy/albedo.md",
            ],
            "Coupling to atmosphere" => [
                "Atmosphere interface" => "processes/atmosphere/atmosphere.md",
                "Aerodynamics" => "processes/atmosphere/aerodynamics.md",
            ],
            "Utilities" => [
                "Constants" => "processes/utils/physical_constants.md",
                "Physics" => "processes/utils/physics_utils.md",
            ],
        ],
        "Contributing" => "contributing.md",
        "Index of API" => "api_index.md",
        "References" => "references.md",
    ],
    linkcheck = CHECK_LINKS,
    warnonly = [:cross_references],
    draft = IS_DRAFT,
)

deployconfig = Documenter.auto_detect_deploy_system()

deploydocs(
    repo = "github.com/NumericalEarth/Terrarium.jl.git",
    push_preview = true,
    versions = ["v0" => "v^", "v#.#", "dev" => "dev"],
    deploy_config = deployconfig,
)
