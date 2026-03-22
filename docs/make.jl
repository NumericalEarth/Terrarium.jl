using ArgParse
using Documenter
using Literate
using PlutoStaticHTML

using Terrarium

const EXAMPLE_NOTEBOOK_DIR = joinpath(dirname(@__DIR__), "examples", "notebooks")
const DOCS_EXAMPLE_DIR = joinpath(@__DIR__, "src", "notebooks")
const EXAMPLE_DIR_RELATIVE = joinpath("notebooks")

s = ArgParseSettings()
@add_arg_table! s begin
    "--local", "-l"
    action = :store_true
    help = "Local docs build mode"
    "--draft", "-d"
    action = :store_true
    help = "Whether to build docs in draft mode, i.e. skipping execution of examples and doctests"
end
parsed_args = parse_args(ARGS, s)

IS_LOCAL = parsed_args["local"] || parse(Bool, get(ENV, "LOCALDOCS", "false"))
IS_DRAFT = parsed_args["draft"] || parse(Bool, get(ENV, "DRAFTDOCS", "false"))
BUILD_DOCS_NOTEBOOKS = !IS_DRAFT && parse(Bool, get(ENV, "BUILD_DOCS_NOTEBOOKS", "true"))
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

# lookup table for all Pluto notebooks to be included
notebook_lookup = if BUILD_DOCS_NOTEBOOKS
    Dict(
        "Model Interface" => "example_model_notebook.md",
        #    "Differentiating Terrarium" => "differentiate-notebook.md",
    )
else
    Dict()
end

# notebooks to be build (.jl files)
notebooks_files = []
for (title, name) in notebook_lookup
    push!(notebooks_files, replace(name, ".md" => ".jl"))
end

"""
Run all Pluto notebooks (".jl" files) in `NOTEBOOK_DIR`.
"""
function build_notebook_doc_pages()
    println("Building notebooks in $EXAMPLE_NOTEBOOK_DIR and moving them to $DOCS_EXAMPLE_DIR")
    oopts = OutputOptions(; append_build_context = false)
    output_format = documenter_output
    bopts = BuildOptions(EXAMPLE_NOTEBOOK_DIR; output_format)
    build_notebooks(bopts, notebooks_files, oopts)

    # move to docs/src/notebooks because for some reason that's needed
    mkpath(DOCS_EXAMPLE_DIR)
    for (_, file_name) in notebook_lookup
        mv(joinpath(EXAMPLE_NOTEBOOK_DIR, file_name), joinpath(DOCS_EXAMPLE_DIR, file_name), force = true)
    end

    return nothing
end

# Build the notebooks; defaults to true.
if BUILD_DOCS_NOTEBOOKS
    build_notebook_doc_pages()
end

# Dict for makedocs for notebooks to be included
notebook_docpages = Pair{String, String}[]
for (title, name) in notebook_lookup
    push!(notebook_docpages, title => joinpath(EXAMPLE_DIR_RELATIVE, name))
end

makedocs(
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        ansicolor = true,
        collapselevel = 1,
        canonical = "https://tum-pik-esm.github.io/Terrarium.jl/stable/",
        size_threshold = 600_000,
        # Using MathJax3 since Pluto uses that engine too.
        mathengine = Documenter.MathJax3(),
    ),      # in bytes
    sitename = "Terrarium.jl",
    authors = "Brian Groenke, Maximillian Galbrecht, Maha Badri, and Contributors",
    modules = [Terrarium],
    pages = [
        "Home" => "index.md",
        "Overview" => [
            "Baisc concepts" => "introduction/basic_concepts.md",
            "Numerical core" => "introduction/numerical_core.md",
            "Mathematical formulation" => "introduction/mathematical_formulation.md",
        ],
        "Extending Terrarium" => [
            "Core interfaces" => "extending/core_interfaces.md",
        ],
        "Processes" => [
            "Soil" => [
                "Soil stratigraphy" => "processes/soil/soil_stratigraphy.md",
                "Soil hydrology" => "processes/soil/soil_hydrology.md",
                "Soil energy" => "processes/soil/soil_energy.md",
            ],
            "Vegetation" => [
                "Photosynthesis and gas exchange" => "processes/vegetation/photosynthesis.md",
                "Stomatal conductance" => "processes/vegetation/stomatal_conductance.md",
                "Plant available water" => "processes/vegetation/plant_available_water.md",
                "Autotrophic respiration" => "processes/vegetation/autotrophic_respiration.md",
                "Carbon dynamics" => "processes/vegetation/carbon_dynamics.md",
                "Vegetation dynamics" => "processes/vegetation/vegetation_dynamics.md",
                "Vegetation phenology" => "processes/vegetation/vegetation_phenology.md",
                "Root distribution" => "processes/vegetation/root_distribution.md",
            ],
            "Surface hydrology" => [
                "Surface hydrology" => "processes/surface_hydrology/surface_hydrology.md",
                "Canopy interception" => "processes/surface_hydrology/canopy_interception.md",
                "Evapotranspiration" => "processes/surface_hydrology/evapotranspiration.md",
                "Surface runoff" => "processes/surface_hydrology/surface_runoff.md",
            ],
            "Surface energy balance" => [
                "Surface energy balance" => "processes/surface_energy/surface_energy_balance.md",
                "Radiative fluxes" => "processes/surface_energy/radiative_fluxes.md",
                "Turbulent fluxes" => "processes/surface_energy/turbulent_fluxes.md",
                "Skin temperature and ground heat" => "processes/surface_energy/skin_temperature.md",
                "Albedo and emissivity" => "processes/surface_energy/albedo.md",
            ],
        ],
        "Models" => [

        ],
        "Examples" => [
            "Overview" => "examples_overview.md",
            notebook_docpages...,
        ],
        "Contributing" => "contributing.md",
        "API Reference" => "api_reference.md",
    ],
    draft = IS_DRAFT,
)

deployconfig = Documenter.auto_detect_deploy_system()

# remove gitignore from build files
# rm(joinpath(@__DIR__, "build", ".gitignore"))

deploydocs(
    repo = "github.com/NumericalEarth/Terrarium.jl.git",
    push_preview = true,
    versions = ["v0" => "v^", "v#.#", "dev" => "dev"],
    deploy_config = deployconfig,
)
