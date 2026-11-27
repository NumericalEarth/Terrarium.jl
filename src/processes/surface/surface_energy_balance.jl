"""
    $TYPEDEF

Standard implementation of the surface energy balance (SEB) that computes the radiative,
turbulent, and ground energy fluxes at the surface. The SEB is also responsible for defining
and solving the so-called *skin temperature* (effective emission temperature of the land surface)
as well as the albedo.

Each flux group is an exchangeable sub-process with (at least) a `Prescribed` and a `Diagnosed`
implementation, so that any subset of the surface fluxes may be supplied externally. Note that the
type parameters are declared in the same order as the fields.
"""
struct SurfaceEnergyBalance{
        NF,
        SkinTemperature <: AbstractSkinTemperature{NF},
        RadiativeFluxes <: AbstractRadiativeFluxes{NF},
        TurbulentFluxes <: AbstractTurbulentFluxes{NF},
        GroundHeatFlux <: AbstractGroundHeatFlux{NF},
        Albedo <: AbstractAlbedo{NF},
    } <: AbstractSurfaceEnergyBalance{NF}
    "Scheme for determining the skin temperature"
    skin_temperature::SkinTemperature

    "Scheme for determining the net radiation budget"
    radiative_fluxes::RadiativeFluxes

    "Scheme for computing turbulent (sensible and latent) heat fluxes"
    turbulent_fluxes::TurbulentFluxes

    "Scheme for determining the ground heat flux"
    ground_heat_flux::GroundHeatFlux

    "Scheme for parameterizing surface albedo"
    albedo::Albedo
end

"""
    $TYPEDSIGNATURES

Construct a `SurfaceEnergyBalance` with number format `NF`. The defaults reproduce the standalone
land model configuration: an implicitly solved skin temperature with all surface fluxes diagnosed
internally.

Not every combination of `skin_temperature` and `ground_heat_flux` is valid; the supported
pairs are documented on the surface energy balance documentation page.
"""
function SurfaceEnergyBalance(
        ::Type{NF};
        radiative_fluxes::AbstractRadiativeFluxes = DiagnosedRadiativeFluxes(NF),
        turbulent_fluxes::AbstractTurbulentFluxes = DiagnosedTurbulentFluxes(NF),
        skin_temperature::AbstractSkinTemperature = ImplicitSkinTemperature(NF),
        ground_heat_flux::AbstractGroundHeatFlux = DiagnosedGroundHeatFlux(NF),
        albedo::AbstractAlbedo = DiagnosticAlbedo(NF)
    ) where {NF}
    return SurfaceEnergyBalance(skin_temperature, radiative_fluxes, turbulent_fluxes, ground_heat_flux, albedo)
end

"""
    PrescribedSurfaceEnergyBalance{NF, Albedo}

Type alias for a [`SurfaceEnergyBalance`](@ref) in which *all four* surface energy fluxes, i.e. the
skin temperature, the radiative fluxes, the turbulent fluxes, and the ground heat flux, are supplied
externally as input variables. This is the configuration used when an external coupler owns the
atmosphere-land interface and Terrarium should not re-close the surface energy budget itself.

The albedo is deliberately left free: with prescribed radiative fluxes it only feeds a computation
that is itself prescribed, so it is inert.

!!! warning
    With all fluxes prescribed, the residual identity `G = R_net + H_s + H_l` is not enforced by
    construction. It is the caller's responsibility to supply a mutually consistent set of fluxes.
"""
const PrescribedSurfaceEnergyBalance{NF, Albedo} = SurfaceEnergyBalance{
    NF,
    <:PrescribedSkinTemperature{NF},
    <:PrescribedRadiativeFluxes{NF},
    <:PrescribedTurbulentFluxes{NF},
    <:PrescribedGroundHeatFlux{NF},
    Albedo,
}

"""
    $TYPEDSIGNATURES

Construct a [`PrescribedSurfaceEnergyBalance`](@ref) with number format `NF`.
"""
function PrescribedSurfaceEnergyBalance(::Type{NF}; albedo::AbstractAlbedo = ConstantAlbedo(NF)) where {NF}
    return SurfaceEnergyBalance(
        NF;
        skin_temperature = PrescribedSkinTemperature(NF),
        radiative_fluxes = PrescribedRadiativeFluxes(NF),
        turbulent_fluxes = PrescribedTurbulentFluxes(NF),
        ground_heat_flux = PrescribedGroundHeatFlux(NF),
        albedo
    )
end

variables(seb::SurfaceEnergyBalance) = tuplejoin(
    variables(seb.albedo),
    variables(seb.skin_temperature),
    variables(seb.radiative_fluxes),
    variables(seb.turbulent_fluxes),
    variables(seb.ground_heat_flux)
)

# Without snow, the surface is not split into tiles.
variables(seb::AbstractSurfaceEnergyBalance, ::Nothing) = variables(seb)

"""
    $TYPEDSIGNATURES

Variables of the surface energy balance coupled to an optional `snow` component. Without snow
(`nothing`), these are just the variables of `seb`. With snow, each grid cell is split into a snow-free
and a snow-covered surface tile, and the variable namespaces `snow_free` and `snow` (one per tile)
replicate the auxiliary variables declared by the surface energy balance (its skin temperature, fluxes,
albedo, and emissivity). Its input variables (e.g. `ground_temperature` or the inputs of prescribed
sub-processes) stay at the root and are shared by both tiles. The root
variables keep their grid-cell meaning (area-weighted over the tiles), and the root auxiliary
`snow_melt_energy_flux` is added for the surface-melt energy flux into the snowpack.
"""
function variables(seb::SurfaceEnergyBalance, ::AbstractSnow)
    root_vars = variables(seb)
    tile_vars = auxiliary_variables(root_vars)
    snow_vars = (
        auxiliary(
            :snow_melt_energy_flux,
            Surface(XY()),
            units = u"W/m^2",
            desc = "Grid-cell-mean surface-melt energy flux into the snowpack beyond snow-top conduction (positive upward)"
        ),
    )
    tile_namespaces = (
        namespace(:snow_free, tile_vars),
        namespace(:snow, tile_vars),
    )
    return tuplejoin(root_vars, snow_vars, tile_namespaces)
end

"""
    $TYPEDSIGNATURES

With all surface fluxes prescribed, the coupler owns the surface energy balance, so the surface is not
split into tiles and the variables are those of `seb` alone, with or without snow.
"""
variables(seb::PrescribedSurfaceEnergyBalance, ::AbstractSnow) = variables(seb)

# Tile dispatches of the state field accessors (see `get_fields` and `auxiliary_fields` in state_variables.jl)

"""
    $TYPEDSIGNATURES

Retrieves a single `NamedTuple` of `Field`s for evaluating kernel functions on the surface `tile`
(see [`SnowFree`](@ref) and [`SnowCovered`](@ref)). Kernel functions read tile variables by name (e.g.
`fields.skin_temperature` or `fields.albedo`), so a tile launch needs a single `fields` tuple in which
those names resolve to the fields of the tile namespace [`tile_name`](@ref)`(tile)`, while shared
quantities (the atmospheric forcing, `ground_temperature`, the snow state, or the evapotranspiration
conductances) still resolve to the root. This lets the unchanged kernel functions be evaluated per tile.

The returned tuple holds the non-tendency fields of `components` from the root of `state` (except those
named in `except`), merged with the prognostic and auxiliary fields of the tile namespace. Tile fields
shadow root fields of the same name (e.g. `albedo` or `skin_temperature`), so that kernel functions see the
tile values of the tile variables and the shared root values of everything else.
"""
@inline function get_fields(state, tile::AbstractSurfaceTile, components...; except = (;))
    root_fields = get_fields(state, components...; except)
    tile_state = getproperty(state, tile_name(tile))
    own_fields = merge(getfield(tile_state, :prognostic), getfield(tile_state, :auxiliary))
    return merge(root_fields, own_fields)
end

"""
    $TYPEDSIGNATURES

Retrieves all `Field`s corresponding to auxiliary variables defined on the given `components` from the
namespace of the surface `tile` (see [`tile_name`](@ref)) in `state`, e.g. as the output fields of a
kernel evaluated on that tile.
"""
@inline auxiliary_fields(state, tile::AbstractSurfaceTile, components...) =
    auxiliary_fields(getproperty(state, tile_name(tile)), components...)

""" $TYPEDSIGNATURES """
@inline function compute_auxiliary!(
        state, grid,
        seb::SurfaceEnergyBalance,
        vegetation::Optional{AbstractVegetation} = nothing,
        snow::Optional{AbstractSnow} = nothing,
        soil::Optional{AbstractSoil} = nothing,
        args...
    )
    # diagnose the (optionally snow-, vegetation-, and soil-moisture-aware) albedo
    compute_auxiliary!(state, grid, seb.albedo, vegetation, snow, soil)
    return nothing
end

""" $TYPEDSIGNATURES """
initialize!(state, grid, seb::SurfaceEnergyBalance, args...) = initialize!(state, grid, seb.skin_temperature, args...)

"""
    $TYPEDSIGNATURES

Solve the surface energy balance for skin temperature on `grid` based on the current atmospheric
and surface hydrology state.
"""
function solve_surface_energy_balance!(
        state, grid,
        seb::SurfaceEnergyBalance{NF},
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        snow::Optional{AbstractSnow} = nothing,
        args...
    ) where {NF}
    out = auxiliary_fields(state, seb)
    # Merge the snow thermal auxiliaries so the (optionally snow-aware) conduction target can be evaluated
    fields = get_fields(state, seb, atmos, hydrology, snow)
    launch!(grid, XY, solve_surface_energy_balance_kernel!, out, fields, seb, constants, atmos, hydrology, snow, args...)
    return nothing
end

"""
    $TYPEDSIGNATURES

Compute the surface energy fluxes on `grid` based on the current atmospheric state.
"""
function compute_surface_energy_fluxes!(
        state, grid,
        seb::SurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        args...
    )
    out = auxiliary_fields(state, seb)
    fields = get_fields(state, seb, atmos, hydrology)
    launch!(grid, XY, compute_surface_energy_fluxes_kernel!, out, fields, seb, constants, atmos, hydrology, args...)
    return nothing
end

# Kernel functions

"""
    $TYPEDSIGNATURES

Fused kernel function that computes the radiative and turbulent fluxes, as well as the ground heat flux based on the current
skin temperature and humidity fluxes.
"""
@propagate_inbounds function compute_surface_energy_fluxes!(
        out, i, j, grid, fields,
        seb::SurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        snow::Optional{AbstractSnow} = nothing,
        args...
    )
    # Each flux group is stored by a per-process mutating variant, so a prescribed process (whose
    # fluxes are supplied as input fields) is a no-op while a diagnosed one computes and stores.
    # Radiative fluxes (net radiation, and upwelling SW/LW where diagnosed).
    compute_radiative_fluxes!(out, i, j, grid, fields, seb.radiative_fluxes, seb.skin_temperature, seb.albedo, constants, atmos)
    # Turbulent fluxes; `snow` partitions the latent flux (evaporation vs. sublimation) by area fraction.
    compute_turbulent_fluxes!(out, i, j, grid, fields, seb.turbulent_fluxes, seb.skin_temperature, constants, atmos, hydrology, snow)
    # Ground heat flux, evaluated last since it depends on the other three flux terms.
    compute_ground_heat_flux!(out, i, j, grid, fields, seb.ground_heat_flux, seb.skin_temperature, seb)
    return out
end

# Kernels (fused)

@kernel inbounds = true function solve_surface_energy_balance_kernel!(
        out, grid, fields,
        seb::SurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        snow::Optional{AbstractSnow} = nothing,
        args...
    )
    i, j = @index(Global, NTuple)

    # Solve for skin temperature; `snow` (after `seb`) makes the conduction target and latent flux snow-aware
    solve_skin_temperature!(out, i, j, grid, fields, seb.skin_temperature, seb, constants, atmos, hydrology, snow, args...)
    if !isnothing(hydrology)
        # Recompute evapotranspiration component fluxes from final skin temperature
        evtr = get_evapotranspiration(hydrology)
        out_evtr = auxiliary_fields(fields, evtr)
        compute_evapotranspiration_fluxes!(out_evtr, i, j, grid, fields, evtr, constants, atmos, snow)
    end
    # Recompute fluxes from final skin temperature; `snow` partitions the latent flux (evaporation vs. sublimation)
    compute_surface_energy_fluxes!(out, i, j, grid, fields, seb, constants, atmos, hydrology, snow, args...)
end

@kernel inbounds = true function compute_surface_energy_fluxes_kernel!(
        out, grid, fields,
        seb::SurfaceEnergyBalance,
        constants::PhysicalConstants,
        atmos::AbstractAtmosphere,
        hydrology::Optional{AbstractSurfaceHydrology} = nothing,
        args...
    )
    i, j = @index(Global, NTuple)

    # Compute fluxes based on current skin temperature
    compute_surface_energy_fluxes!(out, i, j, grid, fields, seb, constants, atmos, hydrology, args...)
end
