# TODO: should this be a process of its own?
# also, define abstract types for these parameter structs if there may be multiple implementations
Base.@kwdef struct SoilCarbonRespiration{NF}
    "Reference decomposition rate [1/s]"
    k_ref::NF = ustrip(u"s^-1", 0.1u"yr^-1")

    "Temperature sensitivity of decomposition rate (Q10)"
    Q10::NF = 2.0

    "Reference temperature for k_ref [°C]"
    T_ref::NF = 10.0
end

SoilCarbonRespiration(::Type{NF}; kwargs...) where {NF} = SoilCarbonRespiration{NF}(; kwargs...)

function compute_respiration_rate(resp::SoilCarbonRespiration{NF}, C, T) where {NF}
    (; k_ref, Q10, T_ref) = resp # unpack properties
    R = k_ref * Q10^((T - T_ref) / NF(10)) * C
    return R
end

# TODO: should this be a process of its own?
Base.@kwdef struct SoilCarbonTransport{NF}
    "Advection velocity, constant for now"
    ω::NF = 0.0

    "Diffusion coefficient, constant for now"
    D_b::NF = 0.1
end

SoilCarbonTransport(::Type{NF}; kwargs...) where {NF} = SoilCarbonTransport{NF}(; kwargs...)

"""
    $TYPEDEF

Represents a simple one pool soil carbon model.
"""
struct OnePoolSoilCarbon{NF} <: AbstractSoilBiogeochemistry{NF}
    "Paramteters for SOC transport"
    transport::SoilCarbonTransport{NF}

    "Decomposition dynamics / biogeochemistry"
    respiration::SoilCarbonRespiration{NF}

    "Pure organic matter density [kg/m^3]"
    ρ_org::NF
end

function OnePoolSoilCarbon(
        ::Type{NF};
        transport::SoilCarbonTransport = SoilCarbonTransport(NF),
        respiration::SoilCarbonRespiration = SoilCarbonRespiration(NF),
        ρ_org::NF = NF(1300.0) # kg/m^3
    ) where {NF}
    return OnePoolSoilCarbon(transport, respiration, ρ_org)
end

variables(::OnePoolSoilCarbon) = (
    prognostic(:density_soc, XYZ(), units = u"kg/m^3"),
    auxiliary(:respiration_rate, XYZ(), units = u"kg/m^3/s"), # TODO: does this really need to be auxiliary?
    input(:litter, XY(), units = u"kg/m^2/s"), # TODO: this should be later replaced with a flux boundary condition
)

# Implementation of the SOC density getter method for the one-pool scheme;
# this will be called by the other soil components to determine porosity
@inline density_soc(i, j, k, grid, fields, bgc::OnePoolSoilCarbon) = fields.density_soc[i, j, k]

function compute_auxiliary!(state, grid, soc::OnePoolSoilCarbon, soil::AbstractSoil, args...)
    out = auxiliary_fields(state, soc)
    fields = get_fields(state, soc, soil; except = out)
    launch!(grid, XYZ, compute_auxiliary_kernel!, out, fields, soc, soil)
    return nothing
end

function compute_tendencies!(state, grid, soc::OnePoolSoilCarbon, soil::AbstractSoil, args...)
    out = tendency_fields(state, soc)
    fields = get_fields(state, soc, soil)
    launch!(grid, XYZ, compute_tendencies_kernel!, out, fields, soc, soil)
    return nothing
end

# Kernel functions

@propagate_inbounds function compute_respiration!(out, i, j, k, grid, fields, soc::OnePoolSoilCarbon{NF}, soil::AbstractSoil) where {NF}
    T = fields.temperature[i, j, k] # defined by soil energy balance
    C = fields.density_soc[i, j, k]
    # Compute respiration rate
    out.respiration_rate[i, j, k] = compute_respiration_rate(soc.respiration, T, C)
    return out
end

@propagate_inbounds function compute_soc_tendency!(tend, i, j, k, grid, fields, soc::OnePoolSoilCarbon, soil::AbstractSoil)
    tend.density_soc[i, j, k] = compute_soc_tendency(i, j, k, grid, fields, soc, soil)
    return nothing
end

# TODO: the soil argument with the other soil processes is currently unused but I guess it will be used...?
@propagate_inbounds function compute_soc_tendency(i, j, k, grid, fields, soc::OnePoolSoilCarbon, soil::AbstractSoil)
    # Operators require the underlying Oceananigans grid
    field_grid = get_field_grid(grid)
    litter = fields.litter[i, j, k]
    decomposition = fields.respiration_rate[i, j, k]
    # Compute soil carbon flux
    # runic: off
    ∂C∂t = (
        - ∂zᵃᵃᶜ(i, j, k, field_grid, compute_soc_diffusive_flux, fields, soc.transport)
        - ∂zᵃᵃᶜ(i, j, k, field_grid, compute_soc_advective_flux, fields, soc.transport)
        + litter
        - decomposition
    )
    # runic: on
    return ∂C∂t
end

@propagate_inbounds function compute_soc_diffusive_flux(i, j, k, grid, fields, transport::SoilCarbonTransport)
    C = fields.density_soc
    # TODO: compute and interpolate conductvitiy to grid cell faces
    # D_b = ℑzᵃᵃᶠ(i, j, k, grid, compute_soc_conductivity, fields, args...)
    D_b = transport.D_b
    ∇C = ∂zᵃᵃᶠ(i, j, k, grid, C)
    q_d = D_b * ∇C
    return q_d
end

@propagate_inbounds function compute_soc_advective_flux(i, j, k, grid, fields, transport::SoilCarbonTransport)
    C = fields.density_soc[i, j, k]
    # TODO: compute depth-varying advection; also use Oceananigans advection operator
    ω = transport.ω
    q_a = ω * C
    return q_a
end

# Kernels

@kernel inbounds = true function compute_auxiliary_kernel!(out, grid, fields, soc::OnePoolSoilCarbon, args...)
    i, j, k = @index(Global, NTuple)
    compute_respiration!(out, i, j, k, grid, fields, soc, args...)
end

@kernel inbounds = true function compute_tendencies_kernel!(out, grid, fields, soc::OnePoolSoilCarbon, args...)
    i, j, k = @index(Global, NTuple)
    compute_soc_tendency!(out, i, j, k, grid, fields, soc, args...)
end
