# Energy BCs

"""
Alias for `FluxBoundaryCondition` on `internal_energy` with name `soil_heat_flux` representing the net heat flux into the top of the soil column. Without snow this equals the surface energy balance `ground_heat_flux`; with snow it is the (blended) conductive flux across the snow base.
"""
SoilHeatFlux(value = var(:soil_heat_flux, XY(), u"W/m^2"); kwargs...) = (internal_energy = (top = FluxBoundaryCondition(value; kwargs...),),)

"""
Alias for `FluxBoundaryCondition` on `internal_energy` with name `geothermal_heat_flux` representing the geothermal heat flux at the bottom
boundary of the soil column.
"""
GeothermalHeatFlux(value = var(:geothermal_heat_flux, XY(), u"W/m^2"); kwargs...) = (internal_energy = (bottom = FluxBoundaryCondition(value; kwargs...),),)

"""
Alias for `ValueBoundaryCondition` on top `temperature` (in °C) with the given variable name.
"""
PrescribedSurfaceTemperature(name::Symbol, value = var(name, XY(), u"°C"); kwargs...) = (temperature = (top = ValueBoundaryCondition(value; kwargs...),),)

"""
Alias for `ValueBoundaryCondition` on top `temperature` (in °C) with the given variable name.
"""
PrescribedBottomTemperature(name::Symbol, value = var(name, XY(), u"°C"); kwargs...) = (temperature = (bottom = ValueBoundaryCondition(value; kwargs...),),)

# Hydrology BCs

"""
Alias for `PrescribedFlux` with name `infiltration` representing liquid water infiltration at the soil surface.
"""
InfiltrationFlux(value = var(:infiltration, XY(), u"m/s"); kwargs...) = (saturation_water_ice = (top = FluxBoundaryCondition(value; kwargs...),),)

"""
Alias for a discrete-form `FluxBoundaryCondition` on `saturation_water_ice` that normalizes the
physical `infiltration` flux (m/s of water depth) by the top-of-soil porosity at evaluation time,
since `saturation_water_ice` is the dimensionless saturation (VWC / porosity). See
[`saturation_infiltration_bc`](@ref).
"""
function InfiltrationFlux(strat::AbstractStratigraphy, bgc::AbstractSoilBiogeochemistry; kwargs...)
    parameters = (; strat, bgc)
    condition = FluxBoundaryCondition(saturation_infiltration_bc; discrete_form = true, parameters, kwargs...)
    return (saturation_water_ice = (top = condition,),)
end

"""
Alias for `NoFlux` representing a zero-flux bottom boundary condition for water flow (prognostic variable `saturation_water_ice`).
"""
ImpermeableBoundary() = (saturation_water_ice = (bottom = NoFluxBoundaryCondition(),),)

"""
Alias for `PrescribedGradient` representing a Neumann-type zero pressure gradient at the bottom of the soil
column, thereby allowing free drainage of water.
"""
FreeDrainage() = (pressure_head = (bottom = GradientBoundaryCondition(0),),)
