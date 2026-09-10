# Utilities

export PhysicalConstants, ThermodynamicConstants, MaterialConstants, UniversalConstants
include("constants.jl")
include("unit_conversions.jl")

# Abstract types and methods

include("thermodynamics/abstract_types.jl")
include("atmosphere/abstract_types.jl")
# snow abstract types precede surface so the surface energy balance can dispatch on `AbstractSnow`
include("snow/abstract_types.jl")
include("surface/abstract_types.jl")
include("soil/abstract_types.jl")
include("vegetation/abstract_types.jl")

# Thermodynamics

include("thermodynamics/thermodynamics.jl")
include("thermodynamics/enthalpy.jl")
include("thermodynamics/heat_conduction.jl")

# Atmosphere

export ConstantAerodynamics
include("atmosphere/aerodynamics.jl")
export PrescribedAtmosphere, RainSnow, LongShortWaveRadiation, TracerGas, TracerGases, AmbientCO2
export Windspeed, WindVelocity, SpecificHumidity
include("atmosphere/prescribed_atmosphere.jl")

# Ground (soil and other subsurface media)

export SoilTexture, normalize_texture!
include("soil/stratigraphy/soil_texture.jl")

export ConstantSoilPorosity, SoilPorositySURFEX
include("soil/stratigraphy/soil_porosity.jl")

export SoilComposition, MineralOrganic, volumetric_fractions
include("soil/stratigraphy/soil_composition.jl")

export ConstantSoilHorizon, PrescribedSoilHorizon
include("soil/stratigraphy/soil_horizon.jl")

export SoilStratigraphy, HomogeneousSoilStratigraphy, SoilGridsStratigraphy
include("soil/stratigraphy/soil_stratigraphy.jl")

export ConstantSoilCarbonDensity
include("soil/biogeochem/constant_soil_carbon.jl")

export ConstantSoilHydraulics, SoilHydraulicsSURFEX, UnsatKLinear, UnsatKVanGenuchten
export saturated_hydraulic_conductivity, mineral_porosity, field_capacity, wilting_point
include("soil/hydrology/soil_hydraulic_properties.jl")

export SoilHydrology, NoFlow
include("soil/hydrology/soil_hydrology.jl")

export RichardsEq
include("soil/hydrology/soil_hydrology_rre.jl")

export SoilSaturationPressureClosure
include("soil/hydrology/soil_hydraulic_closures.jl")

export SoilThermalConductivities, SoilHeatCapacities, SoilThermalProperties, InverseQuadratic
export compute_thermal_conductivity, heat_capacity
include("soil/energy/soil_thermal_properties.jl")

export SoilThermodynamics, SoilEnergyTemperatureClosure
include("soil/energy/soil_energy.jl")

include("soil/energy/soil_energy_closures.jl")

export SoilEnergyWaterCarbon
include("soil/soil_coupled.jl")

include("soil/soil_diffusion_timescales.jl")

# Snow

export ConstantSnowHydraulics
include("snow/snow_hydraulic_properties.jl")

export SingleLayerSnow
include("snow/snow_single_layer.jl")
include("snow/snow_interfaces.jl")
include("snow/snow_albedo.jl")

export FractionalSnowCover
include("snow/mass/snow_cover.jl")

export ConstantSnowDensity
include("snow/mass/snow_density.jl")
include("snow/mass/snow_mass.jl")

export PowerLawSnowThermalConductivity, LogarithmicSnowThermalConductivity, QuadraticSnowThermalConductivity
include("snow/energy/snow_thermal_conductivity.jl")

export SnowEnergyTemperatureClosure
include("snow/energy/snow_energy_closures.jl")
include("snow/energy/snow_energy.jl")
include("snow/snow_diffusion_timescales.jl")

# Vegetation

include("vegetation/vegetation_base.jl")

export PlantTraits
include("vegetation/plant_traits.jl")

export PALADYNCarbonDynamics
include("vegetation/dynamics/carbon_dynamics.jl")

export PALADYNVegetationDynamics
include("vegetation/dynamics/vegetation_dynamics.jl")

export PALADYNPhenology, PrescribedPhenology
include("vegetation/phenology/paladyn_phenology.jl")
include("vegetation/phenology/prescribed_phenology.jl")

export StaticExponentialRootDistribution
include("vegetation/hydraulics/root_distribution.jl")

export FieldCapacityLimitedPAW
include("vegetation/hydraulics/plant_available_water.jl")

export LUEPhotosynthesis
include("vegetation/photosynthesis/lue_photosynthesis.jl")

export MedlynStomatalConductance
include("vegetation/stomatal_conductance/medlyn_stomatal_conductance.jl")

export PALADYNAutotrophicRespiration
include("vegetation/respiration/autotrophic_respiration.jl")

export VegetationCarbonCycle
include("vegetation/vegetation_carbon_cycle.jl")

export PrescribedVegetation
include("vegetation/prescribed_vegetation.jl")

# Surface

export PrescribedAlbedo, ConstantAlbedo, DiagnosticAlbedo
include("surface/albedo.jl")

export PrescribedRadiativeFluxes, DiagnosedRadiativeFluxes
include("surface/radiative_fluxes.jl")

export PrescribedSkinTemperature, ImplicitSkinTemperature
include("surface/skin_temperature.jl")

export PrescribedTurbulentFluxes, DiagnosedTurbulentFluxes
include("surface/turbulent_fluxes.jl")

export SurfaceEnergyBalance
include("surface/surface_energy_balance.jl")

export NoCanopyInterception, PALADYNCanopyInterception
include("surface/canopy_interception/canopy_interception.jl")

include("surface/evapotranspiration/evapotranspiration_base.jl")

export SoilMoistureResistanceFactor, ConstantEvaporationResistanceFactor
include("surface/evapotranspiration/ground_resistance_factor.jl")

export BareGroundEvaporation
include("surface/evapotranspiration/bare_ground_evaporation.jl")

export PALADYNCanopyEvapotranspiration
include("surface/evapotranspiration/canopy_evapotranspiration.jl")

export DirectSurfaceRunoff
include("surface/runoff/direct_surface_runoff.jl")

export SurfaceHydrology
include("surface/surface_hydrology.jl")

# Default debug hooks
@inline debughook!(::typeof(compute_auxiliary_kernel!), out, args...) = checkfinite!(out)
@inline debughook!(::typeof(compute_tendencies_kernel!), out, args...) = checkfinite!(out)
