# Utilities

export PhysicalConstants
include("physical_constants.jl")
include("physics_utils.jl")

# Abstract types and methods

include("atmosphere/abstract_types.jl")
include("surface_energy/abstract_types.jl")
include("surface_hydrology/abstract_types.jl")
include("soil/abstract_types.jl")
include("vegetation/abstract_types.jl")

# Atmosphere

export ConstantAerodynamicResistance
include("atmosphere/aerodynamics.jl")
export PrescribedAtmosphere, RainSnow, LongShortWaveRadiation, TracerGas, TracerGases, AmbientCO2
include("atmosphere/prescribed_atmosphere.jl")

# Soil

export SoilTexture
include("soil/stratigraphy/soil_texture.jl")
export ConstantSoilPorosity, SoilPorositySURFEX
include("soil/stratigraphy/soil_porosity.jl")
export SoilVolume, MineralOrganic, volumetric_fractions
include("soil/stratigraphy/soil_volume.jl")
export HomogeneousStratigraphy
include("soil/stratigraphy/homogeneous_strat.jl")

export ConstantSoilCarbonDensity
include("soil/biogeochem/constant_soil_carbon.jl")
export OnePoolSoilCarbon
include("soil/biogeochem/one_pool_soil_carbon.jl")

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

export SoilEnergyBalance, SoilEnergyTemperatureClosure
include("soil/energy/soil_energy.jl")
include("soil/energy/soil_energy_closures.jl")

export SoilEnergyWaterCarbon
include("soil/soil_coupled.jl")

# Vegetation

export PALADYNCarbonDynamics
include("vegetation/carbon_dynamics.jl")

export PALADYNVegetationDynamics
include("vegetation/vegetation_dynamics.jl")

export PALADYNPhenology
include("vegetation/phenology.jl")

export StaticExponentialRootDistribution
include("vegetation/root_distribution.jl")

export FieldCapacityLimitedPAW
include("vegetation/plant_available_water.jl")

export LUEPhotosynthesis
include("vegetation/photosynthesis.jl")

export MedlynStomatalConductance
include("vegetation/stomatal_conductance.jl")

export PALADYNAutotrophicRespiration
include("vegetation/autotrophic_respiration.jl")

export VegetationCarbon
include("vegetation/vegetation_coupled.jl")

# Surface Energy Balance

export PrescribedAlbedo, ConstantAlbedo
include("surface_energy/albedo.jl")

export PrescribedRadiativeFluxes, DiagnosedRadiativeFluxes
include("surface_energy/radiative_fluxes.jl")

export PrescribedSkinTemperature, ImplicitSkinTemperature
include("surface_energy/skin_temperature.jl")

export PrescribedTurbulentFluxes, DiagnosedTurbulentFluxes
include("surface_energy/turbulent_fluxes.jl")

export SurfaceEnergyBalance
include("surface_energy/surface_energy_balance.jl")

# Suface Hydrology

export GroundEvaporation
include("surface_hydrology/ground_evaporation.jl")

export PALADYNCanopyEvapotranspiration
include("surface_hydrology/canopy_evapotranpsiration.jl")

export PALADYNCanopyInterception
include("surface_hydrology/canopy_interception.jl")

export DirectSurfaceRunoff
include("surface_hydrology/surface_runoff.jl")

export SurfaceHydrology
include("surface_hydrology/surface_hydrology.jl")

# Default debug hooks
@inline debughook!(::typeof(compute_auxiliary_kernel!), out, args...) = nancheck!(out)
@inline debughook!(::typeof(compute_tendencies_kernel!), out, args...) = nancheck!(out)
