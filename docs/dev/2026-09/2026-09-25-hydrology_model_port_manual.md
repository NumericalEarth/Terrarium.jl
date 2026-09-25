
Thinking about where to port which part of the model.

Model description (the `Model equations version clean`) can be found in [this markdown file](https://ugentbe-my.sharepoint.com/:t:/g/personal/olivier_bonte_ugent_be/IQCCkyLYvVRURJS8ms5S2BD8ARP5vZtTJoLGOWwN1tuEdKo?e=6mmYrU) or the [rendered PDF](https://ugentbe-my.sharepoint.com/:b:/g/personal/olivier_bonte_ugent_be/IQDjDkFLW5lFR4XnmkGEyYA9AbjFmUxn0m_O0KOpUO3jcBc?e=YSZhpv)

- [[Model equations version clean#Surface run-off]] in `src/processes/surface/runoff`. name related to HBV.
	- only compatible with the force restore type model
- [[Model equations version clean#Net radiation]]: net radiation is prescribed → `PrescribedRadiativeFluxes` should allow for setting $R_n$ and $G$ directly OR just pass them via InputSource? 
	- Extinction coefficient already set a a traint in `PlantTraits()`
	- Lambert beer law not yet implemetned
- [[Model equations version clean#Ground heat flux]]: not sure yet if this should be inside Terrarium
	- Insolation.jl should be used for the calculation of seconds since solar noon
- [[Model equations version clean#Latent heat flux]]
	- The whole scheme should have a new name subtyping `AbstractEvapotranspiration`. Its name could be `ThreeSourceEvapotranspiration` (for the tree fluxes that are combined at the canopy air space node: ground + canopy + transpiration)
	- Penman monteith equation itself (allows to set r_s=0 of course): `src/processes/surface/evapotranspiration` Also the more complex $\lambda E$ formulation from 2012 Lhomme should be placed here. 
		- Reference for classic penman-monteith = Bigleaf.jl. Can be used in tests as a reference implementation 
	- vpd calculation already in `src/processes/atmosphere` → applied at both the canopy air space node and the atmospheric reference level
		- the `VPD_m` (at canopy air space nodes) is only used for the transpiration calculation so also here (definitely not in atmosphere)
- Resistances
	- aerodynamic resistances
		- between canopy air space and atmosphere → `NeutralStabilityAerodynamics` subtyping `AbstractAerodynamics`
		- between leaf/canopy and canopy air space:  ~~extend `aerodynamic_resistance` in `src/processes/surface/evapotranspiration/canopy_evapotranspiration.jl`?~~ New method called `leaf_canopy_air_space_aerodynamic_resistance`
		- between surface and canopy air space: new method `ground_canopy_air_space_aerodynamic resistance` 
		- Suggestion: move all resistances to a new .jl file inside of `src/processes/surface`
		- Note that the surface roughness info is required here: 
			- Maybe it makes more sens to just move everything related to aerodynamics to this surface (so out of atmosphere) 
			- The `SurfaceHydrology` scheme would then have a new field in its struct related to these roughness properties. Scheme 1 could be static (in function of vegetation height), scheme 2 dynamic in function of LAI (so needs vegetation, can be prescribed of course)
	- surface resistance: 
		- soil resistance ($\beta$ factor): `ground_evaporation_resistance_factor` can be reused, but new kernel function will have to be defined specializing on a new type of soil (see below)
			- ~~Add new method `ground_evaporation_resistance` that uses the aerodynamic resistance (between soil and canopy air space) $\beta$  factor to calculate this as an actual resistance that can be used in a Penman-Monteith type framework.~~ First try mulitplying $\beta$ factor times the potential evaporation 
		- surface resistance: add a new `JarvisStomatalConductance` subtyping `AbstractStomatalConductance` in `src/processes/vegetation/stomatal_conductance`.
			- Ideally the soil moisture constraint can be used both with plant available water from a richards type model OR with a force restore type model. 
- Given that conductance is in vegetation, we should use `PrescribedVegetation` with:
	- `PrescribedPhenology`: LAI is given as inputs
	- No `PhotoSynthesis` (if this is possible)
	- `RootDistribution`:  all roots are in the second layer of the simple model, not sure how to deal with this. should specialise on the new soil type or be none
	- `PlantTraits` can be reused, no harm in this. 
- [[Model equations version clean#Interception]]. subtype AbstractCanopyInterception, `DeRidderInterception`
	- fraction that is wet → new method for `compute_canopy_saturation_fraction`
	- drainage → new method for `compute_canopy_water_removal`
	- rainfall partitioning
		- method from Paladyn can be reused:
			- $\alpha$ to zero
			- abstract the labmert beer extinction out because we need an `f_veg` in different places. 
			- set stem area index to 0, LAI is prescribed
		- Canopy evaporation= reusing Penman Monteith with zero surface resistance
- The `SurfaceHydrology` struct has options for
	- Evapotranspiration: use `ThreeSourceEvapotranspiration`
	- canopy_interception use `DeRidderInterception`
	- runoff: new type related to HBV type system
- Soil scheme:
	- Not useful to reuse `SoilEnergyWaterCarbon`. A new struct here could be `ForceRestoreWater`. It could consider:
		- `StratiGraphy`?
		- `Hydrology`
	- There are 2 states `w_1` and `w_2`.  Maybe there is a way to use `StratiGraphy` to define their depths? 2 options to define the new state:
		- 1 prognostic variable `XYZ` with 2 layers. depth is then set in the discretisation part, not in the stratigraphy
		- 2 prognostic variables of type `XY`
	- `AbstractSoilHydraulics` should ideally be subtyped for the specific constants of the force restorce scheme: $C_1, C_2, C_3$.  However, there is no unsaturated hydraulic conductivity needed here, so it should be set to a sort of none type. Brooks-Corey can be used as a retention curve. 
	- `ForceRestoreVeritcalFlow` can subtype `AbstractVeritcalFlow` and be the set of processes that actually calculates the vertical fluxes $D_1$ and $K_2$. I suppose also the tendency functions are best added in the same `src/processes/hydrology/soil_hydrology.jl`

