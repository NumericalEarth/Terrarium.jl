# Texture-dependent soil thermal conductivity

> Status: **completed**. This document describes the intended Option A implementation and
> records the deferred Option B refinement.

Date of initial draft: 2026-06-24

Base revision: ac4b8ede34ab570734911061a8426c0eb40d1c5d

## Originating prompt

> (Not recorded — this plan predates the implementation-planning workflow.)

## Revision log

> (Not recorded — this plan predates the implementation-planning workflow.)

## Problem

Soil texture (sand/silt/clay) currently influences porosity and hydraulic properties but
**not** the thermal properties. The mineral solid constituent uses a single, texture-invariant
thermal conductivity (`3.8 W/m/K`) and heat capacity (`2.0 MJ m⁻³ K⁻¹`) in
[`soil_thermal_properties.jl`](../../src/processes/soil/energy/soil_thermal_properties.jl),
flagged by the in-code TODO at line 14:

```julia
# Note: Should sand, silt, and clay have separate thermal properties?
```

This omits a first-order effect: quartz (dominant in sand) conducts heat far better
(~7.7 W/m/K) than clay minerals (~2.0 W/m/K), so sandy soils have substantially higher solid
conductivity than clayey soils at equal porosity and saturation.

## Background: why not the SURFEX CGsat coefficient

The motivating SURFEX/ISBA relation,

```
CGsat = (−1.557e-2·SAND − 1.441e-2·CLAY + 4.7021) × 10⁻⁶   [K m² J⁻¹]   (SAND, CLAY in %)
```

is the **force-restore** soil thermal coefficient (Noilhan & Planton 1989). In force-restore
the entire column collapses to one surface-temperature equation and `CG` is inversely
proportional to the surface thermal inertia `μ = √(λ·C_v)`:

```
CG ∝ 1 / √(λ·C_v)
```

`CGsat` therefore lumps conductivity `λ` and volumetric heat capacity `C_v` into a single
diurnal-timescale inertia parameter. Terrarium uses an **explicit multi-layer conduction**
scheme that needs `λ` and `C_v` *separately, per layer*, so `CGsat` cannot be plugged in
directly: inverting it yields only the product `λ·C_v` and is tied to a different numerical
scheme. The correct SURFEX analogue is **ISBA-DIF** (the multi-layer diffusion version), which
computes `λ` from constituents with a Johansen-type formulation and does **not** use `CGsat`.

## Current thermal-property pipeline

Bulk thermal properties are computed per soil volume in
[`soil_thermal_properties.jl`](../../src/processes/soil/energy/soil_thermal_properties.jl):

```julia
# heat capacity: weighted average  C = Σ θᵢ cᵢ
compute_heat_capacity(props, soil)        # WeightedAverage over getproperties(props.heat_capacities)

# conductivity: inverse-quadratic mix   λ = (Σ θᵢ √λᵢ)²
compute_thermal_conductivity(props, soil) # props.conductivity_weighting over getproperties(props.conductivities)
```

The constituent conductivities `κs = (water, ice, air, mineral, organic)` are matched by
key against `volumetric_fractions(soil) = (water, ice, air, organic, mineral)` (the `fastmap`
over `NamedTuple`s is key-based, so ordering is irrelevant). The texture is reachable from the
soil volume via `mineral_texture(soil)` → `soil.solid.texture` (a `SoilTexture` with `.sand`,
`.silt`, `.clay`); see [`soil_composition.jl`](../../src/processes/soil/stratigraphy/soil_composition.jl).

**Heat capacity is left untouched.** Mineral grain volumetric heat capacity is nearly
texture-independent (~2.0 MJ m⁻³ K⁻¹ across sand/silt/clay), so the meaningful texture signal
lives almost entirely in `λ`. Option A modifies only the conductivity path.

## Design (Option A): quartz-weighted solid conductivity, quartz = sand

Make the **mineral constituent conductivity** a function of texture using the standard
geometric-mean (Johansen 1975; Peters-Lidard et al. 1998; used in Noah/CLM):

```
λ_solid = λ_q^q · λ_o^(1 − q)
```

with `q` the quartz volume fraction of the solid grains. For this first pass we assume
**q = sand fraction** (acceptable for most siliceous soils; see Future work for the caveat).
`λ_q` is the quartz endpoint (~7.7 W/m/K) and `λ_o` the non-quartz mineral endpoint (~2.0 W/m/K).

Sanity check for a loam (sand ≈ 0.4): `λ_solid = 7.7^0.4 · 2.0^0.6 ≈ 3.4 W/m/K`, close to the
current fixed `3.8`; pure sand → 7.7, pure clay → 2.0. The bulk `mineral` conductivity is thus
no longer a fixed parameter but a derived, texture-dependent quantity.

### Parameter changes

In `SoilThermalConductivities`
([`soil_thermal_properties.jl:13`](../../src/processes/soil/energy/soil_thermal_properties.jl#L13)),
replace the single `mineral` endpoint with the two geometric-mean endpoints:

```julia
"Thermal conductivity of quartz (sand) mineral grains"
@param quartz::NF = 7.7 (units = u"W/m/K", bounds = Positive)

"Thermal conductivity of non-quartz (silt/clay) mineral grains"
@param mineral::NF = 2.0 (units = u"W/m/K", bounds = Positive)
```

`quartz` is a parameter of the conductivity *model*, not a soil constituent — it must not enter
the constituent mixing as its own component (there is no `quartz` volumetric fraction).

### Code changes

1. Add a helper that computes the texture-dependent solid conductivity:

   ```julia
   @inline mineral_thermal_conductivity(c::SoilThermalConductivities, texture::SoilTexture) =
       c.quartz ^ texture.sand * c.mineral ^ (1 - texture.sand)
   ```

2. Rework `compute_thermal_conductivity` to build the constituent `NamedTuple` explicitly with
   exactly the five mixing keys, injecting the computed `mineral` value. This replaces the bare
   `getproperties(props.conductivities)` call, which would otherwise leak the new `quartz` field
   into the mix and break the key-matched `fastmap`:

   ```julia
   @inline function compute_thermal_conductivity(props::SoilThermalProperties, soil::SoilVolume)
       c = props.conductivities
       κs = (; c.water, c.ice, c.air, c.organic,
               mineral = mineral_thermal_conductivity(c, mineral_texture(soil)))
       return props.conductivity_weighting(κs, volumetric_fractions(soil))
   end
   ```

   (`compute_heat_capacity` is unchanged.)

### GPU / type-stability notes

- `mineral_thermal_conductivity` is scalar arithmetic on `NF`; the `^` with a non-integer exponent must
  use `NF`-typed values (`texture.sand` is already `NF`) to stay `Float32`-clean on the GPU.
- The constituent `NamedTuple` is built from concrete fields, so the existing key-matched
  `fastmap` mixing remains type-stable and kernel-compatible.

### Verification

- Unit: `mineral_thermal_conductivity` returns `quartz` at `sand=1`, `mineral` at `sand=0`, and the
  geometric mean in between; bulk `λ` increases monotonically with sand at fixed porosity/saturation.
- Regression: a mid-range loam reproduces ≈ the previous fixed-`mineral` bulk `λ` (within ~10%),
  so existing single-texture results do not shift dramatically.
- Integration: the global SoilGrids example
  ([`soil_heat_global_soilgrids.jl`](../../examples/simulations/soil_heat_global_soilgrids.jl))
  runs on CPU and GPU, and surface-temperature fields show the expected sand/clay contrast.

## Future work (Option B): full texture+state conductivity model

Option A only makes the *solid* conductivity texture-dependent and keeps the inverse-quadratic
constituent mix for the saturation/phase dependence. The more faithful approach used by modern
LSMs computes bulk `λ` directly from texture, porosity, bulk density, and saturation:

- **Johansen (1975)**: `λ = (λ_sat − λ_dry)·Ke + λ_dry`, with `λ_dry` from bulk density,
  `λ_sat = λ_solid^(1−φ)·λ_water^φ`, and a saturation-dependent Kersten number `Ke(Sr)` whose
  form depends on texture (coarse vs fine) and frozen/unfrozen state.
- **Refinements**: Lu et al. (2007), Balland & Arp (2005), Côté & Konrad (2005) — texture- and
  bulk-density-dependent `Ke` parameters.

This warrants its own `AbstractBulkWeighting` (or a higher-level thermal-conductivity component
that supersedes the constituent mix), since it computes bulk `λ` rather than per-constituent
conductivities. When implementing, also revisit the architectural shortcut from Option A: the
`quartz`/`mineral` endpoints would more cleanly live in a dedicated solid-conductivity
sub-component rather than as loose fields on `SoilThermalConductivities`.

### Deferred refinements

- **Quartz ≠ sand.** Quartz content is only approximately the sand fraction; volcanic,
  carbonate, and some tropical soils have non-quartz sand. A separately specifiable quartz
  fraction (input or per-horizon parameter) would improve accuracy. Johansen also lowers the
  non-quartz endpoint to `λ_o ≈ 3.0 W/m/K` when `q < 0.2`; this two-regime endpoint is omitted
  in Option A.
- **Frozen vs unfrozen Kersten number.** Option A's saturation/phase dependence comes entirely
  from the inverse-quadratic mix; Johansen uses distinct `Ke` curves for frozen and unfrozen
  soil, relevant for permafrost applications.
