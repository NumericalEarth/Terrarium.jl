"""
    $TYPEDEF

Marker type for state variable spatial *domains*; currently, five domains are considered:
`Ground`, `Snow`, `Canopy`, `Surface`, and `Atmosphere`.

The first three are *vertical* domains, each of which a land grid may discretize in its own right.
`Surface` instead represents the land-atmosphere interface, whose physical position depends on the
surface tile (the top of the soil column over bare ground, the top of the snowpack under snow, the
canopy where there is vegetation). A variable declared on `Surface` therefore never carries a
vertical dimension and must always be 2D (`XY`).

`Atmosphere` is likewise not a vertical domain; it is the domain of the atmospheric forcings a land
model reads rather than computes.

A variable need not state a domain at all; see [`VarLocation`](@ref).
"""
abstract type VarDomain end

"""
    $TYPEDEF

The ground (soil) domain: the vertically resolved subsurface column. This is the domain on which a
land grid's shared horizontal discretization is defined.
"""
struct Ground <: VarDomain end

"""
    $TYPEDEF

The snow domain, i.e. the snowpack above the ground surface. A grid which does not discretize the
snowpack vertically (a single-layer scheme) still carries its 2D variables.
"""
struct Snow <: VarDomain end

"""
    $TYPEDEF

The canopy domain, i.e. the vegetation layer. A grid which does not discretize the canopy vertically
(a big-leaf scheme) still carries its 2D variables.
"""
struct Canopy <: VarDomain end

"""
    $TYPEDEF

The land-atmosphere interface. Unlike [`Ground`](@ref), [`Snow`](@ref), and [`Canopy`](@ref), the
surface is not a vertical domain: its physical position depends on the surface tile, being the top
of the soil column over bare ground, the top of the snowpack under snow, and the canopy where there
is vegetation. `Surface` variables therefore never carry a vertical dimension and must be declared
as `Surface(XY())`; `Surface(XYZ())` raises an error.
"""
struct Surface <: VarDomain end

"""
    $TYPEDEF

The near-surface atmosphere: the domain of *atmospheric forcing* variables, i.e. quantities a land
model reads rather than computes, such as air temperature, wind, humidity, precipitation, and
downwelling radiation. `Atmosphere` is intended for forcing variables only. Like [`Surface`](@ref),
`Atmosphere` carries no vertical discretization, so its variables must be declared as `Atmosphere(XY())`;
`Atmosphere(XYZ())` raises an error.
"""
struct Atmosphere <: VarDomain end
