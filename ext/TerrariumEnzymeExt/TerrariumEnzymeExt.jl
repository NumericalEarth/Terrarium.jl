module TerrariumEnzymeExt

# Enzyme.jl support for Terrarium.
#
# This extension holds hand-written `EnzymeRules` for operations that Enzyme's activity analysis
# cannot handle on its own. It depends only on `EnzymeCore`, not on the full `Enzyme` package, so
# loading it is cheap and it applies to both `Enzyme` and any other consumer of `EnzymeRules`.

using DocStringExtensions

using EnzymeCore
using EnzymeCore: Annotation, Const, Duplicated, Active
using EnzymeCore.EnzymeRules

using Terrarium
using Oceananigans.TimeSteppers: Clock, tick!

include("clock.jl")

end # module
