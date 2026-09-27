# Custom reverse rules for the Oceananigans `Clock`.
#
# `tick!` advances the clock in place. Since `StateVariables` is a mutable struct, the clock is
# reachable by reference from the differentiated integrator state, and Enzyme's activity analysis
# therefore descends into `tick!`. There it meets the integer bookkeeping updates
# (`clock.iteration += 1`, `clock.stage = 1`) and bails out with
#
#     EnzymeNoDerivativeError: cannot handle unknown binary operator: add i64 ..., 1
#
# which breaks reverse-mode AD through `timestep!` (most visibly under Checkpointing.jl).
#
# The rules below keep the integer fields out of the analysis entirely while still propagating the
# derivative of `clock.time`, which is a genuine floating-point dependency: time-dependent forcing
# reads it, and it is differentiable with respect to the step size `Δt`. Marking `tick!` wholly
# inactive would be simpler but would silently zero that derivative.

"""
    $TYPEDSIGNATURES

Augmented forward pass for `Oceananigans.TimeSteppers.tick!`. The primal is run
unchanged and no tape is needed: every field the reverse pass touches is either overwritten by
`Δt` or accumulates the incoming adjoint directly.
"""
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(tick!)},
        ::Type{<:Const},
        clock::Annotation{<:Clock},
        Δt::Annotation,
    )
    tick!(clock.val, Δt.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

"""
    $TYPEDSIGNATURES

Reverse pass for `Oceananigans.TimeSteppers.tick!`, accumulating the adjoint of the
step size `Δt` from the three clock fields that depend on it:

- `time += Δt` accumulates, so `d̄Δt += d̄time` and `d̄time` is left untouched;
- `last_Δt = Δt` and `last_stage_Δt = Δt` overwrite, so each contributes its adjoint to `d̄Δt` and is
  then reset to zero.

The `iteration` and `stage` fields are integer counters with no derivative and are deliberately
never referenced here.
"""
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(tick!)},
        ::Type{<:Const},
        tape,
        clock::Annotation{<:Clock},
        Δt::Annotation,
    )
    accumulated = false
    if clock isa Duplicated
        dclock = clock.dval
        accumulated = dclock.time + dclock.last_Δt + dclock.last_stage_Δt
        dclock.last_Δt = zero(dclock.last_Δt)
        dclock.last_stage_Δt = zero(dclock.last_stage_Δt)
    end
    dΔt = Δt isa Active ? convert(typeof(Δt.val), accumulated) : nothing
    return (nothing, dΔt)
end
