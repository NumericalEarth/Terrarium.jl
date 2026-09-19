using Terrarium
using Terrarium: Ground, Top, Bottom
using Test

# The tests below drive `explicit_step!` with a lightweight `NamedTuple` mock standing in for a
# `StateVariables` object. `explicit_step!` reads the prognostic names via `prognostic_names(state)`,
# which is only defined for `StateVariables`, so provide a `NamedTuple` method for the mock here
Terrarium.prognostic_names(state::NamedTuple) = keys(state.prognostic)

struct TestClosure
    varname::Symbol
end

Terrarium.variables(closure::TestClosure) = (
    Terrarium.auxiliary(closure.varname, Ground(XYZ())),
)

@testset "Forward Euler" begin
    Δt = 10.0
    euler = ForwardEuler(; Δt)
    @test !is_adaptive(euler)
    @test default_dt(euler) == Δt
    # set up grid and fields
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = 10))
    clock = Clock(time = 0.0)
    # here we mock the structure of a `StateVariables` object
    # for a model with prognostic variables at the top level and
    # in a nested namespace.
    state = (
        prognostic = (x = Field(grid, Ground(XYZ())), y = Field(grid, Ground(XYZ()))),
        auxiliary = (z = Field(grid, Ground(XYZ())),),
        tendencies = (
            x = Field(grid, Ground(XYZ())),
            y = Field(grid, Ground(XYZ())),
        ),
        namespaces = (
            inner = (
                prognostic = (x = Field(grid, Ground(XYZ())),),
                auxiliary = (;),
                tendencies = (
                    x = Field(grid, Ground(XYZ())),
                ),
                namespaces = (;),
                clock = clock,
            ),
        ),
        clock = clock,
    )
    dxdt = 0.1
    dydt = 0.2
    set!(state.tendencies.x, dxdt)
    set!(state.tendencies.y, dydt)
    set!(state.namespaces.inner.tendencies.x, dxdt * 2)

    Terrarium.explicit_step!(state, grid, ForwardEuler(; Δt), Δt, (:x, :y))
    @test all(state.prognostic.x .≈ Δt * dxdt)
    @test all(state.prognostic.y .≈ Δt * dydt)
    # the explicit step recurses into namespaces, stepping namespaced prognostics whose name is in `names`
    @test all(state.namespaces.inner.prognostic.x .≈ Δt * dxdt * 2)
    # check that z was not changed (inverse closure not evaluated)
    @test all(iszero.(state.auxiliary.z))
end

@testset "Forward Euler on vertically sliced fields" begin
    # A variable declared at the `Top` or `Bottom` of a domain occupies a single vertical index but
    # still carries a `Center`/`Face` vertical location, so it is indistinguishable from a fully
    # resolved variable by its location parameters alone. `explicit_step!` must dispatch on the
    # field's `indices` instead and step only the index the slice occupies.
    Δt = 10.0
    Nz = 10
    grid = ColumnGrid(CPU(), Float64, ExponentialSpacing(N = Nz))
    clock = Clock(time = 0.0)

    for (name, loc, k) in (
            ("Top(Face)", Ground(Top()), Nz + 1),
            ("Top(Center)", Ground(Top(z = Center())), Nz),
            ("Bottom(Face)", Ground(Bottom()), 1),
        )
        state = (
            prognostic = (x = Field(grid, loc),),
            auxiliary = (;),
            tendencies = (x = Field(grid, loc),),
            namespaces = (;),
            clock = clock,
        )
        dxdt = 0.1
        set!(state.tendencies.x, dxdt)

        Terrarium.explicit_step!(state, grid, ForwardEuler(; Δt), Δt, (:x,))

        # the slice is stepped, whichever vertical index it occupies
        @test axes(state.prognostic.x, 3) == k:k
        @test all(interior(state.prognostic.x) .≈ Δt * dxdt)
        # and nothing outside the slice is touched, i.e. the step did not walk the whole column
        @test count(!iszero, parent(state.prognostic.x)) == length(interior(state.prognostic.x))
    end
end
