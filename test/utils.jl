using Terrarium
using Test

using Dates
using Unitful

using Terrarium:
    tuplejoin, merge_recursive, safediv, fastmap, piecewise_linear, timestamp, convert_dt

@testset "Tuple utilities" begin
    # tuplejoin
    @test tuplejoin() == ()
    @test tuplejoin((1,)) == (1,)
    @test tuplejoin((1,), (2,)) == (1, 2)
    @test tuplejoin((1,), (1,)) == (1, 1)
    @test tuplejoin((1, 2), (3, 4, 5), (6, 7)) == (1, 2, 3, 4, 5, 6, 7)

    # merge recursive
    @test merge_recursive((;), (;)) == merge((;), (;)) == (;)
    @test merge_recursive((;), (a = 1,)) == merge((;), (a = 1,)) == (a = 1,)
    @test merge_recursive((a = 1,), (;)) == merge((a = 1,), (;)) == (a = 1,)
    @test merge_recursive((a = 1,), (b = 2,)) == merge((a = 1,), (b = 2,)) == (a = 1, b = 2)
    @test merge_recursive((a = 1,), (a = 2,)) == merge((a = 1,), (a = 2,)) == (a = 2,)
    @test merge_recursive((a = 1, b = 2), (a = 2, c = 3)) == merge((a = 1, b = 2), (a = 2, c = 3)) == (a = 2, b = 2, c = 3)
    @test merge_recursive((a = 1, b = (x = 2, y = 3)), (a = 2, b = (x = 3, z = 4))) == (a = 2, b = (x = 3, y = 3, z = 4))
    @test merge_recursive(
        (a = 1, b = (x = (u = -1, v = -2), y = 3)),
        (a = 2, b = (x = (v = -3, w = 0), z = 4))
    ) == (a = 2, b = (x = (u = -1, v = -3, w = 0), y = 3, z = 4))
    @test merge_recursive((a = 1,), nothing) == (a = 1,)
    @test merge_recursive(nothing, (a = 1,)) == (a = 1,)
    @test_throws MethodError merge_recursive(nothing, nothing)

    # fastmap
    @test fastmap(x -> x + 1, (1, 2, 3)) == (2, 3, 4)
    @test fastmap(+, (1, 2, 3), (2, 3, 4)) == (3, 5, 7)
    @test fastmap(+, (a = 1, b = 2, c = 3), (b = 3, c = 4, a = 2)) == (a = 3, b = 5, c = 7)
    @inferred fastmap(*, ("a", 3.0, 1), ("b", 2.0, missing))

    # ntdiff
    @test Terrarium.ntdiff((a = 1, b = 2, c = 3), (b = 0,)) == (a = 1, c = 3)
    @test Terrarium.ntdiff((a = 1, b = 2), (;)) == (a = 1, b = 2)
    @test Terrarium.ntdiff((a = 1, b = 2), (a = 0, b = 0)) == (;)
    @test Terrarium.ntdiff((a = 1, b = 2), (c = 0,)) == (a = 1, b = 2)
    ## large `NamedTuple`s, as returned by `get_fields` for several components, must still infer
    names = Tuple(Symbol(:x, i) for i in 1:40)
    nt = NamedTuple{names}(Tuple(1:40))
    @test @inferred(Terrarium.ntdiff(nt, (x2 = 0, x40 = 0))) == NamedTuple{filter(∉((:x2, :x40)), names)}(Tuple(filter(∉((2, 40)), 1:40)))
end

@testset "Math utilities" begin
    # safediv
    @test safediv(1.0, 2.0) ≈ 1 / 2
    @test safediv(1.0, 0.0) == Inf
    @test safediv(0.0, 0.0) == Inf
    @test safediv(0.0, 1.0) == 0

    # Piecewise linear interpolation
    f = piecewise_linear(
        1.0u"m" => 1.0,
        0.0u"m" => -1.0,
        -2.0u"m" => -2.0
    )
    @test f(2.0) ≈ 1.0
    @test f(1.0) ≈ 1.0
    @test f(0.5) ≈ 0.0
    @test f(-1.0) ≈ -1.5
    @test f(-3.0) ≈ -2.0
end

@testset "Time utilities" begin
    reftime = DateTime(2020, 1, 1)
    Δt = 24 * 3600.0

    # convert_dt
    @test convert_dt(Δt) == Δt
    @test convert_dt(Float32, Second(Δt)) == Float32(Δt)
    @test convert_dt(Float32, Hour(1)) == Float32(3600)
    @test convert_dt(Second, Hour(1)) == Second(3600)
    @test convert_dt(Second, 1.0) == Second(1)
    @test convert_dt(Hour, Second(Δt)) == Hour(24)
    @test_throws InexactError convert_dt(Second, 1.5) == Second(1.5)

    # timestamp
    @test timestamp(DateTime, reftime, Δt) == reftime + Second(Δt)
    @test timestamp(Date, reftime, Δt) == reftime + Day(Second(Δt))
    @test timestamp(Second, reftime, Δt) == Second(Δt)
    @test timestamp(Float32, reftime, Δt) == Float32(Δt)
    @test timestamp(Second, Second(1), Second(Δt)) == Second(Δt + 1)
    @test timestamp(Float32, Second(1), Float32(Δt)) == Δt + 1
    @test timestamp(Float32, 1.0f0, Float32(Δt)) == Float32(Δt + 1)
end
