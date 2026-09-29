using Terrarium
using Test

import Oceananigans
import Oceananigans: CenterField, Center, set!, interior
import Oceananigans.Grids: RectilinearGrid, z_domain, halo_size, total_size, nodes, xnodes, znodes, isrectilinear
import Terrarium.RingGrids
import Terrarium.RingGrids: FullHEALPixGrid, get_npoints

@testset "AbstractGrid interface" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 2)
    @test grid isa Oceananigans.AbstractGrid
    @test eltype(grid) == Float32
    @test size(grid) == (2, 1, 5)
    @test Oceananigans.Grids.topology(grid) == (Oceananigans.Grids.Periodic, Oceananigans.Grids.Flat, Oceananigans.Grids.Bounded)
    @test isrectilinear(grid)
    @test znodes(grid, Oceananigans.Center()) ≈ collect(-0.45f0:0.1f0:-0.05f0)

    # A land grid forwards each of these to its ground domain.
    land_grid = LandGrid(grid)
    @test size(land_grid) == size(grid)
    @test eltype(land_grid) == eltype(grid)
    @test Oceananigans.Grids.topology(land_grid) == Oceananigans.Grids.topology(grid)
    @test halo_size(land_grid) == halo_size(grid)
    @test isrectilinear(land_grid)
    @test znodes(land_grid, Oceananigans.Center()) ≈ znodes(grid, Oceananigans.Center())

    ring_grid = FullHEALPixGrid(4)
    col_ring_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5f0, N = 3), ring_grid)
    @test col_ring_grid isa Oceananigans.AbstractGrid
    @test eltype(col_ring_grid) == Float32
    @test halo_size(col_ring_grid) == halo_size(col_ring_grid.grid)
    @test isrectilinear(col_ring_grid)
end

@testset "Grid property forwarding" begin
    # Property forwarding applies to the grid *wrappers*: `LandGrid` and `ColumnRingGrid`.
    # `ColumnGrid` is an alias for `RectilinearGrid` and so carries these fields itself.
    grid = LandGrid(ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 3))
    ground_grid = ground_domain(grid)

    # Dimension/halo fields Oceananigans accesses directly (e.g. grid.Nx) forward to the
    # underlying field grid rather than throwing a missing-field error.
    for name in (:Nx, :Ny, :Nz, :Hx, :Hy, :Hz)
        @test getproperty(grid, name) === getproperty(ground_grid, name)
    end
    @test grid.Nx == 3
    @test grid.Ny == 1
    @test grid.Nz == 5

    # Coordinate arrays are forwarded generically as well.
    @test grid.z === ground_grid.z

    # The wrapper's own struct fields still resolve via getfield (not shadowed by forwarding).
    @test grid.ground === ground_grid
    @test isnothing(grid.snow)

    # propertynames advertises both the wrapper's own fields and the forwarded ones.
    @test :ground in propertynames(grid)
    @test :Nx in propertynames(grid)
    @test :Nz in propertynames(grid)

    # ColumnRingGrid's extra own fields (rings, mask) are not shadowed by the forwarding.
    col_ring_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5f0, N = 3), FullHEALPixGrid(4))
    @test col_ring_grid.grid isa RectilinearGrid
    @test col_ring_grid.rings isa RingGrids.AbstractGrid
    @test col_ring_grid.Nx == col_ring_grid.grid.Nx
    @test :rings in propertynames(col_ring_grid)
    @test :mask in propertynames(col_ring_grid)
end

@testset "Oceananigans operations on land grids" begin
    # Exercise real Oceananigans code paths that consume grid fields directly (grid.Nx/Ny/Nz,
    # halos): total_size and Field construction/reduction must work on the land grid itself.
    column_grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 3)
    grid = LandGrid(column_grid)

    @test total_size(grid) == total_size(column_grid)

    # `CenterField(grid)` dispatches into Oceananigans' generic AbstractGrid constructor, which
    # allocates data from the grid's dimensions/halos — only possible if the fields forward.
    field = CenterField(grid)
    @test size(field) == size(grid)
    set!(field, 2)
    @test sum(interior(field)) ≈ 2 * prod(size(grid))
end

@testset "Vertical discretizations" begin
    # Uniformly spaced columns are given directly as a range of cell interfaces.
    @test num_layers(UniformSpacing(Δz = 0.1, N = 1)) == 1
    @test num_layers(UniformSpacing(Δz = 0.1, N = 10)) == 10
    @test diff(collect(UniformSpacing(Δz = 0.1, N = 10))) ≈ repeat([0.1], 10)

    # Prescribed layer thicknesses become a vector of cell interfaces.
    Δz = [0.1, 0.2, 0.3]
    faces = vcat(-reverse(cumsum(Δz)), 0.0)
    @test num_layers(faces) == 3
    @test diff(faces) ≈ reverse(Δz)   # cell interfaces run bottom-up

    # `ExponentialSpacing` reproduces the geometric progression of layer thicknesses implied by
    # `Δz_min`, `Δz_max` and `N`. Oceananigans orders cell interfaces bottom-up, so the spacings
    # come out in the reverse of Terrarium's surface-down ordering.
    z = ExponentialSpacing(Δz_min = 0.1, Δz_max = 1.0, N = 2)
    @test num_layers(z) == 2
    @test reverse(diff(z.faces)) ≈ [0.1, 1.0]

    z = ExponentialSpacing(Δz_min = 0.1, Δz_max = 1.0, N = 3)
    @test reverse(diff(z.faces)) ≈ exp2.(LinRange(log2(0.1), log2(1.0), 3))

    # The column extends exactly as deep as the layer thicknesses imply.
    Δz_min, Δz_max, N = 0.05, 100.0, 50
    ρ = (Δz_max / Δz_min)^(1 / (N - 1))
    z = ExponentialSpacing(; Δz_min, Δz_max, N)
    @test num_layers(z) == N
    @test z.faces[end] == 0
    @test z.faces[1] ≈ -Δz_min * (ρ^N - 1) / (ρ - 1)
    @test first(diff(z.faces)) ≈ Δz_max   # bottom layer
    @test last(diff(z.faces)) ≈ Δz_min    # surface layer

    # A degenerate or ill-posed column is rejected on the host rather than producing NaN interfaces.
    @test_throws ArgumentError ExponentialSpacing(Δz_min = 0.1, Δz_max = 0.1, N = 10)
    @test_throws ArgumentError ExponentialSpacing(Δz_min = 0.1, Δz_max = 1.0, N = 1)
    @test_throws ArgumentError ExponentialSpacing(Δz_min = 0.0, Δz_max = 1.0, N = 10)
end

@testset "ColumnGrid" begin
    # 2-column grid with 5 linearly spaced points
    num_columns = 2
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 5), num_columns)
    @test isa(grid, RectilinearGrid)
    @test grid.Nx == num_columns
    @test grid.Ny == 1
    @test grid.Nz == 5
    @test z_domain(grid) == (-0.5, 0.0)
end

@testset "ColumnRingGrid" begin
    # test with 10-ring HEALPix
    ring_grid = FullHEALPixGrid(8)
    grid = ColumnRingGrid(UniformSpacing(Δz = 0.5, N = 10), ring_grid)
    rectilinear_grid = grid.grid
    @test isa(rectilinear_grid, RectilinearGrid)
    @test rectilinear_grid.Nx == get_npoints(ring_grid)
    @test rectilinear_grid.Ny == 1
    @test rectilinear_grid.Nz == 10
    @test z_domain(rectilinear_grid) == (-5.0, 0.0)

    # Test RingGrids.Field to Oceananigans.Field conversion
    @testset "RingGrids to Oceananigans Field conversion" begin
        # Create a 2D RingGrids field with test data
        ring_grid = FullHEALPixGrid(8)
        grid = ColumnRingGrid(UniformSpacing(Δz = 0.5, N = 10), ring_grid)

        ring_field_2d = rand(ring_grid)

        # Convert to Oceananigans field
        oceananigans_field_2d = Terrarium.Field(ring_field_2d, grid)
        @test isa(oceananigans_field_2d, Terrarium.Field)
        @test size(oceananigans_field_2d) == (get_npoints(ring_grid), 1, 1)

        # Check that values were copied correctly
        ocean_data = Terrarium.interior(oceananigans_field_2d)
        @test all(ocean_data[:, 1, 1] .== ring_field_2d.data[grid.mask.data])

        # Test with 3D field (horizontal + vertical)
        ring_field_3d = rand(ring_grid, 10)

        ocean_field_3d = Terrarium.Field(ring_field_3d, grid)
        @test isa(ocean_field_3d, Terrarium.Field)
        @test size(ocean_field_3d) == (get_npoints(ring_grid), 1, 10)

        # Check that values were copied correctly for each vertical level
        ocean_data_3d = Terrarium.interior(ocean_field_3d)
        for k in 1:10
            @test all(ocean_data_3d[:, 1, k] .== ring_field_3d.data[grid.mask.data, k])
        end

        # Test with masked grid (some points inactive)

        # Create a new ring grid for this test
        mask = rand(Bool, ring_grid)
        # Create a masked ColumnRingGrid using the correct constructor
        masked_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5, N = 10), mask)

        ring_field = rand(ring_grid)

        # Convert to Oceananigans field
        ocean_field_masked = Terrarium.Field(ring_field, masked_grid)
        @test isa(ocean_field_masked, Terrarium.Field)
        @test size(ocean_field_masked) == (sum(mask), 1, 1)

        # Verify only masked (active) points were copied
        ocean_data_masked = Terrarium.interior(ocean_field_masked)

        expected_values = ring_field.data[mask]
        @test all(ocean_data_masked[:, 1, 1] .== expected_values)

        # Test that conversion throws error for RingGrids fields with ndims >= 3
        ring_field_3d_plus = rand(ring_grid, 10, 5)  # horizontal × vertical × extra dimension
        @test_throws ErrorException Terrarium.Field(ring_field_3d_plus, grid)
    end

    @testset "Oceananigans Field to RingGrids Field" begin
        ring_grid = FullHEALPixGrid(8)
        npoints = get_npoints(ring_grid)
        nz = 10

        full_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5, N = nz), ring_grid)
        rectilinear_grid = full_grid.grid
        ncols = rectilinear_grid.Nx  # == npoints (no mask)

        # 2D Oceananigans field (Field{Center,Center,Nothing}) → RingGrids.Field
        field_2d = Terrarium.Field{Center, Center, Nothing}(rectilinear_grid)
        Terrarium.interior(field_2d)[:, 1, 1] .= Float32.(1:ncols)

        ring_2d = RingGrids.Field(field_2d, full_grid)
        @test ring_2d.data[full_grid.mask.data, 1] ≈ Float32.(1:ncols)

        # 3D Oceananigans field (Field{Center,Center,Center}) → RingGrids.Field
        field_3d = Terrarium.Field{Center, Center, Center}(rectilinear_grid)
        for k in 1:nz
            Terrarium.interior(field_3d)[:, 1, k] .= Float32(k)
        end

        ring_3d = RingGrids.Field(field_3d, full_grid)
        @test size(ring_3d, 2) == nz
        for k in 1:nz
            @test all(ring_3d.data[full_grid.mask.data, k] .≈ Float32(k))
        end

        # fill_value is set for non-masked (inactive) points
        mask = rand(Bool, ring_grid)
        masked_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5, N = nz), mask)
        masked_rectilinear_grid = masked_grid.grid

        field_masked = Terrarium.Field{Center, Center, Nothing}(masked_rectilinear_grid)
        fill!(field_masked, 1.0f0)

        ring_fill = RingGrids.Field(field_masked, masked_grid; fill_value = -1.0)
        @test all(ring_fill.data[masked_grid.mask.data, 1] .≈ 1.0f0)
        @test all(ring_fill.data[.!masked_grid.mask.data, 1] .== -1.0)
    end
end

@testset "Grid type hierarchy" begin
    column_grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 2)
    ring_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5f0, N = 3), FullHEALPixGrid(4))
    land_grid = LandGrid(column_grid)

    # `ColumnGrid` is an alias for a `RectilinearGrid` with a `Flat` lateral dimension, not a
    # distinct type, and both column grids are spatial discretizations rather than land grids.
    @test column_grid isa RectilinearGrid
    @test column_grid isa ColumnGrid
    @test ground_domain(column_grid) === column_grid
    @test !(column_grid isa Terrarium.AbstractLandGrid)
    @test !(ring_grid isa Terrarium.AbstractLandGrid)
    @test ring_grid isa Oceananigans.AbstractGrid

    # A rectilinear grid with a non-`Flat` lateral dimension is not a `ColumnGrid`.
    @test !(RectilinearGrid(size = (2, 2, 3), x = (0, 1), y = (0, 1), z = (-1, 0)) isa ColumnGrid)

    # `LandGrid` is the only `AbstractLandGrid`.
    @test land_grid isa Terrarium.AbstractLandGrid
    @test land_grid isa Oceananigans.AbstractGrid
end

@testset "LandGrid" begin
    column_grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 2)

    # Ground-only land grid: the snow and canopy domains are not vertically resolved.
    grid = LandGrid(column_grid)
    @test ground_domain(grid) === column_grid
    @test isnothing(snow_domain(grid))
    @test isnothing(canopy_domain(grid))
    @test (@inferred ground_domain(grid)) === column_grid

    # The ground domain defines the grid on which `Field`s are allocated.
    @test size(grid) == size(column_grid)
    @test eltype(grid) == eltype(column_grid)
    @test Terrarium.num_layers(grid) == 5
    @test Oceananigans.Grids.topology(grid) == Oceananigans.Grids.topology(column_grid)
    @test architecture(grid) == architecture(column_grid)
    @test size(Field(grid, Terrarium.Ground(XYZ()))) == size(grid)

    # Grids which are not land grids are their own ground discretization.
    @test ground_domain(column_grid) === column_grid

    # All three domains may be given explicitly, sharing the horizontal discretization.
    snow = ColumnGrid(UniformSpacing(Δz = 0.05f0, N = 3), 2)
    canopy = ColumnGrid(UniformSpacing(Δz = 1.0f0, N = 1), 2)
    multi_domain = LandGrid(column_grid; snow, canopy)
    @test ground_domain(multi_domain) === column_grid
    @test snow_domain(multi_domain) === snow
    @test canopy_domain(multi_domain) === canopy
    @test Terrarium.num_layers(snow_domain(multi_domain)) == 3
    @test Terrarium.num_layers(canopy_domain(multi_domain)) == 1

    # Domain grids must be compatible with the ground domain.
    @test_throws ArgumentError LandGrid(column_grid; snow = ColumnGrid(UniformSpacing(Δz = 0.05f0, N = 3), 3))
    @test_throws ArgumentError LandGrid(column_grid; snow = ColumnGrid(UniformSpacing(Δz = 0.05, N = 3), 2))

    # Any Oceananigans grid can serve as the underlying spatial discretization.
    rect_grid = RectilinearGrid(size = (2, 1, 5), x = (0, 1), y = (0, 1), z = (-1, 0))
    rect_land_grid = LandGrid(rect_grid)
    @test ground_domain(rect_land_grid) === rect_grid
    @test size(rect_land_grid) == size(rect_grid)

    ring_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5f0, N = 3), FullHEALPixGrid(4))
    ring_land_grid = LandGrid(ring_grid)
    @test ground_domain(ring_land_grid) === ring_grid
    @test ring_land_grid.rings === ring_grid.rings

    # Transferring to the same architecture preserves the domain structure.
    cpu_grid = on_architecture(CPU(), multi_domain)
    @test cpu_grid isa LandGrid
    @test size(ground_domain(cpu_grid)) == size(column_grid)
    @test size(snow_domain(cpu_grid)) == size(snow)
    @test size(canopy_domain(cpu_grid)) == size(canopy)

    @test occursin("LandGrid{Float32}", sprint(show, MIME"text/plain"(), grid))
end

@testset "create_land_grid" begin
    column_grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 2)

    # The default builds a ground-only land grid from any spatial discretization.
    grid = create_land_grid(column_grid)
    @test grid isa LandGrid
    @test ground_domain(grid) === column_grid
    @test isnothing(snow_domain(grid))
    @test isnothing(canopy_domain(grid))

    # Component-aware form; all currently implemented snow and vegetation schemes are 0D, so the
    # snow and canopy domains remain unresolved.
    soil = SoilEnergyWaterCarbon(Float32)
    snow = SingleLayerSnow(Float32)
    vegetation = VegetationCarbonCycle(Float32)
    with_components = create_land_grid(column_grid, soil, snow, vegetation)
    @test with_components isa LandGrid
    @test ground_domain(with_components) === column_grid
    @test isnothing(snow_domain(with_components))

    # `create_land_grid` is idempotent on land grids.
    @test create_land_grid(grid) === grid
    @test create_land_grid(grid, soil, snow, vegetation) === grid
end

@testset "Model construction from spatial discretizations" begin
    column_grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 2)

    # Single-domain models keep whatever discretization they are given: an ordinary spatial
    # discretization is no longer wrapped in a land grid, so a model which resolves only the ground
    # can be built on a plain Oceananigans grid.
    model = SoilModel(column_grid)
    @test get_grid(model) === column_grid

    # A pre-built land grid is likewise stored as-is.
    land_grid = LandGrid(column_grid)
    @test get_grid(SoilModel(land_grid)) === land_grid
    @test ground_domain(get_grid(SoilModel(land_grid))) === column_grid

    # ... including a bare `RectilinearGrid`.
    rect_grid = RectilinearGrid(Float32, size = (2, 1, 5), x = (0, 1), y = (0, 1), z = (-1, 0))
    @test get_grid(SoilModel(rect_grid)) === rect_grid

    # `LandModel` couples several vertical domains, so it is the exception: it builds a land grid
    # from an ordinary discretization and stores a pre-built one unchanged.
    @test get_grid(LandModel(column_grid; vegetation = nothing)) isa LandGrid
    @test ground_domain(get_grid(LandModel(column_grid; vegetation = nothing))) === column_grid
    @test get_grid(LandModel(land_grid; vegetation = nothing)) === land_grid
end
