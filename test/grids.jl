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
    @test halo_size(grid) == halo_size(get_field_grid(grid))
    @test isrectilinear(grid)
    @test znodes(grid, Oceananigans.Center()) ≈ znodes(get_field_grid(grid), Oceananigans.Center())

    ring_grid = FullHEALPixGrid(4)
    col_ring_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5f0, N = 3), ring_grid)
    @test col_ring_grid isa Oceananigans.AbstractGrid
    @test eltype(col_ring_grid) == Float32
    @test halo_size(col_ring_grid) == halo_size(get_field_grid(col_ring_grid))
    @test isrectilinear(col_ring_grid)
end

@testset "Grid property forwarding" begin
    grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 3)
    field_grid = get_field_grid(grid)

    # Dimension/halo fields Oceananigans accesses directly (e.g. grid.Nx) forward to the
    # underlying field grid rather than throwing a missing-field error.
    for name in (:Nx, :Ny, :Nz, :Hx, :Hy, :Hz)
        @test getproperty(grid, name) === getproperty(field_grid, name)
    end
    @test grid.Nx == 3
    @test grid.Ny == 1
    @test grid.Nz == 5

    # Coordinate arrays are forwarded generically as well.
    @test grid.z === field_grid.z

    # The wrapper's own struct field still resolves via getfield (not shadowed by forwarding).
    @test grid.grid === field_grid

    # propertynames advertises both the wrapper's own fields and the forwarded ones.
    @test :grid in propertynames(grid)
    @test :Nx in propertynames(grid)
    @test :Nz in propertynames(grid)

    # ColumnRingGrid's extra own fields (rings, mask) are not shadowed by the forwarding.
    col_ring_grid = ColumnRingGrid(UniformSpacing(Δz = 0.5f0, N = 3), FullHEALPixGrid(4))
    @test col_ring_grid.grid === get_field_grid(col_ring_grid)
    @test col_ring_grid.rings isa RingGrids.AbstractGrid
    @test col_ring_grid.Nx == get_field_grid(col_ring_grid).Nx
    @test :rings in propertynames(col_ring_grid)
    @test :mask in propertynames(col_ring_grid)
end

@testset "Oceananigans operations on land grids" begin
    # Exercise real Oceananigans code paths that consume grid fields directly (grid.Nx/Ny/Nz,
    # halos): total_size and Field construction/reduction must work on the land grid itself.
    grid = ColumnGrid(UniformSpacing(Δz = 0.1f0, N = 5), 3)

    @test total_size(grid) == total_size(get_field_grid(grid))

    # `CenterField(grid)` dispatches into Oceananigans' generic AbstractGrid constructor, which
    # allocates data from the grid's dimensions/halos — only possible if the fields forward.
    field = CenterField(grid)
    @test size(field) == size(grid)
    set!(field, 2)
    @test sum(interior(field)) ≈ 2 * prod(size(grid))
end

@testset "Vertical discretizations" begin
    # Uniform spacing
    @test Terrarium.get_spacing(UniformSpacing(Δz = 0.1, N = 1)) == [0.1]
    @test Terrarium.get_spacing(UniformSpacing(Δz = 0.1, N = 10)) == repeat([0.1], 10)

    # ExponentialSpacing
    @test Terrarium.get_spacing(ExponentialSpacing(Δz_min = 0.1, Δz_max = 1.0, N = 2)) == [0.1, 1.0]
    @test Terrarium.get_spacing(ExponentialSpacing(Δz_min = 0.1, Δz_max = 1.0, N = 3, sig = nothing)) ≈ exp2.(LinRange(log2(0.1), log2(1.0), 3))

    # PrescribedSpacing
    @test Terrarium.get_spacing(PrescribedSpacing(Δz = [0.1, 0.2, 0.3])) == [0.1, 0.2, 0.3]
end

@testset "ColumnGrid" begin
    # 2-column grid with 5 linearly spaced points
    num_columns = 2
    grid = ColumnGrid(UniformSpacing(Δz = 0.1, N = 5), num_columns)
    field_grid = get_field_grid(grid)
    @test isa(field_grid, RectilinearGrid)
    @test field_grid.Nx == num_columns
    @test field_grid.Ny == 1
    @test field_grid.Nz == 5
    @test z_domain(field_grid) == (-0.5, 0.0)
end

@testset "ColumnRingGrid" begin
    # test with 10-ring HEALPix
    ring_grid = FullHEALPixGrid(8)
    grid = ColumnRingGrid(UniformSpacing(Δz = 0.5, N = 10), ring_grid)
    field_grid = get_field_grid(grid)
    @test isa(field_grid, RectilinearGrid)
    @test field_grid.Nx == get_npoints(ring_grid)
    @test field_grid.Ny == 1
    @test field_grid.Nz == 10
    @test z_domain(field_grid) == (-5.0, 0.0)

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
        fgrid = get_field_grid(full_grid)
        ncols = fgrid.Nx  # == npoints (no mask)

        # 2D Oceananigans field (Field{Center,Center,Nothing}) → RingGrids.Field
        field_2d = Terrarium.Field{Center, Center, Nothing}(fgrid)
        Terrarium.interior(field_2d)[:, 1, 1] .= Float32.(1:ncols)

        ring_2d = RingGrids.Field(field_2d, full_grid)
        @test ring_2d.data[full_grid.mask.data, 1] ≈ Float32.(1:ncols)

        # 3D Oceananigans field (Field{Center,Center,Center}) → RingGrids.Field
        field_3d = Terrarium.Field{Center, Center, Center}(fgrid)
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
        masked_fgrid = get_field_grid(masked_grid)

        field_masked = Terrarium.Field{Center, Center, Nothing}(masked_fgrid)
        fill!(field_masked, 1.0f0)

        ring_fill = RingGrids.Field(field_masked, masked_grid; fill_value = -1.0)
        @test all(ring_fill.data[masked_grid.mask.data, 1] .≈ 1.0f0)
        @test all(ring_fill.data[.!masked_grid.mask.data, 1] .== -1.0)
    end
end
