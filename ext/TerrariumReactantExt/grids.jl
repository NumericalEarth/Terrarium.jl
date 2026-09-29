# Grid transfers to/from the device.
# ColumnRingGrid: transfer the inner Oceananigans grid to the device; keep `rings`/`mask` on the
# CPU, where they are used only for host-side pre/post-processing (masked-index field conversion).
Terrarium.on_architecture(arch::RARCH, grid::ColumnRingGrid) =
    ColumnRingGrid(grid.rings, grid.mask, on_architecture(arch, grid.grid))

Terrarium.on_architecture(arch::CPU, grid::ColumnRingGrid{<:Any, <:RARCH}) =
    ColumnRingGrid(
    on_architecture(arch, grid.rings), on_architecture(arch, grid.mask),
    on_architecture(arch, grid.grid)
)
