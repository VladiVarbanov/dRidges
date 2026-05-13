"""
Napari tools for viewing and editing image annotations.

This package contains small GUI helpers for:
    - launching napari with project images,
    - displaying source annotation boxes,
    - converting source row/column boxes to napari shapes,
    - eventually saving corrected annotations.

Source annotation format:
    class row_min col_min row_max col_max

Important:
    These tools operate on source annotation labels:
        0, 1, 2, 3

    They do not use TorchVision BG0 labels:
        1, 2, 3, 4
"""