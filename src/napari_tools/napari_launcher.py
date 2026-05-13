"""
Launcher for napari-based annotation viewing/editing tools.

Run from project root:

    python -m src.napari_tools.napari_launcher

This file is intentionally small.
It acts like an entry point and delegates the real viewer logic to:

    src.napari_tools.view_annotated_images
"""

from __future__ import annotations

from src.napari_tools.view_annotated_images import view_napari_annotated_images


def main() -> None:
    """
    Start the default napari annotation-viewing workflow.

    Current default behavior:
        - open one image from the training split,
        - load its matching source annotation txt file,
        - display the image,
        - display annotation boxes as napari shapes.

    Source annotation format:
        class row_min col_min row_max col_max

    Important:
        This viewer works with source labels:
            0, 1, 2, 3

        It does not use TorchVision BG0 labels:
            1, 2, 3, 4
    """
    view_napari_annotated_images()


if __name__ == "__main__":
    main()