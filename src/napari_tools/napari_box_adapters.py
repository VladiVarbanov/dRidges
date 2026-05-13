"""
Napari-specific box format adapters.

Source annotation box format:
    [row_min, col_min, row_max, col_max]

Napari rectangle format:
    four vertices in row/column coordinates:
        [
            [row_min, col_min],
            [row_max, col_min],
            [row_max, col_max],
            [row_min, col_max],
        ]

Important:
    This module does not know about TorchVision.
    It does not convert labels.
    It does not shift labels to BG0 format.
"""

from __future__ import annotations

import numpy as np


def box_rc_to_napari_rectangle(box_rc: np.ndarray) -> np.ndarray:
    """
    Convert one source row/column box to one napari rectangle.

    Source box format:
        [row_min, col_min, row_max, col_max]

    Napari rectangle format:
        shape = (4, 2)
        coordinates are [row, col]
    """
    box_rc = np.asarray(box_rc, dtype=np.float32)

    if box_rc.shape != (4,):
        raise ValueError(f"box_rc must have shape (4,), got {box_rc.shape}")

    row_min, col_min, row_max, col_max = box_rc

    if row_max <= row_min:
        raise ValueError(
            f"Invalid box height: row_max={row_max} must be > row_min={row_min}"
        )

    if col_max <= col_min:
        raise ValueError(
            f"Invalid box width: col_max={col_max} must be > col_min={col_min}"
        )

    return np.asarray(
        [
            [row_min, col_min],  # top-left
            [row_max, col_min],  # bottom-left
            [row_max, col_max],  # bottom-right
            [row_min, col_max],  # top-right
        ],
        dtype=np.float32,
    )


def boxes_rc_to_napari_rectangles(boxes_rc: np.ndarray) -> list[np.ndarray]:
    """
    Convert multiple source row/column boxes to napari rectangles.

    Input:
        boxes_rc shape = (N, 4)

    Output:
        list of N rectangles
        each rectangle has shape = (4, 2)
    """
    boxes_rc = np.asarray(boxes_rc, dtype=np.float32)

    if boxes_rc.ndim != 2 or boxes_rc.shape[1] != 4:
        raise ValueError(
            f"boxes_rc must have shape (N, 4), got {boxes_rc.shape}"
        )

    return [
        box_rc_to_napari_rectangle(box_rc)
        for box_rc in boxes_rc
    ]


def napari_rectangle_to_box_rc(rectangle: np.ndarray) -> np.ndarray:
    """
    Convert one edited napari rectangle back to source row/column box format.

    Napari rectangle:
        shape = (N, 2)
        coordinates are [row, col]

    Output:
        [row_min, col_min, row_max, col_max]

    This uses min/max of vertices, so it stays valid after moving/resizing.
    """
    rectangle = np.asarray(rectangle, dtype=np.float32)

    if rectangle.ndim != 2 or rectangle.shape[1] != 2:
        raise ValueError(
            f"rectangle must have shape (N, 2), got {rectangle.shape}"
        )

    rows = rectangle[:, 0]
    cols = rectangle[:, 1]

    row_min = float(np.min(rows))
    col_min = float(np.min(cols))
    row_max = float(np.max(rows))
    col_max = float(np.max(cols))

    if row_max <= row_min:
        raise ValueError(
            f"Invalid edited rectangle height: row_max={row_max} "
            f"must be > row_min={row_min}"
        )

    if col_max <= col_min:
        raise ValueError(
            f"Invalid edited rectangle width: col_max={col_max} "
            f"must be > col_min={col_min}"
        )

    return np.asarray(
        [row_min, col_min, row_max, col_max],
        dtype=np.float32,
    )


def napari_rectangles_to_boxes_rc(rectangles: list[np.ndarray]) -> np.ndarray:
    """
    Convert multiple napari rectangles back to source row/column boxes.

    Input:
        list of napari rectangles

    Output:
        boxes_rc shape = (N, 4)
    """
    boxes_rc = [
        napari_rectangle_to_box_rc(rectangle)
        for rectangle in rectangles
    ]

    if not boxes_rc:
        return np.empty((0, 4), dtype=np.float32)

    return np.stack(boxes_rc, axis=0).astype(np.float32, copy=False)