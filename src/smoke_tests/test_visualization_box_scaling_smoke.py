from __future__ import annotations

import numpy as np

from src.visualization import (
    scale_xywh_boxes_about_center,
    paint_labeled_xywh_boxes_in_place,
)


def test_scale_xywh_boxes_about_center_keeps_center() -> None:
    boxes_xywh = np.asarray(
        [
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=np.float32,
    )

    scaled_boxes = scale_xywh_boxes_about_center(
        boxes_xywh,
        scale_factor=1.2,
    )

    original_center_x = boxes_xywh[:, 0] + 0.5 * boxes_xywh[:, 2]
    original_center_y = boxes_xywh[:, 1] + 0.5 * boxes_xywh[:, 3]

    scaled_center_x = scaled_boxes[:, 0] + 0.5 * scaled_boxes[:, 2]
    scaled_center_y = scaled_boxes[:, 1] + 0.5 * scaled_boxes[:, 3]

    assert np.allclose(original_center_x, scaled_center_x)
    assert np.allclose(original_center_y, scaled_center_y)
    assert np.allclose(scaled_boxes[:, 2], boxes_xywh[:, 2] * 1.2)
    assert np.allclose(scaled_boxes[:, 3], boxes_xywh[:, 3] * 1.2)


def test_paint_labeled_xywh_boxes_accepts_scale_factor() -> None:
    image_rgba = np.zeros((80, 80, 4), dtype=np.uint8)

    boxes_xywh = np.asarray(
        [
            [20.0, 20.0, 20.0, 20.0],
        ],
        dtype=np.float32,
    )

    labels = np.asarray([0], dtype=np.int64)

    paint_labeled_xywh_boxes_in_place(
        image_rgba=image_rgba,
        boxes_xywh=boxes_xywh,
        labels=labels,
        class_colors={0: (255, 0, 0)},
        line_width=2,
        scale_factor=1.2,
    )

    assert image_rgba[..., 3].max() > 0