from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .debug_io import save_rgba_tiff
from .preprocessing import to_gray_normalized
from .utilities import rgba_from_gray, save_rgba_tiff_from_gray, load_image, collect_images_paths, ensure_dir
from .nn_input_prepare import parse_annotation_txt_rc
from nn_adapters import rows_cols_to_xywh

# Optional: if you keep ALFA_VALUE in config (normalized in [0,1])
try:
    from configs.config import ALFA_VALUE as _ALFA_VALUE
except Exception:
    _ALFA_VALUE = 0.5


def seeds_to_mask(
    shape_hw: Tuple[int, int],
    seeds: Iterable,
) -> np.ndarray:
    """Return a 2D boolean mask with True at (row,col) seed locations."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=bool)

    for s in seeds:
        r = int(s.row)
        c = int(s.col)
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True

    return mask


def overlay_from_mask(
    bool_mask: np.ndarray,
    *,
    half_width: int = 2,
    overlay_form: str = "square",
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha_u8: int = 128,
) -> np.ndarray:
    """
    Build an RGBA overlay image (uint8) that is transparent everywhere except
    where bool_mask is True (expanded to a square if requested).

    - bool_mask: (H,W) bool mask.
    - half_width: square marker half-width in pixels (square side = 2*half_width+1).
    - alpha_u8: 0..255 opacity used on the overlay where mask is True.
    """
    if bool_mask.ndim != 2:
        raise ValueError(f"bool_mask must be 2D, got shape={bool_mask.shape}")

    h, w = bool_mask.shape
    hw = max(int(half_width), 0)

    # Expand mask in-place on a copy, so callers keep the original.
    mask = bool_mask.copy()

    if overlay_form == "square" and hw > 0:
        rr, cc = np.nonzero(mask)
        for r, c in zip(rr.tolist(), cc.tolist()):
            r0 = max(0, r - hw)
            r1 = min(h, r + hw + 1)
            c0 = max(0, c - hw)
            c1 = min(w, c + hw + 1)
            mask[r0:r1, c0:c1] = True
    elif overlay_form != "square":
        raise ValueError(f"Unsupported overlay_form={overlay_form!r}")

    r_col, g_col, b_col = (int(color[0]), int(color[1]), int(color[2]))
    a = int(np.clip(alpha_u8, 0, 255))

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask, 0] = np.uint8(np.clip(r_col, 0, 255))
    overlay[mask, 1] = np.uint8(np.clip(g_col, 0, 255))
    overlay[mask, 2] = np.uint8(np.clip(b_col, 0, 255))
    overlay[mask, 3] = np.uint8(a)

    return overlay


def apply_overlay(
    image_base_rgba: np.ndarray,
    overlay_rgba: np.ndarray,
    *,
    alfa_value: float | None = None,
) -> np.ndarray:
    """
    Alpha-blend overlay over base using the overlay alpha channel.

    Assumptions:
    - image_base_rgba: uint8 (H,W,4)
    - overlay_rgba:    uint8 (H,W,4), with alpha=0 where overlay is inactive

    If alfa_value is provided (0..1), it scales overlay opacity (global knob).
    If not provided, falls back to ALFA_VALUE from config when available.

    Returns a new uint8 (H,W,4) RGBA image.
    """
    if image_base_rgba.ndim != 3 or image_base_rgba.shape[-1] != 4:
        raise ValueError(f"image_base_rgba must be (H,W,4), got {image_base_rgba.shape}")
    if overlay_rgba.shape != image_base_rgba.shape:
        raise ValueError(
            f"overlay_rgba shape {overlay_rgba.shape} must match base {image_base_rgba.shape}"
        )

    alpha_scale = _ALFA_VALUE if alfa_value is None else float(alfa_value)
    alpha_scale = float(np.clip(alpha_scale, 0.0, 1.0))

    base = image_base_rgba.astype(np.float32)
    over = overlay_rgba.astype(np.float32)

    # Per-pixel alpha from overlay (0..1), scaled by alpha_scale.
    a = (over[..., 3] / 255.0) * alpha_scale  # (H,W)

    # Blend only where overlay is active.
    mask = a > 0.0
    if np.any(mask):
        a3 = a[mask][:, None]  # (N,1)
        base[mask, 0:3] = (1.0 - a3) * base[mask, 0:3] + a3 * over[mask, 0:3]

    # Keep base alpha as-is (usually 255).
    out = np.clip(base, 0.0, 255.0).astype(np.uint8)
    return out

def paint_points_in_place(
    image_rgba: np.ndarray,
    rows,
    cols,
    *,
    color: Tuple,
    half_width=1,
    alfa_value: float | None = None,
    round_coords=True,
) -> None:
    h, w = image_rgba.shape[:2]

    if alfa_value is None:
        alfa_value = _ALFA_VALUE

    for row, col in zip(rows, cols):
        r = int(round(row)) if round_coords else int(row)
        c = int(round(col)) if round_coords else int(col)

        if not (0 <= r < h and 0 <= c < w):
            continue

        r0 = max(0, r - half_width)
        r1 = min(h, r + half_width + 1)
        c0 = max(0, c - half_width)
        c1 = min(w, c + half_width + 1)

        image_rgba[r0:r1, c0:c1, 0] = color[0]
        image_rgba[r0:r1, c0:c1, 1] = color[1]
        image_rgba[r0:r1, c0:c1, 2] = color[2]
        image_rgba[r0:r1, c0:c1, 3] = int(round(255.0 * float(np.clip(alfa_value, 0.0, 1.0))))




# TODO: clear this in future
def visualize_suppressed_grid_views(
    base_image: np.ndarray,
    *,
    grid0_seeds: Iterable,
    grid1_seeds: Iterable,
    both_grid_seeds: Iterable | None = None,
    half_width: int = 2,
    alpha_u8: int = 160,
    alfa_value: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Returns overlays:
        - grid0_only_view
        - grid1_only_view
        - both_only_view
        - combined_grid01_view

    In combined_grid01_view:
        - grid0 is red
        - grid1 is green
        - overlap mixes visually
    """
    base_rgba = rgba_from_gray(base_image)
    shape_hw = base_rgba.shape[:2]

    grid0_mask = seeds_to_mask(shape_hw, grid0_seeds)
    grid1_mask = seeds_to_mask(shape_hw, grid1_seeds)
    both_mask = seeds_to_mask(shape_hw, both_grid_seeds or [])

    grid0_overlay = overlay_from_mask(
        grid0_mask,
        half_width=half_width,
        color=(255, 0, 0),
        alpha_u8=alpha_u8,
    )
    grid1_overlay = overlay_from_mask(
        grid1_mask,
        half_width=half_width,
        color=(0, 255, 0),
        alpha_u8=alpha_u8,
    )
    both_overlay = overlay_from_mask(
        both_mask,
        half_width=half_width,
        color=(0, 0, 255),
        alpha_u8=alpha_u8,
    )

    grid0_view = apply_overlay(base_rgba.copy(), grid0_overlay, alfa_value=alfa_value)
    grid1_view = apply_overlay(base_rgba.copy(), grid1_overlay, alfa_value=alfa_value)
    both_view = apply_overlay(base_rgba.copy(), both_overlay, alfa_value=alfa_value)

    combined = apply_overlay(base_rgba.copy(), grid0_overlay, alfa_value=alfa_value)
    combined = apply_overlay(combined, grid1_overlay, alfa_value=alfa_value)

    return {
        "grid0_only_view": grid0_view,
        "grid1_only_view": grid1_view,
        "both_only_view": both_view,
        "combined_grid01_view": combined,
    }


def paint_xywh_boxes_in_place(
        #TODO: Check the channels, derive #top,bottom, left, right
    image_rgba: np.ndarray,
    boxes_xywh: np.ndarray,
    *,
    color: tuple[int, int, int],
    alfa_value: float | None = None,
    line_width: int = 2,
) -> None:
    """
    Paint XYWH bounding boxes into an RGBA image in place.

    boxes_xywh format:
        [x, y, w, h]
    """
    if alfa_value is None:
        alfa_value = _ALFA_VALUE

    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)

    if boxes_xywh.ndim != 2 or boxes_xywh.shape[1] != 4:
        raise ValueError(f"boxes_xywh must have shape (N, 4), got {boxes_xywh.shape}")

    h_img, w_img = image_rgba.shape[:2]
    alpha_u8 = int(round(255.0 * float(np.clip(alfa_value, 0.0, 1.0))))

    for box in boxes_xywh:
        x, y, w, h = box

        col_start = int(round(x))
        row_start = int(round(y))
        col_end = int(round(x + w))
        row_end = int(round(y + h))

        col_start = max(0, min(w_img - 1, col_start))
        col_end = max(0, min(w_img - 1, col_end))
        row_start = max(0, min(h_img - 1, row_start))
        row_end = max(0, min(h_img - 1, row_end))

        if col_end <= col_start or row_end <= row_start:
            continue

        lw = max(int(line_width), 1)

        # top
        image_rgba[row_start:min(row_start + lw, h_img), col_start:col_end + 1, 0] = color[0]
        image_rgba[row_start:min(row_start + lw, h_img), col_start:col_end + 1, 1] = color[1]
        image_rgba[row_start:min(row_start + lw, h_img), col_start:col_end + 1, 2] = color[2]
        image_rgba[row_start:min(row_start + lw, h_img), col_start:col_end + 1, 3] = alpha_u8

        # bottom
        image_rgba[max(row_end - lw + 1, 0):row_end + 1, col_start:col_end + 1, 0] = color[0]
        image_rgba[max(row_end - lw + 1, 0):row_end + 1, col_start:col_end + 1, 1] = color[1]
        image_rgba[max(row_end - lw + 1, 0):row_end + 1, col_start:col_end + 1, 2] = color[2]
        image_rgba[max(row_end - lw + 1, 0):row_end + 1, col_start:col_end + 1, 3] = alpha_u8

        # left
        image_rgba[row_start:row_end + 1, col_start:min(col_start + lw, w_img), 0] = color[0]
        image_rgba[row_start:row_end + 1, col_start:min(col_start + lw, w_img), 1] = color[1]
        image_rgba[row_start:row_end + 1, col_start:min(col_start + lw, w_img), 2] = color[2]
        image_rgba[row_start:row_end + 1, col_start:min(col_start + lw, w_img), 3] = alpha_u8

        # right
        image_rgba[row_start:row_end + 1, max(col_end - lw + 1, 0):col_end + 1, 0] = color[0]
        image_rgba[row_start:row_end + 1, max(col_end - lw + 1, 0):col_end + 1, 1] = color[1]
        image_rgba[row_start:row_end + 1, max(col_end - lw + 1, 0):col_end + 1, 2] = color[2]
        image_rgba[row_start:row_end + 1, max(col_end - lw + 1, 0):col_end + 1, 3] = alpha_u8


def paint_labeled_xywh_boxes_in_place(
    image_rgba: np.ndarray,
    boxes_xywh: np.ndarray,
    labels: np.ndarray,
    *,
    class_colors: dict[int, tuple[int, int, int]],
    default_color: tuple[int, int, int] = (255, 255, 255),
    alfa_value: float | None = None,
    line_width: int = 2,
) -> None:
    """
    Paint XYWH boxes into an RGBA image in place, using one color per class label.

    Parameters
    ----------
    image_rgba
        RGBA image of shape (H, W, 4), dtype uint8.

    boxes_xywh
        Boxes of shape (N, 4), format:
            [x, y, width, height]

    labels
        Class labels of shape (N,).

    class_colors
        Dict mapping class id to RGB color:
            {class_id: (R, G, B)}

    default_color
        Fallback RGB color used when label is missing from class_colors.

    alfa_value
        Box opacity in [0, 1]. If None, uses module default.

    line_width
        Box line width in pixels.
    """
    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
    labels = np.asarray(labels)

    if boxes_xywh.ndim != 2 or boxes_xywh.shape[1] != 4:
        raise ValueError(
            f"boxes_xywh must have shape (N, 4), got {boxes_xywh.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(f"labels must have shape (N,), got {labels.shape}")

    if len(labels) != len(boxes_xywh):
        raise ValueError(
            f"labels and boxes_xywh must have same length, "
            f"got {len(labels)} and {len(boxes_xywh)}"
        )

    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_int = int(label)
        color = class_colors.get(label_int, default_color)

        class_mask = labels == label
        class_boxes_xywh = boxes_xywh[class_mask]

        paint_xywh_boxes_in_place(
            image_rgba,
            class_boxes_xywh,
            color=color,
            alfa_value=alfa_value,
            line_width=line_width,
        )