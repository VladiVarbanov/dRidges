from __future__ import annotations

from configs.config import (NN_ANN_EXT,
                            NN_MIN_BOX_WIDTH,
                            NN_MIN_BOX_HEIGHT,
                            NN_MIN_BOX_AREA,
                            NN_EXPECTED_NUM_CHANNELS, FRAMEWORK_TORCHVISION, FRAMEWORK_NUMPY,
                            ROI_FORMAT_XYWH,
                            NN_LOCAL_NORM_SIGMA,
                            NN_GAUSSIAN_SMOOTH_SIGMA,
                            NN_HESSIAN_SCALE_PX,
                            )
from pathlib import Path
from .utilities import collect_images_paths, load_image
from typing import Callable
from .preprocessing import to_gray_normalized, local_normalize_HOG_style, gaussian_smoothing
from .ridges import RidgeMap
from typing import Any
from dataclasses import dataclass, field
import numpy as np
import torch
from torchvision import tv_tensors

from skimage.filters import gaussian


def rows_cols_to_xywh(boxes_rc: np.ndarray) -> np.ndarray:
    """
    Convert boxes from source row/col corner format:

        [y_min, x_min, y_max, x_max]

    to internal xywh format:

        [x, y, w, h]

    Parameters
    ----------
    boxes_rc
        np.ndarray of shape (N, 4)

    Returns
    -------
    boxes
        np.ndarray of shape (N, 4), dtype float32
    """
    boxes_rc = np.asarray(boxes_rc, dtype=np.float32)

    if boxes_rc.ndim != 2 or boxes_rc.shape[1] != 4:
        raise ValueError(
            f"rows_cols_to_xywh expects shape (N, 4), got {boxes_rc.shape}"
        )

    y_min = boxes_rc[:, 0]
    x_min = boxes_rc[:, 1]
    y_max = boxes_rc[:, 2]
    x_max = boxes_rc[:, 3]

    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    return np.stack((x, y, w, h), axis=1).astype(np.float32, copy=False)


def rows_cols_to_xyxy(boxes_rc: np.ndarray) -> np.ndarray:
    """
    Convert boxes from source row/col corner format:

        [y_min, x_min, y_max, x_max]

    to XYXY format:

        [x_min, y_min, x_max, y_max]

    This is the format expected by TorchVision detection models.
    """
    boxes_rc = np.asarray(boxes_rc, dtype=np.float32)

    if boxes_rc.ndim != 2 or boxes_rc.shape[1] != 4:
        raise ValueError(
            f"rows_cols_to_xyxy expects shape (N, 4), got {boxes_rc.shape}"
        )

    y_min = boxes_rc[:, 0]
    x_min = boxes_rc[:, 1]
    y_max = boxes_rc[:, 2]
    x_max = boxes_rc[:, 3]

    return np.stack((x_min, y_min, x_max, y_max), axis=1).astype(
        np.float32,
        copy=False,
    )


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)

    if boxes_xywh.ndim != 2 or boxes_xywh.shape[1] != 4:
        raise ValueError(
            f"boxes_xywh must have shape (N, 4), got {boxes_xywh.shape}"
        )

    x = boxes_xywh[:, 0]
    y = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]

    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    return np.stack((x_min, y_min, x_max, y_max), axis=1).astype(
        np.float32,
        copy=False,
    )


def xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Convert boxes from XYXY format:

        [x_min, y_min, x_max, y_max]

    to XYWH format:

        [x, y, width, height]

    Parameters
    ----------
    boxes_xyxy
        np.ndarray of shape (N, 4)

    Returns
    -------
    boxes_xywh
        np.ndarray of shape (N, 4), dtype float32
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)

    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError(
            f"xyxy_to_xywh expects shape (N, 4), got {boxes_xyxy.shape}"
        )

    x_min = boxes_xyxy[:, 0]
    y_min = boxes_xyxy[:, 1]
    x_max = boxes_xyxy[:, 2]
    y_max = boxes_xyxy[:, 3]

    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    return np.stack((x, y, w, h), axis=1).astype(np.float32, copy=False)


def define_rois_for_framework(
    # Legacy/future framework adapter.
    # Not used by the first TorchVision Faster R-CNN Dataset path.
    # Current training code builds the TorchVision target dict directly.
    boxes: np.ndarray,
    height: int,
    width: int,
    framework: str,
    roi_format: str

,
):
    boxes = boxes.astype(np.float32, copy=False)

    if framework == FRAMEWORK_TORCHVISION:
        return tv_tensors.BoundingBoxes(
            torch.as_tensor(boxes, dtype=torch.float32),
            format=roi_format,
            canvas_size=(height, width),
        )

    if framework == FRAMEWORK_NUMPY:
        return boxes

    raise ValueError(f"Unsupported framework: {framework}")


def label_ids_to_bg0_format(label_ids: np.ndarray) -> np.ndarray:
    """
    Convert annotation/internal label IDs to BG0 detection label format.

    BG0 format:
        0 = background
        1, 2, 3, ... = foreground classes

    Internal label IDs:
        0, 1, 2, ... = foreground classes

    Therefore:
        bg0_label = label_id + 1
    """
    label_ids = np.asarray(label_ids, dtype=np.int64)

    if label_ids.ndim != 1:
        raise ValueError(
            f"label_ids_to_bg0_format expects shape (N,), got {label_ids.shape}"
        )

    if np.any(label_ids < 0):
        raise ValueError(
            f"label IDs must be >= 0, got {label_ids}"
        )

    return label_ids + 1


def label_ids_from_bg0_format(bg0_labels: np.ndarray) -> np.ndarray:
    """
    Convert BG0 foreground labels back to annotation/internal label IDs.

    BG0 format:
        0 = background
        1, 2, 3, ... = foreground classes

    Therefore:
        label_id = bg0_label - 1

    This expects foreground labels only.
    """
    bg0_labels = np.asarray(bg0_labels, dtype=np.int64)

    if bg0_labels.ndim != 1:
        raise ValueError(
            f"label_ids_from_bg0_format expects shape (N,), got {bg0_labels.shape}"
        )

    if np.any(bg0_labels <= 0):
        raise ValueError(
            f"BG0 foreground labels must be > 0, got {bg0_labels}"
        )

    return bg0_labels - 1