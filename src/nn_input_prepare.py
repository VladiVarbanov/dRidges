from __future__ import annotations

from configs.config import (NN_ANN_EXT,
                            NN_MIN_BOX_WIDTH,
                            NN_MIN_BOX_HEIGHT,
                            NN_MIN_BOX_AREA,
                            ROI_FORMAT_XYWH,
                            NN_LOCAL_NORM_SIGMA,
                            NN_GAUSSIAN_SMOOTH_SIGMA,
                            NN_HESSIAN_SCALE_PX,
                            )

from .annotation_io import resolve_annotation_path, pair_split_images_with_annotations
from .utilities import collect_images_paths, load_image, ensure_dir
from typing import Callable
from .preprocessing import to_gray_normalized, local_normalize_HOG_style
from .ridges import RidgeMap
from typing import Any
from dataclasses import dataclass, field
import numpy as np
import json
from skimage.filters import gaussian

@dataclass
class NNSample:
    """
    Neutral NN sample container.

    It stores prepared data and metadata.
    It does not build channels.
    It does not parse annotations.
    It does not convert boxes to a framework-specific target.
    """
    nn_input_image: np.ndarray

    boxes: np.ndarray | None = None
    labels: np.ndarray | None = None

    image_path: Path | None = None
    annotation_path: Path | None = None
    image_id: int | None = None

    roi_framework: str | None = None
    roi_format: str | None = None
    nn_architecture: str | None = None

    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DerivedAnnotationConfig:
    split_path: Path
    image_dir: Path
    src_annotation_dir: Path
    dst_annotation_dir: Path
    dst_annotation_format_path: Path
    source_annotation_format_path: Path

    class_ids_initial: tuple[int, ...]
    class_ids_merged: tuple[int, ...]
    derived_class_names: dict[int, str]

    bbox_scale_rules: list[dict[str, Any]] = field(default_factory=list)
    annotation_format_name: str = "derived_annotations_v1"
    ann_ext: str = NN_ANN_EXT
    max_images: int | None = None


from pathlib import Path


def pair_images_with_annotations(
    input_dir: str | Path,
    *,
    recursive: bool = True,
    require_annotation: bool = True,
) -> list[tuple[Path, Path | None]]:
    """

    TODO: OBSOLETE DELETE THIS FUNCTION.
    TODO:2 I guess it is not that obsolete after all haha
    Scan a directory for image files and pair each image with its matching
    annotation path.

    Parameters
    ----------
    input_dir
        Directory containing images.
    recursive
        If True, recurse into subfolders.
    require_annotation
        If True, keep only images whose matching annotation file exists.
        If False, return None for missing annotation files.

    Returns
    -------
    pairs
        List of (image_path, annotation_path_or_none) tuples.
    """

    image_paths = collect_images_paths(
        input_dir=input_dir,
        recursive=recursive,
    )

    pairs: list[tuple[Path, Path | None]] = []

    for image_path in image_paths:
        ann_path = resolve_annotation_path(image_path)

        if ann_path.exists():
            pairs.append((image_path, ann_path))
        elif not require_annotation:
            pairs.append((image_path, None))

    return pairs


"""
Utilities for deriving copied/redacted box annotations from source annotations.

Original annotation files are never overwritten.

Source annotation format:
    class row_min col_min row_max col_max
"""


def keep_xywh_boxes(
    boxes_xywh: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    min_width: float = NN_MIN_BOX_WIDTH,
    min_height: float = NN_MIN_BOX_HEIGHT,
    min_area: float = NN_MIN_BOX_AREA,
    require_finite: bool = True,
):
    """
    Keep xywh boxes by size rules.
    TODO: Overlaps with other functions clean it later
    Parameters
    ----------
    boxes_xywh
        np.ndarray of shape (N, 4) in [x, y, w, h] format
    labels
        Optional np.ndarray of shape (N,). If provided, filtered labels are
        returned together with filtered boxes.
    min_width
        Minimum allowed width
    min_height
        Minimum allowed height
    min_area
        Minimum allowed area = w * h
    require_finite
        If True, reject boxes containing NaN or inf

    Returns
    -------
    If labels is None:
        filtered_boxes_xywh
    else:
        filtered_labels, filtered_boxes_xywh
    """
    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)

    if boxes_xywh.ndim != 2 or boxes_xywh.shape[1] != 4:
        raise ValueError(
            f"boxes must have shape (N, 4), got {boxes_xywh.shape}"
        )

    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape (N,), got {labels.shape}")
        if len(labels) != len(boxes_xywh):
            raise ValueError(
                f"labels and boxes must have the same length, got {len(labels)} and {len(boxes_xywh)}"
            )

    keep = np.ones(len(boxes_xywh), dtype=bool)

    if require_finite:
        keep &= np.isfinite(boxes_xywh).all(axis=1)

    widths = boxes_xywh[:, 2]
    heights = boxes_xywh[:, 3]
    areas = widths * heights

    keep &= widths >= min_width
    keep &= heights >= min_height
    keep &= areas >= min_area

    filtered_boxes = boxes_xywh[keep]

    if labels is None:
        return filtered_boxes

    return labels[keep], filtered_boxes



def discard_xywh_boxes_by_size(
    boxes_xywh: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    min_width: float | None = None,
    min_height: float | None = None,
    min_area: float | None = None,
):
    """
    Discard boxes that fail enabled size criteria.

    Parameters
    ----------
    boxes_xywh
        Array of shape (N, 4) in [x, y, w, h] format.
    labels
        Optional array of shape (N,). If provided, matching labels are discarded too.
    min_width
        Discard boxes with w < min_width.
        If None, width is not checked.
    min_height
        Discard boxes with h < min_height.
        If None, height is not checked.
    min_area
        Discard boxes with w * h < min_area.
        If None, area is not checked.

    Returns
    -------
    kept_boxes_xywh
        If labels is None.
    kept_labels, kept_boxes_xywh
        If labels is provided.
    """
    boxes_xywh = boxes_xywh.astype(np.float32, copy=False)

    keep_mask = np.ones(len(boxes_xywh), dtype=bool)

    widths = None
    heights = None

    if min_width is not None:
        widths = boxes_xywh[:, 2]
        keep_mask &= widths >= min_width

    if min_height is not None:
        heights = boxes_xywh[:, 3]
        keep_mask &= heights >= min_height

    if min_area is not None:
        if widths is None:
            widths = boxes_xywh[:, 2]
        if heights is None:
            heights = boxes_xywh[:, 3]

        areas = widths * heights
        keep_mask &= areas >= min_area

    kept_boxes_xywh = boxes_xywh[keep_mask]

    if labels is None:
        return kept_boxes_xywh

    kept_labels = labels[keep_mask]
    return kept_labels, kept_boxes_xywh


def build_nn_gray_channel(img_raw: np.ndarray) -> np.ndarray:
    """
        TODO: use cache later. Add some different logic
        Build the grayscale normalized channel from the raw input image.

        Returns
        -------
        gray
            2D float32 image, normalized for downstream processing.
        """

    return to_gray_normalized(img_raw)

def build_nn_hog_norm_channel(img_raw: np.ndarray) -> np.ndarray:
    """
    Build the HOG-style locally normalized channel from the raw input image.

    Returns
    -------
    hog_norm
        2D float32 image for downstream NN input.
    """
    gray = to_gray_normalized(img_raw,)
    return local_normalize_HOG_style(gray, local_norm_sigma=NN_LOCAL_NORM_SIGMA)


def build_nn_vesselness_channel(img_raw: np.ndarray) -> np.ndarray:
    """
    Build NN-specific vesselness channel.

    This is separate from the classical ridge pipeline.
    """
    gray = to_gray_normalized(img_raw)

    smoothed = gaussian(
        gray,
        sigma=NN_GAUSSIAN_SMOOTH_SIGMA,
        preserve_range=True,
    )

    ridge_map = RidgeMap(
        smoothed,
        hessian_scale_px=NN_HESSIAN_SCALE_PX,
    )

    if ridge_map.vesselness is None:
        raise RuntimeError("RidgeMap did not produce vesselness")

    return ridge_map.vesselness.astype(np.float32, copy=False)


def build_nn_input_img(
    img_raw: np.ndarray,
    channel_fns: list[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    """
    Build the final NN input image from explicit channel builder functions.

    Parameters
    ----------
    img_raw
        Raw input image as loaded from disk.

    channel_fns
        List of callables. Each function must:
            - take img_raw
            - return one 2D channel of shape (H, W)

    Returns
    -------
    nn_input_image
        Stacked image of shape (C, H, W), dtype float32.
    """
    if not channel_fns:
        raise ValueError("channel_fns must not be empty")

    channels: list[np.ndarray] = []

    for fn in channel_fns:
        fn_name = getattr(fn, "__name__", repr(fn))

        ch = np.asarray(fn(img_raw), dtype=np.float32)

        if ch.ndim != 2:
            raise ValueError(
                f"Channel function {fn_name} must return a 2D array, "
                f"got shape {ch.shape}"
            )

        if not np.isfinite(ch).all():
            raise ValueError(
                f"Channel function {fn_name} returned NaN or Inf values"
            )

        channels.append(ch)

    base_shape = channels[0].shape

    for i, ch in enumerate(channels[1:], start=1):
        if ch.shape != base_shape:
            raise ValueError(
                f"All channels must have the same shape. "
                f"Channel 0 has shape {base_shape}, "
                f"channel {i} has shape {ch.shape}"
            )

    nn_input_image = np.stack(channels, axis=0).astype(np.float32, copy=False)

    return nn_input_image


def save_nn_input_to_npy(
    nn_input_image: np.ndarray,
    *,
    npy_path: str | Path,
) -> dict:
    """
    Save one stacked NN input image to a .npy file.

    Expected input format:
        shape = (C, H, W)
        dtype = float32 or convertible to float32

    This function only:
        1. validates the stacked array
        2. converts it to float32
        3. saves it as .npy
        4. returns a small receipt

    It does not:
        - build channels
        - parse annotations
        - write JSON metadata
        - convert boxes or labels
    """
    npy_path = Path(npy_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(nn_input_image)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected stacked NN input with shape (C, H, W), got {arr.shape}"
        )

    if arr.shape[0] <= 0:
        raise ValueError(f"Expected at least one channel, got shape {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError(f"{npy_path}: NN input contains NaN or Inf")

    arr = arr.astype(np.float32, copy=False)

    np.save(npy_path, arr)

    return {
        "npy_path": npy_path,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
    }


def prepare_nn_input_cache_from_split(
    split_path: str | Path,
    image_dir: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
    *,
    channel_fns: list[Callable[[np.ndarray], np.ndarray]],
    channel_names: list[str] | None = None,
    ann_ext: str = NN_ANN_EXT,
    max_images: int | None = None,
    cache_version: str = "nn_input_cache_v1",
    preparation: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Build the final stacked NN input cache from a dataset split.

    This function owns the full cache-preparation context:

        split file
        image path
        annotation path
        channel functions
        output .npy path
        output .json metadata path

    It writes, for each image:

        DataSetFinal/nn_input_npy/<image_stem>.npy
        DataSetFinal/nn_input_npy/<image_stem>.json

    The .npy file contains only the stacked NN input image:

        shape = (C, H, W)
        dtype = float32
        layout = CHW

    The .json file describes the cached image input only.
    It does not store boxes.
    It does not store labels.
    It does not describe TorchVision targets.

    Annotation txt files remain the source of truth for boxes/classes.
    Box and label conversion belongs later in the target builder / adapter layer.
    """
    if not channel_fns:
        raise ValueError("channel_fns must not be empty")

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(len(channel_fns))]

    if len(channel_names) != len(channel_fns):
        raise ValueError(
            f"channel_names length must match channel_fns length, "
            f"got {len(channel_names)} names for {len(channel_fns)} functions"
        )

    output_dir = ensure_dir(output_dir)

    pairs = pair_split_images_with_annotations(
        split_path=split_path,
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        ann_ext=ann_ext,
        require_annotation=True,
        max_images=max_images,
    )

    channels_meta = [
        {
            "id": int(channel_id),
            "name": str(channel_name),
            "function": getattr(channel_fn, "__name__", repr(channel_fn)),
        }
        for channel_id, (channel_name, channel_fn)
        in enumerate(zip(channel_names, channel_fns))
    ]

    preparation_payload = {} if preparation is None else dict(preparation)
    preparation_payload.setdefault("version", cache_version)

    records: list[dict] = []

    for image_path, annotation_path in pairs:
        if annotation_path is None:
            raise RuntimeError(
                f"Internal error: annotation_path is None for {image_path}"
            )

        img_raw = load_image(image_path)

        nn_input_image = build_nn_input_img(
            img_raw=img_raw,
            channel_fns=channel_fns,
        )

        npy_path = output_dir / f"{image_path.stem}.npy"
        json_path = output_dir / f"{image_path.stem}.json"

        save_record = save_nn_input_to_npy(
            nn_input_image,
            npy_path=npy_path,
        )

        payload = {
            "image_name": image_path.name,

            "image_path": str(image_path),
            "annotation_path": str(annotation_path),
            "npy_path": str(npy_path),

            "nn_input_shape": list(save_record["shape"]),
            "nn_input_dtype": save_record["dtype"],
            "image_layout": "CHW",

            "channels": channels_meta,

            "preparation": preparation_payload,
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        records.append(
            {
                "image_path": image_path,
                "annotation_path": annotation_path,
                "npy_path": npy_path,
                "json_path": json_path,
                "shape": save_record["shape"],
                "dtype": save_record["dtype"],
            }
        )

    return records

def create_nn_sample(
    nn_input_image: np.ndarray,
    *,
    boxes: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    image_path: str | Path | None = None,
    annotation_path: str | Path | None = None,
    image_id: int | None = None,
    roi_framework: str | None = None,
    roi_format: str = ROI_FORMAT_XYWH,
    nn_architecture: str | None = None,
    meta: dict[str, Any] | None = None,
) -> NNSample:
    return NNSample(
        nn_input_image=nn_input_image,
        boxes=boxes,
        labels=labels,
        image_path=None if image_path is None else Path(image_path),
        annotation_path=None if annotation_path is None else Path(annotation_path),
        image_id=image_id,
        roi_framework=roi_framework,
        roi_format=roi_format,
        nn_architecture=nn_architecture,
        meta={} if meta is None else dict(meta),
    )

