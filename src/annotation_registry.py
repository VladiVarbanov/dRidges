from __future__ import annotations

from collections.abc import Sequence

from configs.config import (NN_ANN_EXT,
                            NN_MIN_BOX_WIDTH,
                            NN_MIN_BOX_HEIGHT,
                            NN_MIN_BOX_AREA,
                            ROI_FORMAT_XYWH,
                            NN_LOCAL_NORM_SIGMA,
                            NN_GAUSSIAN_SMOOTH_SIGMA,
                            NN_HESSIAN_SCALE_PX,
                            )
from pathlib import Path
from .utilities import collect_images_paths, load_image, ensure_dir
from typing import Callable, Any
from .preprocessing import to_gray_normalized, local_normalize_HOG_style
from .ridges import RidgeMap
from typing import Any
from dataclasses import dataclass, field
import numpy as np
import json
from skimage.filters import gaussian


def make_object_uid(image_id: int, row_idx: int) -> str:
    """
    Create a stable object UID from image_id and annotation row index.

    The UID must depend only on stable indexing, not on label or box geometry,
    because labels and boxes may change during processing.
    """
    if image_id < 0:
        raise ValueError(f"image_id must be non-negative, got {image_id}")

    if row_idx < 0:
        raise ValueError(f"row_idx must be non-negative, got {row_idx}")

    return f"img{int(image_id):05d}_box{int(row_idx):05d}"





def write_derived_annotation_format_json(
    *,
    output_path: str | Path,
    annotation_format_name: str,
    annotation_format_role: str = "derived_source_annotations",
    class_names: dict[int, str],
    source_annotation_format: str | Path,
    class_ids_initial: tuple[int, ...],
    class_ids_merged: tuple[int, ...],
    bbox_edit_notes: list[dict[str, Any]] | None = None,
) -> None:
    """
    Write JSON describing a derived source-annotation set.

    This JSON describes the copied/edited annotation txt files,
    not TorchVision targets.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_ids_remap = {
        str(int(initial)): int(merged)
        for initial, merged in zip(class_ids_initial, class_ids_merged)
    }

    payload = {
        "annotation_format_name": annotation_format_name,
        "annotation_format_role": annotation_format_role,

        "annotation_format": "class row_min col_min row_max col_max",
        "delimiter": "whitespace",
        "has_header": False,

        "class_column": 0,
        "box_columns": [1, 2, 3, 4],

        "source_box_format": "ROW_COL_MINMAX", #TODO: make it config param later
        "source_label_base": 0,

        "class_names": {
            str(int(class_id)): str(class_name)
            for class_id, class_name in class_names.items()
        },

        "source_annotation_format": str(source_annotation_format),

        "derived_annotation_notes": {
            "class_remap": class_ids_remap,
            "bbox_edits": bbox_edit_notes or [],
            "overwrites_originals": False,
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def load_annotation_format_json(annotation_format_path: str | Path) -> dict[str, Any]:
    """
    Load the dataset-level annotation format JSON.

    This file describes how the source annotation txt files should be read.
    It does not describe TorchVision targets.
    """
    annotation_format_path = Path(annotation_format_path)

    if not annotation_format_path.exists():
        raise FileNotFoundError(
            f"Annotation format JSON does not exist: {annotation_format_path}"
        )

    if not annotation_format_path.is_file():
        raise ValueError(
            f"Annotation format path is not a file: {annotation_format_path}"
        )

    with annotation_format_path.open("r", encoding="utf-8") as f:
        annotation_format = json.load(f)

    if not isinstance(annotation_format, dict):
        raise ValueError(
            f"Annotation format JSON must contain an object/dict, "
            f"got {type(annotation_format).__name__}"
        )

    return annotation_format


def validate_source_annotation_format(annotation_format: dict[str, Any]) -> None:
        """
        Validate that the source annotation format is supported by the current parser.

        Current supported source annotation format:

            class row_min col_min row_max col_max

        Internally, parse_annotation_txt_rc expects:
            labels    from column 0
            boxes_rc  from columns [1, 2, 3, 4]
            boxes_rc order = [row_min, col_min, row_max, col_max]
        """
        required = {
            "delimiter",
            "has_header",
            "class_column",
            "box_columns",
            "source_box_format",
            "source_label_base",
        }

        missing = sorted(required - set(annotation_format.keys()))
        if missing:
            raise ValueError(
                f"Annotation format is missing required fields: {missing}"
            )

        if annotation_format["delimiter"] != "whitespace":
            raise ValueError(
                f"Only whitespace-delimited annotations are supported for now, "
                f"got {annotation_format['delimiter']!r}"
            )

        if bool(annotation_format["has_header"]) is not False:
            raise ValueError(
                f"Only headerless annotation txt files are supported for now, "
                f"got has_header={annotation_format['has_header']!r}"
            )

        if int(annotation_format["class_column"]) != 0:
            raise ValueError(
                f"Current parser expects class_column=0, "
                f"got {annotation_format['class_column']!r}"
            )
#TODO fix annotation registry
        if list(annotation_format["box_columns"]) != [1, 2, 3, 4]:
            raise ValueError(
                f"Current parser expects box_columns=[1, 2, 3, 4], "
                f"got {annotation_format['box_columns']!r}"
            )

        if annotation_format["source_box_format"] != "ROW_COL_MINMAX":
            raise ValueError(
                f"Current parser expects source_box_format='ROW_COL_MINMAX', "
                f"got {annotation_format['source_box_format']!r}"
            )

        if int(annotation_format["source_label_base"]) != 0:
            raise ValueError(
                f"Current label adapter expects source_label_base=0, "
                f"got {annotation_format['source_label_base']!r}"
            )


def probability_vector_to_str(probability_vector) -> str:
    """
    Convert a 1D probability/confidence vector to a whitespace-delimited string.

    Example:
        [0, 0, 1, 0] -> "0 0 1 0"
        [0.8, 0.2]   -> "0.8 0.2"

    This is used for pandas/CSV-friendly object-table storage.
    """
    arr = np.asarray(probability_vector, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(
            f"probability_vector_to_str expects a 1D vector, got shape {arr.shape}"
        )

    if arr.size == 0:
        raise ValueError("probability_vector_to_str expects a non-empty vector")

    if not np.isfinite(arr).all():
        raise ValueError("probability vector contains NaN or Inf")

    if np.any(arr < 0.0):
        raise ValueError("probability vector contains negative values")

    values: list[str] = []

    for value in arr:
        value = float(value)

        if value == 0.0:
            values.append("0")
        elif value.is_integer():
            values.append(str(int(value)))
        else:
            values.append(f"{value:.6g}")

    return " ".join(values)


def make_label_probability(label: int, num_classes: int) -> str:
    """
    Convert a zero-based hard class label into a one-hot probability string.

    Example:
        label=2, num_classes=4 -> "0 0 1 0"

    Notes
    -----
    - No background class is added here.
    - Background conversion belongs to NN adapters, not the annotation registry.
    - Output uses the object-table probability-vector string convention.
    """
    label_int = int(label)
    num_classes_int = int(num_classes)

    if num_classes_int <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if label_int < 0 or label_int >= num_classes_int:
        raise ValueError(
            f"label must be in [0, {num_classes_int - 1}], got {label_int}"
        )

    probability_vector = np.zeros(num_classes_int, dtype=np.float64)
    probability_vector[label_int] = 1.0

    return probability_vector_to_str(probability_vector)