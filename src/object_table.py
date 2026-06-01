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
from typing import Callable
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
        "annotation_format_role": "derived_source_annotations",

        "annotation_format": "class row_min col_min row_max col_max",
        "delimiter": "whitespace",
        "has_header": False,

        "class_column": 0,
        "box_columns": [1, 2, 3, 4],

        "source_box_format": "ROW_COL_MINMAX",
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

        records.append(
            {
                "image_id": int(image_id),
                "image_path": image_path,
                "src_annotation_path": src_annotation_path,
                "dst_annotation_path": dst_annotation_path,
                "num_boxes": int(len(labels_src)),
                "num_black_dot_boxes_expanded": int(len(black_dot_idxs)),
                "image_shape_hw": (int(image_height), int(image_width)),
            }
        )

        write_derived_annotation_format_json(
            output_path=dst_annotation_format_path,
            annotation_format_name=annotation_format_name,
            class_names=derived_class_names,
            source_annotation_format=source_annotation_format_path,
            class_ids_initial=class_ids_initial,
            class_ids_merged=class_ids_merged,
            bbox_edit_notes=[
                {
                    "target_class_after_merge": int(black_dot_class_id_after_merge),
                    "class_name": derived_class_names[int(black_dot_class_id_after_merge)],
                    "width_scale": float(black_dot_width_scale),
                    "height_scale": float(black_dot_height_scale),
                    "center_preserved": True,
                    "clamped_to_image_boundaries": True,
                }
            ],
        )

        return records
