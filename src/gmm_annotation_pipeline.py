"""
src/gmm_annotation_pipeline.py

GMM-based scale separation of merged oval loops and black dots.

Purpose
-------
Original annotations had many wrong class labels — oval loops and black dots
were confused by human annotators. To correct this unsupervised:

    Step 1 (done): Merge the two loop classes (0+1 → 0) into merged_oval_loop
                   and keep black_dot as class 1. Stored in bounding_boxes_redacted/.

    Step 2 (this module): Pool all boxes from redacted classes 0 (merged_oval_loop)
                          and 1 (black_dot). Class 2 (other_defect) is excluded —
                          it is a placeholder category.
                          Fit a 2-component GMM on log(box_area) globally across all
                          images. GMM components are ordered by mean log_area:
                              GMM label 0 → small → black_dot  → output class 1
                              GMM label 1 → large → oval_loop  → output class 0

    Step 3: Visualise inner (original human label) + outer (GMM label) bounding
            boxes per image for human inspection.

    Step 4 (after inspection): If GMM results are promising, GMM-corrected labels
                               become the training annotations (bounding_boxes_gmm/).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs.config import NN_ANN_EXT
from configs.object_features_config import (
    GMM_SCALE_FEATURE_COLUMNS,
    GMM_SCALE_LABEL_ORDER_FEATURE,
    GMM_N_LABELS,
    GMM_COVARIANCE_TYPE,
    GMM_RANDOM_STATE,
    GMM_N_INIT,
    GMM_SCALE_EVIDENCE_NAME,
)

from .annotation_io import read_split_image_names, parse_annotation_txt_rc
from .annotation_registry import (
    make_object_uid,
    write_derived_annotation_format_json,
)
from .roi_instance_features import compute_box_geometry_rc
from .gmm_wrapper import build_gmm_evidence_dataframe
from .utilities import ensure_dir


# ---------------------------------------------------------------------------
# Class mapping constants
# ---------------------------------------------------------------------------

# Redacted input classes that participate in GMM separation.
# Class 2 (other_defect) is excluded — it is a placeholder.
GMM_INPUT_CLASSES: tuple[int, ...] = (0, 1)

# GMM components ordered by mean log_area (ascending):
#   GMM label 0 = smaller boxes → black_dot  → output class 1
#   GMM label 1 = larger  boxes → oval_loop  → output class 0
GMM_LABEL_TO_OUTPUT_CLASS: dict[int, int] = {0: 1, 1: 0}

GMM_OUTPUT_CLASS_NAMES: dict[int, str] = {
    0: "oval_loop",
    1: "black_dot",
}

GMM_ANNOTATION_FORMAT_NAME = "gmm_scale_separation_v1"


# ---------------------------------------------------------------------------
# Step 1 — Build global annotation table
# ---------------------------------------------------------------------------

def build_gmm_annotation_table(
    split_path: str | Path,
    redacted_annotation_dir: str | Path,
    *,
    classes_for_gmm: tuple[int, ...] = GMM_INPUT_CLASSES,
    ann_ext: str = NN_ANN_EXT,
    max_images: int | None = None,
) -> pd.DataFrame:
    """
    Build a global per-object DataFrame from redacted annotations.

    Loads all redacted annotation txt files listed in split_path.
    Only includes boxes whose class is in classes_for_gmm.
    Computes geometry features via compute_box_geometry_rc.

    Returns
    -------
    table : pd.DataFrame
        One row per valid box. Columns:
            image_id, image_name, annotation_path,
            row_idx, object_uid, original_class,
            row_min, col_min, row_max, col_max,
            width, height, area, sqrt_area, log_area,
            aspect_ratio, center_row, center_col
    """
    split_path = Path(split_path)
    redacted_annotation_dir = Path(redacted_annotation_dir)

    image_names = read_split_image_names(split_path)

    if max_images is not None:
        image_names = image_names[:max_images]

    records: list[dict[str, Any]] = []

    for image_id, image_name in enumerate(image_names):
        image_stem = Path(image_name).stem
        annotation_path = redacted_annotation_dir / f"{image_stem}{ann_ext}"

        if not annotation_path.exists():
            continue

        labels, boxes_rc = parse_annotation_txt_rc(annotation_path)

        if len(labels) == 0:
            continue

        for row_idx in range(len(labels)):
            class_id = int(labels[row_idx])

            if class_id not in classes_for_gmm:
                continue

            box_arr = boxes_rc[row_idx : row_idx + 1]  # shape (1, 4)
            features, keep_mask = compute_box_geometry_rc(box_arr)

            if not keep_mask[0]:
                continue  # degenerate box — skip

            records.append({
                "image_id": image_id,
                "image_name": image_name,
                "annotation_path": str(annotation_path),
                "row_idx": row_idx,
                "object_uid": make_object_uid(image_id, row_idx),
                "original_class": class_id,
                "row_min": float(features["row_min"][0]),
                "col_min": float(features["col_min"][0]),
                "row_max": float(features["row_max"][0]),
                "col_max": float(features["col_max"][0]),
                "width": float(features["width"][0]),
                "height": float(features["height"][0]),
                "area": float(features["area"][0]),
                "sqrt_area": float(features["sqrt_area"][0]),
                "log_area": float(features["log_area"][0]),
                "aspect_ratio": float(features["aspect_ratio"][0]),
                "center_row": float(features["center_row"][0]),
                "center_col": float(features["center_col"][0]),
            })

    if not records:
        raise ValueError(
            f"No valid boxes found for classes {classes_for_gmm} "
            f"in {redacted_annotation_dir}"
        )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Step 2 — GMM separation
# ---------------------------------------------------------------------------

def run_gmm_scale_separation(table: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a 2-component GMM on log_area and add GMM evidence columns.

    Reads GMM hyperparameters from object_features_config.

    Appended columns (prefixed with gmm_scale_):
        gmm_scale_labels              — ordered GMM label (0=small, 1=large)
        gmm_scale_probabilities       — probability string per component
        gmm_scale_highest_probability — max probability across components
        gmm_scale_feature_columns     — feature column names used
        gmm_scale_label_order_feature — ordering feature name

    Also appends:
        output_class — final annotation class after GMM_LABEL_TO_OUTPUT_CLASS
                       mapping (0=oval_loop, 1=black_dot)
    """
    table_with_gmm = build_gmm_evidence_dataframe(
        table,
        feature_columns=GMM_SCALE_FEATURE_COLUMNS,
        gmm_label_order_feature=GMM_SCALE_LABEL_ORDER_FEATURE,
        n_gmm_labels=GMM_N_LABELS,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=GMM_RANDOM_STATE,
        n_init=GMM_N_INIT,
        evidence_name=GMM_SCALE_EVIDENCE_NAME,
    )

    gmm_labels_col = f"{GMM_SCALE_EVIDENCE_NAME}_labels"

    table_with_gmm["output_class"] = (
        table_with_gmm[gmm_labels_col]
        .map(GMM_LABEL_TO_OUTPUT_CLASS)
        .astype(int)
    )

    return table_with_gmm


# ---------------------------------------------------------------------------
# Step 3 — Write output annotation txt files
# ---------------------------------------------------------------------------

def write_gmm_annotation_txts(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    float_precision: int = 4,
) -> dict[str, Path]:
    """
    Write per-image annotation txt files using GMM output_class labels.

    Format: class row_min col_min row_max col_max  (ROW_COL_MINMAX)

    Box coordinates are unchanged from the redacted annotations.
    Only the class label is updated based on GMM prediction.

    Only images that have at least one GMM box are written.

    Returns
    -------
    written : dict mapping image_name → written annotation Path
    """
    output_dir = ensure_dir(output_dir)
    written: dict[str, Path] = {}

    fmt = f"{{:.{float_precision}f}}"

    for image_name, group in table.groupby("image_name", sort=False):
        image_stem = Path(str(image_name)).stem
        annotation_path = output_dir / f"{image_stem}.txt"

        group_sorted = group.sort_values("row_idx")

        with annotation_path.open("w", encoding="utf-8") as f:
            for _, row in group_sorted.iterrows():
                line = (
                    f"{int(row['output_class'])} "
                    f"{fmt.format(float(row['row_min']))} "
                    f"{fmt.format(float(row['col_min']))} "
                    f"{fmt.format(float(row['row_max']))} "
                    f"{fmt.format(float(row['col_max']))}\n"
                )
                f.write(line)

        written[str(image_name)] = annotation_path

    return written


def write_gmm_annotation_format_json(
    output_path: str | Path,
    *,
    source_annotation_format: str | Path,
) -> None:
    """
    Write the annotation format JSON sidecar for GMM-corrected annotations.

    Describes output class mapping and GMM parameters used.
    """
    write_derived_annotation_format_json(
        output_path=output_path,
        annotation_format_name=GMM_ANNOTATION_FORMAT_NAME,
        annotation_format_role="gmm_scale_corrected_annotations",
        class_names=GMM_OUTPUT_CLASS_NAMES,
        source_annotation_format=source_annotation_format,
        class_ids_initial=(0, 1),  # redacted: merged_oval_loop, black_dot
        class_ids_merged=(0, 1),   # same IDs, but data-driven re-assignment
        bbox_edit_notes=[
            {
                "note": (
                    "Class re-assignment by 2-component GMM on log_area. "
                    "Box coordinates are unchanged."
                ),
                "feature": "log_area",
                "n_components": GMM_N_LABELS,
                "ordering": "ascending mean log_area",
                "gmm_label_0_small": "black_dot  → output class 1",
                "gmm_label_1_large": "oval_loop  → output class 0",
            }
        ],
    )
