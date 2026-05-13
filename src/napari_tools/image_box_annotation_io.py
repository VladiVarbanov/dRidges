"""
Image + box-annotation IO helpers for napari tools.

This module connects project dataset files to the napari viewer.

It reuses existing project functions for:
    - split-based image/annotation pairing,
    - image loading,
    - source annotation txt parsing.

Source annotation format:
    class row_min col_min row_max col_max

Important:
    This module works with source annotation labels:
        0, 1, 2, 3

    It does not convert labels to TorchVision BG0 format:
        1, 2, 3, 4
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np

from configs.config import (
    NN_IMAGE_DIR,
    NN_DATASET_ROOT,
    NN_ANNOTATION_DIR,
    NN_TRAIN_SPLIT_TXT,
    NN_CLASS_NAMES,
)

from src.utilities import load_image
from src.nn_input_prepare import (
    pair_split_images_with_annotations,
    parse_annotation_txt_rc,
)


def load_source_annotation_format(
    annotation_format_path: str | Path = NN_DATASET_ROOT / "annotation_format.json",
) -> dict[str, Any]:
    """
    Load the dataset-level source annotation format JSON.

    This describes the txt files, not TorchVision targets.
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
            f"Annotation format JSON must contain a dict, "
            f"got {type(annotation_format).__name__}"
        )

    return annotation_format


def validate_source_annotation_format_for_napari(
    annotation_format: dict[str, Any],
) -> None:
    """
    Validate that the source annotation format matches what napari tools support.

    Supported format:
        class row_min col_min row_max col_max
    """
    required_keys = {
        "delimiter",
        "has_header",
        "class_column",
        "box_columns",
        "source_box_format",
        "source_label_base",
    }

    missing_keys = sorted(required_keys - set(annotation_format.keys()))
    if missing_keys:
        raise ValueError(
            f"Annotation format JSON is missing required keys: {missing_keys}"
        )

    if annotation_format["delimiter"] != "whitespace":
        raise ValueError(
            f"Only whitespace-delimited annotations are supported, "
            f"got {annotation_format['delimiter']!r}"
        )

    if bool(annotation_format["has_header"]) is not False:
        raise ValueError(
            f"Only headerless annotation txt files are supported, "
            f"got has_header={annotation_format['has_header']!r}"
        )

    if int(annotation_format["class_column"]) != 0:
        raise ValueError(
            f"Expected class_column=0, "
            f"got {annotation_format['class_column']!r}"
        )

    if list(annotation_format["box_columns"]) != [1, 2, 3, 4]:
        raise ValueError(
            f"Expected box_columns=[1, 2, 3, 4], "
            f"got {annotation_format['box_columns']!r}"
        )

    if annotation_format["source_box_format"] != "ROW_COL_MINMAX":
        raise ValueError(
            f"Expected source_box_format='ROW_COL_MINMAX', "
            f"got {annotation_format['source_box_format']!r}"
        )

    if int(annotation_format["source_label_base"]) != 0:
        raise ValueError(
            f"Expected source_label_base=0, "
            f"got {annotation_format['source_label_base']!r}"
        )

def select_image_annotation_pair(
    *,
    split_path: str | Path = NN_TRAIN_SPLIT_TXT,
    image_dir: str | Path = NN_IMAGE_DIR,
    annotation_dir: str | Path = NN_ANNOTATION_DIR,
    image_index: int = 0,
) -> tuple[Path, Path]:
    """
    Select one image and its matching annotation file from a split file.

    The split file is treated as the source of truth.

    Parameters
    ----------
    split_path
        Path to trainimages.txt / testimages.txt / allimages.txt.

    image_dir
        Directory containing source images.

    annotation_dir
        Directory containing source annotation txt files.

    image_index
        Which image/annotation pair to select from the split.

    Returns
    -------
    image_path, annotation_path
        Matching paths for one sample.

    Notes
    -----
    This reuses pair_split_images_with_annotations(...), which already exists
    in the project and already understands your dataset layout.
    """
    pairs = pair_split_images_with_annotations(
        split_path=split_path,
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        require_annotation=True,
        max_images=None,
    )

    if not pairs:
        raise ValueError(f"No image/annotation pairs found from split: {split_path}")

    if image_index < 0 or image_index >= len(pairs):
        raise IndexError(
            f"image_index={image_index} is out of range for {len(pairs)} pairs"
        )

    image_path, annotation_path = pairs[image_index]

    if annotation_path is None:
        raise RuntimeError(f"Annotation path is None for image: {image_path}")

    return image_path, annotation_path


def load_image_and_box_annotations(
    *,
    image_path: str | Path,
    annotation_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one image and its matching source box annotations.

    Returns
    -------
    image
        Raw image array loaded through src.utilities.load_image(...).

    labels_src
        Source class labels from the annotation txt file.
        These are foreground labels starting at 0.

    boxes_rc
        Source boxes in row/column minmax format:
            [row_min, col_min, row_max, col_max]

    Important
    ---------
    No TorchVision conversion happens here.
    No BG0 label shift happens here.
    """
    image_path = Path(image_path)
    annotation_path = Path(annotation_path)

    image = load_image(image_path)

    labels_src, boxes_rc = parse_annotation_txt_rc(annotation_path)

    return image, labels_src, boxes_rc


def validate_annotation_labels(
    labels_src: np.ndarray,
    *,
    annotation_path: str | Path,
    class_names: dict[int, str] = NN_CLASS_NAMES,
) -> None:
    """
    Validate that annotation labels are known source classes.

    Expected source labels:
        0, 1, 2, 3

    This catches corrupted or unexpected class IDs before opening napari.
    """
    labels_src = np.asarray(labels_src, dtype=np.int64)

    if labels_src.ndim != 1:
        raise ValueError(
            f"labels_src must have shape (N,), got {labels_src.shape}"
        )
    #TODO: DUPLICATED IN veiw_annotated_images. DO big refactoring
    known_labels = set(int(label_id) for label_id in class_names.keys())
    observed_labels = set(int(label_id) for label_id in labels_src.tolist())

    unknown_labels = sorted(observed_labels - known_labels)

    if unknown_labels:
        raise ValueError(
            f"{annotation_path}: unknown class labels {unknown_labels}. "
            f"Known labels are {sorted(known_labels)}"
        )


def load_validated_image_box_sample(
    *,
    split_path: str | Path = NN_TRAIN_SPLIT_TXT,
    image_dir: str | Path = NN_IMAGE_DIR,
    annotation_dir: str | Path = NN_ANNOTATION_DIR,
    image_index: int = 0,
) -> tuple[Path, Path, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one complete image + annotation sample for napari viewing.

    This is a convenience wrapper used by the napari launcher/viewer.

    Returns
    -------
    image_path
        Path to source image.

    annotation_path
        Path to matching source annotation txt.

    image
        Raw image array.

    labels_src
        Source class labels, still 0-based.

    boxes_rc
        Source boxes:
            [row_min, col_min, row_max, col_max]
    """
    annotation_format = load_source_annotation_format()

    validate_source_annotation_format_for_napari(annotation_format)

    image_path, annotation_path = select_image_annotation_pair(
        split_path=split_path,
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        image_index=image_index,
    )

    image, labels_src, boxes_rc = load_image_and_box_annotations(
        image_path=image_path,
        annotation_path=annotation_path,
    )

    validate_annotation_labels(
        labels_src,
        annotation_path=annotation_path,
    )

    return image_path, annotation_path, image, labels_src, boxes_rc