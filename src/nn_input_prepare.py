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


@dataclass(frozen=True)
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

    bbox_scale_rules: tuple[BBoxScaleRule, ...] = ()

    annotation_format_name: str = "derived_annotations_v1"
    ann_ext: str = NN_ANN_EXT
    max_images: int | None = None


def resolve_annotation_path(image_path: Path, ann_ext: str = NN_ANN_EXT) -> Path:
    """
    Return the matching annotation path for one image path.

    Example:
        sample.jpg  -> sample.txt
        sample.tif  -> sample.txt

    Notes:
        - This function only maps the path.
        - It does not check whether the annotation file exists.
    """
    if not ann_ext.startswith("."):
        ann_ext = f".{ann_ext}"
    return image_path.with_suffix(ann_ext)

from pathlib import Path


def read_split_image_names(split_path: str | Path) -> list[str]:
    """
    Read image names from a dataset split file.

    The split file is expected to contain one image name per line, for example:

        image_001.tif
        image_002.tif
        image_003.tif

    Empty lines are ignored.

    Notes
    -----
    This function only reads the names listed in the split file.
    It does not scan the image folder.
    It does not check whether the image files exist.
    It does not resolve annotation paths.

    Path resolution is handled later by split-aware resolver functions, e.g.:

        image_path = image_dir / image_name
        annotation_path = annotation_dir / f"{Path(image_name).stem}.txt"

    Parameters
    ----------
    split_path
        Path to a split file such as:
            trainimages.txt
            testimages.txt
            allimages.txt

    Returns
    -------
    image_names
        List of image names exactly as written in the split file,
        stripped of leading/trailing whitespace.
    """
    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")

    if not split_path.is_file():
        raise ValueError(f"Split path is not a file: {split_path}")

    image_names: list[str] = []

    with split_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            image_name = raw_line.strip()

            if not image_name:
                continue

            image_names.append(image_name)

    return image_names



def pair_split_images_with_annotations(
    split_path: str | Path,
    image_dir: str | Path,
    annotation_dir: str | Path,
    *,
    ann_ext: str = NN_ANN_EXT,
    require_annotation: bool = True,
    max_images: int | None = None,
) -> list[tuple[Path, Path | None]]:
    """
    Pair images listed in a dataset split file with their annotation files.

    This function is split-driven, not folder-scan-driven.

    The split file is treated as the source of truth. Only image names listed in
    split_path are used. The function does not scan image_dir and does not try to
    discover extra images.

    Expected dataset layout:

        DataSetFinal/
            images/
                image_001.tif
                image_002.tif
            bounding_boxes/
                image_001.txt
                image_002.txt
            trainimages.txt
            testimages.txt
            allimages.txt

    For each image name in the split file:

        image_001.tif

    the function resolves:

        image_path = image_dir / "image_001.tif"
        annotation_path = annotation_dir / "image_001.txt"

    The annotation name is derived from the image stem and ann_ext:

        Path(image_name).with_suffix(ann_ext)

    Parameters
    ----------
    split_path
        Path to a split file, for example:
            trainimages.txt
            testimages.txt
            allimages.txt

    image_dir
        Directory containing the real image files.

    annotation_dir
        Directory containing the annotation txt files.

    ann_ext
        Annotation file extension. Default is NN_ANN_EXT, usually ".txt".

    require_annotation
        If True, missing annotation files raise FileNotFoundError.
        If False, images without annotation files are still returned with
        annotation_path set to None.

    max_images
        Optional limit for smoke tests. If provided, only the first max_images
        names from the split file are resolved.

    Returns
    -------
    pairs
        List of tuples:

            (image_path, annotation_path)

        where annotation_path is None only when require_annotation=False and the
        corresponding annotation file is missing.

    Raises
    ------
    FileNotFoundError
        If an image listed in the split file does not exist.
        If require_annotation=True and the matching annotation file does not exist.

    ValueError
        If max_images is negative.
    """
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)

    if not ann_ext.startswith("."):
        ann_ext = f".{ann_ext}"

    image_names = read_split_image_names(split_path)

    if max_images is not None:
        if max_images < 0:
            raise ValueError(f"max_images must be >= 0 or None, got {max_images}")
        image_names = image_names[:max_images]

    pairs: list[tuple[Path, Path | None]] = []

    for image_name in image_names:
        image_rel = Path(image_name)

        image_path = image_dir / image_rel
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image listed in split file does not exist: {image_path}"
            )

        annotation_rel = image_rel.with_suffix(ann_ext)
        annotation_path = annotation_dir / annotation_rel

        if annotation_path.exists():
            pairs.append((image_path, annotation_path))
            continue

        if require_annotation:
            raise FileNotFoundError(
                f"Annotation file does not exist for image {image_path}: "
                f"{annotation_path}"
            )

        pairs.append((image_path, None))

    return pairs


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


def parse_annotation_txt_rc(
    annotation_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one annotation txt file in source row/col corner format:

        class  y_min  x_min  y_max  x_max

    Returns
    -------
    labels
        np.ndarray of shape (N,), dtype int64
    boxes_rc
        np.ndarray of shape (N, 4), dtype float32
        order = [y_min, x_min, y_max, x_max]
    """
    labels: list[int] = []
    boxes_rc: list[list[float]] = []

    with annotation_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{annotation_path}: line {line_num} must have 5 fields, got {len(parts)}: {line!r}"
                )

            try:
                class_id = int(float(parts[0]))
                y_min = float(parts[1])
                x_min = float(parts[2])
                y_max = float(parts[3])
                x_max = float(parts[4])
            except ValueError as exc:
                raise ValueError(
                    f"{annotation_path}: line {line_num} contains non-numeric values: {line!r}"
                ) from exc

            labels.append(class_id)
            boxes_rc.append([y_min, x_min, y_max, x_max])

    labels_arr = np.asarray(labels, dtype=np.int64)

    if boxes_rc:
        boxes_arr = np.asarray(boxes_rc, dtype=np.float32)
    else:
        boxes_arr = np.empty((0, 4), dtype=np.float32)

    return labels_arr, boxes_arr


"""
Utilities for deriving copied/redacted box annotations from source annotations.

Original annotation files are never overwritten.

Source annotation format:
    class row_min col_min row_max col_max
"""

def select_annotation_rows(
    source,
    selection: None | int | Sequence[int | Sequence[int]] = None,
) -> list[int]:
    """
    Select annotation row indexes from a source annotation container.

    Parameters
    ----------
    source
        Annotation container used only for its length.

        Examples:
            labels_src
            boxes_rc
            annotation_rows

    selection
        Which annotation rows to select.

        Supported forms:

            None
                Select all rows.

            int
                Select one row.

            [start, end]
                Select an inclusive range:
                    start, start + 1, ..., end

            [idx0, idx1, idx2, ...]
                If length > 2, treat as explicit row indexes.

            [[start, end], idx, [start2, end2]]
                Mixed inclusive ranges and single row indexes.

    Returns
    -------
    selected_indexes
        Sorted unique list of selected annotation row indexes.

    Notes
    -----
    Ranges are inclusive.
    Therefore [10, 20] means:
        10, 11, 12, ..., 20
    """
    num_annotations = len(source)

    if selection is None:
        return list(range(num_annotations))

    selected_indexes: list[int] = []

    def append_single_index(index: int) -> None:
        index = int(index)

        if index < 0 or index >= num_annotations:
            raise IndexError(
                f"Annotation row index {index} is out of range for "
                f"{num_annotations} annotations"
            )

        selected_indexes.append(index)

    def append_inclusive_range(start_index: int, end_index: int) -> None:
        start_index = int(start_index)
        end_index = int(end_index)

        if end_index < start_index:
            raise ValueError(
                f"Inclusive range end must be >= start, "
                f"got [{start_index}, {end_index}]"
            )

        while start_index <= end_index:
            append_single_index(start_index)
            start_index += 1

    if isinstance(selection, int):
        append_single_index(selection)
        return sorted(set(selected_indexes))

    if len(selection) == 2 and all(isinstance(item, int) for item in selection):
        start_index, end_index = selection
        append_inclusive_range(start_index, end_index)
        return sorted(set(selected_indexes))

    for item in selection:
        if isinstance(item, int):
            append_single_index(item)
            continue

        if len(item) == 2:
            start_index, end_index = item
            append_inclusive_range(start_index, end_index)
            continue

        for nested_item in item:
            append_single_index(nested_item)

    return sorted(set(selected_indexes))


def merge_classes(
    labels_src: np.ndarray,
    *,
    class_ids_initial: tuple[int, ...],
    class_ids_merged: tuple[int, ...],
    idxs_to_edit: list[int] | None = None,
    annotation_path: str | Path | None = None,
) -> np.ndarray:
    """
    Remap / merge source class IDs for selected annotation rows.

    Parameters
    ----------
    labels_src
        Source labels from one annotation txt file.

        Shape:
            (N,)

        Example:
            [0, 1, 2, 3, 2]

    class_ids_initial
        Original class IDs before remapping.

        Example:
            (0, 1, 2, 3)

    class_ids_merged
        New class IDs after remapping.

        Example:
            (0, 0, 1, 2)

        Together with class_ids_initial, this means:
            0 -> 0
            1 -> 0
            2 -> 1
            3 -> 2

    idxs_to_edit
        Annotation row indexes to edit.

        These are row indexes inside labels_src, not class IDs.

        If None, all annotation rows are edited.

        Example:
            idxs_to_edit = [0, 2, 5]

        means:
            edit labels_src[0], labels_src[2], labels_src[5]

    annotation_path
        Optional annotation file path.

        Used only for clearer error messages.

    Returns
    -------
    labels_out
        Remapped labels array, dtype int64.

        Shape:
            (N,)

    Notes
    -----
    This function edits class labels only.

    It does not:
        - edit bounding boxes,
        - delete annotation rows,
        - save files,
        - convert labels to TorchVision BG0 format.

    The remapping relation is internally represented as:

        class_ids_remap = {
            old_class_id: new_class_id
        }

    For the current redacted annotation plan:

        class_ids_initial = (0, 1, 2, 3)
        class_ids_merged  = (0, 0, 1, 2)

    resulting derived classes are:

        0 = merged_oval_loop
        1 = black_dot
        2 = other_defect
    """
    labels_src = np.asarray(labels_src, dtype=np.int64)

    if labels_src.ndim != 1:
        raise ValueError(
            f"labels_src must have shape (N,), got {labels_src.shape}"
        )

    if len(class_ids_initial) != len(class_ids_merged):
        raise ValueError(
            "class_ids_initial and class_ids_merged must have the same length, "
            f"got {len(class_ids_initial)} and {len(class_ids_merged)}"
        )

    class_ids_remap = {
        int(initial): int(merged)
        for initial, merged in zip(class_ids_initial, class_ids_merged)
    }

    labels_out = labels_src.copy()

    if idxs_to_edit is None:
        idxs_to_edit = select_annotation_rows(labels_out)

    for row_idx in idxs_to_edit:
        row_idx = int(row_idx)

        if row_idx < 0 or row_idx >= len(labels_out):
            raise IndexError(
                f"Annotation row row_idx {row_idx} is out of range for "
                f"{len(labels_out)} annotations"
            )

        old_class_id = int(labels_out[row_idx])

        if old_class_id not in class_ids_remap:
            where = f" in {annotation_path}" if annotation_path is not None else ""
            raise ValueError(
                f"Class id {old_class_id} at annotation row {row_idx}{where} "
                f"is not present in class_ids_initial={class_ids_initial}"
            )

        labels_out[row_idx] = class_ids_remap[old_class_id]

    return labels_out.astype(np.int64, copy=False)


def edit_bbox_size(
    boxes_rc: np.ndarray,
    *,
    idxs_to_edit: list[int] | None = None,
    width_scale: float = 1.0,
    height_scale: float = 1.0,
    image_shape_hw: tuple[int, int] | None = None,
    annotation_path: str | Path | None = None,
) -> np.ndarray:
    """
    Scale selected boxes around their original center.
    Parameters
    ----------
    boxes_rc
        Box array with shape (N, 4):
            [row_min, col_min, row_max, col_max]
    idxs_to_edit
        Annotation row indexes to edit.
        If None, all boxes are edited.
    width_scale
        Multiplier for box width.
    height_scale
        Multiplier for box height.
    image_shape_hw
        Optional image shape:
            (height, width)
        If provided, edited boxes are clamped to image boundaries.
    annotation_path
        Optional path used only for clearer error messages.
    Returns
    -------
    boxes_out
        Edited box array with shape (N, 4), dtype float32.
    Notes
    -----
    The box center stays fixed.
    Extra width/height is split equally on both sides.
    """
    boxes_rc = np.asarray(boxes_rc, dtype=np.float32)

    if boxes_rc.ndim != 2 or boxes_rc.shape[1] != 4:
        raise ValueError(
            f"boxes_rc must have shape (N, 4), got {boxes_rc.shape}"
        )

    if width_scale <= 0:
        raise ValueError(f"width_scale must be > 0, got {width_scale}")

    if height_scale <= 0:
        raise ValueError(f"height_scale must be > 0, got {height_scale}")

    boxes_out = boxes_rc.copy()

    if idxs_to_edit is None:
        idxs_to_edit = select_annotation_rows(boxes_out)

    if image_shape_hw is not None:
        image_height, image_width = image_shape_hw

        if image_height <= 0 or image_width <= 0:
            raise ValueError(
                f"image_shape_hw must contain positive values, got {image_shape_hw}"
            )

    for row_idx in idxs_to_edit:
        row_idx = int(row_idx)

        if row_idx < 0 or row_idx >= len(boxes_out):
            where = f" in {annotation_path}" if annotation_path is not None else ""
            raise IndexError(
                f"Annotation row index {row_idx}{where} is out of range for "
                f"{len(boxes_out)} boxes"
            )

        row_min, col_min, row_max, col_max = boxes_out[row_idx]

        box_height = row_max - row_min
        box_width = col_max - col_min

        if box_height <= 0:
            where = f" in {annotation_path}" if annotation_path is not None else ""
            raise ValueError(
                f"Box at row {row_idx}{where} has non-positive height: "
                f"row_min={row_min}, row_max={row_max}"
            )

        if box_width <= 0:
            where = f" in {annotation_path}" if annotation_path is not None else ""
            raise ValueError(
                f"Box at row {row_idx}{where} has non-positive width: "
                f"col_min={col_min}, col_max={col_max}"
            )

        row_center = 0.5 * (row_min + row_max)
        col_center = 0.5 * (col_min + col_max)

        new_height = box_height * float(height_scale)
        new_width = box_width * float(width_scale)

        new_row_min = row_center - 0.5 * new_height
        new_row_max = row_center + 0.5 * new_height
        new_col_min = col_center - 0.5 * new_width
        new_col_max = col_center + 0.5 * new_width

        if image_shape_hw is not None:
            new_row_min = max(0.0, min(float(image_height), new_row_min))
            new_row_max = max(0.0, min(float(image_height), new_row_max))
            new_col_min = max(0.0, min(float(image_width), new_col_min))
            new_col_max = max(0.0, min(float(image_width), new_col_max))

        boxes_out[row_idx] = [
            new_row_min,
            new_col_min,
            new_row_max,
            new_col_max,
        ]

    return boxes_out.astype(np.float32, copy=False)

def write_annotation_txt_rc(
    *,
    labels_src: np.ndarray,
    boxes_rc: np.ndarray,
    output_path: str | Path,
    float_precision: int = 4,
) -> None:
    """
    Write annotation txt in source row/col format:

        class row_min col_min row_max col_max
    """
    labels_src = np.asarray(labels_src, dtype=np.int64)
    boxes_rc = np.asarray(boxes_rc, dtype=np.float32)

    if labels_src.ndim != 1:
        raise ValueError(
            f"labels_src must have shape (N,), got {labels_src.shape}"
        )

    if boxes_rc.ndim != 2 or boxes_rc.shape[1] != 4:
        raise ValueError(
            f"boxes_rc must have shape (N, 4), got {boxes_rc.shape}"
        )

    if len(labels_src) != len(boxes_rc):
        raise ValueError(
            f"labels_src and boxes_rc must have same length, "
            f"got {len(labels_src)} and {len(boxes_rc)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = f"{{:.{float_precision}f}}"

    with output_path.open("w", encoding="utf-8") as f:
        for label, box in zip(labels_src.tolist(), boxes_rc.tolist()):
            row_min, col_min, row_max, col_max = box

            line = (
                f"{int(label)} "
                f"{fmt.format(float(row_min))} "
                f"{fmt.format(float(col_min))} "
                f"{fmt.format(float(row_max))} "
                f"{fmt.format(float(col_max))}\n"
            )
            f.write(line)


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

def derive_box_annotations_from_src(
    *,
    split_path: str | Path,
    image_dir: str | Path,
    src_annotation_dir: str | Path,
    dst_annotation_dir: str | Path,
    dst_annotation_format_path: str | Path,
    source_annotation_format_path: str | Path,
    class_ids_initial: tuple[int, ...],
    class_ids_merged: tuple[int, ...],
    derived_class_names: dict[int, str],
    black_dot_class_id_after_merge: int = 1,
    black_dot_width_scale: float = 1.3,
    black_dot_height_scale: float = 1.3,
    annotation_format_name: str = "redacted_merged_ovals_expanded_black_dots_v1",
    ann_ext: str = NN_ANN_EXT,
    max_images: int | None = None,
) -> list[dict]:
    """
    Derive copied box annotations from source annotations.

    Current intended edit policy:
        1. merge source classes:
               0 -> 0
               1 -> 0
               2 -> 1
               3 -> 2

        2. expand black-dot boxes after merge:
               derived class 1
               width_scale  = 1.3
               height_scale = 1.3

    Original annotations are never overwritten.
    """
    dst_annotation_dir = ensure_dir(dst_annotation_dir)

    pairs = pair_split_images_with_annotations(
        split_path=split_path,
        image_dir=image_dir,
        annotation_dir=src_annotation_dir,
        ann_ext=ann_ext,
        require_annotation=True,
        max_images=max_images,
    )

    records: list[dict] = []

    for image_id, (image_path, src_annotation_path) in enumerate(pairs):
        if src_annotation_path is None:
            raise RuntimeError(
                f"Internal error: annotation_path is None for {image_path}"
            )

        image = load_image(image_path)
        image_height, image_width = image.shape[:2]

        labels_src, boxes_rc = parse_annotation_txt_rc(src_annotation_path)

        idxs_all = select_annotation_rows(labels_src)

        labels_merged = merge_classes(
            labels_src,
            class_ids_initial=class_ids_initial,
            class_ids_merged=class_ids_merged,
            idxs_to_edit=idxs_all,
            annotation_path=src_annotation_path,
        )

        black_dot_idxs = np.where(
            labels_merged == int(black_dot_class_id_after_merge)
        )[0].tolist()

        boxes_edited = edit_bbox_size(
            boxes_rc,
            idxs_to_edit=black_dot_idxs,
            width_scale=black_dot_width_scale,
            height_scale=black_dot_height_scale,
            image_shape_hw=(int(image_height), int(image_width)),
            annotation_path=src_annotation_path,
        )

        dst_annotation_path = dst_annotation_dir / src_annotation_path.name

        write_annotation_txt_rc(
            labels_src=labels_merged,
            boxes_rc=boxes_edited,
            output_path=dst_annotation_path,
        )

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

