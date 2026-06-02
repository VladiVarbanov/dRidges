from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from configs.config import NN_ANN_EXT
from utilities import ensure_dir, load_image


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
