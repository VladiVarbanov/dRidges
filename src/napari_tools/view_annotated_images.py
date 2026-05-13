"""
View project images with source box annotations in napari.

This module builds the napari viewer:
    image layer + editable box annotation layer

It does not save edited annotations yet.
Saving will be handled later in save_edited_annotations.py.

Run through:
    python -m src.napari_tools.napari_launcher
"""

from __future__ import annotations

from pathlib import Path

import napari
import numpy as np

from configs.config import (
    NN_IMAGE_DIR,
    NN_ANNOTATION_DIR,
    NN_TRAIN_SPLIT_TXT,
    NN_CLASS_COLORS,
    NN_CLASS_NAMES,
)

from src.utilities import load_image
from src.nn_input_prepare import (
    pair_split_images_with_annotations,
    parse_annotation_txt_rc,
)

from src.napari_tools.napari_box_adapters import (
    boxes_rc_to_napari_rectangles,
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

    The split file is the source of truth.

    Current behavior:
        - read pairs from the training split,
        - pick one pair by image_index.

    Later we can add:
        - command-line image_index,
        - image name selection,
        - next/previous image navigation.
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
    Load one image and its source box annotations.

    Returns
    -------
    image
        Raw image array loaded through the project loader.

    labels_src
        Source class labels from the txt file.
        These are foreground labels starting at 0.

    boxes_rc
        Source boxes in row/column minmax format:
            [row_min, col_min, row_max, col_max]

    Important
    ---------
    This function does not convert labels to TorchVision BG0 format.
    Napari editing must preserve the source annotation format.
    """
    image = load_image(image_path)
    labels_src, boxes_rc = parse_annotation_txt_rc(Path(annotation_path))

    return image, labels_src, boxes_rc


def validate_annotation_labels(
    labels_src: np.ndarray,
    *,
    annotation_path: str | Path,
    class_names: dict[int, str] = NN_CLASS_NAMES,
) -> None:
    """
    Validate that all annotation labels are known project classes.

    Source annotation labels should be:
        0, 1, 2, 3
    TODO: LAter add the expexted format, based on the json file
    If an unknown label appears, fail loudly before opening the viewer.
    """
    known_labels = set(int(label_id) for label_id in class_names.keys())
    observed_labels = set(int(label_id) for label_id in labels_src.tolist())

    unknown_labels = sorted(observed_labels - known_labels)

    if unknown_labels:
        raise ValueError(
            f"{annotation_path}: unknown class labels {unknown_labels}. "
            f"Known labels are {sorted(known_labels)}"
        )


def add_napari_image_layer(
    viewer: napari.Viewer,
    image: np.ndarray,
    *,
    image_name: str,
) -> None:
    """
    Add the source image to napari.

    For now, this displays the image exactly as loaded by load_image(...).
    TODO: I am sure there was an json file with that data
    Later we may choose to display:
        - normalized grayscale,
        - RGB-converted image,
        - cached NN channels,
        - preprocessed channel views.

    """
    viewer.add_image(
        image,
        name=image_name,
    )


def add_napari_box_annotation_layer(
    viewer: napari.Viewer,
    *,
    boxes_rc: np.ndarray,
    labels_src: np.ndarray,
    class_names: dict[int, str] = NN_CLASS_NAMES,
    class_colors: dict[int, tuple[int, int, int]] = NN_CLASS_COLORS,
) -> None:
    """
    Add source annotation boxes as an editable napari Shapes layer.

    Source boxes:
        [row_min, col_min, row_max, col_max]

    Napari rectangles:
        four vertices in row/column coordinates.

    Labels are stored as napari features, not converted to TorchVision labels.
    """
    rectangles = boxes_rc_to_napari_rectangles(boxes_rc)

    class_id_values = [int(label_id) for label_id in labels_src.tolist()]
    class_name_values = [
        class_names.get(int(label_id), f"unknown_{int(label_id)}")
        for label_id in class_id_values
    ]

    features = {
        "class_id": class_id_values,
        "class_name": class_name_values,
    }

    # For first version, use a single edge color.
    # Class-specific coloring can be added after we confirm basic loading.
    text_parameters = {
        "string": "class {class_id}",
        "size": 10,
        "color": "yellow",
        "anchor": "upper_left",
        "translation": [-3, 0],
    }

    viewer.add_shapes(
        rectangles,
        shape_type="rectangle",
        name="source_boxes",
        edge_color="green",
        face_color="transparent",
        features=features,
        text=text_parameters,
    )


def view_napari_annotated_images(
    *,
    image_index: int = 0,
) -> None:
    """
    Open napari with one image and its matching source annotation boxes.

    Current default:
        image_index=0 from trainimages.txt

    This is intentionally small and conservative:
        - no saving,
        - no overwrite,
        - no TorchVision conversion,
        - no model predictions.
    """
    image_path, annotation_path = select_image_annotation_pair(
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

    viewer = napari.Viewer()

    add_napari_image_layer(
        viewer,
        image,
        image_name=image_path.name,
    )

    add_napari_box_annotation_layer(
        viewer,
        boxes_rc=boxes_rc,
        labels_src=labels_src,
    )

    print("Opened annotated image:")
    print(f"  image:      {image_path}")
    print(f"  annotation: {annotation_path}")
    print(f"  boxes:      {len(boxes_rc)}")
    print(f"  labels:     {sorted(set(labels_src.tolist()))}")

    napari.run()