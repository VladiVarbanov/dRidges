from __future__ import annotations

from pathlib import Path

import numpy as np

from configs.config import (
    NN_ANN_EXT,
    NN_REDACTED_ANNOTATION_DIR,
    NN_IMAGE_DIR,
    NN_DATASET_ROOT,
    RESULTS_VIS_DIR,
)
from configs.object_features_config import (
    GMM_CORRECTED_ANNOTATION_DIR,
)

from src.debug_io import save_rgba_tiff
from src.nn_adapters import rows_cols_to_xywh
from src.annotation_io import parse_annotation_txt_rc
from src.preprocessing import to_gray_normalized
from src.utilities import collect_images_paths, load_image, rgba_from_gray, ensure_dir
from src.visualization import paint_labeled_xywh_boxes_in_place


REDACTED_CLASS_COLORS = {
    0: (160, 20, 60),    # merged_oval_loop
    1: (0, 170, 110),    # black_dot
    2: (230, 180, 40),   # other_defect
}


def build_image_lookup_by_stem(image_dir: Path) -> dict[str, Path]:
    """
    Build image lookup by filename stem.

    This avoids assuming that every image is .jpg.
    """
    image_paths = collect_images_paths(
        input_dir=image_dir,
        recursive=False,
    )

    lookup: dict[str, Path] = {}

    for image_path in image_paths:
        if image_path.stem in lookup:
            raise ValueError(
                f"Duplicate image stem found: {image_path.stem}\n"
                f"Existing: {lookup[image_path.stem]}\n"
                f"New:      {image_path}"
            )

        lookup[image_path.stem] = image_path

    return lookup


def main() -> None:
    image_dir = NN_IMAGE_DIR

    # Try GMM corrected annotations first, fall back to redacted
    if GMM_CORRECTED_ANNOTATION_DIR.exists():
        annotation_dir = GMM_CORRECTED_ANNOTATION_DIR
        print("Using GMM-corrected annotations")
    else:
        annotation_dir = NN_REDACTED_ANNOTATION_DIR
        print("Using original redacted annotations")

    output_dir = ensure_dir(RESULTS_VIS_DIR / "test_smoke_draw_gmm_corrected_annotations")

    image_lookup = build_image_lookup_by_stem(image_dir)

    annotation_paths = sorted(annotation_dir.glob(f"*{NN_ANN_EXT}"))

    print(f"annotation files found: {len(annotation_paths)}")
    print(f"images found: {len(image_lookup)}")

    for annotation_path in annotation_paths:
        image_path = image_lookup.get(annotation_path.stem)

        if image_path is None:
            print(f"missing image for: {annotation_path.name}")
            continue

        labels_source, boxes_rc = parse_annotation_txt_rc(annotation_path)
        boxes_xywh = rows_cols_to_xywh(boxes_rc)

        source_raw = load_image(image_path)
        source_gray = to_gray_normalized(source_raw)
        source_rgba = rgba_from_gray(source_gray)

        unknown_labels = sorted(
            set(int(label) for label in labels_source.tolist())
            - set(REDACTED_CLASS_COLORS.keys())
        )

        if unknown_labels:
            raise ValueError(
                f"{annotation_path}: unknown redacted labels {unknown_labels}. "
                f"Known labels are {sorted(REDACTED_CLASS_COLORS.keys())}"
            )

        paint_labeled_xywh_boxes_in_place(
            source_rgba,
            boxes_xywh,
            labels_source,
            class_colors=REDACTED_CLASS_COLORS,
            default_color=(255, 255, 255),
            alfa_value=0.35,
            line_width=4,
        )

        output_path = (
            output_dir /
            f"{annotation_path.stem}_gmm_corrected_annotations.tif"
        )

        save_rgba_tiff(source_rgba, output_path)

        unique_labels = np.unique(labels_source)

        print(
            f"saved: {output_path.name} "
            f"boxes={len(boxes_xywh)} "
            f"labels={unique_labels.tolist()}"
        )

    print("done")


if __name__ == "__main__":
    main()
