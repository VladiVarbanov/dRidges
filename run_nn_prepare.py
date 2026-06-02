from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from configs.config import (
    NN_IMAGE_DIR,
    NN_ANNOTATION_DIR,
    NN_TRAIN_SPLIT_TXT,
    NN_ALL_SPLIT_TXT,
    RESULTS_VIS_DIR,
    NN_CLASS_COLORS,
    NN_INPUT_NPY_DIR,
    NN_EXPECTED_NUM_CHANNELS,
    NN_LOCAL_NORM_SIGMA,
    NN_GAUSSIAN_SMOOTH_SIGMA,
    NN_HESSIAN_SCALE_PX,
)

from src.nn_prepare_pipeline_helpers import prepare_small_nn_input_cache_test

from src.nn_input_prepare import (
    build_nn_gray_channel,
    build_nn_hog_norm_channel,
    build_nn_vesselness_channel,
    build_nn_input_img,
)
from nn_anotation_io import pair_split_images_with_annotations, parse_annotation_txt_rc
from nn_adapters import rows_cols_to_xywh

from src.utilities import (
    load_image,
    rgba_from_gray,
    ensure_dir,
)

from src.preprocessing import to_gray_normalized
from src.visualization import paint_labeled_xywh_boxes_in_place
from src.debug_io import save_rgba_tiff


CHANNEL_FNS = [
    build_nn_gray_channel,
    build_nn_hog_norm_channel,
    build_nn_vesselness_channel,
]


def main() -> None:
    # ------------------------------------------------------------
    # 1) Small NN input cache smoke test
    # ------------------------------------------------------------
    # This saves separate channel inspection files through the old helper.
    # It is not the final training cache.
    records = prepare_small_nn_input_cache_test(
        image_dir=NN_IMAGE_DIR,
        channel_fns=CHANNEL_FNS,
        max_images=5,
    )

    print("\nNN input channel smoke test:")
    for record in records:
        print(record["image_path"], record["shape"], record["dtype"])

    # ------------------------------------------------------------
    # 2) Annotation overlay plumbing test from train split
    # ------------------------------------------------------------
    # This visualizes source annotations as-is.
    # Some source annotations may be incomplete or wrong; this step does not fix them.
    output_dir = ensure_dir(RESULTS_VIS_DIR / "nn_source_annotation_boxes")

    pairs = pair_split_images_with_annotations(
        split_path=NN_TRAIN_SPLIT_TXT,
        image_dir=NN_IMAGE_DIR,
        annotation_dir=NN_ANNOTATION_DIR,
        require_annotation=True,
        max_images=5,
    )

    print("\nAnnotation overlay test:")

    for image_id, (image_path, annotation_path) in enumerate(pairs):
        if annotation_path is None:
            raise RuntimeError(
                f"Internal error: annotation_path is None for {image_path}"
            )

        # Read source annotations:
        # class row_min col_min row_max col_max
        labels, boxes_rc = parse_annotation_txt_rc(annotation_path)

        # Convert to internal XYWH:
        # x, y, width, height
        boxes_xywh = rows_cols_to_xywh(boxes_rc)

        # Crash loudly on unknown classes.
        unknown_labels = sorted(set(labels.tolist()) - set(NN_CLASS_COLORS.keys()))
        if unknown_labels:
            raise ValueError(
                f"{annotation_path}: unknown class labels {unknown_labels}. "
                f"Known labels are {sorted(NN_CLASS_COLORS.keys())}"
            )

        # Load raw image.
        img_raw = load_image(image_path)

        # Convert raw image to stable grayscale visualization base.
        gray = to_gray_normalized(img_raw)
        image_rgba = rgba_from_gray(gray)

        # Draw class-colored boxes.
        paint_labeled_xywh_boxes_in_place(
            image_rgba=image_rgba,
            boxes_xywh=boxes_xywh,
            labels=labels,
            class_colors=NN_CLASS_COLORS,
            alfa_value=None,
            line_width=2,
        )

        # Save debug overlay TIFF.
        overlay_path = output_dir / f"{image_path.stem}_ann_boxes.tif"
        save_rgba_tiff(image_rgba, overlay_path)

        print(
            f"[{image_id}]",
            "\n image:     ", image_path,
            "\n annotation:", annotation_path,
            "\n boxes:     ", len(boxes_xywh),
            "\n labels:    ", sorted(set(labels.tolist())),
            "\n saved:     ", overlay_path,
        )

    # ------------------------------------------------------------
    # 3) Build real stacked NN input .npy cache from all split images
    # ------------------------------------------------------------
    # This is the useful training cache:
    # one .npy per image, shape = (C, H, W), dtype = float32.
    #
    # The split txt files remain the source of train/test separation.
    # The annotation txt files remain the source of truth for boxes/labels.
    cache_dir = ensure_dir(NN_INPUT_NPY_DIR)

    cache_pairs = pair_split_images_with_annotations(
        split_path=NN_ALL_SPLIT_TXT,
        image_dir=NN_IMAGE_DIR,
        annotation_dir=NN_ANNOTATION_DIR,
        require_annotation=True,
        max_images=None,
    )

    print("\nBuilding stacked NN input cache:")

    for image_id, (image_path, annotation_path) in enumerate(cache_pairs):
        if annotation_path is None:
            raise RuntimeError(
                f"Internal error: annotation_path is None for {image_path}"
            )

        # Build stacked NN image from raw:
        # output shape: (C, H, W)
        img_raw = load_image(image_path)

        nn_input_image = build_nn_input_img(
            img_raw=img_raw,
            channel_fns=CHANNEL_FNS,
        )

        if nn_input_image.ndim != 3:
            raise ValueError(
                f"{image_path}: expected NN input shape (C, H, W), "
                f"got {nn_input_image.shape}"
            )

        if nn_input_image.shape[0] != NN_EXPECTED_NUM_CHANNELS:
            raise ValueError(
                f"{image_path}: expected {NN_EXPECTED_NUM_CHANNELS} channels, "
                f"got {nn_input_image.shape[0]}"
            )

        if nn_input_image.dtype != np.float32:
            nn_input_image = nn_input_image.astype(np.float32, copy=False)

        if not np.isfinite(nn_input_image).all():
            raise ValueError(f"{image_path}: NN input contains NaN or Inf")

        npy_path = cache_dir / f"{image_path.stem}.npy"
        json_path = cache_dir / f"{image_path.stem}.json"

        np.save(npy_path, nn_input_image)

        # Metadata only. Do not store heavy arrays in JSON.
        labels, boxes_rc = parse_annotation_txt_rc(annotation_path)
        boxes_xywh = rows_cols_to_xywh(boxes_rc)

        metadata = {
            "image_id": int(image_id),
            "image_name": image_path.name,
            "image_stem": image_path.stem,
            "image_path": str(image_path),
            "annotation_path": str(annotation_path),
            "npy_path": str(npy_path),

            "nn_input_shape": list(nn_input_image.shape),
            "nn_input_dtype": str(nn_input_image.dtype),
            "channel_axis": 0,
            "image_layout": "CHW",

            "channel_fns": [
                getattr(fn, "__name__", repr(fn))
                for fn in CHANNEL_FNS
            ],

            "source_annotation_format": "class row_min col_min row_max col_max",
            "box_format_in_memory": "XYWH",
            "label_dtype": str(labels.dtype),
            "num_boxes": int(len(boxes_xywh)),
            "labels_present": sorted(int(x) for x in set(labels.tolist())),

            "preparation": {
                "version": "nn_input_cache_v1",
                "nn_expected_num_channels": int(NN_EXPECTED_NUM_CHANNELS),
                "nn_local_norm_sigma": float(NN_LOCAL_NORM_SIGMA),
                "nn_gaussian_smooth_sigma": float(NN_GAUSSIAN_SMOOTH_SIGMA),
                "nn_hessian_scale_px": float(NN_HESSIAN_SCALE_PX),
            },
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        print(
            f"[{image_id}]",
            image_path.name,
            "shape:",
            nn_input_image.shape,
            "saved:",
            npy_path,
        )


if __name__ == "__main__":
    main()