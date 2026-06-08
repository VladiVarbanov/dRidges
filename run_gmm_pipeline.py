from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imsave

from configs.config import (
    NN_IMAGE_DIR,
    NN_ALL_SPLIT_TXT,
    NN_ANN_EXT,
    NN_CLASS_COLORS_GT,
    NN_REDACTED_ANNOTATION_DIR,
    GMM_SCALE_EVIDENCE_OUTPUT_DIR,
    GMM_SCALE_OVERLAY_DIR,
)

from configs.object_features_config import (
    GMM_SCALE_FEATURE_COLUMNS,
    GMM_SCALE_LABEL_ORDER_FEATURE,
    GMM_N_LABELS,
    GMM_COVARIANCE_TYPE,
    GMM_RANDOM_STATE,
    GMM_N_INIT,
    GMM_SCALE_EVIDENCE_NAME,
    GMM_SCALE_LABEL_TO_REDACTED_CLASS,
    GMM_OVERLAY_TOP_DISAGREEMENTS,
    GMM_OVERLAY_MAX_UNIQUE_IMAGES,
)

from src.annotation_io import (
    pair_split_images_with_annotations,
    parse_annotation_txt_rc,
)
from src.annotation_registry import make_object_uid
from src.gmm_wrapper import build_gmm_evidence_dataframe
from src.nn_adapters import rows_cols_to_xywh
from src.roi_instance_features import compute_box_geometry_rc
from src.utilities import ensure_dir, load_image, rgba_from_gray
from src.visualization import paint_labeled_xywh_boxes_in_place
from src.debug_io import save_rgba_tiff


REDACTED_CLASS_COLORS_GT = {
    0: NN_CLASS_COLORS_GT[0],
    1: NN_CLASS_COLORS_GT[2],
}


def main() -> None:
    output_dir = ensure_dir(GMM_SCALE_EVIDENCE_OUTPUT_DIR)
    overlay_dir = ensure_dir(GMM_SCALE_OVERLAY_DIR)

    # --------------------------------------------------------
    # 1. Build feature table inline
    # --------------------------------------------------------
    pairs = pair_split_images_with_annotations(
        split_path=NN_ALL_SPLIT_TXT,
        image_dir=NN_IMAGE_DIR,
        annotation_dir=NN_REDACTED_ANNOTATION_DIR,
        ann_ext=NN_ANN_EXT,
        require_annotation=True,
    )

    image_tables: list[pd.DataFrame] = []

    for image_id, (image_path, annotation_path) in enumerate(pairs):
        if annotation_path is None:
            continue

        source_labels, boxes_rc = parse_annotation_txt_rc(annotation_path)

        if len(source_labels) == 0:
            continue

        features, keep_mask = compute_box_geometry_rc(boxes_rc)

        if not keep_mask.any():
            continue

        original_row_indices = np.where(keep_mask)[0]
        kept_labels = source_labels[keep_mask]

        redacted_mask = np.isin(
            kept_labels, list(REDACTED_CLASS_COLORS_GT.keys())
        )

        if not redacted_mask.any():
            continue

        df = pd.DataFrame(features)
        df["source_label"] = kept_labels.astype(int)
        df["image_id"] = image_id
        df["image_name"] = image_path.name
        df["image_stem"] = image_path.stem
        df["image_path"] = str(image_path)
        df["annotation_path"] = str(annotation_path)
        df["row_idx"] = original_row_indices.astype(int)
        df["object_uid"] = [
            make_object_uid(image_id, int(idx))
            for idx in original_row_indices
        ]

        image_tables.append(df[redacted_mask].copy())

    if not image_tables:
        raise ValueError(
            f"No valid redacted class boxes found in {NN_REDACTED_ANNOTATION_DIR}"
        )

    feature_table = pd.concat(image_tables, ignore_index=True)

    print(f"Feature table: {len(feature_table)} boxes from {feature_table['image_id'].nunique()} images")

    # --------------------------------------------------------
    # 2. Run GMM
    # --------------------------------------------------------
    table_with_gmm = build_gmm_evidence_dataframe(
        feature_table,
        feature_columns=GMM_SCALE_FEATURE_COLUMNS,
        gmm_label_order_feature=GMM_SCALE_LABEL_ORDER_FEATURE,
        n_gmm_labels=GMM_N_LABELS,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=GMM_RANDOM_STATE,
        n_init=GMM_N_INIT,
        evidence_name=GMM_SCALE_EVIDENCE_NAME,
    )

    gmm_labels_col = f"{GMM_SCALE_EVIDENCE_NAME}_labels"
    gmm_highest_probability_col = (
        f"{GMM_SCALE_EVIDENCE_NAME}_highest_probability"
    )
    gmm_suggested_class_col = (
        f"{GMM_SCALE_EVIDENCE_NAME}_suggested_class"
    )

    table_with_gmm[gmm_suggested_class_col] = (
        table_with_gmm[gmm_labels_col]
        .map(GMM_SCALE_LABEL_TO_REDACTED_CLASS)
        .astype(int)
    )

    # --------------------------------------------------------
    # 3. Disagreement table
    # --------------------------------------------------------
    disagreement_mask = (
        table_with_gmm["source_label"] != table_with_gmm[gmm_suggested_class_col]
    )
    gmm_scale_disagreement_table = table_with_gmm[disagreement_mask].copy()

    table_path = output_dir / "gmm_scale_table.csv"
    disagreement_path = output_dir / "gmm_scale_disagreement_table.csv"

    table_with_gmm.to_csv(table_path, index=False)
    gmm_scale_disagreement_table.to_csv(disagreement_path, index=False)

    print()
    print("Saved:")
    print(f"  {table_path}")
    print(f"  {disagreement_path}")

    # --------------------------------------------------------
    # 4. Print stats
    # --------------------------------------------------------
    print()
    print("Source label counts:")
    print(table_with_gmm["source_label"].value_counts().sort_index())

    print()
    print("GMM scale label counts:")
    print(table_with_gmm[gmm_labels_col].value_counts().sort_index())

    print()
    print("Source label × GMM suggested class:")
    print(
        pd.crosstab(
            table_with_gmm["source_label"],
            table_with_gmm[gmm_suggested_class_col],
            rownames=["source_label"],
            colnames=[gmm_suggested_class_col],
        )
    )

    print()
    print("Mean log_area by GMM scale label:")
    print(table_with_gmm.groupby(gmm_labels_col)["log_area"].mean())

    print()
    print("Disagreement count:")
    print(len(gmm_scale_disagreement_table))

    # --------------------------------------------------------
    # 5. Overlays — top disagreements
    # --------------------------------------------------------
    overlay_candidates = (
        gmm_scale_disagreement_table.sort_values(
            gmm_highest_probability_col,
            ascending=False,
        )
        .head(GMM_OVERLAY_TOP_DISAGREEMENTS)
    )

    unique_images = (
        overlay_candidates["image_name"].drop_duplicates().head(GMM_OVERLAY_MAX_UNIQUE_IMAGES)
    )

    saved_overlay_count = 0

    for image_name in unique_images:
        table_for_image = table_with_gmm[
            table_with_gmm["image_name"] == image_name
        ].copy()

        image_stem = Path(str(image_name)).stem
        output_path = (
            overlay_dir / f"{image_stem}_source_and_gmm.png"
        )

        # Load image
        image_path = Path(str(table_for_image["image_path"].iloc[0]))
        image = load_image(image_path)
        image_rgba = rgba_from_gray(image)

        # Boxes
        boxes_rc = table_for_image[
            ["row_min", "col_min", "row_max", "col_max"]
        ].to_numpy(dtype=np.float32)
        boxes_xywh = rows_cols_to_xywh(boxes_rc)

        source_labels = table_for_image["source_label"].to_numpy(dtype=np.int64)
        gmm_suggested_labels = table_for_image[
            gmm_suggested_class_col
        ].to_numpy(dtype=np.int64)

        # GMM suggested class (outer, larger)
        paint_labeled_xywh_boxes_in_place(
            image_rgba=image_rgba,
            boxes_xywh=boxes_xywh,
            labels=gmm_suggested_labels,
            class_colors=REDACTED_CLASS_COLORS_GT,
            line_width=2,
            scale_factor=1.08,
        )

        # Source/redacted class (inner, smaller)
        paint_labeled_xywh_boxes_in_place(
            image_rgba=image_rgba,
            boxes_xywh=boxes_xywh,
            labels=source_labels,
            class_colors=REDACTED_CLASS_COLORS_GT,
            line_width=2,
            scale_factor=0.92,
        )

        save_rgba_tiff(image_rgba, output_path)
        saved_overlay_count += 1

    print()
    print(f"Saved overlay images: {saved_overlay_count}")
    print(f"Overlay folder: {overlay_dir}")


if __name__ == "__main__":
    main()
