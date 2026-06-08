from __future__ import annotations

from pathlib import Path

import pandas as pd

from configs.config import (
    NN_IMAGE_DIR,
    NN_ALL_SPLIT_TXT,
    NN_ANN_EXT,
    NN_REDACTED_ANNOTATION_DIR,
    GMM_SCALE_EVIDENCE_OUTPUT_DIR,
)

from configs.object_features_config import (
    GMM_BOX_SCALE_FACTOR,
    GMM_CORRECTED_ANNOTATION_DIR,
)

from src.annotation_io import (
    pair_split_images_with_annotations,
    parse_annotation_txt_rc,
    write_annotation_txt_rc,
    edit_bbox_size,
)
from src.utilities import ensure_dir, load_csv_table


def main() -> None:
    # Load GMM evidence
    table_path = GMM_SCALE_EVIDENCE_OUTPUT_DIR / "gmm_scale_table.csv"
    table_with_gmm = load_csv_table(table_path)

    # Identify disagreed boxes by object_uid
    disagreement_mask = (
        table_with_gmm["source_label"]
        != table_with_gmm["gmm_scale_suggested_class"]
    )
    disagreed_uids = set(
        table_with_gmm.loc[disagreement_mask, "object_uid"].tolist()
    )

    print(f"Total boxes in GMM table: {len(table_with_gmm)}")
    print(f"Disagreed boxes to scale up: {len(disagreed_uids)}")

    # Setup output directory
    output_dir = ensure_dir(GMM_CORRECTED_ANNOTATION_DIR)

    # Re-pair images with their original redacted annotations
    pairs = pair_split_images_with_annotations(
        split_path=NN_ALL_SPLIT_TXT,
        image_dir=NN_IMAGE_DIR,
        annotation_dir=NN_REDACTED_ANNOTATION_DIR,
        ann_ext=NN_ANN_EXT,
        require_annotation=True,
    )

    processed_count = 0
    scaled_count = 0

    for image_id, (image_path, annotation_path) in enumerate(pairs):
        if annotation_path is None:
            continue

        # Load original redacted annotations
        labels, boxes_rc = parse_annotation_txt_rc(annotation_path)

        if len(labels) == 0:
            continue

        # Build object_uids for this image's boxes to match with GMM table
        image_uids = [
            f"img{image_id:05d}_box{row_idx:05d}"
            for row_idx in range(len(labels))
        ]

        # Find which boxes in this image are disagreed
        idxs_to_scale = [
            idx for idx, uid in enumerate(image_uids)
            if uid in disagreed_uids
        ]

        if not idxs_to_scale:
            # No disagreements in this image, copy as-is
            output_path = output_dir / annotation_path.name
            write_annotation_txt_rc(
                labels_src=labels,
                boxes_rc=boxes_rc,
                output_path=output_path,
            )
            processed_count += 1
            continue

        # Scale up disagreed boxes
        boxes_scaled = edit_bbox_size(
            boxes_rc,
            idxs_to_edit=idxs_to_scale,
            width_scale=GMM_BOX_SCALE_FACTOR,
            height_scale=GMM_BOX_SCALE_FACTOR,
        )

        # Write corrected annotations
        output_path = output_dir / annotation_path.name
        write_annotation_txt_rc(
            labels_src=labels,
            boxes_rc=boxes_scaled,
            output_path=output_path,
        )

        processed_count += 1
        scaled_count += len(idxs_to_scale)

    print()
    print(f"Processed {processed_count} annotation files")
    print(f"Scaled {scaled_count} boxes by {GMM_BOX_SCALE_FACTOR}x")
    print(f"Output directory: {GMM_CORRECTED_ANNOTATION_DIR}")


if __name__ == "__main__":
    main()
