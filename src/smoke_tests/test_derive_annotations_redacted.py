from __future__ import annotations

from configs.config import (
    NN_ALL_SPLIT_TXT,
    NN_IMAGE_DIR,
    NN_ANNOTATION_DIR,
    NN_DATASET_ROOT,
)

from nn_input_prepare import derive_box_annotations_from_src


def main() -> None:
    records = derive_box_annotations_from_src(
        split_path=NN_ALL_SPLIT_TXT,
        image_dir=NN_IMAGE_DIR,
        src_annotation_dir=NN_ANNOTATION_DIR,
        dst_annotation_dir=NN_DATASET_ROOT / "bounding_boxes_redacted",
        dst_annotation_format_path=NN_DATASET_ROOT / "annotation_format_redacted.json",
        source_annotation_format_path=NN_DATASET_ROOT / "annotation_format.json",

        class_ids_initial=(0, 1, 2, 3),
        class_ids_merged=(0, 0, 1, 2),
        derived_class_names={
            0: "merged_oval_loop",
            1: "black_dot",
            2: "other_defect",
        },

        black_dot_class_id_after_merge=1,
        black_dot_width_scale=1.3,
        black_dot_height_scale=1.3,

        annotation_format_name="redacted_merged_ovals_expanded_black_dots_v1",
        max_images=None,
    )

    print("redacted annotation export complete")
    print(f"files processed: {len(records)}")

    if records:
        first = records[0]
        print("first record:")
        for key, value in first.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()