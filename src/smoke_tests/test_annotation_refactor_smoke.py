from __future__ import annotations

from pathlib import Path

import numpy as np

from configs.config import (
    NN_IMAGE_DIR,
    NN_ANNOTATION_DIR,
    NN_TRAIN_SPLIT_TXT,
)

from src.annotation_io import (
    pair_split_images_with_annotations,
    parse_annotation_txt_rc,
    write_annotation_txt_rc,
    merge_classes,
    edit_bbox_size,
)

from src.annotation_registry import (
    make_object_uid,
    write_derived_annotation_format_json,
    load_annotation_format_json,
    validate_source_annotation_format,
    probability_vector_to_str,
    make_label_probability,
)

from src.nn_adapters import rows_cols_to_xywh

from src.nn_input_prepare import (
    keep_xywh_boxes,
    discard_xywh_boxes_by_size,
)

from src import roi_instance_features as rif


def test_annotation_refactor_smoke(tmp_path: Path) -> None:
    # ------------------------------------------------------------
    # 1) Find one real image/annotation pair from the train split
    # ------------------------------------------------------------
    pairs = pair_split_images_with_annotations(
        split_path=NN_TRAIN_SPLIT_TXT,
        image_dir=NN_IMAGE_DIR,
        annotation_dir=NN_ANNOTATION_DIR,
        require_annotation=True,
        max_images=1,
    )

    assert len(pairs) == 1

    image_path, annotation_path = pairs[0]

    assert image_path.exists()
    assert annotation_path is not None
    assert annotation_path.exists()

    # ------------------------------------------------------------
    # 2) Parse source annotation txt
    # ------------------------------------------------------------
    labels_src, boxes_rc = parse_annotation_txt_rc(annotation_path)

    assert labels_src.ndim == 1
    assert boxes_rc.ndim == 2
    assert boxes_rc.shape[1] == 4
    assert len(labels_src) == len(boxes_rc)
    assert len(labels_src) > 0

    # ------------------------------------------------------------
    # 3) Mechanical annotation transform
    # ------------------------------------------------------------
    class_ids_initial = (0, 1, 2, 3)
    class_ids_merged = (0, 0, 1, 2)

    labels_merged = merge_classes(
        labels_src,
        class_ids_initial=class_ids_initial,
        class_ids_merged=class_ids_merged,
        idxs_to_edit=None,
        annotation_path=annotation_path,
    )

    boxes_edited = edit_bbox_size(
        boxes_rc,
        idxs_to_edit=None,
        width_scale=1.3,
        height_scale=1.3,
        image_shape_hw=None,
        annotation_path=annotation_path,
    )

    assert labels_merged.shape == labels_src.shape
    assert boxes_edited.shape == boxes_rc.shape

    # ------------------------------------------------------------
    # 4) Write derived annotation txt, then read it back
    # ------------------------------------------------------------
    derived_annotation_path = tmp_path / "derived_annotations" / annotation_path.name

    write_annotation_txt_rc(
        labels_src=labels_merged,
        boxes_rc=boxes_edited,
        output_path=derived_annotation_path,
    )

    assert derived_annotation_path.exists()

    labels_back, boxes_back = parse_annotation_txt_rc(derived_annotation_path)

    assert np.array_equal(labels_back, labels_merged)
    assert np.allclose(boxes_back, boxes_edited, atol=1e-3)

    # ------------------------------------------------------------
    # 5) Registry metadata JSON
    # ------------------------------------------------------------
    derived_format_path = tmp_path / "derived_annotation_format.json"

    derived_class_names = {
        0: "merged_oval_loop",
        1: "black_dot",
        2: "other_defect",
    }

    write_derived_annotation_format_json(
        output_path=derived_format_path,
        annotation_format_name="smoke_derived_annotations_v1",
        annotation_format_role="derived_source_annotations",
        class_names=derived_class_names,
        source_annotation_format="annotation_format.json",
        class_ids_initial=class_ids_initial,
        class_ids_merged=class_ids_merged,
        bbox_edit_notes=[
            {
                "note": "smoke test only; no size change",
                "width_scale": 1.0,
                "height_scale": 1.0,
            }
        ],
    )

    assert derived_format_path.exists()

    annotation_format = load_annotation_format_json(derived_format_path)
    validate_source_annotation_format(annotation_format)

    assert annotation_format["annotation_format_role"] == "derived_source_annotations"
    assert annotation_format["derived_annotation_notes"]["class_remap"] == {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 2,
    }

    # ------------------------------------------------------------
    # 6) Registry identity/probability helpers
    # ------------------------------------------------------------
    assert make_object_uid(37, 12) == "img00037_box00012"
    assert probability_vector_to_str([0.8, 0.2]) == "0.8 0.2"
    assert make_label_probability(2, 4) == "0 0 1 0"

    # ------------------------------------------------------------
    # 7) NN-side XYWH box filters still work
    # ------------------------------------------------------------
    boxes_xywh = rows_cols_to_xywh(boxes_rc)

    kept_labels, kept_boxes = keep_xywh_boxes(
        boxes_xywh,
        labels_src,
    )

    assert kept_labels.ndim == 1
    assert kept_boxes.ndim == 2
    assert kept_boxes.shape[1] == 4
    assert len(kept_labels) == len(kept_boxes)

    discarded_boxes = discard_xywh_boxes_by_size(
        boxes_xywh,
        min_width=1.0,
        min_height=1.0,
        min_area=1.0,
    )

    assert discarded_boxes.ndim == 2
    assert discarded_boxes.shape[1] == 4

    # ------------------------------------------------------------
    # 8) ROI instance feature helpers still work
    # ------------------------------------------------------------
    features, keep_mask = rif.compute_box_geometry_rc(boxes_rc)

    assert keep_mask.ndim == 1
    assert keep_mask.shape[0] == boxes_rc.shape[0]
    assert "area" in features
    assert "log_area" in features
    assert "aspect_ratio" in features

    if len(features["area"]) > 0:
        hist = rif.compute_log_area_histogram(features["area"])

        assert "counts" in hist
        assert "bin_edges" in hist
        assert "bin_centers" in hist
        assert "log_area" in hist

        peaks = rif.find_area_histogram_peaks(
            hist["counts"],
            hist["bin_centers"],
        )

        valleys = rif.find_area_histogram_valleys(
            hist["counts"],
            hist["bin_centers"],
        )

        assert "peak_indices" in peaks
        assert "valley_indices" in valleys