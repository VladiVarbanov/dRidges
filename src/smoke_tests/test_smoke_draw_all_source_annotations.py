from pathlib import Path
from debug_io import save_rgba_tiff
from nn_adapters import rows_cols_to_xywh
from nn_input_prepare import parse_annotation_txt_rc
from utilities import load_image, rgba_from_gray
from visualization import paint_labeled_xywh_boxes_in_place
from configs.config import NN_CLASS_COLORS, NN_CLASS_COLORS_GT, NN_ANN_EXT
import numpy as np

def main():
    root = Path("../../DataSetFinal")

    image_dir = root / "images"
    annotation_dir = root / "bounding_boxes"

    output_dir = Path("../../results/test_smoke_draw_all_source_annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_paths = sorted(annotation_dir.glob(f"*{NN_ANN_EXT}"))

    print(f"annotation files found: {len(annotation_paths)}")

    for annotation_path in annotation_paths:
        image_path = image_dir / f"{annotation_path.stem}.jpg"

        if not image_path.exists():










            print(f"missing image for: {annotation_path.name}")
            continue

        labels_source, boxes_rc = parse_annotation_txt_rc(annotation_path)
        boxes_xywh = rows_cols_to_xywh(boxes_rc)

        source_gray = load_image(image_path)
        source_rgba = rgba_from_gray(source_gray)

        paint_labeled_xywh_boxes_in_place(
            source_rgba,
            boxes_xywh,
            labels_source,
            class_colors=NN_CLASS_COLORS_GT,
            default_color=(255, 255, 255),
            alfa_value=0.35,
            line_width=4,
        )

        output_path = (
            output_dir /
            f"{annotation_path.stem}_source_annotations_gt.tif"
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