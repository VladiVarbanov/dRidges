from pathlib import Path

from configs.config import NN_CLASS_COLORS_GT
from debug_io import save_rgba_tiff
from nn_adapters import rows_cols_to_xywh
from annotation_io import parse_annotation_txt_rc
from torch_vision_dataset import TorchVisionDataset
from utilities import load_image, rgba_from_gray
from visualization import paint_labeled_xywh_boxes_in_place


def main():
    root = Path("../../DataSetFinal")

    dataset = TorchVisionDataset(
        split_path=root / "trainimages.txt",
        image_dir=root / "images",
        annotation_dir=root / "bounding_boxes",
        npy_dir=root / "nn_input_npy",
        annotation_format_path=root / "annotation_format.json",
        max_images=1,
    )

    image_path, annotation_path = dataset.pairs[0]

    source_gray = load_image(image_path)
    source_rgba = rgba_from_gray(source_gray)

    labels_source, boxes_rc = parse_annotation_txt_rc(annotation_path)
    boxes_xywh = rows_cols_to_xywh(boxes_rc)

    paint_labeled_xywh_boxes_in_place(
        source_rgba,
        boxes_xywh,
        labels_source,
        class_colors=NN_CLASS_COLORS_GT,
        default_color=(255, 255, 255),
        alfa_value=0.35,
        line_width=4,
    )

    output_dir = Path("../../results/smoke_source_annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_source_annotations_gt.tif"
    save_rgba_tiff(source_rgba, output_path)

    print(f"source image: {image_path}")
    print(f"annotation: {annotation_path}")
    print(f"source image shape: {source_gray.shape}")
    print(f"raw annotation boxes: {len(boxes_xywh)}")
    print(f"labels source min/max: {labels_source.min()} / {labels_source.max()}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()