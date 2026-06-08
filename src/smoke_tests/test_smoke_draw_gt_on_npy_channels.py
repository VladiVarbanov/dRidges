from pathlib import Path

import numpy as np

from configs.config import NN_CLASS_COLORS_GT
from src.debug_io import save_rgba_tiff
from src.nn_adapters import label_ids_from_bg0_format, xyxy_to_xywh
from src.torch_vision_dataset import TorchVisionDataset
from src.utilities import rgba_from_gray
from src.visualization import paint_labeled_xywh_boxes_in_place


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

    image_tensor, target = dataset[0]
    image_path, annotation_path = dataset.pairs[0]
    npy_path = root / "nn_input_npy" / f"{image_path.stem}.npy"

    nn_input = np.load(npy_path)

    boxes_xyxy = target["boxes"].detach().cpu().numpy()
    boxes_xywh = xyxy_to_xywh(boxes_xyxy)

    labels_bg0 = target["labels"].detach().cpu().numpy()
    labels_source = label_ids_from_bg0_format(labels_bg0)

    output_dir = Path("../../results/smoke_npy_channels_gt")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"source image: {image_path}")
    print(f"annotation: {annotation_path}")
    print(f"npy path: {npy_path}")
    print(f"npy shape: {nn_input.shape}")
    print(f"image tensor shape: {tuple(image_tensor.shape)}")
    print(f"target boxes: {len(boxes_xywh)}")
    print(f"labels BG0 min/max: {labels_bg0.min()} / {labels_bg0.max()}")
    print(f"labels source min/max: {labels_source.min()} / {labels_source.max()}")

    for channel_index in range(nn_input.shape[0]):
        channel = nn_input[channel_index]
        channel_rgba = rgba_from_gray(channel)

        paint_labeled_xywh_boxes_in_place(
            channel_rgba,
            boxes_xywh,
            labels_source,
            class_colors=NN_CLASS_COLORS_GT,
            default_color=(255, 255, 255),
            alfa_value=0.35,
            line_width=4,
        )

        output_path = output_dir / f"{image_path.stem}_channel_{channel_index}_gt.tif"
        save_rgba_tiff(channel_rgba, output_path)

        print(f"saved channel {channel_index} GT overlay: {output_path}")


if __name__ == "__main__":
    main()