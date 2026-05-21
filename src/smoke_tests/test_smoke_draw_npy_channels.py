from pathlib import Path

import numpy as np

from debug_io import save_rgba_tiff
from torch_vision_dataset import TorchVisionDataset
from utilities import rgba_from_gray


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
    npy_path = root / "nn_input_npy" / f"{image_path.stem}.npy"

    nn_input = np.load(npy_path)

    print(f"source image: {image_path}")
    print(f"annotation: {annotation_path}")
    print(f"npy path: {npy_path}")
    print(f"npy shape: {nn_input.shape}")
    print(f"npy dtype: {nn_input.dtype}")

    output_dir = Path("../../results/smoke_npy_channels")
    output_dir.mkdir(parents=True, exist_ok=True)

    for channel_index in range(nn_input.shape[0]):
        channel = nn_input[channel_index]
        channel_rgba = rgba_from_gray(channel)

        output_path = output_dir / f"{image_path.stem}_channel_{channel_index}.tif"
        save_rgba_tiff(channel_rgba, output_path)

        print(
            f"saved channel {channel_index}: {output_path} "
            f"shape={channel.shape} "
            f"dtype={channel.dtype} "
            f"min={float(np.nanmin(channel)):.6f} "
            f"max={float(np.nanmax(channel)):.6f}"
        )


if __name__ == "__main__":
    main()