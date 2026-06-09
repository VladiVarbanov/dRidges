from __future__ import annotations

import numpy as np
import torch

from configs.config import (
    NN_REDACTED_ANNOTATION_DIR,
    NN_DATASET_ROOT,
    NN_INPUT_NPY_DIR,
    NN_ANN_EXT,
)
from configs.object_features_config import GMM_CORRECTED_ANNOTATION_DIR
from nn_augmentation import get_augmentations, convert_boxes_to_torchvision, apply_transform_and_save
from src.annotation_io import parse_annotation_txt_rc
from src.utilities import ensure_dir

# Target image size (our images are 1024x1024)
TARGET_SIZE = 1024
NUM_CHANNELS = 3

# Augmentation settings
AUGMENTATION_SCALE_RANGE = (0.8, 1.2)
AUGMENTATION_ROTATION_ANGLES = [0, 90, 180, 270] #TODO: Add 

# Output directory
AUGMENTED_DATA_DIR = NN_DATASET_ROOT / "augmented_v1"
AUGMENTED_NPY_DIR = AUGMENTED_DATA_DIR / "nn_input_npy"
AUGMENTED_ANNOTATION_DIR = AUGMENTED_DATA_DIR / "bounding_boxes"


#TODO: Check if duplicated in torchvision Dataset


def main():
    # Setup directories
    ensure_dir(AUGMENTED_NPY_DIR)
    ensure_dir(AUGMENTED_ANNOTATION_DIR)

    # Get augmentations
    augmentations = get_augmentations()

    # Use GMM corrected annotations if available, else use redacted
    if GMM_CORRECTED_ANNOTATION_DIR.exists():
        annotation_dir = GMM_CORRECTED_ANNOTATION_DIR
        print(f"Using GMM-corrected annotations: {annotation_dir}")
    else:
        annotation_dir = NN_REDACTED_ANNOTATION_DIR
        print(f"Using redacted annotations: {annotation_dir}")

    # List of annotation files
    annotation_paths = sorted(annotation_dir.glob(f"*{NN_ANN_EXT}"))
    print(f"Found {len(annotation_paths)} annotation files")

    #TODO conversion in annnotations format need to happen need to happen here


    # Base random seed for reproducibility across runs
    base_seed = 42

    for ann_path in annotation_paths:
        image_name = ann_path.stem
        npy_path = NN_INPUT_NPY_DIR / f"{image_name}.npy"

        if not npy_path.exists():
            print(f"Skipping {ann_path.name}: no matching .npy file found")
            continue

        # Load 3-channel numpy array (C, H, W)
        image_np = np.load(npy_path)
        if image_np.shape[0] != NUM_CHANNELS:
            print(f"Skipping {image_name}: expected {NUM_CHANNELS} channels, got {image_np.shape[0]}")
            continue

        # Convert to torch tensor (C, H, W)
        image_tensor = torch.from_numpy(image_np).to(torch.float32)

        # Load annotations
        labels, boxes_rc = parse_annotation_txt_rc(ann_path)

        # Convert to torchvision target format
        # canvas_size should be (height, width) for the tensor
        canvas_size = (image_tensor.shape[1], image_tensor.shape[2])
        target = convert_boxes_to_torchvision(boxes_rc, labels, canvas_size)

        # Apply each augmentation and save
        for aug_name, transform in augmentations:
            rng_seed = hash(f"{base_seed}_{image_name}_{aug_name}") % (2**32)

            output_npy_path = AUGMENTED_NPY_DIR / f"{image_name}_{aug_name}.npy"
            output_annotation_path = AUGMENTED_ANNOTATION_DIR / f"{image_name}_{aug_name}{NN_ANN_EXT}"

            try:
                apply_transform_and_save(
                    image_tensor=image_tensor,
                    target=target,
                    transform=transform,
                    name=aug_name,
                    output_npy_path=output_npy_path,
                    output_annotation_path=output_annotation_path,
                    rng_seed=rng_seed,
                )
                print(f"  Saved {output_npy_path.name} and {output_annotation_path.name}")
            except Exception as e:
                print(f"  ERROR processing {aug_name} for {image_name}: {e}")

    print("\nAugmentation complete.")
    print(f"Augmented dataset saved to: {AUGMENTED_DATA_DIR}")
    print(f"NPY files: {len(list(AUGMENTED_NPY_DIR.glob('*.npy')))}")
    print(f"Annotations: {len(list(AUGMENTED_ANNOTATION_DIR.glob(f'*{NN_ANN_EXT}')))}")


if __name__ == "__main__":
    main()
