from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes

from .annotation_io import write_annotation_txt_rc
from .nn_adapters import rows_cols_to_xyxy, xyxy_to_rows_cols
from .torch_vision_dataset import build_torchvision_target


# Augmentation constants (to be imported from config)
AUGMENTATION_ROTATION_ANGLES = [0, 90, 180, 270]
TARGET_SIZE = 1024
AUGMENTATION_SCALE_RANGE = (0.8, 1.2)


def get_augmentation_transforms() -> list[tuple[str, Callable]]:
    """
    Returns a list of (name, transform) tuples for offline augmentation.

    Each transform is callable taking (image_tensor, target_dict) and returning
    (image_tensor, target_dict) with synchronized geometry changes.

    Transformations:
    - original: identity (no change)
    - rotate{angle}: fixed-angle rotations (90, 180, 270)
    - hflip: horizontal flip
    - vflip: vertical flip
    - scale08: scale down to 80% + pad to original size
    - scale12: scale up to 120% + center crop to original size
    """
    augmentations = []

    # 1. Original (Identity)
    augmentations.append(("original", T.Identity()))

    # 2. Fixed-angle rotations
    for angle in AUGMENTATION_ROTATION_ANGLES:
        if angle == 0:
            continue
        augmentations.append((
            f"rotate{angle}",
            T.RandomRotation(degrees=(angle, angle), expand=False)
        ))

    # 3. Flips
    augmentations.append(("hflip", T.RandomHorizontalFlip(p=1.0)))
    augmentations.append(("vflip", T.RandomVerticalFlip(p=1.0)))

    # 4. Scale down 80% + pad to original size
    scale_down_size = int(TARGET_SIZE * AUGMENTATION_SCALE_RANGE[0])
    pad = TARGET_SIZE - scale_down_size
    pad_half = pad // 2
    pad_rest = pad - pad_half
    augmentations.append((
        "scale08",
        T.Compose([
            T.Resize((scale_down_size, scale_down_size)),
            T.Pad((pad_half, pad_half, pad_rest, pad_rest), fill=0)
        ])
    ))

    # 5. Scale up 120% + center crop to original size
    scale_up_size = int(TARGET_SIZE * AUGMENTATION_SCALE_RANGE[1])
    augmentations.append((
        "scale12",
        T.Compose([
            T.Resize((scale_up_size, scale_up_size)),
            T.CenterCrop((TARGET_SIZE, TARGET_SIZE))
        ])
    ))

    return augmentations


def convert_to_torchvision_target(
    image_tensor: torch.Tensor,
    labels_src: np.ndarray,
    boxes_rc: np.ndarray,
    image_id: int = 0
) -> dict[str, torch.Tensor | BoundingBoxes]:
    """
    Convert source annotation format to TorchVision target with BoundingBoxes.

    Uses build_torchvision_target() which handles the BG0 label shift internally.

    Parameters
    ----------
    image_tensor
        torch tensor with shape (C, H, W)
    labels_src
        Project labels: 0, 1, 2 (foreground only)
    boxes_rc
        Source annotation boxes: [row_min, col_min, row_max, col_max]
    image_id
        Identifier for the image (default 0 for single-image processing)

    Returns
    -------
    target
        TorchVision target dict with boxes as BoundingBoxes, labels shifted to BG0
    """
    return build_torchvision_target(
        labels_src=labels_src,
        boxes_rc=boxes_rc,
        image_tensor=image_tensor,
        image_id=image_id,
        use_tv_tensors=True
    )


def apply_transform(
    image_tensor: torch.Tensor,
    target: dict[str, torch.Tensor | BoundingBoxes],
    transform: Callable
) -> tuple[torch.Tensor, dict[str, torch.Tensor | BoundingBoxes]]:
    """
    Apply a TorchVision v2 transform to image and target together.

    The transform synchronizes geometry changes between the image and bounding boxes.
    Labels are preserved unchanged (including BG0 shift from target building).

    Parameters
    ----------
    image_tensor
        torch tensor with shape (C, H, W), dtype torch.float32
    target
        TorchVision target dict with BoundingBoxes
    transform
        Callable that takes (image, target) and returns (image, target)

    Returns
    -------
    transformed_image
        Transformed image tensor with same shape (C, H, W)
    transformed_target
        Transformed target dict with updated boxes
    """
    return transform(image_tensor, target)


def target_to_annotation_format(
    target: dict[str, torch.Tensor | BoundingBoxes],
    canvas_height: int,
    canvas_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert TorchVision target back to source annotation format.

    Parameters
    ----------
    target
        Transformed target dict with BoundingBoxes in XYXY format
    canvas_height
        Height of the image canvas
    canvas_width
        Width of the image canvas

    Returns
    -------
    labels_project
        Project labels (0, 1, 2) - BG0 labels converted back
    boxes_rc
        Boxes in source format: [row_min, col_min, row_max, col_max]
    """
    boxes_xyxy = target["boxes"].detach().cpu().numpy()
    labels_bg0 = target["labels"].detach().cpu().numpy()

    # Convert BG0 labels back to project labels
    labels_project = labels_bg0 - 1

    # Convert XYXY back to row/col format
    boxes_rc = xyxy_to_rows_cols(boxes_xyxy)

    return labels_project.astype(np.int64), boxes_rc.astype(np.float32)


def save_augmented_sample(
    image_tensor: torch.Tensor,
    target: dict[str, torch.Tensor | BoundingBoxes],
    output_npy_path: str | Path,
    output_annotation_path: str | Path,
    canvas_size: tuple[int, int]
) -> None:
    """
    Save augmented image and annotation to disk.

    Parameters
    ----------
    image_tensor
        Augmented image tensor (C, H, W)
    target
        Augmented target dict
    output_npy_path
        Path to save the augmented .npy file
    output_annotation_path
        Path to save the augmented annotation .txt file
    canvas_size
        (height, width) of the augmented image
    """
    output_npy_path = Path(output_npy_path)
    output_annotation_path = Path(output_annotation_path)

    # Ensure parent directories exist
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    output_annotation_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert target back to annotation format
    labels_project, boxes_rc = target_to_annotation_format(
        target, canvas_size[0], canvas_size[1]
    )

    # Save annotation
    write_annotation_txt_rc(
        labels_src=labels_project,
        boxes_rc=boxes_rc,
        output_path=output_annotation_path
    )

    # Save image as .npy
    image_np = image_tensor.detach().cpu().numpy().astype(np.float32)
    np.save(output_npy_path, image_np)
