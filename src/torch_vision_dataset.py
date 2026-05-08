from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
import numpy as np
import torch

from configs.config import (
    NN_ANN_EXT,
    NN_EXPECTED_NUM_CHANNELS,
)

from .nn_input_prepare import (
    pair_split_images_with_annotations, parse_annotation_txt_rc,
)

from torchvision import tv_tensors

from .nn_adapters import (
    rows_cols_to_xyxy,
    label_ids_to_bg0_format,
)


def load_annotation_format(annotation_format_path: str | Path) -> dict[str, Any]:
    """
    Load the dataset-level annotation format JSON.

    This file describes how the source annotation txt files should be read.
    It does not describe TorchVision targets.
    """
    annotation_format_path = Path(annotation_format_path)

    if not annotation_format_path.exists():
        raise FileNotFoundError(
            f"Annotation format JSON does not exist: {annotation_format_path}"
        )

    if not annotation_format_path.is_file():
        raise ValueError(
            f"Annotation format path is not a file: {annotation_format_path}"
        )

    with annotation_format_path.open("r", encoding="utf-8") as f:
        annotation_format = json.load(f)

    if not isinstance(annotation_format, dict):
        raise ValueError(
            f"Annotation format JSON must contain an object/dict, "
            f"got {type(annotation_format).__name__}"
        )

    return annotation_format

def validate_source_annotation_format(annotation_format: dict[str, Any]) -> None:    
        """
        Validate that the source annotation format is supported by the current parser.

        Current supported source annotation format:

            class row_min col_min row_max col_max

        Internally, parse_annotation_txt_rc expects:
            labels    from column 0
            boxes_rc  from columns [1, 2, 3, 4]
            boxes_rc order = [row_min, col_min, row_max, col_max]
        """
        required = {
            "delimiter",
            "has_header",
            "class_column",
            "box_columns",
            "source_box_format",
            "source_label_base",
        }

        missing = sorted(required - set(annotation_format.keys()))
        if missing:
            raise ValueError(
                f"Annotation format is missing required fields: {missing}"
            )

        if annotation_format["delimiter"] != "whitespace":
            raise ValueError(
                f"Only whitespace-delimited annotations are supported for now, "
                f"got {annotation_format['delimiter']!r}"
            )

        if bool(annotation_format["has_header"]) is not False:
            raise ValueError(
                f"Only headerless annotation txt files are supported for now, "
                f"got has_header={annotation_format['has_header']!r}"
            )

        if int(annotation_format["class_column"]) != 0:
            raise ValueError(
                f"Current parser expects class_column=0, "
                f"got {annotation_format['class_column']!r}"
            )

        if list(annotation_format["box_columns"]) != [1, 2, 3, 4]:
            raise ValueError(
                f"Current parser expects box_columns=[1, 2, 3, 4], "
                f"got {annotation_format['box_columns']!r}"
            )

        if annotation_format["source_box_format"] != "ROW_COL_MINMAX":
            raise ValueError(
                f"Current parser expects source_box_format='ROW_COL_MINMAX', "
                f"got {annotation_format['source_box_format']!r}"
            )

        if int(annotation_format["source_label_base"]) != 0:
            raise ValueError(
                f"Current label adapter expects source_label_base=0, "
                f"got {annotation_format['source_label_base']!r}"
            )


def npy_to_torch_tensor(    
            npy_path: str | Path,
            *,
            expected_num_channels: int | None = NN_EXPECTED_NUM_CHANNELS,
    ) -> torch.Tensor:
        """
        Load one cached stacked NN input .npy file and convert it to a torch tensor.

        Expected cached format:
            shape = (C, H, W)
            dtype = float32 or float32-convertible

        Returns
        -------
        image_tensor
            torch.float32 tensor with shape (C, H, W)
        """
        npy_path = Path(npy_path)

        if not npy_path.exists():
            raise FileNotFoundError(f"Cached .npy file does not exist: {npy_path}")

        if not npy_path.is_file():
            raise ValueError(f"Cached .npy path is not a file: {npy_path}")

        arr = np.load(npy_path)

        if arr.ndim != 3:
            raise ValueError(
                f"Expected cached NN input shape (C, H, W), got {arr.shape} "
                f"from {npy_path}"
            )

        if expected_num_channels is not None:
            if arr.shape[0] != int(expected_num_channels):
                raise ValueError(
                    f"Expected {expected_num_channels} channels, got {arr.shape[0]} "
                    f"from {npy_path}"
                )

        if not np.isfinite(arr).all():
            raise ValueError(f"Cached .npy contains NaN or Inf: {npy_path}")

        arr = arr.astype(np.float32, copy=False)

        image_tensor = torch.as_tensor(arr, dtype=torch.float32)

        return image_tensor


def build_torchvision_target( #TODO: read the source annotation format and then choose the rigtht adapter
    *,
    labels_src: np.ndarray,
    boxes_rc: np.ndarray,
    image_tensor: torch.Tensor,
    image_id: int,
    use_tv_tensors: bool = True,
) -> dict[str, torch.Tensor | tv_tensors.BoundingBoxes]:
    """
    Build one TorchVision detection target from source annotation arrays.

    Source annotation format:
        labels_src: foreground labels starting at 0
        boxes_rc:   [row_min, col_min, row_max, col_max]

    TorchVision target format:
        boxes:    [x_min, y_min, x_max, y_max]
        labels:   foreground labels starting at 1, because 0 is background
        image_id: tensor([image_id])
        area:     box area
        iscrowd:  zeros
    """
    labels_src = np.asarray(labels_src, dtype=np.int64)
    boxes_rc = np.asarray(boxes_rc, dtype=np.float32)

    if labels_src.ndim != 1:
        raise ValueError(
            f"labels_src must have shape (N,), got {labels_src.shape}"
        )

    if boxes_rc.ndim != 2 or boxes_rc.shape[1] != 4:
        raise ValueError(
            f"boxes_rc must have shape (N, 4), got {boxes_rc.shape}"
        )

    if len(labels_src) != len(boxes_rc):
        raise ValueError(
            f"labels_src and boxes_rc must have the same length, "
            f"got {len(labels_src)} and {len(boxes_rc)}"
        )

    boxes_xyxy_np = rows_cols_to_xyxy(boxes_rc)
    labels_bg0_np = label_ids_to_bg0_format(labels_src)

    boxes_tensor = torch.as_tensor(boxes_xyxy_np, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels_bg0_np, dtype=torch.int64)

    x_min = boxes_tensor[:, 0]
    y_min = boxes_tensor[:, 1]
    x_max = boxes_tensor[:, 2]
    y_max = boxes_tensor[:, 3]

    box_widths = x_max - x_min
    box_heights = y_max - y_min
    box_area = box_widths * box_heights

    if torch.any(box_widths <= 0):
        raise ValueError("All boxes must have positive width after XYXY conversion")

    if torch.any(box_heights <= 0):
        raise ValueError("All boxes must have positive height after XYXY conversion")


    iscrowd = torch.zeros((len(boxes_tensor),), dtype=torch.int64)
    image_id_tensor = torch.tensor([int(image_id)], dtype=torch.int64)

    _, image_height, image_width = image_tensor.shape
    if use_tv_tensors:
        boxes_out = tv_tensors.BoundingBoxes(
            boxes_tensor,
            format="XYXY",
            canvas_size=(int(image_height), int(image_width)),
        )
    else:
        boxes_out = boxes_tensor

    target = {
        "boxes": boxes_out,
        "labels": labels_tensor,
        "image_id": image_id_tensor,
        "area": box_area,
        "iscrowd": iscrowd,
    }

    return target



class TorchVisionDataset(Dataset):
    """
    TorchVision object-detection dataset backed by the prepared NN input cache.

    The dataset is split-driven. The split file is the source of truth.
    """

    def __init__(
        self,
        *,
        split_path: str | Path,
        image_dir: str | Path,
        annotation_dir: str | Path,
        npy_dir: str | Path,
        annotation_format_path: str | Path,
        ann_ext: str = NN_ANN_EXT,
        expected_num_channels: int | None = NN_EXPECTED_NUM_CHANNELS,
        use_tv_tensors: bool = True,
        max_images: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        split_path
            Path to trainimages.txt / testimages.txt / allimages.txt.

        image_dir
            Directory with original source images.

        annotation_dir
            Directory with source annotation txt files.

        npy_dir
            Directory with prepared stacked NN input cache:
                <image_stem>.npy

        annotation_format_path
            Dataset-level JSON file describing the source annotation format.

        ann_ext
            Annotation extension, usually ".txt".

        expected_num_channels
            Expected channel count in cached .npy files.
            Use None to disable the channel-count check.

        use_tv_tensors
            If True, targets will later use tv_tensors.BoundingBoxes.
            If False, targets will use plain torch.Tensor boxes.

        max_images
            Optional smoke-test limit.
        """
        self.split_path = Path(split_path)
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.npy_dir = Path(npy_dir)
        self.annotation_format_path = Path(annotation_format_path)

        self.ann_ext = ann_ext
        self.expected_num_channels = expected_num_channels
        self.use_tv_tensors = bool(use_tv_tensors)
        self.max_images = max_images

        if not self.npy_dir.exists():
            raise FileNotFoundError(
                f"NN input cache directory does not exist: {self.npy_dir}"
            )

        if not self.annotation_format_path.exists():
            raise FileNotFoundError(
                f"Annotation format JSON does not exist: {self.annotation_format_path}"
            )

        if not self.annotation_format_path.is_file():
            raise ValueError(
                f"Annotation format path is not a file: {self.annotation_format_path}"
            )

        with self.annotation_format_path.open("r", encoding="utf-8") as f:
            self.annotation_format: dict[str, Any] = json.load(f)

        self.pairs = pair_split_images_with_annotations(
            split_path=self.split_path,
            image_dir=self.image_dir,
            annotation_dir=self.annotation_dir,
            ann_ext=self.ann_ext,
            require_annotation=True,
            max_images=self.max_images,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, annotation_path = self.pairs[idx]

        if annotation_path is None:
            raise RuntimeError(
                f"Internal error: annotation_path is None for image {image_path}"
            )

        npy_path = self.npy_dir / f"{image_path.stem}.npy"

        image_tensor = npy_to_torch_tensor(
            npy_path,
            expected_num_channels=self.expected_num_channels,
        )

        labels_src, boxes_rc = parse_annotation_txt_rc(annotation_path)

        target = build_torchvision_target(
            labels_src=labels_src,
            boxes_rc=boxes_rc,
            image_tensor=image_tensor,
            image_id=idx,
            use_tv_tensors=self.use_tv_tensors,
        )

        return image_tensor, target


