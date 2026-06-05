from __future__ import annotations

import json
from pathlib import Path

import torch

from configs.config import (
    NN_IMAGE_DIR,
    NN_ANNOTATION_DIR,
    NN_TRAIN_SPLIT_TXT,
    NN_EXPECTED_NUM_CHANNELS,
)

from src.annotation_io import pair_split_images_with_annotations

from src.utilities import load_image

from src.nn_input_prepare import (
    build_nn_gray_channel,
    build_nn_hog_norm_channel,
    build_nn_vesselness_channel,
    build_nn_input_img,
)

from src.torch_vision_dataset import TorchVisionDataset


CHANNEL_FNS = [
    build_nn_gray_channel,
    build_nn_hog_norm_channel,
    build_nn_vesselness_channel,
]


def test_torchvision_dataset_smoke(tmp_path: Path) -> None:
    # ------------------------------------------------------------
    # 1) Pick one real image/annotation pair
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
    # 2) Build one temporary NN input cache file
    # ------------------------------------------------------------
    npy_dir = tmp_path / "nn_input_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    img_raw = load_image(image_path)

    nn_input_image = build_nn_input_img(
        img_raw=img_raw,
        channel_fns=CHANNEL_FNS,
    )

    assert nn_input_image.ndim == 3
    assert nn_input_image.shape[0] == NN_EXPECTED_NUM_CHANNELS

    npy_path = npy_dir / f"{image_path.stem}.npy"

    import numpy as np

    np.save(npy_path, nn_input_image)

    assert npy_path.exists()

    # ------------------------------------------------------------
    # 3) Write temporary source annotation-format JSON
    # ------------------------------------------------------------
    annotation_format_path = tmp_path / "annotation_format.json"

    annotation_format = {
        "annotation_format": "class row_min col_min row_max col_max",
        "delimiter": "whitespace",
        "has_header": False,
        "class_column": 0,
        "box_columns": [1, 2, 3, 4],
        "source_box_format": "ROW_COL_MINMAX",
        "source_label_base": 0,
        "class_names": {
            "0": "a0_half_111_loop",
            "1": "a0_100_loop",
            "2": "black_dot",
            "3": "other_defect",
        },
    }

    with annotation_format_path.open("w", encoding="utf-8") as f:
        json.dump(annotation_format, f, indent=4, ensure_ascii=False)

    assert annotation_format_path.exists()

    # ------------------------------------------------------------
    # 4) Construct dataset and read first item
    # ------------------------------------------------------------
    dataset = TorchVisionDataset(
        split_path=NN_TRAIN_SPLIT_TXT,
        image_dir=NN_IMAGE_DIR,
        annotation_dir=NN_ANNOTATION_DIR,
        npy_dir=npy_dir,
        annotation_format_path=annotation_format_path,
        expected_num_channels=NN_EXPECTED_NUM_CHANNELS,
        use_tv_tensors=False,
        max_images=1,
    )

    assert len(dataset) == 1

    image_tensor, target = dataset[0]

    # ------------------------------------------------------------
    # 5) Check image tensor
    # ------------------------------------------------------------
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.ndim == 3
    assert image_tensor.shape[0] == NN_EXPECTED_NUM_CHANNELS
    assert image_tensor.dtype == torch.float32
    assert torch.isfinite(image_tensor).all()

    # ------------------------------------------------------------
    # 6) Check target dictionary
    # ------------------------------------------------------------
    expected_target_keys = {
        "boxes",
        "labels",
        "image_id",
        "area",
        "iscrowd",
    }

    assert set(target.keys()) == expected_target_keys

    boxes = target["boxes"]
    labels = target["labels"]
    image_id = target["image_id"]
    area = target["area"]
    iscrowd = target["iscrowd"]

    assert isinstance(boxes, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(image_id, torch.Tensor)
    assert isinstance(area, torch.Tensor)
    assert isinstance(iscrowd, torch.Tensor)

    assert boxes.ndim == 2
    assert boxes.shape[1] == 4

    assert labels.ndim == 1
    assert area.ndim == 1
    assert iscrowd.ndim == 1

    assert len(labels) == boxes.shape[0]
    assert len(area) == boxes.shape[0]
    assert len(iscrowd) == boxes.shape[0]

    assert torch.all(labels > 0)
    assert torch.all(area > 0)
    assert torch.all(iscrowd == 0)

    assert image_id.shape == (1,)
    assert int(image_id.item()) == 0