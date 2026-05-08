from __future__ import annotations

from pathlib import Path

from configs.config import NN_ANN_EXT
from . utilities import (
    load_image,
    collect_images_paths,
    rgba_from_gray,
    ensure_dir,
    save_outputs_with_metadata,
    SAVE_NPY,
    SAVE_TIFF2D_F32,
)

from .debug_io import save_rgba_tiff

from . nn_input_prepare import (
    build_nn_input_img,
    parse_annotation_txt_rc)
from nn_adapters import rows_cols_to_xywh

from .visualization import paint_labeled_xywh_boxes_in_place


def prepare_small_nn_input_cache_test(
    image_dir: str | Path,
    *,
    channel_fns: list,
    max_images: int = 5,
    recursive: bool = True,
) -> list[dict]:
    """
    Build NN input images for a small number of files and save each channel.

    Saves through existing utilities:
        - .npy
        - float32 .tif
        - metadata .json

    This is only a smoke-test / inspection function.
    """
    image_paths = collect_images_paths(
        input_dir=image_dir,
        recursive=recursive,
        max_files=max_images,
    )

    records: list[dict] = []

    for image_path in image_paths:
        img_raw = load_image(image_path)

        nn_input_image = build_nn_input_img(
            img_raw=img_raw,
            channel_fns=channel_fns,
        )

        saved_channels = []

        for ch_id, channel in enumerate(nn_input_image):
            out = save_outputs_with_metadata(
                image2d=channel,
                input_path=image_path,
                config={
                    "stage": "nn_input_cache_test",
                    "channel_id": ch_id,
                    "channel_fns": [
                        getattr(fn, "__name__", repr(fn))
                        for fn in channel_fns
                    ],
                },
                modes=[SAVE_NPY, SAVE_TIFF2D_F32],
                name_override=f"{image_path.stem}_nn_ch{ch_id}",
            )

            saved_channels.append(out)

        records.append(
            {
                "image_path": image_path,
                "shape": nn_input_image.shape,
                "dtype": str(nn_input_image.dtype),
                "saved_channels": saved_channels,
            }
        )

    return records
