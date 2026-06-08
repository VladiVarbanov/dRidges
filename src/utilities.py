
import numpy as np

from pathlib import Path
import pandas as pd
from numpy import dtype, floating, ndarray
from numpy._typing import _64Bit
from skimage.io import imsave
import json
import hashlib
from dataclasses import dataclass
from skimage.util import img_as_ubyte
from typing import Any, Iterable
import cv2

from configs.config import (
    DATA_DIR,
    INTERMEDIATE_NPY_DIR,
    INTERMEDIATE_TIFF2D_DIR,
    RESULTS_VIS_DIR,
)

# ---- Save modes (vector-like flags) ----
SAVE_NPY = 1
SAVE_TIFF2D_F32 = 2
SAVE_RGBA_TIFF = 4  # visualization
DEFAULT_SAVE_MODES = (SAVE_NPY, SAVE_TIFF2D_F32)  # your default
DEFAULT_iMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg",".tif", ".tiff", ".bmp" )


def _as_modes(modes: int | Iterable[int] | None) -> set[int]:
    """
    Accepts:
      - None -> DEFAULT_SAVE_MODES
      - single int -> {int}
      - iterable -> set(iterable)
    """
    if modes is None:
        return set(DEFAULT_SAVE_MODES)
    if isinstance(modes, int):
        return {modes}
    return set(modes)


def hash_config(config: dict[str, Any], hash_len: int = 10) -> str:
    """
    Stable hash of a config dict.
    - sort_keys=True ensures stable hashing
    - default=str handles Path and other non-JSON types
    """
    s = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:hash_len]


def mirrored_relative_path(input_path: str | Path, base_dir: str | Path = DATA_DIR) -> Path:
    """
    Returns the path of input relative to DATA_DIR (or a user override).
    Example:
      input: data/imgs_tem/run1/a.png -> imgs_tem/run1/a.png
    """
    input_path = Path(input_path)
    base_dir = Path(base_dir)
    try:
        return input_path.relative_to(base_dir)
    except ValueError:
        # Not under DATA_DIR -> fall back to using just the filename
        return Path(input_path.name)


@dataclass(frozen=True)
class OutputPaths:
    npy: Path | None
    tiff2d: Path | None
    rgba: Path | None
    meta: Path  # always written


def build_output_paths(
    input_path: str | Path,
    cfg_hash: str,
    base_dir: str | Path = DATA_DIR,
    name_override: str | None = None,
) -> OutputPaths:
    """
    Mirrors folder structure from DATA_DIR into:
      intermediate/npy/
      intermediate/tiff2d/
      results/vis/

    Output filenames:
      <stem>_<hash>.npy
      <stem>_<hash>.tif
      <stem>_<hash>_rgba.tif
      <stem>_<hash>.json  (sidecar metadata)
    """
    rel = mirrored_relative_path(input_path, base_dir=base_dir)
    rel_parent = rel.parent
    stem = name_override if name_override else Path(rel.name).stem

    npy_path = INTERMEDIATE_NPY_DIR / rel_parent / f"{stem}_{cfg_hash}.npy"
    tiff_path = INTERMEDIATE_TIFF2D_DIR / rel_parent / f"{stem}_{cfg_hash}.tif"
    rgba_path = RESULTS_VIS_DIR / rel_parent / f"{stem}_{cfg_hash}_rgba.tif"
    meta_path = INTERMEDIATE_NPY_DIR / rel_parent / f"{stem}_{cfg_hash}.json"

    return OutputPaths(npy=npy_path, tiff2d=tiff_path, rgba=rgba_path, meta=meta_path)


# --- Paths ---#
def ensure_dir(dir_path: str | Path) -> Path:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

# --- Image IO --- #
# TODO:create iterator helper functions, that take all the images in folder, Preserve structure of folders if there is one ---#
def load_image(input_path: str | Path):
    input_path = Path(input_path)
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    return img
    
# ---- Input scanning ----

DEFAULT_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")



def collect_images_paths(
    input_dir: str | Path,
    recursive: bool = True,
    exts: Iterable[str] = DEFAULT_IMAGE_EXTS,
    max_files: int | None = None,
) -> list[Path]:
    """
    Collect image files from a directory in deterministic order.

    Args:
        input_dir: Directory to scan.
        recursive: If True, recurse into subfolders.
        exts: Allowed file extensions (case-insensitive).
        max_files: Limit number of returned files (after sorting). Use 1 for “one image first”.

    Returns:
        Sorted list of Paths.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    exts_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    paths = [p for p in iterator if p.is_file() and p.suffix.lower() in exts_set]

    # Deterministic order matters for debugging now and threads later
    paths = sorted(paths)

    if max_files is not None:
        if max_files < 0:
            raise ValueError(f"max_files must be >= 0 or None, got {max_files}")
        paths = paths[:max_files]

    return paths


# TODO: create helper that add_names to functions and and create folders
def save_tiff2d_float32(image2d: np.ndarray, output_path: str | Path) -> None:
    """
    Save 2D image as float32 TIFF (portable to C++/OpenCV).
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    arr = np.asarray(image2d)
    if arr.ndim != 2:
        raise ValueError(f"save_tiff2d_float32 expects 2D array, got {arr.shape}")
    arr_f32 = arr.astype(np.float32, copy=False)
    imsave(output_path, arr_f32)


def save_rgba_tiff_from_gray(
    image: np.ndarray,
    output_path: str | Path,
    alpha_value: int = 255,
) -> None:
    """
    Save a 2D grayscale image as 8-bit RGBA TIFF for visualization.

    Expected input:
        - (H, W) grayscale
        - (H, W, 1) grayscale

    Output:
        - (H, W, 4) uint8 RGBA
          R = G = B = gray
          A = alpha_value
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    rgba = rgba_from_gray(image, alpha_value)

    imsave(output_path, rgba)


def rgba_from_gray(image: ndarray, alpha_value: int = 255) ->  np.ndarray:
    arr = np.asarray(image)

    # ---- Validate and normalize shape ----
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(
            f"save_rgba_tiff_from_gray expects 2D grayscale input, "
            f"got shape={arr.shape}, dtype={arr.dtype}"
        )

    # ---- Convert to uint8 safely ----
    if arr.dtype == np.uint8:
        gray_u8 = arr
    elif np.issubdtype(arr.dtype, np.floating):
        # noinspection PyTypeChecker
        amin, amax = np.nanmin(arr), np.nanmax(arr)
        if amax > amin:
            # noinspection PyUnresolvedReferences
            arr01 = (arr - amin) / (amax - amin)
        else:
            arr01 = np.zeros_like(arr, dtype=np.float32)
        gray_u8 = img_as_ubyte(arr01)
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr01 = arr.astype(np.float32) / float(info.max)
        gray_u8 = img_as_ubyte(arr01)
    else:
        raise ValueError(f"Unsupported dtype for visualization: {arr.dtype}")

    # ---- Expand to RGBA ----
    h, w = gray_u8.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = gray_u8
    rgba[..., 1] = gray_u8
    rgba[..., 2] = gray_u8
    rgba[..., 3] = alpha_value
    return rgba


def write_metadata_json(
    meta_path: str | Path,
    input_path: str | Path,
    config: dict[str, Any],
    cfg_hash: str,
    outputs: OutputPaths,
) -> None:
    meta_path = Path(meta_path)
    ensure_dir(meta_path.parent)

    payload = {
        "input": str(Path(input_path)),
        "hash": cfg_hash,
        "config": config,
        "outputs": {
            "npy": str(outputs.npy) if outputs.npy else None,
            "tiff2d": str(outputs.tiff2d) if outputs.tiff2d else None,
            "rgba": str(outputs.rgba) if outputs.rgba else None,
            "meta": str(outputs.meta),
        },
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False, default=str)


def save_outputs_with_metadata(
    image2d: np.ndarray,
    input_path: str | Path,
    config: dict[str, Any],
    modes: int | Iterable[int] | None = None,
    base_dir: str | Path = DATA_DIR,
    name_override: str | None = None,
) -> OutputPaths:
    """
    Saves selected outputs AND always writes a JSON sidecar metadata file.
    Default: SAVE_NPY + SAVE_TIFF2D_F32
    Optional: SAVE_RGBA_TIFF

    Mirrors folder structure relative to base_dir (default DATA_DIR).
    """
    modes_set = _as_modes(modes)
    cfg_hash = hash_config(config)

    outputs = build_output_paths(
        input_path=input_path,
        cfg_hash=cfg_hash,
        base_dir=base_dir,
        name_override=name_override,
    )

    # Save .npy
    if SAVE_NPY in modes_set:
        if outputs.npy is None:
            raise ValueError("Internal error: npy path is None")
        arr = np.asarray(image2d)
        if arr.ndim != 2:
            raise ValueError(f"SAVE_NPY expects 2D array, got {arr.shape}")
        ensure_dir(outputs.npy.parent)
        np.save(outputs.npy, arr.astype(np.float32, copy=False))

    # Save float32 TIFF (2D)
    if SAVE_TIFF2D_F32 in modes_set:
        if outputs.tiff2d is None:
            raise ValueError("Internal error: tiff path is None")
        save_tiff2d_float32(image2d, outputs.tiff2d)

    # Save RGBA TIFF (visualization)
    if SAVE_RGBA_TIFF in modes_set:
        if outputs.rgba is None:
            raise ValueError("Internal error: rgba path is None")
        save_rgba_tiff_from_gray(image2d, outputs.rgba)

    # Always write metadata JSON
    write_metadata_json(
        outputs.meta,
        input_path=input_path,
        config=config,
        cfg_hash=cfg_hash,
        outputs=outputs,
    )

    return outputs


# --- Loop parameter saving --- #

def save_processed_parameters(params_df: pd.DataFrame, path: str | Path):
    """
    params_df: DataFrame with columns like:
    ['id', 'center_x', 'center_y', 'major_axis', 'minor_axis',
     'orientation_deg', 'area_px', 'is_overlapping', 'is_concentric',  'id_overlapping', 'id_concentric']
    """
    path = Path(path)
    ensure_dir(path.parent)
    params_df.to_csv(path, index=False)


def load_csv_table(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def save_nn_input_npy_channels_as_tiffs(
            npy_path: str | Path,
            output_dir: str | Path,
    ) -> list[Path]:
        """
        Load one cached NN input .npy file and save each channel as a float32 TIFF.

        Expected cached array format:
            (C, H, W), dtype float32

        This is only a debug / inspection helper.
        It does not modify the cache.
        """
        npy_path = Path(npy_path)
        output_dir = ensure_dir(output_dir)

        arr = np.load(npy_path)

        if arr.ndim != 3:
            raise ValueError(f"Expected cached NN input shape (C,H,W), got {arr.shape}")

        if not np.isfinite(arr).all():
            raise ValueError(f"{npy_path}: contains NaN or Inf")

        saved_paths: list[Path] = []

        for ch_id, channel in enumerate(arr):
            out_path = output_dir / f"{npy_path.stem}_ch{ch_id}.tif"
            save_tiff2d_float32(channel, out_path)
            saved_paths.append(out_path)

        return saved_paths