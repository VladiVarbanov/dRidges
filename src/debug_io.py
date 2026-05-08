# src/debug_io.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile


def save_rgba_tiff(image_rgba: np.ndarray, path: str | Path) -> None:
    """
    Save an RGBA uint8 image as TIFF.
    - image_rgba: (H, W, 4) uint8
    - path: output .tif/.tiff
    Debug-only: no metadata, no sidecar files.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(image_rgba)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"Expected (H,W,4) RGBA, got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGBA, got {arr.dtype}")

    # photometric="rgb" + extrasamples marks the 4th channel as alpha
    tifffile.imwrite(
        str(p),
        arr,
        photometric="rgb",
        planarconfig="contig",
        extrasamples= (2,)#["unassociated alpha"],
    )
