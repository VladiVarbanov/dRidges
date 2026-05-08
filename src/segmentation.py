# src/segmentation.py

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .utilities import (
    save_outputs_with_metadata,
    SAVE_NPY,
    SAVE_TIFF2D_F32,
    SAVE_RGBA_TIFF,
)

# Optional: AtomAI
try:
    import atomai
    HAS_ATOMAI = True
except ImportError:
    HAS_ATOMAI = False


# -----------------------------
# Segmentation primitives
# -----------------------------

def segment_otsu(img: np.ndarray, min_size: int = 20) -> np.ndarray:
    thresh = filters.threshold_otsu(img)
    binary = img > thresh
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    return binary.astype(np.uint8)


def segment_sam(img: np.ndarray) -> np.ndarray:
    if not HAS_ATOMAI:
        raise RuntimeError("AtomAI is not installed")

    analyzer = atomai.models.ParticleAnalyzer(model_type="vit_h", device="auto")
    result = analyzer.analyze(img)
    return result["mask"].astype(np.uint8)


def split_overlapping(binary: np.ndarray) -> np.ndarray:
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        distance,
        labels=binary,
        footprint=np.ones((3, 3)),
        exclude_border=False,
    )

    markers = np.zeros_like(distance, dtype=int)
    if coords.size > 0:
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    return watershed(-distance, markers, mask=binary)


# -----------------------------
# High-level segmentation
# -----------------------------

def run_segmentation(
    img: np.ndarray,
    input_path,
    method: str = "otsu",
    min_size: int = 20,
    save_outputs: bool = False, # save will happen throught  pipeline
):
    """
    Run segmentation and optionally save results with metadata.
    """

    if method == "otsu":
        binary = segment_otsu(img, min_size=min_size)
    elif method == "sam":
        binary = segment_sam(img)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    labels_ws = split_overlapping(binary)

    if save_outputs:
        # Binary mask
        save_outputs_with_metadata(
            binary,
            input_path,
            config={
                "stage": "segmentation_binary",
                "method": method,
                "min_size": min_size,
            },
            modes=[SAVE_NPY, SAVE_TIFF2D_F32, SAVE_RGBA_TIFF],
        )

        # Label image
        save_outputs_with_metadata(
            labels_ws.astype(np.float32),
            input_path,
            config={
                "stage": "segmentation_labels",
                "method": method,
            },
            modes=[SAVE_NPY, SAVE_TIFF2D_F32, SAVE_RGBA_TIFF],
        )

    return binary, labels_ws
