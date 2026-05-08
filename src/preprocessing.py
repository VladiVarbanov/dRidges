# src/preprocessing.py
import cv2
import numpy as np
from skimage import exposure, filters
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from sympy import false
from configs.config import EPS, LOCAL_NORM_SIGMA, GAUSSIAN_SMOOTH_SIGMA
from .utilities import (
    save_outputs_with_metadata,
    SAVE_NPY,
    SAVE_TIFF2D_F32,
    SAVE_RGBA_TIFF,
)

# -----------------------------
# Core preprocessing primitives
# -----------------------------

def to_gray_normalized(img: np.ndarray) -> np.ndarray:
    """
    Convert input image to grayscale float32 normalized to [0, 1].
    """
    gray = None

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        ch = img.shape[-1]
        if ch == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif ch == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    if gray is None:
        raise ValueError(
            f"Unsupported image shape={img.shape}, dtype={img.dtype}"
        )

    # Normalize. Add EPS to zero values. Will filter if needed
    gray = cv2.normalize(gray, None, EPS, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    return gray

def local_normalize_HOG_style(
    I_gray: np.ndarray,
    *,
    local_norm_sigma: float = LOCAL_NORM_SIGMA,
    eps: float = EPS,
) -> np.ndarray:
    I = np.asarray(I_gray, dtype=np.float32)

    if not np.isfinite(I).all():
        raise ValueError("Image contains NaN or Inf")

    mu = gaussian(
        I,
        sigma=local_norm_sigma,
        preserve_range=True,
    )
    mu2 = gaussian(
        I * I,
        sigma=local_norm_sigma,
        preserve_range=True,
    )

    var_local = np.maximum(mu2 - mu * mu, 0.0)
    sigma_local = np.sqrt(var_local)

    I_hat = (I - mu) / (sigma_local + eps)
    out = rescale_intensity(I_hat, in_range="image", out_range=(0.0, 1.0))

    return np.asarray(out, dtype=np.float32)


def local_contrast_normalization_CLAHE(img: np.ndarray) -> np.ndarray:
    """
    Gamma + CLAHE contrast normalization.
    """
    # sanity: finite values only
    if not np.isfinite(img).all():
        raise ValueError("Image contains NaN or Inf before local contrast normalization")

    mx = img.max()
    mn = img.min()
    # sanity: image is not degenerate
    if (mx - mn) < EPS:
        # flat or empty image → nothing to enhance
        raise ValueError("Image is Flat, (min={mn:.3e}, max={mx:.3e}, EPS={EPS})")

    else:
        img_gamma = np.sqrt(img)
    return exposure.equalize_adapthist(img_gamma, clip_limit=0.01)


def gaussian_smoothing(img: np.ndarray) -> np.ndarray:
    return filters.gaussian(img, sigma=GAUSSIAN_SMOOTH_SIGMA)


# -----------------------------
# High-level preprocessing step
# -----------------------------

