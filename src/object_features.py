# src/object_features.py
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


from configs.object_features_config import (
    EPS,
    AREA_LOG_EPS,
    MIN_VALID_BOX_AREA,
    AREA_PERCENTILE_EPS,
    AREA_HISTOGRAM_NUM_BINS,
    LOG_AREA_HISTOGRAM_NUM_BINS,
    AREA_HISTOGRAM_PEAK_PROMINENCE,
    AREA_HISTOGRAM_VALLEY_PROMINENCE,
    AREA_HISTOGRAM_MIN_DISTANCE_BINS, LOG_AREA_GMM_COMPONENTS, LOG_AREA_GMM_COVARIANCE_TYPE, LOG_AREA_GMM_RANDOM_STATE,

)

def compute_box_geometry_rc(boxes_rc: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Compute valid box geometry features from RC minmax boxes.

    Input format:
        [row_min, col_min, row_max, col_max]

    Invalid boxes are discarded.

    Returns:
        features
        keep_mask
    """
    boxes_rc = np.asarray(boxes_rc, dtype=np.float32)

    if boxes_rc.ndim != 2 or boxes_rc.shape[1] != 4:
        raise ValueError(
            f"boxes_rc must have shape (N, 4), got {boxes_rc.shape}"
        )

    row_min = boxes_rc[:, 0]
    col_min = boxes_rc[:, 1]
    row_max = boxes_rc[:, 2]
    col_max = boxes_rc[:, 3]

    height = row_max - row_min
    width = col_max - col_min
    area = width * height

    keep_mask = (
        np.isfinite(boxes_rc).all(axis=1)
        & (width > 0.0)
        & (height > 0.0)
        & (area >= MIN_VALID_BOX_AREA)
    )

    row_min = row_min[keep_mask]
    col_min = col_min[keep_mask]
    row_max = row_max[keep_mask]
    col_max = col_max[keep_mask]
    width = width[keep_mask]
    height = height[keep_mask]
    area = area[keep_mask]

    aspect_ratio = np.maximum(width, height) / np.minimum(width, height)
    sqrt_area = np.sqrt(area)
    log_area = np.log(area + AREA_LOG_EPS)

    center_row = 0.5 * (row_min + row_max)
    center_col = 0.5 * (col_min + col_max)

    features = {
        "row_min": row_min.astype(np.float32, copy=False),
        "col_min": col_min.astype(np.float32, copy=False),
        "row_max": row_max.astype(np.float32, copy=False),
        "col_max": col_max.astype(np.float32, copy=False),
        "width": width.astype(np.float32, copy=False),
        "height": height.astype(np.float32, copy=False),
        "area": area.astype(np.float32, copy=False),
        "sqrt_area": sqrt_area.astype(np.float32, copy=False),
        "log_area": log_area.astype(np.float32, copy=False),
        "aspect_ratio": aspect_ratio.astype(np.float32, copy=False),
        "center_row": center_row.astype(np.float32, copy=False),
        "center_col": center_col.astype(np.float32, copy=False),
    }

    return features, keep_mask


def compute_area_histogram(
    area: np.ndarray,
    *,
    bins: int = AREA_HISTOGRAM_NUM_BINS,
) -> dict[str, np.ndarray]:
    """
    Compute histogram of already-valid box areas.

    Input:
        area: 1D array of box areas

    Returns:
        counts
        bin_edges
        bin_centers
    """
    area = np.asarray(area, dtype=np.float32)

    if area.ndim != 1:
        raise ValueError(f"area must have shape (N,), got {area.shape}")

    counts, bin_edges = np.histogram(area, bins=bins)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "counts": counts.astype(np.int64, copy=False),
        "bin_edges": bin_edges.astype(np.float32, copy=False),
        "bin_centers": bin_centers.astype(np.float32, copy=False),
    }


def compute_log_area_histogram(
    area: np.ndarray,
    *,
    bins: int = LOG_AREA_HISTOGRAM_NUM_BINS,
) -> dict[str, np.ndarray]:
    """
    Compute histogram of log-transformed box areas.

    Input:
        area: 1D array of already-valid box areas

    Returns:
        counts
        bin_edges
        bin_centers
        log_area
    """

    area = np.asarray(area, dtype=np.float32)

    if area.ndim != 1:
        raise ValueError(...)

    log_area = np.log(area + AREA_LOG_EPS)

    counts, bin_edges = np.histogram(log_area, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "counts": counts.astype(np.int64, copy=False),
        "bin_edges": bin_edges.astype(np.float32, copy=False),
        "bin_centers": bin_centers.astype(np.float32, copy=False),
        "log_area": log_area.astype(np.float32, copy=False),
    }



def find_area_histogram_peaks(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    *,
    prominence: float = AREA_HISTOGRAM_PEAK_PROMINENCE,
    distance: int = AREA_HISTOGRAM_MIN_DISTANCE_BINS,
) -> dict[str, np.ndarray]:
    """
    Find peaks in an area/log-area histogram.
    """
    counts = np.asarray(counts)
    bin_centers = np.asarray(bin_centers, dtype=np.float32)

    if counts.ndim != 1:
        raise ValueError(f"counts must have shape (N,), got {counts.shape}")
    if bin_centers.ndim != 1:
        raise ValueError(f"bin_centers must have shape (N,), got {bin_centers.shape}")
    if len(counts) != len(bin_centers):
        raise ValueError("counts and bin_centers must have the same length")

    peak_indices, properties = find_peaks(
        counts,
        prominence=prominence,
        distance=distance,
    )

    return {
        "peak_indices": peak_indices.astype(np.int64, copy=False),
        "peak_centers": bin_centers[peak_indices].astype(np.float32, copy=False),
        "peak_counts": counts[peak_indices].astype(np.int64, copy=False),
        "prominences": properties["prominences"].astype(np.float32, copy=False),
    }



def find_area_histogram_valleys(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    *,
    prominence: float = AREA_HISTOGRAM_VALLEY_PROMINENCE,
    distance: int = AREA_HISTOGRAM_MIN_DISTANCE_BINS,
) -> dict[str, np.ndarray]:
    """
    Find valleys in an area/log-area histogram.

    Valleys are peaks in the negative histogram counts.
    """
    counts = np.asarray(counts)
    bin_centers = np.asarray(bin_centers, dtype=np.float32)

    if counts.ndim != 1:
        raise ValueError(f"counts must have shape (N,), got {counts.shape}")
    if bin_centers.ndim != 1:
        raise ValueError(f"bin_centers must have shape (N,), got {bin_centers.shape}")
    if len(counts) != len(bin_centers):
        raise ValueError("counts and bin_centers must have the same length")

    valley_indices, properties = find_peaks(
        -counts,
        prominence=prominence,
        distance=distance,
    )

    return {
        "valley_indices": valley_indices.astype(np.int64, copy=False),
        "valley_centers": bin_centers[valley_indices].astype(np.float32, copy=False),
        "valley_counts": counts[valley_indices].astype(np.int64, copy=False),
        "prominences": properties["prominences"].astype(np.float32, copy=False),
    }


def fit_log_area_distribution(
    area: np.ndarray,
    *,
    n_components: int = LOG_AREA_GMM_COMPONENTS,
    random_state: int = LOG_AREA_GMM_RANDOM_STATE,
    covariance_type: str = LOG_AREA_GMM_COVARIANCE_TYPE,
) -> GaussianMixture:
    """
    Fit a Gaussian mixture model to log-transformed box areas.

    Input:
        area: 1D array of already-valid box areas

    Returns:
        fitted GaussianMixture model
    """
    area = np.asarray(area, dtype=np.float32)

    if area.ndim != 1:
        raise ValueError(f"area must have shape (N,), got {area.shape}")

    if len(area) < n_components:
        raise ValueError(
            f"Need at least {n_components} areas to fit GMM, got {len(area)}"
        )

    log_area = np.log(area + AREA_LOG_EPS).reshape(-1, 1)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )

    model.fit(log_area)

    return model

