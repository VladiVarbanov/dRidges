from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from configs.object_features_config import (
    GMM_N_LABELS,
    GMM_COVARIANCE_TYPE,
    GMM_RANDOM_STATE,
    GMM_N_INIT,
)

from .annotation_registry import probability_vector_to_str


def make_feature_matrix(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    require_finite: bool = True,
) -> np.ndarray:
    """
    Extract selected feature columns from a DataFrame as a 2D float matrix.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"table must be a pandas DataFrame, got {type(table).__name__}"
        )

    if not feature_columns:
        raise ValueError("feature_columns must not be empty")

    feature_columns = tuple(feature_columns)

    missing_columns = [
        column
        for column in feature_columns
        if column not in table.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    feature_matrix = table.loc[:, feature_columns].to_numpy(dtype=np.float64)

    if feature_matrix.ndim != 2:
        raise ValueError(
            f"feature_matrix must be 2D, got shape {feature_matrix.shape}"
        )

    if feature_matrix.shape[0] == 0:
        raise ValueError("feature_matrix must contain at least one sample")

    if feature_matrix.shape[1] == 0:
        raise ValueError("feature_matrix must contain at least one feature")

    if require_finite and not np.isfinite(feature_matrix).all():
        raise ValueError("feature_matrix contains NaN or Inf values")

    return feature_matrix


def fit_gmm_on_features(
    feature_matrix: np.ndarray,
    *,
    n_gmm_labels: int,
    covariance_type: str,
    random_state: int,
    n_init: int,
) -> GaussianMixture:
    """
    Fit a sklearn GaussianMixture on a prepared feature matrix.

    In sklearn terminology, n_components is the number of Gaussian mixture
    components. In this project-facing wrapper, we call the same concept
    n_gmm_labels.
    """
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)

    if feature_matrix.ndim != 2:
        raise ValueError(
            f"feature_matrix must be 2D, got shape {feature_matrix.shape}"
        )

    n_samples, n_features = feature_matrix.shape

    if n_samples == 0:
        raise ValueError("feature_matrix must contain at least one sample")

    if n_features == 0:
        raise ValueError("feature_matrix must contain at least one feature")

    if not np.isfinite(feature_matrix).all():
        raise ValueError("feature_matrix contains NaN or Inf values")

    if n_gmm_labels < 1:
        raise ValueError(
            f"n_gmm_labels must be >= 1, got {n_gmm_labels}"
        )

    if n_samples < n_gmm_labels:
        raise ValueError(
            f"n_samples must be >= n_gmm_labels, got "
            f"n_samples={n_samples}, n_gmm_labels={n_gmm_labels}"
        )

    if n_init < 1:
        raise ValueError(
            f"n_init must be >= 1, got {n_init}"
        )

    gmm = GaussianMixture(
        n_components=int(n_gmm_labels),
        covariance_type=covariance_type,
        random_state=int(random_state),
        n_init=int(n_init),
    )

    gmm.fit(feature_matrix)

    return gmm


def get_gmm_label_order_by_mean(
    gmm: GaussianMixture,
    *,
    feature_index: int,
) -> np.ndarray:
    """
    Return raw sklearn GMM label indices sorted by mean value along one feature.

    Example:
        If feature_index points to "log_area", ordered GMM label 0 corresponds
        to the smaller mean log_area group.
    """
    if not hasattr(gmm, "means_"):
        raise ValueError(
            "gmm must be fitted before gmm_label_order can be computed"
        )

    gmm_label_means = np.asarray(gmm.means_, dtype=np.float64)

    if gmm_label_means.ndim != 2:
        raise ValueError(
            f"gmm.means_ must be 2D, got shape {gmm_label_means.shape}"
        )

    _, n_features = gmm_label_means.shape

    if feature_index < 0 or feature_index >= n_features:
        raise ValueError(
            f"feature_index must be in [0, {n_features - 1}], got {feature_index}"
        )

    gmm_label_order = np.argsort(
        gmm_label_means[:, feature_index],
        kind="mergesort",
    )

    return gmm_label_order.astype(np.int64, copy=False)


def reorder_gmm_probabilities(
    probabilities: np.ndarray,
    gmm_label_order: np.ndarray,
    *,
    n_gmm_labels: int,
) -> np.ndarray:
    """
    Reorder GMM probability columns into deterministic GMM-label order.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    gmm_label_order = np.asarray(gmm_label_order, dtype=np.int64)

    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must be 2D, got shape {probabilities.shape}"
        )

    if gmm_label_order.ndim != 1:
        raise ValueError(
            f"gmm_label_order must be 1D, got shape {gmm_label_order.shape}"
        )

    if n_gmm_labels < 1:
        raise ValueError(
            f"n_gmm_labels must be >= 1, got {n_gmm_labels}"
        )

    if probabilities.shape[1] != n_gmm_labels:
        raise ValueError(
            f"probabilities must have {n_gmm_labels} columns, "
            f"got {probabilities.shape[1]}"
        )

    if len(gmm_label_order) != n_gmm_labels:
        raise ValueError(
            f"gmm_label_order length must equal n_gmm_labels, "
            f"got len(gmm_label_order)={len(gmm_label_order)}, "
            f"n_gmm_labels={n_gmm_labels}"
        )

    expected_gmm_label_indices = np.arange(n_gmm_labels, dtype=np.int64)

    if not np.array_equal(np.sort(gmm_label_order), expected_gmm_label_indices):
        raise ValueError(
            f"gmm_label_order must contain each GMM label index exactly once, "
            f"expected {expected_gmm_label_indices.tolist()}, "
            f"got {gmm_label_order.tolist()}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contains NaN or Inf values")

    ordered_probabilities = probabilities[:, gmm_label_order]

    return ordered_probabilities.astype(np.float64, copy=False)


def predict_gmm_probabilities(
    gmm: GaussianMixture,
    feature_matrix: np.ndarray,
    *,
    gmm_label_order: np.ndarray,
    n_gmm_labels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict ordered GMM labels, probabilities, and highest probabilities.
    """
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)

    if feature_matrix.ndim != 2:
        raise ValueError(
            f"feature_matrix must be 2D, got shape {feature_matrix.shape}"
        )

    if feature_matrix.shape[0] == 0:
        raise ValueError("feature_matrix must contain at least one sample")

    if feature_matrix.shape[1] == 0:
        raise ValueError("feature_matrix must contain at least one feature")

    if not np.isfinite(feature_matrix).all():
        raise ValueError("feature_matrix contains NaN or Inf values")

    if not hasattr(gmm, "means_"):
        raise ValueError("gmm must be fitted before predicting GMM labels")

    raw_probabilities = gmm.predict_proba(feature_matrix)

    probabilities = reorder_gmm_probabilities(
        probabilities=raw_probabilities,
        gmm_label_order=gmm_label_order,
        n_gmm_labels=n_gmm_labels,
    )

    gmm_labels = np.argmax(probabilities, axis=1).astype(
        np.int64,
        copy=False,
    )

    highest_probability = np.max(probabilities, axis=1).astype(
        np.float64,
        copy=False,
    )

    return gmm_labels, probabilities, highest_probability


def build_gmm_evidence_dataframe(
    table: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    gmm_label_order_feature: str,
    n_gmm_labels: int,
    covariance_type: str,
    random_state: int,
    n_init: int,
    evidence_name: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with appended GMM evidence columns.

    This is the local GMM wrapper:
        table
        -> feature matrix
        -> fitted GMM
        -> ordered GMM labels
        -> ordered probabilities
        -> evidence columns

    This function does not read files, save files, draw visualizations,
    or overwrite source labels.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"table must be a pandas DataFrame, got {type(table).__name__}"
        )

    if not feature_columns:
        raise ValueError("feature_columns must not be empty")

    if gmm_label_order_feature not in feature_columns:
        raise ValueError(
            f"gmm_label_order_feature must be one of feature_columns, "
            f"got {gmm_label_order_feature!r}, feature_columns={feature_columns}"
        )

    if n_gmm_labels < 1:
        raise ValueError(
            f"n_gmm_labels must be >= 1, got {n_gmm_labels}"
        )

    if not evidence_name:
        raise ValueError("evidence_name must not be empty")

    output_columns = [
        f"{evidence_name}_labels",
        f"{evidence_name}_probabilities",
        f"{evidence_name}_highest_probability",
        f"{evidence_name}_feature_columns",
        f"{evidence_name}_label_order_feature",
    ]

    existing_output_columns = [
        column
        for column in output_columns
        if column in table.columns
    ]

    if existing_output_columns:
        raise ValueError(
            f"GMM evidence output columns already exist: {existing_output_columns}"
        )

    feature_matrix = make_feature_matrix(
        table=table,
        feature_columns=feature_columns,
    )

    gmm = fit_gmm_on_features(
        feature_matrix,
        n_gmm_labels=n_gmm_labels,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )

    gmm_label_order_feature_index = feature_columns.index(
        gmm_label_order_feature
    )

    gmm_label_order = get_gmm_label_order_by_mean(
        gmm,
        feature_index=gmm_label_order_feature_index,
    )

    gmm_labels, probabilities, highest_probability = predict_gmm_probabilities(
        gmm,
        feature_matrix,
        gmm_label_order=gmm_label_order,
        n_gmm_labels=n_gmm_labels,
    )

    table_out = table.copy()

    table_out[f"{evidence_name}_labels"] = gmm_labels

    table_out[f"{evidence_name}_probabilities"] = [
        probability_vector_to_str(row)
        for row in probabilities
    ]

    table_out[f"{evidence_name}_highest_probability"] = highest_probability

    table_out[f"{evidence_name}_feature_columns"] = " ".join(
        feature_columns
    )

    table_out[f"{evidence_name}_label_order_feature"] = (
        gmm_label_order_feature
    )

    return table_out