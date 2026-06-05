from __future__ import annotations

import numpy as np
import pandas as pd

from configs.object_features_config import (
    GMM_SCALE_FEATURE_COLUMNS,
    GMM_SCALE_LABEL_ORDER_FEATURE,
    GMM_N_LABELS,
    GMM_COVARIANCE_TYPE,
    GMM_RANDOM_STATE,
    GMM_N_INIT,
    GMM_SCALE_EVIDENCE_NAME,
)

from src.gmm_wrapper import build_gmm_evidence_dataframe


def test_gmm_wrapper_adds_scale_evidence_columns() -> None:
    table = pd.DataFrame(
        {
            "object_uid": [
                "img00000_box00000",
                "img00000_box00001",
                "img00000_box00002",
                "img00000_box00003",
                "img00000_box00004",
                "img00000_box00005",
            ],
            "log_area": [
                2.00,
                2.10,
                2.20,
                6.00,
                6.10,
                6.20,
            ],
        }
    )

    table_with_gmm = build_gmm_evidence_dataframe(
        table=table,
        feature_columns=GMM_SCALE_FEATURE_COLUMNS,
        gmm_label_order_feature=GMM_SCALE_LABEL_ORDER_FEATURE,
        n_gmm_labels=GMM_N_LABELS,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=GMM_RANDOM_STATE,
        n_init=GMM_N_INIT,
        evidence_name=GMM_SCALE_EVIDENCE_NAME,
    )
    print(table_with_gmm)

    expected_columns = {
        "gmm_scale_labels",
        "gmm_scale_probabilities",
        "gmm_scale_highest_probability",
        "gmm_scale_feature_columns",
        "gmm_scale_label_order_feature",
    }

    assert expected_columns.issubset(table_with_gmm.columns)
    assert len(table_with_gmm) == len(table)

    # Small log_area group should be ordered as GMM scale label 0.
    assert table_with_gmm.loc[0, "gmm_scale_labels"] == 0
    assert table_with_gmm.loc[1, "gmm_scale_labels"] == 0
    assert table_with_gmm.loc[2, "gmm_scale_labels"] == 0

    # Large log_area group should be ordered as GMM scale label 1.
    assert table_with_gmm.loc[3, "gmm_scale_labels"] == 1
    assert table_with_gmm.loc[4, "gmm_scale_labels"] == 1
    assert table_with_gmm.loc[5, "gmm_scale_labels"] == 1

    for probability_string in table_with_gmm["gmm_scale_probabilities"]:
        probability_vector = np.fromstring(probability_string, sep=" ")

        assert probability_vector.shape == (GMM_N_LABELS,)
        assert np.isfinite(probability_vector).all()
        assert np.all(probability_vector >= 0.0)
        assert np.isclose(probability_vector.sum(), 1.0)

    assert np.isfinite(table_with_gmm["gmm_scale_highest_probability"]).all()
    assert (table_with_gmm["gmm_scale_highest_probability"] >= 0.0).all()
    assert (table_with_gmm["gmm_scale_highest_probability"] <= 1.0).all()

    assert table_with_gmm["gmm_scale_feature_columns"].iloc[0] == "log_area"
    assert table_with_gmm["gmm_scale_label_order_feature"].iloc[0] == "log_area"


def test_gmm_wrapper_does_not_modify_input_table() -> None:
    table = pd.DataFrame(
        {
            "object_uid": [
                "img00000_box00000",
                "img00000_box00001",
                "img00000_box00002",
                "img00000_box00003",
            ],
            "log_area": [
                2.0,
                2.1,
                6.0,
                6.1,
            ],
        }
    )

    original_columns = list(table.columns)

    _ = build_gmm_evidence_dataframe(
        table=table,
        feature_columns=GMM_SCALE_FEATURE_COLUMNS,
        gmm_label_order_feature=GMM_SCALE_LABEL_ORDER_FEATURE,
        n_gmm_labels=GMM_N_LABELS,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=GMM_RANDOM_STATE,
        n_init=GMM_N_INIT,
        evidence_name=GMM_SCALE_EVIDENCE_NAME,
    )


    assert list(table.columns) == original_columns