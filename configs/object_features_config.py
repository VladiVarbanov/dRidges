from __future__ import annotations

from pathlib import Path

from configs.config import NN_DATASET_ROOT

EPS = 1e-6  #TODO: duplicated

AREA_LOG_EPS = 1e-6   #TODO duplicated
MIN_VALID_BOX_AREA = 1.0

AREA_PERCENTILE_EPS = 1e-12

LOG_AREA_HISTOGRAM_NUM_BINS = 32
AREA_HISTOGRAM_NUM_BINS = 32

AREA_HISTOGRAM_PEAK_PROMINENCE = 1
AREA_HISTOGRAM_VALLEY_PROMINENCE = 1
AREA_HISTOGRAM_MIN_DISTANCE_BINS = 1

LOG_AREA_GMM_COMPONENTS = 2
LOG_AREA_GMM_RANDOM_STATE = 0
LOG_AREA_GMM_COVARIANCE_TYPE = "full"

# ----- GMM -----
GMM_SCALE_FEATURE_COLUMNS = ("log_area",)
GMM_SCALE_LABEL_ORDER_FEATURE = "log_area"

GMM_N_LABELS = 2
GMM_COVARIANCE_TYPE = "full"
GMM_RANDOM_STATE = 0
GMM_N_INIT = 10

GMM_SCALE_EVIDENCE_NAME = "gmm_scale"

# GMM label → redacted class mapping
# GMM label 0 = small boxes → black_dot → redacted class 1
# GMM label 1 = large boxes → merged loops → redacted class 0
GMM_SCALE_LABEL_TO_REDACTED_CLASS = {
    0: 1,
    1: 0,
}

# GMM annotation correction settings
GMM_BOX_SCALE_FACTOR = 1.08
GMM_CORRECTED_ANNOTATION_DIR = NN_DATASET_ROOT / "bounding_boxes_redacted_gmm_corrected"

# GMM overlay inspection settings
GMM_OVERLAY_TOP_DISAGREEMENTS =153 #30
GMM_OVERLAY_MAX_UNIQUE_IMAGES = GMM_OVERLAY_TOP_DISAGREEMENTS #10
