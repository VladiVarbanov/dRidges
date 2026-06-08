# configs/config.py

import os
import sys
from pathlib import Path
#from functools import partial
#from typing import Callable
import numpy as np
#from preprocessing import  (
#   local_contrast_normalization_CLAHE,
#    local_normalize_HOG_style,
#)

# -----Preprocessing Strategies (callables)-----
#LOCAL_CONTRAST_FN: Callable[[np.ndarray], np.ndarray] = local_normalize_HOG_style

# -----Constants-----
# -----preprocessing-----
EPS = 1e-6
LOCAL_NORM_SIGMA = 8.0  # Defines moving window approx -> 6*LOCAL_NORM_SIGMA pixels
GAUSSIAN_SMOOTH_SIGMA = 2.0

# -----ridges + seeds -----
HESSIAN_SCALE_PX= 2.0
RIDGE_SCALE_STEPS = 1
RIDGE_SCALE_FACTOR = np.sqrt(2.0)
QUANTILE = .90  #0.95
MAX_SEEDS = 200
SEEDS_PER_TILE = 8     # TODO: think of different value, might compute or train
TILE_DIVISOR = 8  # tile_size ~= min(row,col)/tile_divisor
TILE_MIN = 32  # 48
TILE_MAX = 128
BASE_SIGMA_N_DIVISOR = 6
BASE_SIGMA_N_PX_MIN = 4     #BASE_SIGMA_N_PX = 6
BASE_SIGMA_N_PX_MAX = 6
PARTNER_OFFSET_PX = 2.0 * HESSIAN_SCALE_PX
PARTNER_GATE_HALF_WIDTH_N_PX = 0.5 * HESSIAN_SCALE_PX
PARTNER_GATE_HALF_WIDTH_T_PX = 1.0 * HESSIAN_SCALE_PX
PARTNER_THETA_TOLERANCE = np.pi / 12.0   # 15 deg TODO: ADD SOME REAL FORMULA HERE
PARTNER_REJECTION_DISTANCE_PX = 0.5 * HESSIAN_SCALE_PX
#TODO: use as parameters later NNs ML
STRENGTH_WEIGHT = 1.0
GATE_CENTER_N_WEIGHT = 1.0
GATE_CENTER_T_WEIGHT = 1.0
THETA_WEIGHT = 1.0
FLAT_AREA_CAP = 2
SUPPRESSION_DISTANCE_THRESHOLD = 0.7   # 0.8  1  1.2

#----- NNs preparation-----
NN_ANN_EXT = ".txt"

NN_MIN_BOX_WIDTH = 1.0
NN_MIN_BOX_HEIGHT = 1.0
NN_MIN_BOX_AREA = 1.0
NN_EXPECTED_NUM_CHANNELS = 3

FRAMEWORK_TORCHVISION = "torchvision"
FRAMEWORK_NUMPY = "numpy"
FRAMEWORK_KERAS = "keras"
FRAMEWORK_OPENCV = "opencv"

ROI_FORMAT_XYWH = "XYWH"
ROI_FORMAT_XYXY = "XYXY"
ROI_FORMAT_YXYX = "YXYX"

NN_LOCAL_NORM_SIGMA = 5.0
NN_GAUSSIAN_SMOOTH_SIGMA = 2.0
NN_HESSIAN_SCALE_PX = 4.0

#----- Visualisation -----
ANCHOR_COLOR = (230, 90, 70)
PARTNER_COLOR = (50, 160, 150)
MIDPOINT_COLOR = (240, 200, 60)
ALFA_VALUE = 0.5
NN_CLASS_COLORS = {
    0: (128, 0, 32),  # burgundy
    1: (75, 0, 130),  # indigo
    2: (85, 107, 47),  # moss green / dark olive green
    3: (255, 191, 0),  # amber
}

NN_CLASS_COLORS_GT = {
    0: (160, 20, 60),     # brighter burgundy / crimson
    1: (30, 90, 220),     # strong royal blue
    2: (0, 170, 110),     # bright emerald
    3: (230, 180, 40),    # warm gold
}

NN_CLASS_NAMES = {
    0: "a0_half_111_loop",
    1: "a0_100_loop",
    2: "black_dot",
    3: "other_defect",
}

# ----- Workspace root -----
# Resolve workspace: ATOMAI_WORKSPACE env var > platform default
def _default_workspace() -> Path:
    if sys.platform == "win32":
        return Path("C:/AtomAI_ws")
    return Path.home() / "AtomAi_ws"

WORKSPACE = Path(os.environ.get("ATOMAI_WORKSPACE", str(_default_workspace())))

# ----- Data subdirectories -----
DATA_DIR = WORKSPACE / "data"
NOTEBOOKS_DIR = WORKSPACE / "notebooks"
RESULTS_DIR = WORKSPACE / "results"
INTERMEDIATE_DIR = NOTEBOOKS_DIR / "intermediate"
INTERMEDIATE_NPY_DIR = INTERMEDIATE_DIR / "npy"
INTERMEDIATE_TIFF2D_DIR = INTERMEDIATE_DIR / "tiff2d"
RESULTS_VIS_DIR = RESULTS_DIR / "vis"

NN_DATASET_ROOT = WORKSPACE / "DataSetFinal"
NN_IMAGE_DIR = NN_DATASET_ROOT / "images"
NN_INPUT_NPY_DIR = NN_DATASET_ROOT / "nn_input_npy"
NN_ANNOTATION_DIR = NN_DATASET_ROOT / "bounding_boxes"
NN_REDACTED_ANNOTATION_DIR = NN_DATASET_ROOT / "bounding_boxes_redacted"
NN_TRAIN_SPLIT_TXT = NN_DATASET_ROOT / "trainimages.txt"
NN_TEST_SPLIT_TXT = NN_DATASET_ROOT / "testimages.txt"
NN_ALL_SPLIT_TXT = NN_DATASET_ROOT / "allimages.txt"

# ----- GMM scale evidence outputs -----
GMM_SCALE_EVIDENCE_OUTPUT_DIR = RESULTS_DIR / "gmm_scale_evidence"
GMM_SCALE_OVERLAY_DIR = GMM_SCALE_EVIDENCE_OUTPUT_DIR / "source_vs_gmm_scale_overlays"


# ----- Default pipeline settings -----
DEFAULT_SEGMENTATION_METHOD = "otsu"
DEFAULT_PREFIX = "sample"

for p in [DATA_DIR, NOTEBOOKS_DIR, RESULTS_DIR, INTERMEDIATE_DIR, INTERMEDIATE_NPY_DIR, INTERMEDIATE_TIFF2D_DIR, RESULTS_VIS_DIR, NN_INPUT_NPY_DIR]:
    p.mkdir(parents=True, exist_ok=True)