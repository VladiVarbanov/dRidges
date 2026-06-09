from __future__ import annotations

from pathlib import Path

from configs.config import WORKSPACE, RESULTS_DIR, NN_INPUT_NPY_DIR, NN_IMAGE_DIR, NN_ANNOTATION_DIR, NN_TRAIN_SPLIT_TXT, NN_TEST_SPLIT_TXT

# ----- Training Hyperparameters -----
TRAINING_BATCH_SIZE = 4
TRAINING_LEARNING_RATE = 0.005
TRAINING_MOMENTUM = 0.9
TRAINING_WEIGHT_DECAY = 0.0005
TRAINING_NUM_EPOCHS = 50

# ----- LR Scheduler -----
TRAINING_LR_STEP_SIZE = 20
TRAINING_LR_GAMMA = 0.1

# ----- Checkpoints -----
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_BEST = CHECKPOINT_DIR / "best_model.pth"

# ----- Augmented Dataset (Optional) -----
# If AUGMENTED_DATA_DIR exists, use it for training.
# Otherwise, fall back to NN_ANNOTATION_DIR.
AUGMENTED_DATA_DIR = NN_IMAGE_DIR.parent / "augmented_v1"

# ----- Number of classes -----
# 4 classes (a0_half_111_loop, a0_100_loop, black_dot, other_defect) + 1 (background) = 5
NUM_CLASSES = 4
