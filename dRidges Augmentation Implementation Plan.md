# dRidges Augmentation Implementation Plan
## Problem Statement
Implement a clean augmentation pipeline for dRidges that separates reusable logic from offline generation, preserves project labels, reuses existing functions, and creates validation reports.
## Current State
* `src/nn_augmentation.py` has circular imports and uses deprecated transforms
* `src/torch_vision_dataset.py` has reusable functions: `npy_to_torch_tensor`, `build_torchvision_target`
* `src/annotation_io.py` has: `parse_annotation_txt_rc`, `write_annotation_txt_rc`
* `src/nn_adapters.py` has: `rows_cols_to_xyxy`, `xyxy_to_rows_cols`
* `DataSetFinal/annotation_format_redacted.json` is the source-of-truth
* `DataSetFinal/augmented_v1/` already contains augmented data
## Key Principles
1. **Label Separation**: Project labels (0, 1, 2) unchanged during augmentation. BG0 conversion only at target-building boundary.
2. **Reuse**: Use existing functions; no reimplementation.
3. **Validation**: Runtime validation reports, not duplicate schema files.
4. **Module Separation**: Reusable logic in `nn_augmentation.py`, offline generation in `run_generate_torch_augmented_data.py`.
## Implementation Steps
### 1. Rewrite src/nn_augmentation.py
Remove circular imports, use torchvision.transforms.v2, preserve project labels.
Key functions: fixed-angle rotation, flips, scale transforms, augmentation applier.
### 2. Create src/run_generate_torch_augmented_data.py
Offline generator: load .npy, parse annotations, apply transforms, save augmented .npy and annotations.
### 3. Add Validation System (src/validation_report.py)
Create validation reports in `DataSetFinal/validation_reports/` with checks for:
* File existence (.npy, annotation txt)
* Label validity
* Box geometry (positive width/height, within bounds)
* Array shape and dtype
* Train/test overlap
### 4. Update configs/training_config.py
Add augmentation constants: rotation angles, scale range, target size.
### 5. Integration
Augmentation applied after `build_torchvision_target` in offline pipeline.
## Output Files
* `src/nn_augmentation.py` (rewritten)
* `src/run_generate_torch_augmented_data.py` (new)
* `src/validation_report.py` (new)
* `configs/training_config.py` (updated)
* `DataSetFinal/validation_reports/train_validation_report.json`
* `DataSetFinal/validation_reports/test_validation_report.json`
