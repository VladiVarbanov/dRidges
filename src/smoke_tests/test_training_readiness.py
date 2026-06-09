"""Smoke test to verify dataset structure, imports, and training readiness."""
from __future__ import annotations

from pathlib import Path
import sys


def test_dataset_readiness():
    root = Path("DataSetFinal")
    checks = {}

    checks["images dir"] = (root / "images").exists()
    checks["npy dir"] = (root / "nn_input_npy").exists()
    checks["annotation dir"] = (root / "bounding_boxes_redacted").exists()
    checks["train split"] = (root / "trainimages.txt").exists()
    checks["test split"] = (root / "testimages.txt").exists()
    checks["annotation format"] = (root / "annotation_format_redacted.json").exists()

    if checks["npy dir"]:
        checks["npy count"] = len(list((root / "nn_input_npy").glob("*.npy")))
    if checks["annotation dir"]:
        checks["ann count"] = len(list((root / "bounding_boxes_redacted").glob("*.txt")))
    if checks["train split"]:
        checks["train images"] = len((root / "trainimages.txt").read_text().strip().splitlines())
    if checks["test split"]:
        checks["test images"] = len((root / "testimages.txt").read_text().strip().splitlines())

    aug_root = root / "augmented_v1"
    if aug_root.exists():
        checks["augmented dir"] = True
        if (aug_root / "nn_input_npy").exists():
            checks["augmented npy count"] = len(list((aug_root / "nn_input_npy").glob("*.npy")))
        if (aug_root / "bounding_boxes").exists():
            checks["augmented ann count"] = len(list((aug_root / "bounding_boxes").glob("*.txt")))

    print("=== DATASET READINESS CHECK ===")
    all_ok = True
    for k, v in checks.items():
        status = "OK" if v else "MISSING"
        if isinstance(v, bool) and not v:
            all_ok = False
        print(f"  {k}: {v}")

    return all_ok


def test_imports():
    print("\n=== IMPORT CHECK ===")
    all_ok = True

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  PyTorch: FAILED - {e}")
        all_ok = False

    try:
        import torchvision
        print(f"  TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"  TorchVision: FAILED - {e}")
        all_ok = False

    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
    except ImportError as e:
        print(f"  NumPy: FAILED - {e}")
        all_ok = False

    return all_ok


def test_code_files():
    print("\n=== CODE FILES CHECK ===")
    all_ok = True
    code_files = [
        "run_training.py",
        "src/torch_vision_dataset.py",
        "src/annotation_io.py",
        "src/nn_adapters.py",
        "configs/config.py",
        "configs/training_config.py",
    ]
    for f in code_files:
        exists = Path(f).exists()
        print(f"  {f}: {'OK' if exists else 'MISSING'}")
        if not exists:
            all_ok = False
    return all_ok


def test_dataset_imports():
    print("\n=== DATASET CLASS CHECK ===")
    all_ok = True
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.torch_vision_dataset import TorchVisionDataset
        print("  TorchVisionDataset: OK")
    except Exception as e:
        print(f"  TorchVisionDataset: FAILED - {e}")
        all_ok = False
    return all_ok


if __name__ == "__main__":
    results = [
        test_dataset_readiness(),
        test_imports(),
        test_code_files(),
        test_dataset_imports(),
    ]
    print(f"\n=== RESULT: {'READY' if all(results) else 'NOT READY'} ===")
    sys.exit(0 if all(results) else 1)
