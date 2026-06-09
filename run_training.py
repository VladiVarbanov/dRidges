from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

from configs.config import (
    NN_IMAGE_DIR,
    NN_TRAIN_SPLIT_TXT,
    NN_TEST_SPLIT_TXT,
    NN_INPUT_NPY_DIR,
    NN_ANNOTATION_DIR,
)
from configs.training_config import (
    TRAINING_BATCH_SIZE,
    TRAINING_LEARNING_RATE,
    TRAINING_MOMENTUM,
    TRAINING_WEIGHT_DECAY,
    TRAINING_NUM_EPOCHS,
    TRAINING_LR_STEP_SIZE,
    TRAINING_LR_GAMMA,
    CHECKPOINT_DIR,
    CHECKPOINT_BEST,
    NUM_CLASSES,
)
from src.torch_vision_dataset import TorchVisionDataset
from src.annotation_io import parse_annotation_txt_rc
from src.nn_adapters import rows_cols_to_xyxy, label_ids_to_bg0_format


class AugmentedDataset(Dataset):
    """Simple dataset for augmented .npy files (3-channel) and annotations."""
    
    def __init__(self, npy_dir, annotation_dir, npy_names, max_images=None):
        self.npy_dir = Path(npy_dir)
        self.annotation_dir = Path(annotation_dir)
        
        if max_images is not None:
            npy_names = npy_names[:max_images]
        
        self.npy_names = npy_names
        self.pairs = []
        
        for npy_name in npy_names:
            npy_stem = Path(npy_name).stem
            # Annotation file name: original stem + extension
            # e.g., image_rotate90.npy -> image_rotate90.txt
            ann_name = npy_stem + ".txt"
            ann_path = self.annotation_dir / ann_name
            if ann_path.exists():
                self.pairs.append((self.npy_dir / npy_name, ann_path))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        npy_path, annotation_path = self.pairs[idx]
        
        # Load 3-channel .npy file (C, H, W)
        image_np = np.load(npy_path)
        if image_np.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, got {image_np.shape[0]} from {npy_path}")
        
        image_tensor = torch.from_numpy(image_np).float()
        
        # Load annotations
        labels_src, boxes_rc = parse_annotation_txt_rc(annotation_path)
        
        # Build target
        boxes_xyxy_np = rows_cols_to_xyxy(boxes_rc)
        labels_bg0_np = label_ids_to_bg0_format(labels_src)
        
        boxes_tensor = torch.as_tensor(boxes_xyxy_np, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels_bg0_np, dtype=torch.int64)
        
        x_min = boxes_tensor[:, 0]
        y_min = boxes_tensor[:, 1]
        x_max = boxes_tensor[:, 2]
        y_max = boxes_tensor[:, 3]
        
        box_widths = x_max - x_min
        box_heights = y_max - y_min
        
        # Filter out invalid boxes (zero or negative width/height)
        valid_mask = (box_widths > 0) & (box_heights > 0)
        if not valid_mask.all():
            boxes_tensor = boxes_tensor[valid_mask]
            labels_tensor = labels_tensor[valid_mask]
            box_widths = box_widths[valid_mask]
            box_heights = box_heights[valid_mask]
        
        # Skip samples with no valid boxes
        if len(boxes_tensor) == 0:
            return None
        
        box_area = box_widths * box_heights
        
        iscrowd = torch.zeros((len(boxes_tensor),), dtype=torch.int64)
        image_id_tensor = torch.tensor([idx], dtype=torch.int64)
        
        _, image_height, image_width = image_tensor.shape
        boxes_out = tv_tensors.BoundingBoxes(
            boxes_tensor,
            format="XYXY",
            canvas_size=(int(image_height), int(image_width)),
        )
        
        target = {
            "boxes": boxes_out,
            "labels": labels_tensor,
            "image_id": image_id_tensor,
            "area": box_area,
            "iscrowd": iscrowd,
        }
        
        return image_tensor, target


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Determine which dataset to use
    # We will check if the augmented dataset exists and use it.
    # Otherwise, we fall back to the original dataset.
    augmented_npy_dir = Path("DataSetFinal/augmented_v1/nn_input_npy")
    augmented_ann = Path("DataSetFinal/augmented_v1/bounding_boxes")

    if augmented_npy_dir.exists() and augmented_ann.exists():
        print(f"Using AUGMENTED dataset: {augmented_npy_dir}")
        all_npy_files = sorted(augmented_npy_dir.glob("*.npy"))
        # Simple split: 80% train, 20% val
        torch.manual_seed(42)
        indices = torch.randperm(len(all_npy_files)).tolist()
        split_idx = int(len(all_npy_files) * 0.8)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_npy_files = [all_npy_files[i] for i in train_indices]
        val_npy_files = [all_npy_files[i] for i in val_indices]
        
        train_npy_names = [f.name for f in train_npy_files]
        val_npy_names = [f.name for f in val_npy_files]
        
        print(f"Split: {len(train_npy_names)} train, {len(val_npy_names)} val")

        train_dataset = AugmentedDataset(
            npy_dir=augmented_npy_dir,
            annotation_dir=augmented_ann,
            npy_names=train_npy_names,
            max_images=None,
        )
        val_dataset = AugmentedDataset(
            npy_dir=augmented_npy_dir,
            annotation_dir=augmented_ann,
            npy_names=val_npy_names,
            max_images=None,
        )
    else:
        print("Using ORIGINAL dataset (no augmentation found).")
        train_dataset = TorchVisionDataset(
            split_path=NN_TRAIN_SPLIT_TXT,
            image_dir=NN_IMAGE_DIR,
            annotation_dir=NN_ANNOTATION_DIR,
            npy_dir=NN_INPUT_NPY_DIR,
            max_images=None,
        )
        val_dataset = TorchVisionDataset(
            split_path=NN_TEST_SPLIT_TXT,
            image_dir=NN_IMAGE_DIR,
            annotation_dir=NN_ANNOTATION_DIR,
            npy_dir=NN_INPUT_NPY_DIR,
            max_images=None,
        )

    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )

    # 3. Define Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)

    # 4. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=TRAINING_LEARNING_RATE, 
        momentum=TRAINING_MOMENTUM, 
        weight_decay=TRAINING_WEIGHT_DECAY
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=TRAINING_LR_STEP_SIZE, gamma=TRAINING_LR_GAMMA)

    # 5. Training Loop
    print("Starting training...")
    best_val_loss = float('inf')
    CHECKPOINT_BEST.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(TRAINING_NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        train_batches = 0
        for images, targets in train_loader:
            if len(images) == 0:
                continue
            
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
            train_batches += 1
        
        avg_train_loss = train_loss / max(train_batches, 1)
        lr_scheduler.step()

        # --- Validate ---
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            model.train()  # Needed to return loss dict instead of predictions
            for images, targets in val_loader:
                if len(images) == 0:
                    continue
                
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_batches += 1
            model.eval()
        
        avg_val_loss = val_loss / max(val_batches, 1)

        print(f"Epoch {epoch+1}/{TRAINING_NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # --- Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, CHECKPOINT_BEST)
            print(f"  -> New Best Model Saved to {CHECKPOINT_BEST}")

    print("Training Complete.")


if __name__ == "__main__":
    main()
