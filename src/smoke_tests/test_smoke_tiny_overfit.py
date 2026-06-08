from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.torch_vision_dataset import TorchVisionDataset


def move_targets_to_device(targets, device):
    """
    Move every tensor-like value inside each target dictionary to the selected device.

    TorchVision detection targets are dictionaries, for example:
        {
            "boxes": tensor,
            "labels": tensor,
            "image_id": tensor,
            "area": tensor,
            "iscrowd": tensor,
        }

    The model and all tensors must be on the same device:
        CPU with CPU, or CUDA with CUDA.

    Some values may not be tensors, so we only call .to(device)
    when the value actually supports it.
    """
    moved_targets = []

    for target in targets:
        moved_target = {}

        for target_key, target_value in target.items():
            if hasattr(target_value, "to"):
                moved_target[target_key] = target_value.to(device)
            else:
                moved_target[target_key] = target_value

        moved_targets.append(moved_target)

    return moved_targets


def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    data_root = ROOT_DIR / "DataSetFinal"
    output_root = ROOT_DIR / "results"

    checkpoint_path = output_root / "checkpoints/test_smoke_tiny_overfit.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = TorchVisionDataset(
        split_path=data_root / "trainimages.txt",
        image_dir=data_root / "images",
        annotation_dir=data_root / "bounding_boxes_redacted_gmm_corrected",
        npy_dir=data_root / "nn_input_npy",
        annotation_format_path=data_root / "annotation_format.json",
        max_images=2,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    num_classes = 5
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    num_epochs = 10

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = move_targets_to_device(targets, device)

            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"epoch {epoch + 1:02d} | avg_loss = {avg_loss:.6f}")
        # Save the trained model parameters.
        #TODO: CREATE SOME STRUCT AND SAVE AS JSON
        # model.state_dict() contains the learned weights.
        # We also save small metadata values that are needed to rebuild
        # the same model structure before loading the weights.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "num_epochs": num_epochs,
            },
            checkpoint_path,
        )

        print(f"saved checkpoint to: {checkpoint_path}")

if __name__ == "__main__":
    main()