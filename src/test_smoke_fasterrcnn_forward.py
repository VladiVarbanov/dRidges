from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.torch_vision_dataset import TorchVisionDataset


def main():
    root = Path("DataSetFinal")

    dataset = TorchVisionDataset(
        split_path=root / "trainimages.txt",
        image_dir=root / "images",
        annotation_dir=root / "bounding_boxes",
        npy_dir=root / "nn_input_npy",
        annotation_format_path=root / "annotation_format.json",
        max_images=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
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

    images, targets = next(iter(loader))

    images = [img.to(device) for img in images]
    targets = [
        {k: v.to(device) if hasattr(v, "to") else v for k, v in target.items()}
        for target in targets
    ]

    loss_dict = model(images, targets)

    print("loss_dict:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.6f}")

    #prove gradients and optimizer update work
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        )

    loss_dict = model(images, targets)
    total_loss = sum(loss_dict.values())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"total_loss = {total_loss.item():.6f}")
    print("backward + optimizer step succeeded")

if __name__ == "__main__":
    main()