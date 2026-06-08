from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.torch_vision_dataset import TorchVisionDataset


def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    data_root = ROOT_DIR / "DataSetFinal"
    output_root = ROOT_DIR / "results"

    dataset = TorchVisionDataset(
        split_path=data_root / "trainimages.txt",
        image_dir=data_root / "images",
        annotation_dir=data_root / "bounding_boxes_redacted_gmm_corrected",
        npy_dir=data_root / "nn_input_npy",
        annotation_format_path=data_root / "annotation_format.json",
        max_images=1,
    )

    image, target = dataset[0]

    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    num_classes = 5
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = output_root / "checkpoints/test_smoke_tiny_overfit.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"loaded checkpoint: {checkpoint_path}")

    model.to(device)
    model.eval()

    image = image.to(device)

    with torch.no_grad():
        predictions = model([image])

    prediction = predictions[0]

    print("Ground truth:")
    print("boxes:")
    print(target["boxes"])
    print("labels:")
    print(target["labels"])

    print("\nPredictions:")
    print("boxes shape:", prediction["boxes"].shape)
    print("labels shape:", prediction["labels"].shape)
    print("scores shape:", prediction["scores"].shape)

    print("\nTop predictions:")
    top_n = min(10, prediction["scores"].shape[0])

    for index in range(top_n):
        score = prediction["scores"][index].item()
        label = prediction["labels"][index].item()
        box = prediction["boxes"][index].detach().cpu().tolist()

        print(
            f"{index:02d} | "
            f"score={score:.4f} | "
            f"label={label} | "
            f"box={box}"
        )


if __name__ == "__main__":
    main()