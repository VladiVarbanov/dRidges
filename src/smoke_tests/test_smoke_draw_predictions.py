from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from configs.config import NN_CLASS_COLORS, NN_CLASS_COLORS_GT
from debug_io import save_rgba_tiff
from nn_adapters import xyxy_to_xywh, label_ids_from_bg0_format
from torch_vision_dataset import TorchVisionDataset
from utilities import load_image, rgba_from_gray
from visualization import paint_labeled_xywh_boxes_in_place


def main():
    root = Path("../../DataSetFinal")

    dataset = TorchVisionDataset(
        split_path=root / "trainimages.txt",
        image_dir=root / "images",
        annotation_dir=root / "bounding_boxes",
        npy_dir=root / "nn_input_npy",
        annotation_format_path=root / "annotation_format.json",
        max_images=1,
    )

    image, target = dataset[0]
    source_image_path, annotation_path = dataset.pairs[0]

    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    num_classes = 5
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path("../../checkpoints/test_smoke_tiny_overfit.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    with torch.no_grad():
        prediction = model([image.to(device)])[0]

    source_gray = load_image(source_image_path)
    image_rgba = rgba_from_gray(source_gray)

    # Ground truth: target boxes are XYXY, labels are BG0.
    gt_boxes_xyxy = target["boxes"].detach().cpu().numpy()
    gt_boxes_xywh = xyxy_to_xywh(gt_boxes_xyxy)
    gt_labels = label_ids_from_bg0_format(
        target["labels"].detach().cpu().numpy()
    )

    paint_labeled_xywh_boxes_in_place(
        image_rgba,
        gt_boxes_xywh,
        gt_labels,
        class_colors=NN_CLASS_COLORS_GT,
        default_color=(255, 255, 255),
        alfa_value=0.35,
        line_width=6,
    )

    # Predictions: boxes are XYXY, labels are BG0.
    score_threshold = 0.05
    scores = prediction["scores"].detach().cpu()
    keep = scores >= score_threshold

    pred_boxes_xyxy = prediction["boxes"][keep].detach().cpu().numpy()
    pred_boxes_xywh = xyxy_to_xywh(pred_boxes_xyxy)
    pred_labels = label_ids_from_bg0_format(
        prediction["labels"][keep].detach().cpu().numpy()
    )

    paint_labeled_xywh_boxes_in_place(
        image_rgba,
        pred_boxes_xywh,
        pred_labels,
        class_colors=NN_CLASS_COLORS,
        default_color=(255, 255, 255),
        alfa_value=1,
        line_width=2,
    )

    output_path = Path("../../results/smoke_predictions/prediction_vs_gt.tif")
    save_rgba_tiff(image_rgba, output_path)

    print(f"source image: {source_image_path}")
    print(f"annotation: {annotation_path}")
    print(f"image tensor shape: {tuple(image.shape)}")
    print(f"source image shape: {source_gray.shape}")
    print(f"GT boxes: {len(gt_boxes_xywh)}")
    print(f"kept predictions: {len(pred_boxes_xywh)}")
    print(f"score threshold: {score_threshold}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()