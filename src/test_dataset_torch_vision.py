from pathlib import Path

from src.torch_vision_dataset import TorchVisionDataset
from configs.config import (
    WORKSPACE,
    NN_DATASET_ROOT,
    NN_IMAGE_DIR,
    NN_INPUT_NPY_DIR,
    NN_ANNOTATION_DIR,
    NN_TRAIN_SPLIT_TXT,
    NN_TEST_SPLIT_TXT,
    NN_ALL_SPLIT_TXT,
)

def main() -> None:

    dataset = TorchVisionDataset(
        split_path = NN_TRAIN_SPLIT_TXT,
        image_dir=NN_IMAGE_DIR,
        annotation_dir=NN_ANNOTATION_DIR,
        npy_dir=NN_INPUT_NPY_DIR,
        annotation_format_path=NN_DATASET_ROOT / "annotation_format.json",
        max_images=3,
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    print("dataset length:", len(dataset))

    image, target = dataset[0]

    print("image shape:", image.shape)
    print("image dtype:", image.dtype)

    print("target keys:", target.keys())
    print("boxes:", target["boxes"])
    print("boxes shape:", target["boxes"].shape)
    print("labels:", target["labels"])
    print("labels shape:", target["labels"].shape)
    print("image_id:", target["image_id"])
    print("area shape:", target["area"].shape)
    print("iscrowd shape:", target["iscrowd"].shape)


if __name__ == "__main__":
    main()