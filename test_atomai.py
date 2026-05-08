import atomai
import torch
#print("AtomAI:", atomai.__version__)
#print("Torch:", torch.__version__)

from src.pipeline import DislocationPipeline
from pathlib import Path
def main():
    pipe = DislocationPipeline(
        # image_path can be omitted to use configs.DATA_DIR / "Figure_9.png"
        input_dir="data",  # or None → defaults to DATA_DIR
        max_files=1,  # one image for now
        segmentation_method="otsu",
        save_vis=True,
        prefix="figure8",
        input_paths=[Path("data/Figure_8.png")]
    )
    result = pipe.run_one()
    print("Done:", result["input_path"])



if __name__ == "__main__":
    main()


