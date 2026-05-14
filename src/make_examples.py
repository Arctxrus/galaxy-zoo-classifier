"""One-off script: extract 10 example galaxy images (one per class) from
the test split and save them as PNGs for the Gradio app's examples panel.

Run once:
    python src/make_examples.py
"""

from pathlib import Path
import h5py
import pandas as pd
from PIL import Image

OUT_DIR = Path("deploy/examples")
H5_PATH = "data/Galaxy10_DECals.h5"
SPLITS_CSV = "data/splits.csv"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLITS_CSV)
    test_df = df[df["split"] == "test"]

    with h5py.File(H5_PATH, "r") as f:
        for cls in range(10):
            # iloc 0 gets first row, converting it to a series then we
            # get the index
            idx = test_df[test_df["label"] == cls].iloc[0]["index"]
            img = f["images"][int(idx)]
            out_path = OUT_DIR / f"class_{cls}.png"
            Image.fromarray(img).save(out_path)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()