"""Create a stratified 70/15/15 train/val/test split and save to CSV.

Run once. The output (data/splits.csv) is committed to the repo so every
training run uses the exact same partition.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

H5_PATH = Path("data/Galaxy10_DECals.h5")
OUT_PATH = Path("data/splits.csv")
RANDOM_STATE = 42

def main():
    # Load
    with h5py.File(H5_PATH, "r") as f:
        labels = np.array(f["ans"])              

    all_indices = np.arange(len(labels))

    # First split: 85% train
    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    # Second split: from trainval, take 15/85 as val, so val is 15% of total
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size= 15/85,
        stratify=labels[trainval_idx],
        random_state=RANDOM_STATE
    )

    # build simple dataframe
    df = pd.DataFrame({
        "index": all_indices,
        "label": labels,
        "split": "",
    })

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx,  "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Quick sanity checks
    assert (df["split"] != "").all(), "Some rows did not get a split assignment"
    assert df["split"].value_counts().sum() == len(labels)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # Report results
    print(f"Saved {len(df)} rows to {OUT_PATH}\n")
    print("Split sizes:")
    print(df["split"].value_counts())
    print("\nClass proportions per split (should be very similar across rows):")
    proportions = (
        df.groupby(["split", "label"]).size()
        .unstack(fill_value=0)
        .div(df["split"].value_counts(), axis=0)
        .round(4)
    )
    print(proportions)


if __name__ == "__main__":
    main()