"""PyTorch Dataset for Galaxy10 DECaLS HDF5 file.

Three design decisions worth understanding:

1. LAZY HDF5 OPENING. The file handle is opened inside __getitem__ on
   first access, NOT in __init__. This is critical for multi-worker
   DataLoaders. On Windows, DataLoader workers use 'spawn', which
   pickles the Dataset and reconstructs it in each child process.
   h5py file handles are NOT picklable. Even on Linux ('fork'), a
   handle shared between processes corrupts under concurrent access.
   Each worker must open its own handle, lazily, on first read.

2. SUBSET BY INDEX. The Dataset takes a list of integer indices into
   the full HDF5 array. Train/val/test splits are passed in this way.
   Same Dataset class for all three splits, just different indices.

3. LABELS LOADED EAGERLY. Labels are tiny (17k integers, ~140 KB) so
   they fit in memory trivially. Only images are loaded on demand.
   Having labels in memory makes class-weight computation in the
   trainer cheap and lets us skip an HDF5 read for every label lookup.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GalaxyDataset(Dataset):
    """Galaxy10 DECaLS image classification dataset.

    Parameters
    ----------
    h5_path : str or Path
        Path to Galaxy10_DECals.h5
    indices : array-like of int
        Indices into the full HDF5 array for this split.
    transform : callable, optional
        Image transform pipeline (from transforms.py).
    """

    def __init__(self, h5_path, indices, transform = None, preload = True):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.preload = preload
        self.h5_file = None # Only used if preload set to false

        # Save the labels for whatever indices we want
        with h5py.File(self.h5_path, "r") as f:
            self.labels = np.array(f["ans"])[self.indices]
            if preload:
                print(f"Loading full HDF5 image array into RAM "
                    f"({len(self.indices)} target images)...")
                # One sequential read of the whole array, then numpy-index.
                # Much faster than h5py fancy indexing across many chunks.
                all_images = np.array(f["images"])
                self._images = all_images[self.indices]
                del all_images  # free the full array, keep only our slice
                print(
                    f"Preloaded {len(self._images)} images "
                    f"({self._images.nbytes / 1e9:.2f} GB) into RAM"
                )
            

    def _ensure_open(self):
        """Open the HDF5 file on first access. Called per worker."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
    
    
    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, i):
        if self.preload:
            image_array = self._images[i]
        else:
            self._ensure_open()
            image_array = self.h5_file["images"][int(self.indices[i])]

        label = int(self.labels[i])
        image = Image.fromarray(image_array)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    

    def __del__(self):
        # Defensive cleanup. h5py usually handles this on GC, but
        # being explicit is harmless.
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
    
    def __getstate__(self):
        # Drop the open HDF5 handle before pickling. Workers will reopen
        # their own handle lazily via _ensure_open on first __getitem__.
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state


def load_split(splits_csv, split_name):
    """Read indices for a named split from splits.csv.

    Returns a numpy array of integer indices into the full dataset.
    """

    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split_name!r}")
    df = pd.read_csv(splits_csv)
    return df.loc[df["split"] == split_name, "index"].to_numpy()