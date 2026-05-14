"""Evaluate a trained Galaxy10 classifier on the held-out test set.

Produces:
  - Headline metrics: accuracy, balanced accuracy, macro F1.
  - Per-class precision/recall/F1 (sklearn classification report).
  - Normalised confusion matrix saved to assets/.
  - Grid of correct and incorrect predictions saved to assets/.

Example:
  python src/evaluate.py --checkpoint checkpoints/stage2_lr1e-4.pt
"""

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from dataset import GalaxyDataset, load_split
from model import build_model
from transforms import get_eval_transform


CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round smooth",
    "In-between smooth",
    "Cigar-shaped smooth",
    "Barred spiral",
    "Unbarred tight spiral",
    "Unbarred loose spiral",
    "Edge-on no bulge",
    "Edge-on with bulge",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--h5", type=str, default="data/Galaxy10_DECals.h5")
    p.add_argument("--splits", type=str, default="data/splits.csv")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="assets")
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking = True)
        logits = model(images)
        all_preds.append(logits.argmax(dim = 1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

def plot_confusion_matrix(cm, class_names, out_path):
    """Save a confusion matrix with raw counts and row-normalised colour."""
    fig, ax = plt.subplots(figsize=(9, 11))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion of true class"},
        ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test set confusion matrix\n"
                "(counts shown, colour normalised by row)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_examples(test_indices, preds, labels, h5_path,
                  class_names, out_path, n_per_row=6):
    """Save a grid: top row correct, bottom row incorrect."""
    # [0] For unwrwapping np.where
    correct_idx = np.where(preds == labels)[0]
    wrong_idx = np.where(preds != labels)[0]

    rng = np.random.default_rng(42)

    # rng.choice(array, size=N, replace=False) picks N random elements from
    # `array` without duplicates. With replace=True (the default) it can
    # pick the same element more than once. 

    pick_correct = rng.choice(
        correct_idx,
        size=min(n_per_row, len(correct_idx)),
        replace=False
        )
    
    pick_wrong = rng.choice(
        wrong_idx,
        size=min(n_per_row, len(wrong_idx)),
        replace=False
        )

    fig, axes = plt.subplots(2, n_per_row, figsize=(n_per_row * 2.5, 6))

    with h5py.File(h5_path, "r") as f:
        for row, (picks, colour) in enumerate(
            [(pick_correct, "green"), (pick_wrong, "red")]
        ):
            for j, i in enumerate(picks):
                img = f["images"][int(test_indices[i])]
                ax = axes[row, j]
                ax.imshow(img)
                ax.set_title(
                    f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}",
                    fontsize=8, color=colour,
                )
                ax.axis("off")

    fig.suptitle(
        "Top row: correct predictions    Bottom row: incorrect predictions",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_indices = load_split(args.splits, "test")
    test_ds = GalaxyDataset(args.h5, test_indices, get_eval_transform())
    test_loader = DataLoader(
        test_ds,
        batch_size= args.batch_size,
        shuffle=False,
        num_workers= args.num_workers,
        pin_memory=True
    )

    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    print(f"Loaded {args.checkpoint}")
    print(f"  trained at epoch {state.get('epoch', '?')}, "
          f"val macro F1 was {state.get('val_macro_f1', float('nan')):.4f}")

    print("\nRunning inference on test set...")
    # Use the predict method defined earlier
    preds, labels = predict(model, test_loader, device)

    # Headline metrics
    acc = (preds == labels).mean()
    bacc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    print("\n=== Test set results ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bacc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    print("\n=== Per-class report ===")
    print(classification_report(labels, preds,
                                target_names=CLASS_NAMES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Example predictions
    examples_path = out_dir / "example_predictions.png"
    plot_examples(test_indices, preds, labels, args.h5,
                  CLASS_NAMES, examples_path)
    print(f"Saved example predictions to {examples_path}")

    # Persist final metrics as a tidy CSV for the README
    metrics_path = Path("logs") / "test_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", round(acc, 4)])
        writer.writerow(["balanced_accuracy", round(bacc, 4)])
        writer.writerow(["macro_f1", round(macro_f1, 4)])
    print(f"Saved final metrics to {metrics_path}")


if __name__ == "__main__":
    main()