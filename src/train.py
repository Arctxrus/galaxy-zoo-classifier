"""Train a ResNet50 galaxy classifier on Galaxy10 DECaLS.

Two stage transfer learning:
  Stage 1: backbone frozen, train classifier head only.
  Stage 2: layer4 + head trainable, fine tune end to end at lower LR.

Examples:
  python src/train.py --stage 1 --epochs 12 --lr 1e-3
  python src/train.py --stage 2 --epochs 8 --lr 1e-4 \
                      --resume checkpoints/stage1_best.pt
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from dataset import GalaxyDataset, load_split
from model import (
    NUM_CLASSES,
    build_model,
    count_trainable_params,
    freeze_backbone,
    unfreeze_layer4,
)
from transforms import get_eval_transform, get_train_transform

def parse_args():
    """Configure script behaviour from the terminal without editing
        the code. Defines what arguments the script accepts and parses
        the values into a Python object."""
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--h5", type=str, default="data/Galaxy10_DECals.h5")
    p.add_argument("--splits", type=str, default="data/splits.csv")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to load (used for stage 2).")
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--log", type=str, default="logs/training_log.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seeds(seed):
    """ Control randomness in a run """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_labels):
    """Inverse-frequency weights for cross-entropy loss.

    sklearn's 'balanced' formula: weight_c = n_samples / (n_classes * n_c)
    Bigger weight for rarer classes. Pass into nn.CrossEntropyLoss(weight=...).
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y = train_labels
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimiser, device):
    """Trains one epoch.
    
    Does so batch by batch, gets predictions every batch and truth values,
    Then calcualtes loss, resets any gradient attributes, computes new gradients,
    and then updates parameter valuse.
    """
    model.train()
    total_loss = 0.0
    total_examples = 0
    # Iterate over each batch from data loader and save the images and labels
    # Tqdm is responsible for a progress bar
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        # Gets the probabilities per class and calculates loss
        logits = model(images)
        loss = criterion(logits, labels)

        # zero_grad: wipe stale gradients from the previous batch off every parameter
        # backward: compute fresh gradients for this batch's loss (writes to .grad)
        # step:     update every parameter's value using those gradients
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Gets average loss and multiplies it by the amount of images per batch
        total_loss += loss.item() * images.size(0)
        total_examples += images.size(0)
    return total_loss/total_examples


# This skips any back propogation/ model tuning, in test mode
@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validates model.
    
    Gets all the predictions and labels per batch, once done calculates metrics.
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="val  ", leave=False):
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        logits = model(images)
        loss = criterion(logits, labels)
        # Still calculate loss, if train loss goes down but val loss goes up
        # it's a sign of overfitting, a good checkpoint, and the metric we care more about

        total_loss += loss.item() * images.size(0)
        total_examples += images.size(0)

        # Get numpy arrays in cpu (memory) of predicted and labels
        # Argmax returns index of highest logit (predicted)
        all_preds.append(logits.argmax(dim = 1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "loss": total_loss / total_examples,
        "accuracy": (all_preds == all_labels).mean(),
        "balanced_acc": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
    }


def log_to_csv(log_path, row):
    """Append a dict as one row to a CSV. Writes the header on first call"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Check exists before opening, mode "a" creates the file otherwise
    is_new = not log_path.exists()

    # newline="" is required for the csv module on Windows
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def main():
    args = parse_args()
    set_seeds(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Numpy array of indices of where train and val images are
    train_indices = load_split(args.splits, "train")
    val_indices = load_split(args.splits, "val")

    # Dataset object, used for getting the image from indicesm, applying transform
    # and fetching the label for the image.
    train_ds = GalaxyDataset(args.h5, train_indices, get_train_transform())
    val_ds = GalaxyDataset(args.h5, val_indices, get_eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size= args.batch_size,
        shuffle=True,
        num_workers = args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size= args.batch_size,
        shuffle=False,
        num_workers = args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

    # Compute weights only from training data to avoid any leakage
    # Labels stored as an attribute
    class_weights = compute_class_weights(train_ds.labels).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    model = build_model().to(device)

    if args.resume is not None:
        print(f"Loading checkpoint from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
    
    # If at first step only train head, otherwise unfreeze layer 4 and have it train
    # to learn large galaxy shapes
    if args.stage == 1:
        freeze_backbone(model)
    else:
        unfreeze_layer4(model)

    trainable, total = count_trainable_params(model)
    print(f"Stage {args.stage}: {trainable:,} / {total:,} trainable")

    # Loss and optimiser
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr = args.lr)

    # Training loop
    best_macro_f1 = 0.0
    best_ckpt_path = out_dir / f"stage{args.stage}_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_metrics = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>2}/{args.epochs} ({elapsed:5.1f}s) | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"bacc={val_metrics['balanced_acc']:.4f} | "
            f"f1={val_metrics['macro_f1']:.4f}"
        )

        log_to_csv(args.log, {
            "stage": args.stage,
            "epoch": epoch,
            "lr": args.lr,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_balanced_acc": round(val_metrics["balanced_acc"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "epoch_seconds": round(elapsed, 2),
        })

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "stage": args.stage,
                "val_macro_f1": best_macro_f1,
            }, best_ckpt_path)
            print(f"  -> new best, saved to {best_ckpt_path}")

    print(f"\nDone. Best val macro F1: {best_macro_f1:.4f}")

if __name__ == "__main__":
    main()