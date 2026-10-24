"""
Full training and evaluation of the GA-selected architecture.

    python train.py --data-dir data --config best_model_config.json --epochs 30

Outputs:
    best_skin_model.pth      checkpoint (weights + config + class order)
    confusion_matrix.png     normalized confusion matrix
    training_report.txt      classification report and final metrics
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from model import build_model, build_optimizer
from preprocess import CLASS_NAMES, build_dataloaders, compute_class_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG: Dict = {
    "backbone": "resnet18",
    "base_filters": 32,
    "dense_units": 256,
    "num_blocks": 4,
    "dropout": 0.4,
    "learning_rate": 3e-4,
    "optimizer": "adamw",
    "batch_size": 32,
    "weight_decay": 1e-4,
    "freeze_backbone": False,
}


class FocalLoss(nn.Module):
    """Class-weighted focal loss, useful for the heavy HAM10000 imbalance."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class EarlyStopping:
    """Stop training when the monitored score stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Return True when the score improved."""
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def load_config(path: str) -> Dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        merged = {**DEFAULT_CONFIG, **config}
        print(f"Loaded GA configuration from {path}")
        return merged
    print("No GA config found; using the default ResNet18 configuration")
    return dict(DEFAULT_CONFIG)


def run_epoch(model: nn.Module, loader, criterion, optimizer=None) -> Tuple[float, List[int], List[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    seen = 0
    predictions: List[int] = []
    targets: List[int] = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        seen += images.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return total_loss / max(seen, 1), predictions, targets


def plot_confusion_matrix(matrix: np.ndarray, path: str) -> None:
    normalized = matrix.astype(float) / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(CLASS_NAMES)), [name.upper() for name in CLASS_NAMES], rotation=45)
    axis.set_yticks(range(len(CLASS_NAMES)), [name.upper() for name in CLASS_NAMES])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Normalized confusion matrix")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            axis.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                      color="white" if normalized[i, j] > 0.5 else "black", fontsize=8)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Confusion matrix saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GA-selected CNN")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="best_model_config.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--checkpoint", default="best_skin_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_dir, batch_size=int(config["batch_size"]), num_workers=args.num_workers
    )

    model = build_model(config).to(DEVICE)
    optimizer = build_optimizer(model, config)
    weights = compute_class_weights(args.data_dir).to(DEVICE)
    criterion = FocalLoss(weights) if args.loss == "focal" else nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    stopper = EarlyStopping(patience=args.patience)

    print(f"Training {config['backbone']} on {DEVICE} for up to {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_pred, val_true = run_epoch(model, val_loader, criterion)
        train_f1 = f1_score(train_true, train_pred, average="macro", zero_division=0)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        val_accuracy = accuracy_score(val_true, val_pred)
        scheduler.step(val_f1)

        improved = stopper.step(float(val_f1))
        marker = "  <- best" if improved else ""
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} f1 {train_f1:.4f} "
              f"| val loss {val_loss:.4f} f1 {val_f1:.4f} acc {val_accuracy:.4f}{marker}")

        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "class_names": CLASS_NAMES,
                    "val_macro_f1": float(val_f1),
                    "epoch": epoch,
                },
                args.checkpoint,
            )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _, test_pred, test_true = run_epoch(model, test_loader, criterion)

    report = classification_report(
        test_true, test_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=[name.upper() for name in CLASS_NAMES], digits=4, zero_division=0
    )
    accuracy = accuracy_score(test_true, test_pred)
    macro_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
    matrix = confusion_matrix(test_true, test_pred, labels=list(range(len(CLASS_NAMES))))

    summary = (
        f"Best epoch: {checkpoint['epoch']} (val macro F1 {checkpoint['val_macro_f1']:.4f})\n"
        f"Test accuracy: {accuracy:.4f}\nTest macro F1: {macro_f1:.4f}\n\n"
        f"{report}\n\nConfusion matrix (rows = true):\n{matrix}\n"
    )
    print("\n" + summary)
    with open("training_report.txt", "w", encoding="utf-8") as handle:
        handle.write(f"Configuration: {json.dumps(config, indent=2)}\n\n{summary}")
    plot_confusion_matrix(matrix, "confusion_matrix.png")
    print(f"Checkpoint: {args.checkpoint} | report: training_report.txt")


if __name__ == "__main__":
    main()
