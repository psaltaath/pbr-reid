"""Checkpointing, progress tables and best-model tracking."""

import os
from pathlib import Path

import torch
from rich import box
from rich.table import Table

_LOSS_LABELS = {
    "head_loss": "Head",
    "fpt_loss": "L_fpt",
    "bht_loss": "L_bht",
    "center_loss": "L_cen",
    "norm_loss": "L_norm",
    "brm_loss": "L_BRM",
    "sep_loss": "L_sep",
    "sep_pairs": "sep pairs",
}


def save_checkpoint(model, optimizer_state, epoch, mAP, rank1_acc, checkpoint_dir):
    """Save ``checkpoint_epoch_<epoch>.pth`` and mirror it to ``latest.pth``."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(optimizer_state, "state_dict"):
        optimizer_state = optimizer_state.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state,
        "mAP": mAP,
        "rank1_acc": rank1_acc,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
    torch.save(checkpoint, os.path.join(checkpoint_dir, "latest.pth"))


def load_checkpoint(checkpoint_path, model, optimizer=None, strict=True):
    """Load weights (and optionally optimizer state); returns the metadata dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint.get("epoch", 0),
        "mAP": checkpoint.get("mAP", 0.0),
        "rank1_acc": checkpoint.get("rank1_acc", 0.0),
    }


def create_training_table(epoch, train_loss, train_loss_dict, all_metrics, lr):
    table = Table(title=f"Training Progress - Epoch {epoch}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Epoch", str(epoch))
    table.add_row("Training Loss", f"{train_loss:.4f}")
    for key, label in _LOSS_LABELS.items():
        if key in train_loss_dict and key != "sep_pairs":
            table.add_row(label, f"{train_loss_dict[key]:.4f}")
    for variant, metrics in all_metrics.items():
        prefix = f"{variant}/" if len(all_metrics) > 1 and variant != "default" else ""
        for m in ("mAP", "R-1", "R-5", "R-10"):
            table.add_row(f"{prefix}{m}", f"{metrics[m]:.2f}%")
    table.add_row("Learning Rate", f"{lr:.2e}")
    return table


def format_loss_string(loss_value, loss_dict):
    parts = [f"Total: {loss_value:.4f}"]
    for key, label in _LOSS_LABELS.items():
        if key in loss_dict:
            v = loss_dict[key]
            parts.append(f"{label}: {v:.4f}" if isinstance(v, float) else f"{label}: {v}")
    return ", ".join(parts)


class BestModelTracker:
    def __init__(self, metric="mAP"):
        self.metric = metric
        self.best_value = 0.0
        self.best_epoch = 0

    def update(self, epoch, mAP, rank1_acc):
        current = mAP if self.metric == "mAP" else rank1_acc
        if current > self.best_value:
            self.best_value = current
            self.best_epoch = epoch
            return True
        return False

    def get_best_info(self):
        return self.best_value, self.best_epoch
