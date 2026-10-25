"""System / run configuration."""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class SystemConfig(BaseConfig):
    """Device, checkpoints, logging and reproducibility."""

    device: str = "cuda"  # cuda | cpu
    mixed_precision: bool = True  # torch.autocast + GradScaler on CUDA

    # Best-mAP checkpoints (checkpoint_epoch_<N>.pth + latest.pth) and the
    # console mirror train.log are written here.
    checkpoint_dir: str = "checkpoints"
    save_logs: bool = True

    seed: int = 42
    deterministic: bool = True
