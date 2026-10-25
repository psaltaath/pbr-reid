"""
Setup utilities for deterministic training and data loading.
"""

import os
import random
import numpy as np
import torch
from rich.console import Console
from config import UnifiedConfig


def setup_deterministic_training(config: UnifiedConfig):
    """
    Setup deterministic training environment for reproducible results.

    Args:
        config: Unified configuration containing system settings
    """
    console = Console()
    seed = config.system.seed
    deterministic = config.system.deterministic

    console.print("[cyan]🎲 Setting up deterministic training environment...[/cyan]")
    console.print(f"  • Seed: {seed}")
    console.print(f"  • Deterministic: {deterministic}")

    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure PyTorch deterministic behavior
    if deterministic:
        # Make PyTorch operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for deterministic behavior
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Enable deterministic operations (PyTorch 1.12+).
        # warn_only=True: still use deterministic algorithms everywhere they
        # exist, but fall back (with a warning) instead of raising for the few
        # ops that lack a deterministic CUDA kernel — notably the bicubic
        # positional-embedding interpolation backward that any ViT run off its
        # pretrain resolution triggers (e.g. CLIP/DINOv2/ImageNet ViT at
        # 384x192 / 392x196). DINOv3 is unaffected (it does not hit that op).
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            console.print("  • [green]✓[/green] Enabled deterministic algorithms")
        except Exception as e:
            console.print(
                f"  • [yellow]⚠[/yellow] Could not enable deterministic algorithms: {e}"
            )
    else:
        # Enable benchmark mode for better performance (non-deterministic)
        torch.backends.cudnn.benchmark = True

    console.print("[green]✅ Deterministic setup complete![/green]\n")


def worker_init_fn(worker_id):
    """
    Initialize worker processes with different seeds for deterministic data loading.
    Each worker gets a different but deterministic seed.

    Args:
        worker_id: Worker process ID from DataLoader
    """
    # Get the initial seed from the main process
    worker_seed = torch.initial_seed() % 2**32
    # Make sure each worker has a different seed
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
