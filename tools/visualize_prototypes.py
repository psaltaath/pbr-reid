#!/usr/bin/env python3
"""
Script to load a trained model from checkpoint and visualize prototypes using t-SNE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
from pathlib import Path

# Import project modules (tools/ is run from the repo root)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ConfigManager  # noqa: E402
from models.backbones import build_backbone_from_config  # noqa: E402
from models.prototype_model import PrototypeReID  # noqa: E402

# Rich imports for beautiful logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize rich console
console = Console()


def load_model_from_checkpoint(checkpoint_path, config):
    """
    Load model from checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Unified configuration

    Returns:
        Loaded model
    """
    console.print(f"[cyan]Loading checkpoint from: {checkpoint_path}[/cyan]")

    # Build the encoder (DINOv3 or a swapped-in backbone)
    model, _ = build_backbone_from_config(config)

    # Load checkpoint; the number of identities is read from the prototype matrix
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint["model_state_dict"]["_prototypes"].shape[0]
    cls_model = PrototypeReID(model, num_classes=num_classes, ema_decay=config.model.ema_decay)

    # Load model state
    missing_keys, unexpected_keys = cls_model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )

    if missing_keys:
        console.print(f"[yellow]⚠️  Missing keys in checkpoint: {missing_keys}[/yellow]")
    if unexpected_keys:
        console.print(
            f"[yellow]⚠️  Unexpected keys in checkpoint: {unexpected_keys}[/yellow]"
        )

    console.print("[green]✓ Model loaded successfully![/green]")
    console.print(f"  • Epoch: {checkpoint.get('epoch', 'Unknown')}")
    console.print(f"  • mAP: {checkpoint.get('mAP', 'Unknown')}")
    console.print(f"  • Rank-1: {checkpoint.get('rank1_acc', 'Unknown')}")

    return cls_model


def visualize_prototypes_tsne(prototypes, save_path, perplexity=30, random_state=42):
    """
    Visualize prototypes using t-SNE.

    Args:
        prototypes: Tensor of shape (num_classes, embed_dim)
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
    """
    console.print(f"[cyan]Running t-SNE on {prototypes.shape[0]} prototypes...[/cyan]")

    # Convert to numpy
    prototypes_np = prototypes.detach().cpu().numpy()

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(
            perplexity, prototypes.shape[0] - 1
        ),  # Ensure perplexity < n_samples
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )

    prototypes_2d = tsne.fit_transform(prototypes_np)

    # Create plot
    plt.figure(figsize=(12, 10))

    # Scatter plot
    scatter = plt.scatter(
        prototypes_2d[:, 0],
        prototypes_2d[:, 1],
        c=np.arange(len(prototypes_2d)),  # Color by class index
        cmap="tab20",
        alpha=0.7,
        s=50,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Class ID")

    # Labels and title
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(
        f"Prototype Visualization (t-SNE)\n{prototypes.shape[0]} Classes, {prototypes.shape[1]}D Features"
    )

    # Add grid
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    console.print(f"[green]✓ Plot saved to: {save_path}[/green]")

    # Show plot info
    console.print("[blue]Plot statistics:[/blue]")
    console.print(f"  • Classes: {prototypes.shape[0]}")
    console.print(f"  • Feature dimension: {prototypes.shape[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model prototypes using t-SNE"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config", type=str, default="market1501", help="Named config (e.g. market1501) or YAML path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prototypes_tsne.png",
        help="Output filename for the plot (default: prototypes_tsne.png)",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    console.print(
        Panel.fit(
            Text("🔍 Prototype Visualization Script", style="bold blue"),
            border_style="blue",
        )
    )

    # Load configuration
    config_manager = ConfigManager()
    try:
        config = config_manager.load_config(config_name=[args.config])
        console.print("[green]✓ Configuration loaded successfully![/green]")
    except Exception as e:
        console.print(f"[red]❌ Error loading configuration: {e}[/red]")
        return

    # Load model from checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[red]❌ Checkpoint file not found: {checkpoint_path}[/red]")
        return

    try:
        model = load_model_from_checkpoint(checkpoint_path, config)
    except Exception as e:
        console.print(f"[red]❌ Error loading model: {e}[/red]")
        return

    # Extract prototypes
    model.eval()
    with torch.no_grad():
        prototypes = model.prototypes  # This is already normalized

    console.print(f"[blue]Prototypes shape: {prototypes.shape}[/blue]")

    # Visualize prototypes
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    try:
        visualize_prototypes_tsne(
            prototypes, output_path, perplexity=args.perplexity, random_state=args.seed
        )
    except Exception as e:
        console.print(f"[red]❌ Error during visualization: {e}[/red]")
        return

    console.print(
        Panel.fit(
            Text("✅ Visualization Complete!", style="bold green"), border_style="green"
        )
    )


if __name__ == "__main__":
    main()
