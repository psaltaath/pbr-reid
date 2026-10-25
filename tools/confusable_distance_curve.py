#!/usr/bin/env python3
"""
Confusable Pairs Distance Distribution Analysis

This script analyzes how BRM training increases the separation between initially
confusable class pairs by visualizing the distribution of prototype distances
at two training stages:
- Early checkpoint (warmup, before BRM has effect)
- Final checkpoint (after full BRM training)

The key insight: The distribution of distances should shift rightward (larger
distances = better separation) for pairs that were confusable at the early stage.

Usage:
    uv run tools/confusable_distance_curve.py \
        --dataset market1501 \
        --early_ckpt checkpoints/checkpoint_epoch_10.pth \
        --final_ckpt checkpoints/checkpoint_epoch_100.pth \
        --config market1501 \
        --similarity_threshold 0.7 \
        --output_dir ablations/results
"""

import argparse
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, ttest_rel, wilcoxon

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ConfigManager  # noqa: E402
from evaluation.reid import extract_features  # noqa: E402
from models.backbones import build_backbone_from_config  # noqa: E402
from models.prototype_model import PrototypeReID  # noqa: E402
from training.data_utils import build_dataset, create_test_loaders, create_transforms  # noqa: E402

console = Console()


class ConfusableDistanceCurve:
    """Analyzes and visualizes distance distribution changes for confusable pairs."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Stage labels shown in plots/report (parametric — the checkpoints are
        # not necessarily epoch 10/100).
        self.early_label = f"Early (Epoch {args.early_epoch})"
        self.final_label = f"Final (Epoch {args.final_epoch})"
        self.config = None
        self.query_loader = None

        # Results storage
        self.features_early = None
        self.features_final = None
        self.pids = None
        self.camids = None
        self.prototypes_early = None
        self.prototypes_final = None
        self.unique_pids = None
        self.confusable_pairs = None
        self.distances_early = None
        self.distances_final = None

    def setup(self):
        """Setup configuration and data loaders."""
        console.print(
            Panel.fit(
                "[bold cyan]Confusable Pairs Distance Distribution Analysis[/bold cyan]\n"
                f"Dataset: {self.args.dataset}\n"
                f"Early checkpoint: {self.args.early_ckpt}\n"
                f"Final checkpoint: {self.args.final_ckpt}\n"
                f"Similarity threshold: {self.args.similarity_threshold}\n"
                f"Top-K pairs: {self.args.top_k_pairs if self.args.top_k_pairs else 'All'}",
                title="Configuration",
            )
        )

        # Load configuration
        console.print(f"[cyan]Loading config from: {self.args.config}[/cyan]")
        config_manager = ConfigManager()
        self.config = config_manager.load_config(config_name=[self.args.config])

        # Override dataset name if needed
        if self.args.dataset.lower() == "market1501":
            self.config.data.dataset_name = "market1501"
        elif self.args.dataset.lower() == "msmt17":
            self.config.data.dataset_name = "msmt17v2"
        elif self.args.dataset.lower() == "dukemtmc":
            self.config.data.dataset_name = "dukemtmcreid"
        elif self.args.dataset.lower() == "cuhk03":
            self.config.data.dataset_name = "cuhk03"
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        # Build data loader (query set only)
        console.print("[cyan]Building data loader for query set...[/cyan]")
        self.backbone, base_transform = build_backbone_from_config(self.config)
        _, val_transforms = create_transforms(self.config, base_transform)
        dataset = build_dataset(self.config)
        test_loaders = create_test_loaders(self.config, dataset, val_transforms)
        # Extract query loader from test_loaders dictionary
        self.query_loader = test_loaders.get("default", list(test_loaders.values())[0])[0]
        # Store num_classes for model creation
        self.num_classes = dataset.num_train_pids
        console.print(f"[green]✓ Query set loaded: {len(self.query_loader.dataset)} samples[/green]")

    def load_checkpoint_and_extract(self, checkpoint_path, stage_name):
        """
        Load a checkpoint and extract features from query set.

        Args:
            checkpoint_path: Path to checkpoint file
            stage_name: Name for progress display (e.g., "Early" or "Final")

        Returns:
            features: (num_samples, feature_dim) tensor
            pids: List of person IDs
            camids: List of camera IDs
        """
        console.print(f"\n[cyan]Processing {stage_name} checkpoint...[/cyan]")

        # Create the prototype model around the (shared) encoder
        model = PrototypeReID(self.backbone, num_classes=self.num_classes, ema_decay=self.config.model.ema_decay)

        # Load checkpoint
        console.print(f"[cyan]Loading checkpoint: {checkpoint_path}[/cyan]")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        console.print("[green]✓ Model loaded successfully[/green]")

        # Extract features
        console.print("[cyan]Extracting features from query set...[/cyan]")
        features, pids, camids = extract_features(
            model, self.query_loader, self.device, desc=f"Extracting {stage_name} features"
        )

        console.print(
            f"[green]✓ Extracted features: {features.shape}[/green]"
        )

        # Clean up
        del model
        torch.cuda.empty_cache()

        return features, pids, camids

    def compute_prototypes(self, features, pids):
        """
        Compute class prototypes (normalized mean features per class).

        Args:
            features: (num_samples, feature_dim) tensor
            pids: List of person IDs

        Returns:
            prototypes: (num_classes, feature_dim) tensor
            unique_pids: List of unique person IDs
        """
        console.print("[cyan]Computing class prototypes...[/cyan]")

        unique_pids = sorted(list(set(pids)))
        num_classes = len(unique_pids)
        feature_dim = features.shape[1]

        prototypes = torch.zeros(num_classes, feature_dim)

        for i, pid in enumerate(unique_pids):
            pid_mask = np.array(pids) == pid
            class_features = features[pid_mask]
            prototypes[i] = class_features.mean(dim=0)

        # Normalize prototypes
        prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)

        console.print(f"[green]✓ Computed {num_classes} prototypes[/green]")
        return prototypes, unique_pids

    def identify_confusable_pairs(self, prototypes, threshold=0.7, top_k=None):
        """
        Identify confusable class pairs based on prototype similarity at EARLY checkpoint.

        Args:
            prototypes: (num_classes, feature_dim) tensor
            threshold: Similarity threshold for confusability
            top_k: If set, return only top-k most confusable pairs

        Returns:
            List of (class_i, class_j, similarity) tuples
        """
        console.print(
            f"[cyan]Identifying confusable pairs (threshold={threshold})...[/cyan]"
        )

        # Compute similarity matrix: S = P @ P^T
        similarity_matrix = prototypes @ prototypes.T
        similarity_matrix = similarity_matrix.cpu().numpy()

        # Find pairs above threshold
        confusable_pairs = []
        num_classes = len(prototypes)

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                sim = similarity_matrix[i, j]
                if sim > threshold:
                    confusable_pairs.append((i, j, sim))

        # Sort by similarity (most confusable first)
        confusable_pairs.sort(key=lambda x: x[2], reverse=True)

        if top_k:
            confusable_pairs = confusable_pairs[:top_k]

        console.print(
            f"[green]✓ Found {len(confusable_pairs)} confusable pairs[/green]"
        )

        if len(confusable_pairs) > 0:
            console.print("[yellow]Top 5 most confusable pairs:[/yellow]")
            for i, (class_i, class_j, sim) in enumerate(confusable_pairs[:5]):
                console.print(
                    f"  {i+1}. Class {class_i} <-> Class {class_j}: {sim:.4f}"
                )
        else:
            console.print("[yellow]⚠ No confusable pairs found with this threshold![/yellow]")

        return confusable_pairs

    def compute_distance_evolution(self, pairs, prototypes_early, prototypes_final):
        """
        Compute distances for confusable pairs at both checkpoints.

        Args:
            pairs: List of (class_i, class_j, similarity_early) tuples
            prototypes_early: Prototypes from early checkpoint
            prototypes_final: Prototypes from final checkpoint

        Returns:
            distances_early: np.array of shape (num_pairs,)
            distances_final: np.array of shape (num_pairs,)
        """
        console.print("[cyan]Computing distance evolution...[/cyan]")

        distances_early = []
        distances_final = []

        for class_i, class_j, sim_early in pairs:
            # Early distance: d = 1 - S
            dist_early = 1.0 - sim_early

            # Final distance: compute similarity then convert
            sim_final = (prototypes_final[class_i] @ prototypes_final[class_j]).item()
            dist_final = 1.0 - sim_final

            distances_early.append(dist_early)
            distances_final.append(dist_final)

        distances_early = np.array(distances_early)
        distances_final = np.array(distances_final)

        console.print(f"[green]✓ Computed distances for {len(pairs)} pairs[/green]")
        console.print(f"  Early: mean={distances_early.mean():.4f}, std={distances_early.std():.4f}")
        console.print(f"  Final: mean={distances_final.mean():.4f}, std={distances_final.std():.4f}")

        return distances_early, distances_final

    def plot_distribution_comparison(self, distances_early, distances_final, output_dir):
        """
        Create overlaid KDE and histogram showing distribution shift.

        Args:
            distances_early: Array of distances at early checkpoint
            distances_final: Array of distances at final checkpoint
            output_dir: Output directory for saving plots
        """
        console.print("[cyan]Creating distribution comparison plot...[/cyan]")

        # Setup style
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Color scheme
        color_early = "#2E86C1"  # Blue
        color_final = "#E67E22"  # Orange

        # ===== Panel A: KDE Curves =====
        ax_kde = axes[0]

        # Compute KDE
        kde_early = gaussian_kde(distances_early, bw_method="scott")
        kde_final = gaussian_kde(distances_final, bw_method="scott")

        # Generate points for plotting
        x_min = min(distances_early.min(), distances_final.min()) - 0.05
        x_max = max(distances_early.max(), distances_final.max()) + 0.05
        x_points = np.linspace(x_min, x_max, 500)

        # Plot KDE curves
        y_early = kde_early(x_points)
        y_final = kde_final(x_points)

        ax_kde.plot(x_points, y_early, color=color_early, linewidth=2.5, label=self.early_label)
        ax_kde.fill_between(x_points, y_early, alpha=0.2, color=color_early)

        ax_kde.plot(x_points, y_final, color=color_final, linewidth=2.5, label=self.final_label)
        ax_kde.fill_between(x_points, y_final, alpha=0.2, color=color_final)

        # Add mean lines
        mean_early = distances_early.mean()
        mean_final = distances_final.mean()

        ax_kde.axvline(mean_early, color=color_early, linestyle="--", linewidth=2, alpha=0.7)
        ax_kde.axvline(mean_final, color=color_final, linestyle="--", linewidth=2, alpha=0.7)

        # Annotate means
        ax_kde.text(mean_early, ax_kde.get_ylim()[1] * 0.95, f"μ={mean_early:.3f}",
                   ha="center", va="top", color=color_early, fontweight="bold", fontsize=10)
        ax_kde.text(mean_final, ax_kde.get_ylim()[1] * 0.95, f"μ={mean_final:.3f}",
                   ha="center", va="top", color=color_final, fontweight="bold", fontsize=10)

        ax_kde.set_xlabel("Prototype Distance (d = 1 - S)", fontsize=12, fontweight="bold")
        ax_kde.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax_kde.set_title("A. Distribution Shift (KDE)", fontsize=14, fontweight="bold", pad=15)
        ax_kde.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax_kde.grid(True, alpha=0.3)

        # ===== Panel B: Histogram =====
        ax_hist = axes[1]

        # Determine bins
        bins = np.linspace(x_min, x_max, 25)

        # Plot histograms
        ax_hist.hist(distances_early, bins=bins, alpha=0.6, color=color_early,
                    label=self.early_label, edgecolor="black", linewidth=0.5)
        ax_hist.hist(distances_final, bins=bins, alpha=0.6, color=color_final,
                    label=self.final_label, edgecolor="black", linewidth=0.5)

        # Add mean lines
        ax_hist.axvline(mean_early, color=color_early, linestyle="--", linewidth=2, alpha=0.9)
        ax_hist.axvline(mean_final, color=color_final, linestyle="--", linewidth=2, alpha=0.9)

        ax_hist.set_xlabel("Prototype Distance (d = 1 - S)", fontsize=12, fontweight="bold")
        ax_hist.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax_hist.set_title("B. Distribution Shift (Histogram)", fontsize=14, fontweight="bold", pad=15)
        ax_hist.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax_hist.grid(True, alpha=0.3, axis="y")

        # Overall title
        fig.suptitle(
            f"Confusable Pairs Distance Distribution - {self.args.dataset.upper()}\n"
            f"Confusable at Early Stage (Similarity > {self.args.similarity_threshold})",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        output_path = Path(output_dir) / f"distance_distribution_{self.args.dataset}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]✓ Saved distribution plot to: {output_path}[/green]")

        if self.args.show_plot:
            plt.show()
        else:
            plt.close()

        # ===== TikZ export of KDE panel (hand-written pgfplots, no tikzplotlib) =====
        console.print("[cyan]Exporting KDE panel as TikZ (.tex)...[/cyan]")

        tikz_path = Path(output_dir) / f"kde_distribution_{self.args.dataset}.tex"

        # Downsample to ~100 points to keep the file manageable
        step = max(1, len(x_points) // 100)
        xs = x_points[::step]
        ye = y_early[::step]
        yf = y_final[::step]
        y_max_kde = max(ye.max(), yf.max())

        def coords(x_arr, y_arr):
            return " ".join(f"({x:.6f},{y:.6f})" for x, y in zip(x_arr, y_arr))

        with open(tikz_path, "w") as f:
            f.write("% KDE Distribution Shift\n")
            f.write("% Generated by confusable_distance_curve.py\n")
            f.write("% Requires: \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}\n\n")
            f.write("\\begin{tikzpicture}\n")
            f.write("\\begin{axis}[\n")
            f.write("  width=\\linewidth,\n")
            f.write("  height=0.55\\linewidth,\n")
            f.write(f"  xmin={xs.min():.4f}, xmax={xs.max():.4f},\n")
            f.write(f"  ymin=0, ymax={y_max_kde * 1.15:.4f},\n")
            f.write("  xlabel={Prototype Distance ($d = 1 - S$)},\n")
            f.write("  ylabel={Density},\n")
            f.write(f"  title={{Distribution Shift (KDE) --- {self.args.dataset.upper()}}},\n")
            f.write("  grid=both,\n")
            f.write("  grid style={line width=0.2pt, draw=gray!30},\n")
            f.write("  legend pos=north east,\n")
            f.write("  legend style={font=\\small},\n")
            f.write("  tick label style={font=\\small},\n")
            f.write("  label style={font=\\small},\n")
            f.write("]\n\n")

            # Early KDE fill
            f.write("% Early KDE fill\n")
            f.write("\\addplot[fill=blue!20, draw=none, forget plot]\n")
            f.write(f"  coordinates {{{coords(xs, ye)}}};\n\n")

            # Early KDE line
            f.write("% Early KDE line\n")
            f.write("\\addplot[blue, thick]\n")
            f.write(f"  coordinates {{{coords(xs, ye)}}};\n")
            f.write(f"\\addlegendentry{{{self.early_label}}}\n\n")

            # Final KDE fill
            f.write("% Final KDE fill\n")
            f.write("\\addplot[fill=orange!20, draw=none, forget plot]\n")
            f.write(f"  coordinates {{{coords(xs, yf)}}};\n\n")

            # Final KDE line
            f.write("% Final KDE line\n")
            f.write("\\addplot[orange, thick]\n")
            f.write(f"  coordinates {{{coords(xs, yf)}}};\n")
            f.write(f"\\addlegendentry{{{self.final_label}}}\n\n")

            # Mean lines
            f.write("% Mean lines\n")
            f.write(f"\\addplot[blue, dashed, thick] coordinates {{({mean_early:.6f},0) ({mean_early:.6f},{y_max_kde:.6f})}};\n")
            f.write(f"\\addplot[orange, dashed, thick] coordinates {{({mean_final:.6f},0) ({mean_final:.6f},{y_max_kde:.6f})}};\n\n")

            # Mean annotations
            f.write("% Mean annotations\n")
            f.write(f"\\node[blue, font=\\small\\bfseries, above] at (axis cs:{mean_early:.6f},{y_max_kde * 0.92:.6f})\n")
            f.write(f"  {{$\\mu={mean_early:.3f}$}};\n")
            f.write(f"\\node[orange, font=\\small\\bfseries, above] at (axis cs:{mean_final:.6f},{y_max_kde * 0.92:.6f})\n")
            f.write(f"  {{$\\mu={mean_final:.3f}$}};\n\n")

            f.write("\\end{axis}\n")
            f.write("\\end{tikzpicture}\n")

        console.print(f"[green]✓ Saved KDE TikZ plot to: {tikz_path}[/green]")
        console.print(
            f"[dim]  Include in LaTeX with: \\input{{{tikz_path.name}}}[/dim]"
        )

    def plot_boxplot_comparison(self, distances_early, distances_final, output_dir):
        """
        Create boxplot/violin plot comparison.

        Args:
            distances_early: Array of distances at early checkpoint
            distances_final: Array of distances at final checkpoint
            output_dir: Output directory for saving plots
        """
        console.print("[cyan]Creating boxplot comparison...[/cyan]")

        # Setup style
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Prepare data in long format
        data = []
        for d in distances_early:
            data.append({"Stage": self.early_label, "Distance": d})
        for d in distances_final:
            data.append({"Stage": self.final_label, "Distance": d})

        import pandas as pd
        df = pd.DataFrame(data)

        # Color palette
        palette = {self.early_label: "#2E86C1", self.final_label: "#E67E22"}

        # Create violin plot
        sns.violinplot(
            data=df,
            x="Stage",
            y="Distance",
            palette=palette,
            inner="box",
            ax=ax,
        )

        # Add mean markers
        means = df.groupby("Stage")["Distance"].mean()
        positions = [0, 1]
        ax.scatter(positions, means.values, color="red", s=100, zorder=3, marker="D", label="Mean")

        # Statistical annotation
        mean_early = distances_early.mean()
        mean_final = distances_final.mean()
        delta = mean_final - mean_early
        pct_change = (delta / mean_early) * 100

        # Add significance indicator
        y_max = df["Distance"].max()
        y_range = df["Distance"].max() - df["Distance"].min()
        y_line = y_max + 0.05 * y_range

        ax.plot([0, 1], [y_line, y_line], "k-", linewidth=1.5)
        ax.text(0.5, y_line + 0.02 * y_range, "***", ha="center", fontsize=16, fontweight="bold")
        ax.text(0.5, y_line + 0.08 * y_range,
               f"Δ = +{delta:.4f} ({pct_change:+.1f}%)",
               ha="center", fontsize=11, fontweight="bold")

        ax.set_ylabel("Prototype Distance (d = 1 - S)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Stage", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Confusable Pairs Separation - {self.args.dataset.upper()}\n"
            f"Pairs Confusable at Early Stage (Similarity > {self.args.similarity_threshold})",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save figure
        output_path = Path(output_dir) / f"boxplot_comparison_{self.args.dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]✓ Saved boxplot to: {output_path}[/green]")

        if self.args.show_plot:
            plt.show()
        else:
            plt.close()

    def plot_trajectory_lines(self, distances_early, distances_final, output_dir, top_n=10):
        """
        Plot individual pair trajectories showing distance changes.

        Args:
            distances_early: Array of distances at early checkpoint
            distances_final: Array of distances at final checkpoint
            output_dir: Output directory for saving plots
            top_n: Number of pairs to highlight
        """
        console.print(f"[cyan]Creating trajectory plot (top {top_n} pairs)...[/cyan]")

        # Setup style
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot all pairs in gray
        for i in range(len(distances_early)):
            ax.plot([0, 1], [distances_early[i], distances_final[i]],
                   color="gray", alpha=0.2, linewidth=0.8)

        # Highlight top N most confusable (smallest early distances)
        top_indices = np.argsort(distances_early)[:top_n]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for rank, idx in enumerate(top_indices):
            color = colors[rank % 10]
            ax.plot([0, 1], [distances_early[idx], distances_final[idx]],
                   color=color, alpha=0.8, linewidth=2.5,
                   marker="o", markersize=8,
                   label=f"Pair {rank+1}")

        ax.set_xticks([0, 1])
        ax.set_xticklabels([self.early_label, self.final_label], fontsize=12)
        ax.set_ylabel("Prototype Distance (d = 1 - S)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Individual Pair Trajectories - {self.args.dataset.upper()}\n"
            f"Top {top_n} Most Confusable Pairs",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(-0.1, 1.1)

        plt.tight_layout()

        # Save figure
        output_path = Path(output_dir) / f"trajectory_lines_{self.args.dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]✓ Saved trajectory plot to: {output_path}[/green]")

        if self.args.show_plot:
            plt.show()
        else:
            plt.close()

    def compute_statistics(self, distances_early, distances_final, output_dir):
        """
        Compute statistical metrics and perform hypothesis tests.

        Args:
            distances_early: Array of distances at early checkpoint
            distances_final: Array of distances at final checkpoint
            output_dir: Output directory for saving report

        Returns:
            Dictionary of statistics
        """
        console.print("\n[cyan]Computing statistical metrics...[/cyan]")

        stats = {}

        # Descriptive statistics
        stats["early_mean"] = distances_early.mean()
        stats["early_std"] = distances_early.std()
        stats["early_median"] = np.median(distances_early)
        stats["early_min"] = distances_early.min()
        stats["early_max"] = distances_early.max()

        stats["final_mean"] = distances_final.mean()
        stats["final_std"] = distances_final.std()
        stats["final_median"] = np.median(distances_final)
        stats["final_min"] = distances_final.min()
        stats["final_max"] = distances_final.max()

        # Improvements
        stats["delta_mean"] = stats["final_mean"] - stats["early_mean"]
        stats["delta_median"] = stats["final_median"] - stats["early_median"]
        stats["pct_change_mean"] = (stats["delta_mean"] / stats["early_mean"]) * 100
        stats["pct_change_median"] = (stats["delta_median"] / stats["early_median"]) * 100

        # Effect size (Cohen's d)
        mean_diff = stats["delta_mean"]
        pooled_std = np.sqrt((distances_early.std()**2 + distances_final.std()**2) / 2)
        stats["cohens_d"] = mean_diff / pooled_std

        # Interpret effect size
        if abs(stats["cohens_d"]) < 0.2:
            effect_label = "negligible"
        elif abs(stats["cohens_d"]) < 0.5:
            effect_label = "small"
        elif abs(stats["cohens_d"]) < 0.8:
            effect_label = "medium"
        else:
            effect_label = "large"
        stats["effect_label"] = effect_label

        # Paired t-test
        t_stat, p_value_t = ttest_rel(distances_final, distances_early)
        stats["t_statistic"] = t_stat
        stats["p_value_ttest"] = p_value_t

        # Wilcoxon signed-rank test (non-parametric)
        w_stat, p_value_w = wilcoxon(distances_final, distances_early)
        stats["w_statistic"] = w_stat
        stats["p_value_wilcoxon"] = p_value_w

        # Significance interpretation
        def significance_stars(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return "n.s."

        stats["significance_ttest"] = significance_stars(p_value_t)
        stats["significance_wilcoxon"] = significance_stars(p_value_w)

        # Print results
        console.print("\n[bold cyan]=" * 35 + "[/bold cyan]")
        console.print("[bold cyan]Statistical Analysis Results[/bold cyan]")
        console.print("[bold cyan]=" * 35 + "[/bold cyan]")

        # Create table for descriptive statistics
        table = Table(title="\nDescriptive Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Early", justify="right", style="yellow")
        table.add_column("Final", justify="right", style="green")
        table.add_column("Δ", justify="right", style="red")

        table.add_row("Mean", f"{stats['early_mean']:.4f}", f"{stats['final_mean']:.4f}",
                     f"{stats['delta_mean']:+.4f} ({stats['pct_change_mean']:+.1f}%)")
        table.add_row("Median", f"{stats['early_median']:.4f}", f"{stats['final_median']:.4f}",
                     f"{stats['delta_median']:+.4f} ({stats['pct_change_median']:+.1f}%)")
        table.add_row("Std Dev", f"{stats['early_std']:.4f}", f"{stats['final_std']:.4f}", "")
        table.add_row("Range", f"[{stats['early_min']:.4f}, {stats['early_max']:.4f}]",
                     f"[{stats['final_min']:.4f}, {stats['final_max']:.4f}]", "")

        console.print(table)

        # Effect size
        console.print("\n[yellow]Effect Size (Cohen's d):[/yellow]")
        console.print(f"  d = {stats['cohens_d']:.4f} ({effect_label} effect)")

        # Statistical tests
        console.print("\n[yellow]Hypothesis Tests:[/yellow]")
        console.print(f"  Paired t-test:       t = {stats['t_statistic']:.4f}, "
                     f"p = {stats['p_value_ttest']:.6f} {stats['significance_ttest']}")
        console.print(f"  Wilcoxon signed-rank: W = {stats['w_statistic']:.0f}, "
                     f"p = {stats['p_value_wilcoxon']:.6f} {stats['significance_wilcoxon']}")

        # Overall interpretation
        if stats['p_value_ttest'] < 0.05 and stats['delta_mean'] > 0:
            console.print("\n[bold green]✓ Result: Highly significant improvement in separation![/bold green]")
        elif stats['p_value_ttest'] < 0.05:
            console.print("\n[bold red]✗ Result: Significant change, but decreased separation.[/bold red]")
        else:
            console.print("\n[yellow]⚠ Result: No significant change in separation.[/yellow]")

        console.print("[bold cyan]=" * 35 + "[/bold cyan]")

        # Save to file
        output_path = Path(output_dir) / f"distance_statistics_{self.args.dataset}.txt"
        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Confusable Pairs Distance Analysis - {self.args.dataset.upper()}\n")
            f.write("=" * 70 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  Dataset: {self.args.dataset}\n")
            f.write(f"  Early checkpoint: {self.args.early_ckpt}\n")
            f.write(f"  Final checkpoint: {self.args.final_ckpt}\n")
            f.write(f"  Similarity threshold: {self.args.similarity_threshold}\n")
            f.write(f"  Number of confusable pairs: {len(distances_early)}\n")
            if self.args.top_k_pairs:
                f.write(f"  Limited to top-K pairs: {self.args.top_k_pairs}\n")
            f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("Descriptive Statistics\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Early Checkpoint (Epoch {self.args.early_epoch}):\n")
            f.write(f"  Mean distance:   {stats['early_mean']:.4f} ± {stats['early_std']:.4f}\n")
            f.write(f"  Median distance: {stats['early_median']:.4f}\n")
            f.write(f"  Range:           [{stats['early_min']:.4f}, {stats['early_max']:.4f}]\n\n")

            f.write(f"Final Checkpoint (Epoch {self.args.final_epoch}):\n")
            f.write(f"  Mean distance:   {stats['final_mean']:.4f} ± {stats['final_std']:.4f}\n")
            f.write(f"  Median distance: {stats['final_median']:.4f}\n")
            f.write(f"  Range:           [{stats['final_min']:.4f}, {stats['final_max']:.4f}]\n\n")

            f.write("Improvement:\n")
            f.write(f"  Δ Mean:          {stats['delta_mean']:+.4f} ({stats['pct_change_mean']:+.1f}%)\n")
            f.write(f"  Δ Median:        {stats['delta_median']:+.4f} ({stats['pct_change_median']:+.1f}%)\n\n")

            f.write("=" * 70 + "\n")
            f.write("Effect Size\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"  Cohen's d:       {stats['cohens_d']:.4f} ({effect_label} effect)\n")
            f.write("  Interpretation:\n")
            f.write("    |d| < 0.2: negligible effect\n")
            f.write("    |d| < 0.5: small effect\n")
            f.write("    |d| < 0.8: medium effect\n")
            f.write("    |d| ≥ 0.8: large effect\n\n")

            f.write("=" * 70 + "\n")
            f.write("Hypothesis Tests\n")
            f.write("=" * 70 + "\n\n")

            f.write("Paired t-test:\n")
            f.write("  H0: Mean distance does not change from early to final checkpoint\n")
            f.write("  H1: Mean distance increases from early to final checkpoint\n")
            f.write(f"  t-statistic:     {stats['t_statistic']:.4f}\n")
            f.write(f"  p-value:         {stats['p_value_ttest']:.6f} {stats['significance_ttest']}\n\n")

            f.write("Wilcoxon Signed-Rank Test (non-parametric):\n")
            f.write("  H0: Median distance does not change from early to final checkpoint\n")
            f.write("  H1: Median distance increases from early to final checkpoint\n")
            f.write(f"  W-statistic:     {stats['w_statistic']:.0f}\n")
            f.write(f"  p-value:         {stats['p_value_wilcoxon']:.6f} {stats['significance_wilcoxon']}\n\n")

            f.write("Significance levels:\n")
            f.write("  ***: p < 0.001 (highly significant)\n")
            f.write("  **:  p < 0.01  (very significant)\n")
            f.write("  *:   p < 0.05  (significant)\n")
            f.write("  n.s.: p ≥ 0.05 (not significant)\n\n")

            f.write("=" * 70 + "\n")
            f.write("Conclusion\n")
            f.write("=" * 70 + "\n\n")

            if stats['p_value_ttest'] < 0.05 and stats['delta_mean'] > 0:
                f.write("✓ BRM training significantly improves separation between confusable pairs.\n")
                f.write(f"  The mean distance increased by {stats['delta_mean']:.4f} ")
                f.write(f"({stats['pct_change_mean']:.1f}%), with a {effect_label} effect size.\n")
                f.write("  Both parametric and non-parametric tests confirm the improvement.\n")
            elif stats['p_value_ttest'] < 0.05:
                f.write("✗ Significant change detected, but separation decreased.\n")
                f.write("  This suggests the model may have degraded during training.\n")
            else:
                f.write("⚠ No significant change in separation detected.\n")
                f.write("  The BRM effect may be too subtle or the sample size may be insufficient.\n")

            f.write("\n" + "=" * 70 + "\n")

        console.print(f"[green]✓ Saved statistical report to: {output_path}[/green]")

        # Optionally save raw data
        if self.args.save_data:
            self.save_raw_data(distances_early, distances_final, output_dir)

        return stats

    def save_raw_data(self, distances_early, distances_final, output_dir):
        """Save raw distance data to CSV."""
        import csv

        output_path = Path(output_dir) / f"distance_data_{self.args.dataset}.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "class_i", "class_j", "similarity_early",
                           "distance_early", "distance_final", "improvement"])

            for idx, (class_i, class_j, sim_early) in enumerate(self.confusable_pairs):
                dist_early = distances_early[idx]
                dist_final = distances_final[idx]
                improvement = dist_final - dist_early

                writer.writerow([
                    idx,
                    self.unique_pids[class_i],
                    self.unique_pids[class_j],
                    sim_early,
                    dist_early,
                    dist_final,
                    improvement
                ])

        console.print(f"[green]✓ Saved raw data to: {output_path}[/green]")

    def run(self):
        """Execute the full analysis pipeline."""
        # Setup
        self.setup()

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Extract features from both checkpoints
        self.features_early, self.pids, self.camids = self.load_checkpoint_and_extract(
            self.args.early_ckpt, "Early"
        )

        self.features_final, pids_final, camids_final = self.load_checkpoint_and_extract(
            self.args.final_ckpt, "Final"
        )

        # Verify consistency
        assert self.pids == pids_final, "Person IDs mismatch between checkpoints!"
        assert self.camids == camids_final, "Camera IDs mismatch between checkpoints!"

        # Compute prototypes
        self.prototypes_early, self.unique_pids = self.compute_prototypes(
            self.features_early, self.pids
        )
        self.prototypes_final, unique_pids_final = self.compute_prototypes(
            self.features_final, pids_final
        )

        assert self.unique_pids == unique_pids_final, "Class IDs mismatch!"

        # Identify confusable pairs at EARLY checkpoint
        self.confusable_pairs = self.identify_confusable_pairs(
            self.prototypes_early,
            threshold=self.args.similarity_threshold,
            top_k=self.args.top_k_pairs,
        )

        if len(self.confusable_pairs) == 0:
            console.print("[red]✗ No confusable pairs found. Try lowering the threshold.[/red]")
            return

        # Compute distance evolution
        self.distances_early, self.distances_final = self.compute_distance_evolution(
            self.confusable_pairs,
            self.prototypes_early,
            self.prototypes_final,
        )

        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        self.plot_distribution_comparison(
            self.distances_early,
            self.distances_final,
            output_dir,
        )

        self.plot_boxplot_comparison(
            self.distances_early,
            self.distances_final,
            output_dir,
        )

        if len(self.confusable_pairs) >= 5:
            self.plot_trajectory_lines(
                self.distances_early,
                self.distances_final,
                output_dir,
                top_n=min(10, len(self.confusable_pairs)),
            )

        # Compute statistics
        stats = self.compute_statistics(
            self.distances_early,
            self.distances_final,
            output_dir,
        )

        console.print("\n[bold green]✓ Analysis complete![/bold green]")
        console.print(f"[cyan]Results saved to: {output_dir}[/cyan]")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Confusable Pairs Distance Distribution Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run tools/confusable_distance_curve.py \\
      --dataset market1501 \\
      --early_ckpt checkpoints/checkpoint_epoch_10.pth \\
      --final_ckpt checkpoints/checkpoint_epoch_100.pth \\
      --config market1501 \\
      --similarity_threshold 0.7

  # Top-10 most confusable pairs only
  uv run tools/confusable_distance_curve.py \\
      --dataset market1501 \\
      --early_ckpt checkpoints/checkpoint_epoch_10.pth \\
      --final_ckpt checkpoints/checkpoint_epoch_100.pth \\
      --config market1501 \\
      --top_k_pairs 10 \\
      --save_data
        """,
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["market1501", "msmt17", "dukemtmc", "cuhk03"],
        help="Dataset name",
    )
    parser.add_argument(
        "--early_ckpt",
        type=str,
        required=True,
        help="Path to early checkpoint (e.g., epoch 10)",
    )
    parser.add_argument(
        "--final_ckpt",
        type=str,
        required=True,
        help="Path to final checkpoint (e.g., epoch 100)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Named config (e.g. market1501) or YAML path",
    )

    # Optional arguments
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for confusability (default: 0.7)",
    )
    parser.add_argument(
        "--early_epoch",
        type=int,
        default=10,
        help="Epoch number of the early checkpoint, for plot/report labels only (default: 10)",
    )
    parser.add_argument(
        "--final_epoch",
        type=int,
        default=100,
        help="Epoch number of the final checkpoint, for plot/report labels only (default: 100)",
    )
    parser.add_argument(
        "--top_k_pairs",
        type=int,
        default=None,
        help="Limit to top-K most confusable pairs (default: None, use all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/confusable_distance",
        help="Output directory for results (default: outputs/confusable_distance)",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Display plots interactively",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Save raw data to CSV",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create and run analysis
    analysis = ConfusableDistanceCurve(args)
    analysis.run()


if __name__ == "__main__":
    main()
