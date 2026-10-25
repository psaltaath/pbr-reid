#!/usr/bin/env python3
"""Retrieval Comparison: Baseline vs BRM

Visualises top-K gallery retrieval for selected queries, comparing a
baseline (no-BRM) checkpoint against a BRM-trained checkpoint.

Layout per figure (one figure = one query):
  Row 0  [No BRM]  : Query | Rank-1 | Rank-2 | ... | Rank-K
  Row 1  [With BRM]: Query | Rank-1 | Rank-2 | ... | Rank-K

  Green border → correct match (same PID as query)
  Red border   → wrong match (different PID)
  Blue border  → query image

Only plots queries where at least one retrieval error exists (not trivially
perfect), ranked by how much BRM improves over the baseline.

Usage
-----
    uv run tools/retrieval_comparison.py \\
        --dataset     market1501 \\
        --no_brm_ckpt runs/market1501_l5_mixup/latest.pth \\
        --brm_ckpt    runs/market1501/latest.pth \\
        --config      market1501 \\
        --top_k       6 \\
        --num_figures 6 \\
        --output_dir  outputs/retrieval_comparison
"""

import argparse
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ConfigManager  # noqa: E402
from models.backbones import build_backbone_from_config  # noqa: E402
from models.prototype_model import PrototypeReID  # noqa: E402
from training.data_utils import (  # noqa: E402
    create_transforms,
    data_root,
    val_collate_fn,
)
from datasets.bases import ImageDataset
from datasets.market1501 import Market1501
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.veri import VeRi
from datasets.msmt17v2 import MSMT17V2
from evaluation.reid import extract_features

console = Console()

_DATASET_MAP: dict[str, type] = {
    "market1501": Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "veri": VeRi,
    "msmt17v2": MSMT17V2,
}
_DATASET_NAME_ALIAS: dict[str, str] = {
    "market1501": "market1501",
    "dukemtmc": "dukemtmcreid",
    "msmt17": "msmt17v2",
}

# Colours (hex without '#') used in LaTeX \definecolor
_HEX_CORRECT   = "27ae60"
_HEX_WRONG     = "e74c3c"
_HEX_QUERY     = "2980b9"
_HEX_MILESTONE = "c07a00"

# Only these rank positions receive a column-header label
_RANK_MILESTONES = {1, 5, 10}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_epoch(ckpt_path: str) -> str:
    stem = Path(ckpt_path).stem
    for part in stem.split("_"):
        if part.isdigit():
            return f"Epoch {part}"
    return stem


def _load_model(
    config,
    num_classes: int,
    ckpt_path: str,
    device: torch.device,
) -> PrototypeReID:
    backbone, _ = build_backbone_from_config(config)
    model = PrototypeReID(backbone, num_classes=num_classes, ema_decay=config.model.ema_decay)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RetrievalComparison:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = None
        self.num_classes: int = 0
        self.query_loader: DataLoader | None = None
        self.gallery_loader: DataLoader | None = None
        # Ordered lists of (img_path, pid, camid, trackid)
        self.query_data: list | None = None
        self.gallery_data: list | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        console.print(
            Panel.fit(
                f"[bold cyan]Retrieval Comparison: No-BRM vs BRM[/bold cyan]\n"
                f"Dataset     : {self.args.dataset}\n"
                f"No-BRM ckpt : {self.args.no_brm_ckpt}\n"
                f"BRM ckpt    : {self.args.brm_ckpt}\n"
                f"Top-K       : {self.args.top_k}\n"
                f"Num figures : {self.args.num_figures}\n"
                f"Device      : {self.device}",
                title="Configuration",
            )
        )

        config_manager = ConfigManager()
        self.config = config_manager.load_config(config_name=[self.args.config])

        ds_name = _DATASET_NAME_ALIAS.get(
            self.args.dataset.lower(), self.args.dataset.lower()
        )
        self.config.data.dataset_name = ds_name

        if ds_name not in _DATASET_MAP:
            raise ValueError(
                f"Unsupported dataset '{ds_name}'. "
                f"Supported: {list(_DATASET_MAP.keys())}"
            )

        console.print("[cyan]Building the encoder transform...[/cyan]")
        _, base_transform = build_backbone_from_config(self.config)
        _, val_transforms = create_transforms(self.config, base_transform)

        raw_dataset = _DATASET_MAP[ds_name](root=data_root(self.config))
        num_classes = raw_dataset.num_train_pids
        self.num_classes = num_classes
        self.query_data = raw_dataset.query    # (path, pid, camid, trackid)
        self.gallery_data = raw_dataset.gallery

        self.query_loader = DataLoader(
            ImageDataset(self.query_data, val_transforms),
            batch_size=self.config.data.test_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=val_collate_fn,
        )
        self.gallery_loader = DataLoader(
            ImageDataset(self.gallery_data, val_transforms),
            batch_size=self.config.data.test_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=val_collate_fn,
        )

        console.print(
            f"[green]✓ Query: {len(self.query_data)}  "
            f"Gallery: {len(self.gallery_data)}  "
            f"Classes: {num_classes}[/green]"
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract(
        self, ckpt_path: str, stage: str
    ) -> tuple[np.ndarray, list[int], list[int], np.ndarray, list[int], list[int]]:
        console.print(f"\n[cyan]Loading {stage}: {ckpt_path}[/cyan]")
        model = _load_model(self.config, self.num_classes, ckpt_path, self.device)

        q_feats, q_pids, q_camids = extract_features(
            model, self.query_loader, self.device, desc=f"Query   [{stage}]"
        )
        g_feats, g_pids, g_camids = extract_features(
            model, self.gallery_loader, self.device, desc=f"Gallery [{stage}]"
        )
        del model
        torch.cuda.empty_cache()

        def _to_int_list(x) -> list[int]:
            return [int(v) for v in x]

        return (
            q_feats.numpy().astype(np.float32),
            _to_int_list(q_pids),
            _to_int_list(q_camids),
            g_feats.numpy().astype(np.float32),
            _to_int_list(g_pids),
            _to_int_list(g_camids),
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(
        self,
        q_feats: np.ndarray,
        q_pids: list[int],
        q_camids: list[int],
        g_feats: np.ndarray,
        g_pids: list[int],
        g_camids: list[int],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Rank gallery per query (cosine similarity, junk removed).

        Returns
        -------
        ranked_indices : list[np.ndarray(top_k,)]
        correct_masks  : list[np.ndarray(top_k, bool)]
        """
        K = self.args.top_k
        g_pids_arr = np.array(g_pids, dtype=np.int32)
        g_camids_arr = np.array(g_camids, dtype=np.int32)

        ranked_indices: list[np.ndarray] = []
        correct_masks: list[np.ndarray] = []

        for qi in range(len(q_feats)):
            q_pid = q_pids[qi]
            q_camid = q_camids[qi]

            # Cosine similarity (features are L2-normalised)
            sim = (q_feats[qi] @ g_feats.T).copy()

            # Junk: same identity captured from the same camera (trivial match)
            junk = (g_pids_arr == q_pid) & (g_camids_arr == q_camid)
            sim[junk] = -np.inf

            ranked = np.argsort(-sim)[:K]
            correct = g_pids_arr[ranked] == q_pid

            ranked_indices.append(ranked)
            correct_masks.append(correct)

        return ranked_indices, correct_masks

    # ------------------------------------------------------------------
    # Query selection
    # ------------------------------------------------------------------

    def _select_queries(
        self,
        correct_nb: list[np.ndarray],
        correct_b: list[np.ndarray],
    ) -> list[int]:
        """
        Pick the most interesting queries:
          1. At least one retrieval error in either model (not trivially perfect).
          2. Ranked by BRM improvement (correct_b - correct_nb), descending.
          3. Break ties: queries where results differ between models.
        """
        scores: list[tuple[int, int, int, int]] = []
        for qi in range(len(correct_nb)):
            nb_sum = int(correct_nb[qi].sum())
            b_sum = int(correct_b[qi].sum())
            improvement = b_sum - nb_sum
            has_error = int(not (correct_nb[qi].all() and correct_b[qi].all()))
            differs = int(not np.array_equal(correct_nb[qi], correct_b[qi]))
            scores.append((improvement, differs, has_error, qi))

        # Sort: improvement desc → differs desc → has_error desc
        scores.sort(key=lambda x: (-x[0], -x[1], -x[2]))

        selected = [qi for _, _, has_err, qi in scores if has_err]
        selected = selected[: self.args.num_figures]

        if len(selected) < self.args.num_figures:
            console.print(
                f"[yellow]⚠ Only {len(selected)} queries with errors found "
                f"(requested {self.args.num_figures})[/yellow]"
            )

        return selected

    # ------------------------------------------------------------------
    # Image collection
    # ------------------------------------------------------------------

    def _collect_images_for_figure(
        self,
        fi: int,
        qi: int,
        ranked_nb: np.ndarray,
        ranked_b: np.ndarray,
        correct_nb: np.ndarray,
        correct_b: np.ndarray,
        images_dir: Path,
        no_brm_label: str,
        brm_label: str,
    ) -> dict:
        """Copy original dataset images into images_dir; return figure spec."""
        K = self.args.top_k
        q_path, q_pid, _, _ = self.query_data[qi]

        # Query image — shared across both rows
        q_ext = Path(q_path).suffix
        q_img_name = f"q{fi + 1:02d}_pid{q_pid:04d}{q_ext}"
        shutil.copy2(q_path, images_dir / q_img_name)

        rows: list[dict] = []
        for row_idx, (ranked, correct, row_label) in enumerate([
            (ranked_nb, correct_nb, f"No BRM\n({no_brm_label})"),
            (ranked_b,  correct_b,  f"With BRM\n({brm_label})"),
        ]):
            cells: list[dict] = []
            for k, (gidx, is_correct) in enumerate(zip(ranked, correct)):
                g_path, g_pid, _, _ = self.gallery_data[gidx]
                ext = Path(g_path).suffix
                status = "c" if is_correct else "w"
                img_name = (
                    f"g{fi + 1:02d}_row{row_idx}_r{k + 1:02d}"
                    f"_{status}_pid{g_pid:04d}{ext}"
                )
                shutil.copy2(g_path, images_dir / img_name)
                cells.append({"img_name": img_name, "pid": g_pid, "correct": bool(is_correct)})
            rows.append({"label": row_label, "cells": cells})

        nb_n = int(correct_nb.sum())
        b_n  = int(correct_b.sum())
        return {
            "fi":         fi,
            "q_img_name": q_img_name,
            "q_pid":      q_pid,
            "nb_n":       nb_n,
            "b_n":        b_n,
            "delta":      b_n - nb_n,
            "rows":       rows,
        }

    # ------------------------------------------------------------------
    # LaTeX export
    # ------------------------------------------------------------------

    # Queries stacked per page, following visualizations-supplementary_template.tex
    _QUERIES_PER_PAGE = 3

    def _write_latex(self, figure_specs: list[dict], output_dir: Path) -> None:
        """Write a self-contained LaTeX document following the supplementary
        template style: each page stacks up to three query cases inside a single
        \\resizebox'd tabular, every case a No-BRM row above a BRM row.
        Compile with: pdflatex retrieval_comparison.tex
        """
        latex_dir = output_dir / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)

        K            = self.args.top_k
        no_brm_label = _parse_epoch(self.args.no_brm_ckpt)
        brm_label    = _parse_epoch(self.args.brm_ckpt)
        ds           = self.args.dataset.upper()

        # Fixed cell box; \resizebox scales the whole table to \textwidth, so
        # only the cell aspect ratio matters. Every image is forced into a 2:1
        # (height:width) box for a uniform grid across datasets.
        imgw = "20mm"
        imgh = "40mm"

        # ---- LaTeX cell helpers -----------------------------------------
        def _img_cell(img_name: str, pid: int, color: str) -> str:
            # Force a fixed 2:1 (height:width) box for every image so all cells
            # are identical in size regardless of the source aspect ratio.
            return (
                "\\begin{tabular}{@{}c@{}}\n"
                f"    \\adjincludegraphics[width=\\imgw,height=\\imgh,cfbox={color} 2pt 0pt]{{images/{img_name}}}\\\\[-1pt]\n"
                f"    {{\\tiny\\textbf{{\\textcolor{{{color}}}{{ID:~{pid}}}}}}}\n"
                "  \\end{tabular}"
            )

        def _row_label(text: str) -> str:
            return (
                "\\rotatebox{90}{\\parbox{20mm}{\\centering\\footnotesize\\textbf{"
                + text
                + "}}}"
            )

        def _query_rows(spec: dict) -> str:
            """Two stacked rows (No BRM, then BRM) for a single query case."""
            rows_tex: list[str] = []
            labels = ["No BRM", "BRM"]
            for row, label in zip(spec["rows"], labels):
                cells = [_row_label(label)]
                cells.append(_img_cell(spec["q_img_name"], spec["q_pid"], "querycol"))
                for cell in row["cells"]:
                    c = "correct" if cell["correct"] else "wrong"
                    cells.append(_img_cell(cell["img_name"], cell["pid"], c))
                rows_tex.append(" &\n    ".join(cells) + r" \\[2pt]")
            return "\n  ".join(rows_tex)

        # tabular column spec: 6mm rotated-label column + Query + K rank columns
        col_spec = "m{6mm}" + "c" * (K + 1)

        # Column headers: only R-1, R-5, R-10 are labelled
        hdr_cells = ["", r"\textbf{Query}"]
        for rank in range(1, K + 1):
            if rank in _RANK_MILESTONES:
                hdr_cells.append(f"\\textcolor{{milestone}}{{\\textbf{{R-{rank}}}}}")
            else:
                hdr_cells.append("")
        header_row = " & ".join(hdr_cells) + r" \\[2pt]"

        # ---- One figure per page, up to three query cases each -----------
        per_page = self._QUERIES_PER_PAGE
        figures_tex: list[str] = []
        for page_no, start in enumerate(
            range(0, len(figure_specs), per_page), start=1
        ):
            page = figure_specs[start : start + per_page]
            blocks = [_query_rows(spec) for spec in page]
            body = "\n  \\hline \\\\[0.1pt]\n\n  ".join(blocks)
            pids = ", ".join(str(spec["q_pid"]) for spec in page)

            fig_tex = (
                "\\begin{figure}[p]\n"
                "\\resizebox{\\textwidth}{!}{%\n"
                "  \\centering\n\n"
                "  {\\setlength{\\tabcolsep}{2pt}%\n"
                f"  \\begin{{tabular}}{{{col_spec}}}\n"
                f"  {header_row}\n"
                f"  {body}\n"
                "  \\end{tabular}}\n\n"
                "}\n"
                "  \\caption{%\n"
                f"    Top-{K} retrieval results for {len(page)} {ds} "
                f"queries (PIDs {pids}), comparing the baseline without BRM "
                f"(upper row of each pair) against the BRM-trained model "
                f"(lower row).\n"
                "    \\textcolor{correct}{Green} = correct match;\\;\n"
                "    \\textcolor{wrong}{red} = incorrect match;\\;\n"
                "    \\textcolor{querycol}{blue} = query image.}\n"
                f"  \\label{{fig:retrieval-page-{page_no:02d}}}\n"
                "\\end{figure}\n"
            )
            figures_tex.append(fig_tex)

        # ---- Preamble ---------------------------------------------------
        preamble = (
            "% Auto-generated by ablations/retrieval_comparison.py\n"
            "% Compile: pdflatex retrieval_comparison.tex\n"
            "\\documentclass[10pt]{article}\n"
            "\\usepackage[margin=1.5cm,a4paper]{geometry}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage[export]{adjustbox}\n"
            "\\usepackage{bm}\n"
            "\\usepackage{caption}\n"
            "\\usepackage{rotating}\n"
            "\\usepackage{array}\n"
            "\\usepackage{xcolor}\n"
            "\\usepackage[hidelinks]{hyperref}\n"
            "\n"
            f"\\definecolor{{correct}}{{HTML}}{{{_HEX_CORRECT}}}\n"
            f"\\definecolor{{wrong}}{{HTML}}{{{_HEX_WRONG}}}\n"
            f"\\definecolor{{querycol}}{{HTML}}{{{_HEX_QUERY}}}\n"
            f"\\definecolor{{milestone}}{{HTML}}{{{_HEX_MILESTONE}}}\n"
            "\n"
            f"\\newlength{{\\imgw}}\\setlength{{\\imgw}}{{{imgw}}}\n"
            f"\\newlength{{\\imgh}}\\setlength{{\\imgh}}{{{imgh}}}\n"
            "\n"
            "\\pagestyle{empty}\n"
            "\n"
            "\\begin{document}\n\n"
        )

        tex_path = latex_dir / "retrieval_comparison.tex"
        tex_path.write_text(
            preamble + "\n".join(figures_tex) + "\n\\end{document}\n",
            encoding="utf-8",
        )
        console.print(f"[green]✓ LaTeX : {tex_path}[/green]")
        console.print(f"[green]✓ Images: {(latex_dir / 'images').resolve()}[/green]")

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.setup()

        # --- Extract features for both models ----------------------------
        (q_feats_nb, q_pids_nb, q_camids_nb,
         g_feats_nb, g_pids_nb, g_camids_nb) = self._extract(
            self.args.no_brm_ckpt, "No-BRM"
        )
        (q_feats_b, q_pids_b, q_camids_b,
         g_feats_b, g_pids_b, g_camids_b) = self._extract(
            self.args.brm_ckpt, "BRM"
        )

        # --- Rank --------------------------------------------------------
        console.print("[cyan]Ranking gallery for both models...[/cyan]")
        ranked_nb, correct_nb = self._rank(
            q_feats_nb, q_pids_nb, q_camids_nb,
            g_feats_nb, g_pids_nb, g_camids_nb,
        )
        ranked_b, correct_b = self._rank(
            q_feats_b, q_pids_b, q_camids_b,
            g_feats_b, g_pids_b, g_camids_b,
        )
        del q_feats_nb, g_feats_nb, q_feats_b, g_feats_b
        console.print("[green]✓ Ranking done[/green]")

        # --- Select interesting queries ----------------------------------
        selected = self._select_queries(correct_nb, correct_b)

        tbl = Table(
            title="Selected queries",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("#", justify="right")
        tbl.add_column("Query idx", justify="right")
        tbl.add_column("PID", justify="right")
        tbl.add_column(f"No-BRM /{self.args.top_k}", justify="right")
        tbl.add_column(f"BRM /{self.args.top_k}", justify="right")
        tbl.add_column("Δ", justify="right")
        for fi, qi in enumerate(selected):
            nb_c = int(correct_nb[qi].sum())
            b_c = int(correct_b[qi].sum())
            d = b_c - nb_c
            sign = "+" if d >= 0 else ""
            clr = "green" if d > 0 else ("red" if d < 0 else "white")
            tbl.add_row(
                str(fi + 1),
                str(qi),
                str(q_pids_nb[qi]),
                str(nb_c),
                str(b_c),
                f"[{clr}]{sign}{d}[/]",
            )
        console.print(tbl)

        # --- Build LaTeX output ------------------------------------------
        output_dir = Path(self.args.output_dir) / "retrieval_comparison"
        images_dir = output_dir / "latex" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        no_brm_label = _parse_epoch(self.args.no_brm_ckpt)
        brm_label    = _parse_epoch(self.args.brm_ckpt)

        figure_specs: list[dict] = []
        for fi, qi in enumerate(selected):
            spec = self._collect_images_for_figure(
                fi, qi,
                ranked_nb[qi], ranked_b[qi],
                correct_nb[qi], correct_b[qi],
                images_dir,
                no_brm_label,
                brm_label,
            )
            figure_specs.append(spec)
            console.print(f"[green]✓ Collected images for figure {fi + 1}/{len(selected)}[/green]")

        self._write_latex(figure_specs, output_dir)

        console.print(
            Panel.fit(
                f"[bold green]Done![/bold green]\n"
                f"LaTeX  : {(output_dir / 'latex' / 'retrieval_comparison.tex').resolve()}\n"
                f"Images : {images_dir.resolve()}\n"
                f"Compile: pdflatex retrieval_comparison.tex",
                title="Complete",
            )
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise retrieval comparison: baseline vs BRM-trained model."
    )
    parser.add_argument(
        "--dataset",
        default="market1501",
        help="Dataset name: market1501 | dukemtmc | msmt17",
    )
    parser.add_argument(
        "--no_brm_ckpt",
        required=True,
        help="Path to baseline (no-BRM) checkpoint .pth",
    )
    parser.add_argument(
        "--brm_ckpt",
        required=True,
        help="Path to BRM-trained checkpoint .pth",
    )
    parser.add_argument(
        "--config",
        default="market1501",
        help="Config name (e.g. market1501) or path to a YAML config file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of retrieved images per row (default: 10)",
    )
    parser.add_argument(
        "--num_figures",
        type=int,
        default=6,
        help="Number of query comparison figures to produce (default: 6)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/retrieval_comparison",
        help="Output root directory (default: outputs/retrieval_comparison)",
    )

    args = parser.parse_args()
    RetrievalComparison(args).run()


if __name__ == "__main__":
    main()
