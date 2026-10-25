#!/usr/bin/env python3
"""Evaluate a PBR-ReID checkpoint (mAP / CMC; optional k-reciprocal re-ranking).

    uv run eval.py market1501 --checkpoint runs/market/latest.pth
    uv run eval.py vehicleid --checkpoint ckpt.pth --data.vehicleid_test_variants S,M,L
    uv run eval.py veri --checkpoint ckpt.pth --rerank

Positional names and ``--section.key`` overrides work exactly as in train.py; the
checkpoint must have been trained with a compatible backbone / input size.
"""

import argparse
import sys

import torch
from rich.console import Console
from rich.table import Table

from evaluation.reid import validate_reid_both, validate_reid_model
from models.prototype_model import PrototypeReID
from training import create_data_loaders, get_configuration, load_checkpoint, setup_deterministic_training


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint (.pth)")
    parser.add_argument("--rerank", action="store_true", help="also report k-reciprocal re-ranked metrics")
    args, rest = parser.parse_known_args()
    config = get_configuration(rest)
    config.system.save_logs = False
    setup_deterministic_training(config)

    console = Console()
    _, test_loaders, num_classes, backbone, _ = create_data_loaders(config)
    device = torch.device("cuda" if config.system.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = PrototypeReID(backbone, num_classes=num_classes, ema_decay=config.model.ema_decay).to(device)
    meta = load_checkpoint(args.checkpoint, model)
    console.print(f"[cyan]Loaded {args.checkpoint} (epoch {meta['epoch']}, train-time mAP {meta['mAP']:.2f}%)[/cyan]")

    E = config.evaluation
    rerank = args.rerank or E.rerank
    table = Table(title=f"{config.data.dataset_name} — {args.checkpoint}")
    for col in ("Test set", "Protocol", "mAP", "R-1", "R-5", "R-10", "R-20"):
        table.add_column(col, justify="right" if col not in ("Test set", "Protocol") else "left")
    for name, (q, g) in test_loaders.items():
        if rerank:
            plain, rr = validate_reid_both(
                model, q, g, device, max_rank=E.max_rank,
                rerank_k1=E.rerank_k1, rerank_k2=E.rerank_k2, rerank_lambda=E.rerank_lambda,
            )
            rows = (("plain", plain), ("re-ranked", rr))
        else:
            rows = (("plain", validate_reid_model(model, q, g, device, max_rank=E.max_rank)),)
        for tag, m in rows:
            table.add_row(name, tag, *(f"{m[k]:.2f}" for k in ("mAP", "R-1", "R-5", "R-10", "R-20")))
    console.print(table)


if __name__ == "__main__":
    sys.exit(main())
