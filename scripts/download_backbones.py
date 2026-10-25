#!/usr/bin/env python
"""
Offline weights downloader for the swappable ReID backbones.

Run ONCE at setup, on a machine with network access. For each backbone it
resolves the pretrained checkpoint via the source library (timm / open_clip,
which pull from the HuggingFace hub), then re-saves a plain ``state_dict`` into
``<out>/<backbone>/<file>.pth`` (default out = weights/backbones). Training then loads strictly from those local
files (see models/backbones.build_backbone; set model.backbone_weights_dir or $PBR_BACKBONE_WEIGHTS to --out) with no network access.

    uv run scripts/download_backbones.py                       # all three
    uv run scripts/download_backbones.py --backbones clip      # just one
    uv run scripts/download_backbones.py --out weights/backbones  # default

Filenames come from models.backbones._WEIGHT_FILES so the loader and this
script can never drift.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbones import (  # noqa: E402
    _CLIP_MODEL,
    _TIMM_MODELS,
    _WEIGHT_FILES,
)


def _save(state: dict, out: str, backbone: str) -> str:
    dst_dir = os.path.join(out, backbone)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, _WEIGHT_FILES[backbone])
    torch.save(state, dst)
    n = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
    print(f"  ✓ {backbone}: {n/1e6:.1f}M params -> {dst}")
    return dst


def download_timm(backbone: str, out: str) -> None:
    import timm

    name = _TIMM_MODELS[backbone]
    print(f"[{backbone}] timm.create_model({name!r}, pretrained=True) …")
    model = timm.create_model(name, pretrained=True)
    _save(model.state_dict(), out, backbone)


def download_clip(out: str) -> None:
    import open_clip

    name, tag = _CLIP_MODEL
    print(f"[clip] open_clip.create_model({name!r}, pretrained={tag!r}) …")
    model = open_clip.create_model(name, pretrained=tag)
    # Save the FULL model state_dict; the loader builds the full model and uses
    # its .visual submodule.
    _save(model.state_dict(), out, "clip")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backbones",
        nargs="+",
        default=["imagenet", "dinov2", "clip"],
        choices=["imagenet", "dinov2", "clip"],
        help="Which backbones to fetch (default: all).",
    )
    ap.add_argument(
        "--out", default="weights/backbones", help="Output root (default: weights/backbones, = model.backbone_weights_dir)."
    )
    args = ap.parse_args()

    for backbone in args.backbones:
        if backbone in _TIMM_MODELS:
            download_timm(backbone, args.out)
        elif backbone == "clip":
            download_clip(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
