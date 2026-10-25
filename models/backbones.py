"""
Swappable vision-encoder backbones for the ReID pipeline.

The whole pipeline is built around a DINOv3 ViT and only ever calls this narrow
interface on the backbone (see models/prototype_model.py):

    - ``.embed_dim``                                            (int attribute)
    - ``forward(x) -> (B, embed_dim)``                          (final-norm CLS)
    - ``get_intermediate_layers(x, n=1, reshape=<bool>,
          return_class_token=True) -> ( (patch, cls), )``
        * reshape=True  -> patch (B, C, Hp, Wp), cls (B, C)
        * reshape=False -> patch (B, N,  C),     cls (B, C)
      patch tokens are taken AFTER the final LayerNorm (DINOv3 ``norm=True``)
      and, when reshaped, laid out row-major (H then W) so the horizontal-stripe
      pooling in the model lines up with the spatial grid.
    - ``parameters()``                                         (for freezing)

This module wraps a timm ViT (ImageNet ViT-B/16, DINOv2 ViT-B/14) or an
open_clip ViT (OpenAI CLIP ViT-B/16) in a thin adapter that re-exposes exactly
that interface, so the rest of the pipeline is backbone-agnostic.

Weights are loaded strictly from local disk (offline) — see
``scripts/download_backbones.py`` which snapshots each checkpoint into
``<backbone_weights_dir>/<backbone>/``. ``_WEIGHT_FILES`` below is the single source of
truth for those filenames, shared between the loader and the download script.
"""

import os
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2

# timm model names for the pip-sourced backbones.
_TIMM_MODELS = {
    "imagenet": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
}
# open_clip (model_name, pretrained-tag) for the CLIP backbone. The OpenAI
# checkpoint was trained with QuickGELU, so the "-quickgelu" variant is required
# — the plain "ViT-B-16" uses nn.GELU and would silently mismatch the weights.
_CLIP_MODEL = ("ViT-B-16-quickgelu", "openai")

# Local checkpoint filenames under ``<other_models_root>/<backbone>/``.
# Imported by scripts/download_backbones.py so paths never drift.
_WEIGHT_FILES = {
    "imagenet": "vit_b16_augreg_in21k_ft_in1k.pth",
    "dinov2": "vit_b14_dinov2_lvd142m.pth",
    "clip": "vit_b16_openai.pth",
}

# Per-backbone ImageNet-style Normalize stats. DINOv2 uses ImageNet stats;
# CLIP uses its own OpenAI stats.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_NORM = {
    "imagenet": (_IMAGENET_MEAN, _IMAGENET_STD),
    "dinov2": (_IMAGENET_MEAN, _IMAGENET_STD),
    "clip": (
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
}


def make_transform(mean, std, resize_size: int = 256) -> v2.Compose:
    """ToImage / Resize / ToDtype / Normalize, same shape as
    the DINOv3 transform. The square Resize is a placeholder:
    the aspect data pipeline strips it and resizes to the (H, W) ReID target,
    keeping only the ToImage/ToDtype/Normalize ops (training/data_utils.py)."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


class TimmViTAdapter(nn.Module):
    """Adapt a timm VisionTransformer to the DINOv3 backbone interface.

    Covers ImageNet ViT-B/16 and DINOv2 ViT-B/14 — they share token mechanics
    (single prefix/CLS token, NCHW intermediate layout). The timm model MUST be
    created with ``dynamic_img_size=True`` so non-224 inputs are accepted (the
    fixed-size patch-embed assert is otherwise fatal at 384x192 / 392x196).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        if getattr(backbone, "num_prefix_tokens", 1) != 1:
            raise ValueError(
                "TimmViTAdapter requires exactly one prefix (CLS) token; got "
                f"num_prefix_tokens={getattr(backbone, 'num_prefix_tokens', None)} "
                "(register-token variants are unsupported — use the no-register "
                "checkpoint, e.g. vit_base_patch14_dinov2.lvd142m)."
            )
        self.backbone = backbone
        self.embed_dim = int(backbone.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features applies the final LayerNorm; [:, 0] is the normed CLS.
        # (Do NOT call backbone(x) — that runs the classifier head / pooling.)
        return self.backbone.forward_features(x)[:, 0]

    def get_intermediate_layers(
        self, x, n=1, reshape=False, return_class_token=False, **kwargs
    ):
        if n != 1 or not return_class_token:
            raise NotImplementedError(
                "Adapter supports only n=1 with return_class_token=True "
                "(the ClsDINOv3 contract)."
            )
        # norm=True is mandatory — timm defaults to False, which would return
        # pre-final-LayerNorm tokens and silently shift the feature statistics.
        out = self.backbone.get_intermediate_layers(
            x, n=1, reshape=reshape, return_prefix_tokens=True, norm=True
        )
        patch, prefix = out[0]
        B = x.shape[0]
        cls = prefix.reshape(B, -1, self.embed_dim)[:, 0]  # (B, 1, C) -> (B, C)
        return ((patch, cls),)


class OpenClipViTAdapter(nn.Module):
    """Adapt an open_clip visual VisionTransformer to the DINOv3 interface.

    open_clip's ``visual(x)`` returns the *projected* (output_dim, e.g. 512)
    embedding; we instead use the pre-projection ``ln_post`` CLS (embed_dim 768)
    and the patch tokens, so the feature dim matches the 768-d prototype matrix.
    open_clip applies no runtime positional-embedding interpolation, so the
    fixed (grid+1) positional_embedding is bilinearly resized to the actual
    token grid here.
    """

    def __init__(self, clip_visual: nn.Module):
        super().__init__()
        self.visual = clip_visual
        self.embed_dim = int(clip_visual.ln_post.normalized_shape[0])
        self.base_grid = tuple(clip_visual.grid_size)  # (gh0, gw0), e.g. (14,14)

    def _interp_pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        """Resize positional_embedding (1+gh0*gw0, C) to (1+gh*gw, C)."""
        pos = self.visual.positional_embedding  # (1 + gh0*gw0, C)
        if (gh, gw) == self.base_grid:
            return pos
        gh0, gw0 = self.base_grid
        cls_pos, grid_pos = pos[:1], pos[1:]
        grid_pos = (
            grid_pos.reshape(1, gh0, gw0, -1)
            .permute(0, 3, 1, 2)  # 1, C, gh0, gw0
            .contiguous()
        )
        grid_pos = F.interpolate(
            grid_pos, size=(gh, gw), mode="bicubic", align_corners=False
        )
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(gh * gw, -1)
        return torch.cat([cls_pos, grid_pos], dim=0)

    def _forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        v = self.visual
        x = v.conv1(x)  # (B, C, gh, gw)
        B, C, gh, gw = x.shape
        x = x.reshape(B, C, gh * gw).permute(0, 2, 1)  # (B, N, C)
        cls = v.class_embedding.to(x.dtype).reshape(1, 1, C).expand(B, 1, C)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, C)
        x = x + self._interp_pos_embed(gh, gw).to(x.dtype)
        x = v.ln_pre(x)
        # open_clip's Transformer takes batch-first (B, N, C) and handles any
        # internal LND transpose itself — do NOT permute here (doing so scrambles
        # the batch and sequence axes and collapses every image to one vector).
        x = v.transformer(x)
        x = v.ln_post(x)  # final norm over all tokens (DINOv3 norm=True)
        return x, gh, gw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        toks, _, _ = self._forward_tokens(x)
        return toks[:, 0]

    def get_intermediate_layers(
        self, x, n=1, reshape=False, return_class_token=False, **kwargs
    ):
        if n != 1 or not return_class_token:
            raise NotImplementedError(
                "Adapter supports only n=1 with return_class_token=True "
                "(the ClsDINOv3 contract)."
            )
        toks, gh, gw = self._forward_tokens(x)
        cls = toks[:, 0]
        patch = toks[:, 1:]  # (B, N, C)
        if reshape:
            B, _, C = patch.shape
            # row-major (H then W) to match DINOv3 reshape(B, h//p, w//p, -1).
            patch = patch.reshape(B, gh, gw, C).permute(0, 3, 1, 2).contiguous()
        return ((patch, cls),)


def _load_local_state_dict(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Backbone weights not found at {path}. Run "
            "`uv run scripts/download_backbones.py` first (offline setup)."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Tolerate checkpoints saved as {'state_dict': ...} or {'model': ...}.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(
        state["model"], dict
    ):
        state = state["model"]
    return state


def _build_timm(backbone: str, other_models_root: str) -> nn.Module:
    import timm

    model = timm.create_model(
        _TIMM_MODELS[backbone],
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
    )
    path = os.path.join(other_models_root, backbone, _WEIGHT_FILES[backbone])
    state = _load_local_state_dict(path)
    # strict=False: the saved checkpoint carries a classifier head we dropped
    # (num_classes=0). Surface anything unexpected beyond that.
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("head.")]
    if missing or unexpected:
        print(
            f"[backbones] {backbone}: loaded with missing={missing} "
            f"unexpected={unexpected}"
        )
    return TimmViTAdapter(model)


def _build_clip(other_models_root: str) -> nn.Module:
    import open_clip

    model = open_clip.create_model(_CLIP_MODEL[0], pretrained=None)
    path = os.path.join(other_models_root, "clip", _WEIGHT_FILES["clip"])
    state = _load_local_state_dict(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[backbones] clip: loaded with missing={missing} "
            f"unexpected={unexpected}"
        )
    return OpenClipViTAdapter(model.visual)

# --------------------------------------------------------------------------- #
# DINOv3 (paper backbone)                                                     #
# --------------------------------------------------------------------------- #
_DINOV3_HUB_KEYS = {
    "vits16": "dinov3_vits16",
    "vits16plus": "dinov3_vits16plus",
    "vitb16": "dinov3_vitb16",
    "vitl16": "dinov3_vitl16",
    "vith16plus": "dinov3_vith16plus",
    "vit7b16": "dinov3_vit7b16",
}
# Official checkpoint filenames (https://github.com/facebookresearch/dinov3).
_DINOV3_WEIGHTS = {
    ("vits16", "lvd"): "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    ("vits16plus", "lvd"): "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    ("vitb16", "lvd"): "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    ("vitl16", "lvd"): "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    ("vitl16", "sat"): "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    ("vith16plus", "lvd"): "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    ("vit7b16", "lvd"): "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    ("vit7b16", "sat"): "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
}
_DINOV3_NORM = {
    "lvd": (_IMAGENET_MEAN, _IMAGENET_STD),
    "sat": ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
}


def dinov3_weights_filename(model_tag: str, pretrain_dataset: str = "lvd") -> str:
    key = (model_tag, pretrain_dataset)
    if key not in _DINOV3_WEIGHTS:
        raise ValueError(
            f"No DINOv3 checkpoint for model_tag={model_tag!r}, "
            f"pretrain_dataset={pretrain_dataset!r}. Known: {sorted(_DINOV3_WEIGHTS)}"
        )
    return _DINOV3_WEIGHTS[key]


def build_dinov3(
    repo_dir: str, weights_dir: str, model_tag: str, pretrain_dataset: str = "lvd"
) -> Tuple[nn.Module, Any]:
    """Load a DINOv3 ViT from the local ``dinov3`` checkout + a local checkpoint.

    ``repo_dir`` is the facebookresearch/dinov3 checkout (git submodule
    ``third_party/dinov3``); ``weights_dir`` holds the official ``*.pth``
    files, which must be downloaded from Meta (gated) — see README.
    """
    if model_tag not in _DINOV3_HUB_KEYS:
        raise ValueError(
            f"Unknown DINOv3 model_tag {model_tag!r}; expected one of {sorted(_DINOV3_HUB_KEYS)}"
        )
    if not os.path.isfile(os.path.join(repo_dir, "hubconf.py")):
        raise FileNotFoundError(
            f"DINOv3 repo not found at {repo_dir!r} (no hubconf.py). Run "
            "`git submodule update --init` or set model.dinov3_repo_dir / $PBR_DINOV3_REPO."
        )
    weights_path = os.path.join(weights_dir, dinov3_weights_filename(model_tag, pretrain_dataset))
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"DINOv3 weights not found at {weights_path!r}. Download the official "
            "checkpoint from https://github.com/facebookresearch/dinov3 into "
            "model.dinov3_weights_dir (or set $PBR_DINOV3_WEIGHTS)."
        )
    model = torch.hub.load(repo_dir, _DINOV3_HUB_KEYS[model_tag], source="local", weights=weights_path)
    mean, std = _DINOV3_NORM[pretrain_dataset]
    return model, make_transform(mean, std)


def build_backbone(
    backbone: str,
    model_tag: str = "vitb16",
    pretrain_dataset: str = "lvd",
    dinov3_repo_dir: str = "third_party/dinov3",
    dinov3_weights_dir: str = "weights/dinov3",
    backbone_weights_dir: str = "weights/backbones",
) -> Tuple[nn.Module, Any]:
    """Construct the selected encoder and its base transform (ToImage/ToDtype/Normalize).

    ``backbone`` in {"dinov3", "dinov2", "clip", "imagenet"}. The returned
    model exposes ``embed_dim``, ``forward(x) -> (B, embed_dim)`` and
    ``get_intermediate_layers``; the transform supplies the Normalize stats
    (its square Resize is replaced by the ReID (H, W) resize in
    training/data_utils.py).
    """
    if backbone == "dinov3":
        return build_dinov3(dinov3_repo_dir, dinov3_weights_dir, model_tag, pretrain_dataset)
    if backbone in _TIMM_MODELS:
        model = _build_timm(backbone, backbone_weights_dir)
    elif backbone == "clip":
        model = _build_clip(backbone_weights_dir)
    else:
        raise ValueError(
            f"Unknown backbone {backbone!r}; expected one of 'dinov3', 'dinov2', 'clip', 'imagenet'."
        )
    mean, std = _NORM[backbone]
    return model, make_transform(mean, std)


def build_backbone_from_config(config) -> Tuple[nn.Module, Any]:
    """``build_backbone`` driven by a UnifiedConfig (paths resolved + env overrides)."""
    from config.manager import resolve_path

    m = config.model
    return build_backbone(
        backbone=m.backbone,
        model_tag=m.model_tag,
        pretrain_dataset=m.pretrain_dataset,
        dinov3_repo_dir=resolve_path(m.dinov3_repo_dir, "PBR_DINOV3_REPO"),
        dinov3_weights_dir=resolve_path(m.dinov3_weights_dir, "PBR_DINOV3_WEIGHTS"),
        backbone_weights_dir=resolve_path(m.backbone_weights_dir, "PBR_BACKBONE_WEIGHTS"),
    )
