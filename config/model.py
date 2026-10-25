"""Model / backbone configuration."""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Encoder backbone and prototype settings (paper Sec. 3.1)."""

    # Encoder family. "dinov3" (paper default) loads a DINOv3 ViT through the
    # ``third_party/dinov3`` submodule; "dinov2" / "clip" / "imagenet" load the
    # ViT-B encoders used in the backbone ablation (Table 3) from local weights.
    backbone: str = "dinov3"  # dinov3 | dinov2 | clip | imagenet

    # DINOv3 variant (only when backbone == "dinov3"): vits16, vits16plus,
    # vitb16 (paper default), vitl16, vith16plus, vit7b16.
    model_tag: str = "vitb16"
    pretrain_dataset: str = "lvd"  # lvd (LVD-1689M) | sat (SAT-493M, vitl16/vit7b16 only)

    # Where weights live. Each path may be overridden by an environment
    # variable (see docs/CONFIG.md): PBR_DINOV3_REPO, PBR_DINOV3_WEIGHTS,
    # PBR_BACKBONE_WEIGHTS. "~" and "$VARS" are expanded.
    dinov3_repo_dir: str = "third_party/dinov3"
    dinov3_weights_dir: str = "weights/dinov3"
    backbone_weights_dir: str = "weights/backbones"

    # Prototypes (paper Sec. 3.1 / 3.2.3): Xavier init gain and the EMA decay
    # gamma_ema used during joint fine-tuning. ema_decay = 0 disables the EMA
    # (Supp. Table 2 "No EMA").
    prototype_init_gain: float = 0.5
    ema_decay: float = 0.999
