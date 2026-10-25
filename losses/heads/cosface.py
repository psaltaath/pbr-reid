"""
CosFace / AM-Softmax — Large Margin Cosine Loss (Wang et al., CVPR 2018).

Standalone, parity implementation kept alongside `losses/arcface.py` so that the
ArcFace-vs-CosFace comparison is *controlled*: identical normalization, identical
logit scale convention, identical (no) label smoothing — the ONLY difference is
where the margin lives. It plugs into the model's "classification slot" via
`loss.cosine_loss_type = "cosface"` (see `training/model_factory.py`), sees the
same `(features, prototypes, targets)` call signature, and is weighted by
`loss.cosine_weight`.

Relationship to the rest of the repo:
    * `CustomCosineLoss` (losses/cosine_loss.py, cosine_loss_type=
      "softmax_cross_entropy") is ALSO a CosFace-style additive-cosine-margin
      loss, but it hardcodes label_smoothing=0.1, defaults to scale s=16, and
      carries difficulty-weighting baggage — fine for its own use, but it does
      NOT match the ArcFace experiment's settings. This module is the clean
      sibling of `ArcFaceLoss` for a fair head-to-head.
    * `ArcFaceLoss` applies an additive *angular* margin `cos(θ + m)`.
      `CosFaceLoss` applies an additive *cosine* margin `cosθ − m`. The cosine
      margin is linear in cosθ (no angle-addition identity, no monotonicity
      threshold needed), which is the practical difference between the two.

The model's per-identity `prototypes` matrix `(num_classes, dim)` is the
class-weight matrix `W`. Both features and prototypes are L2-normalized
internally, so classification depends only on angle.

See dev-docs/COMPARISONS_TO_OTHERS.md for the conceptual comparison against the
PBR-ReID baseline.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CosFaceLoss(nn.Module):
    """
    Large Margin Cosine (CosFace / AM-Softmax) loss.

    For each sample with embedding f and ground-truth identity y, with
    L2-normalized prototypes p_j:

        cosθ_j = p̂_jᵀ f̂                              (cosine logits)
        logit_y  = s · (cosθ_y − m)                   (margin on the target)
        logit_j  = s · cosθ_j           (j ≠ y)       (others unchanged)
        L        = CrossEntropy(logits, y)

    Args:
        scale (float): Logit scale `s`. Normalized cosines live in [−1, 1]; `s`
            rescales them so the softmax can saturate. CosFace literature
            default: 64.0 (kept equal to the ArcFace arm for a fair comparison).
        margin (float): Additive cosine margin `m` (subtracted from the target
            cosine, NOT an angle). CosFace/AM-Softmax literature default: 0.35.
        label_smoothing (float): Cross-entropy label smoothing. CosFace papers
            use 0.0. Default 0.0 (matches the ArcFace arm).
        learnable_scale (bool): If True, `s` is a learnable parameter. Default
            False (fixed buffer), matching the standard recipe.
    """

    def __init__(
        self,
        scale: float = 64.0,
        margin: float = 0.35,
        label_smoothing: float = 0.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if not 0.0 <= margin < 1.0:
            raise ValueError(f"margin must be in [0, 1), got {margin}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

        self.margin = float(margin)
        self.label_smoothing = float(label_smoothing)

    def forward(self, features, prototypes, targets, prototypes_similarity=None):
        """
        Args:
            features: (batch_size, feature_dim) — embeddings (normalized
                internally; need not be pre-normalized).
            prototypes: (num_classes, feature_dim) — class prototypes = CosFace
                weight matrix W (normalized internally).
            targets: (batch_size,) — ground-truth identity indices.
            prototypes_similarity: unused; accepted for call-signature parity
                with CustomCosineLoss / CosineSimilarityLoss / ArcFaceLoss.
        """
        # Guard against NaN/Inf upstream (mirrors CustomCosineLoss behavior).
        if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
            print("Warning: NaN/Inf detected in CosFace inputs; returning zero loss")
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        eps = 1e-7
        features_n = F.normalize(features, p=2, dim=1, eps=eps)
        prototypes_n = F.normalize(prototypes, p=2, dim=1, eps=eps)

        # Cosine logits: (batch_size, num_classes).
        cosine = features_n @ prototypes_n.T
        cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)

        # Additive cosine margin on the ground-truth class column only.
        one_hot = F.one_hot(targets, num_classes=prototypes.shape[0]).to(cosine.dtype)
        logits = (cosine - one_hot * self.margin) * self.scale

        loss = F.cross_entropy(
            logits, targets, label_smoothing=self.label_smoothing
        )

        if not torch.isfinite(loss):
            print("Warning: non-finite CosFace loss; returning zero loss")
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        return loss
