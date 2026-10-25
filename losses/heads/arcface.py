"""
ArcFace — Additive Angular Margin Softmax (Deng et al., CVPR 2019).

This is a standalone, *true* ArcFace implementation kept deliberately outside the
existing prototype-loss machinery. It plugs into the model's "classification
slot" via `loss.cosine_loss_type = "arcface"` (see
`training/model_factory.py`), so it sees the same `(features, prototypes,
targets)` call signature as `CustomCosineLoss` / `CosineSimilarityLoss` and is
weighted by `loss.cosine_weight`.

How it differs from the repo's existing `CustomCosineLoss`:
    * `CustomCosineLoss` applies an additive *cosine* margin (`cosθ − m`) — that
      is CosFace, not ArcFace.
    * `ArcFaceLoss` applies an additive *angular* margin (`cos(θ + m)`), the
      genuine ArcFace objective. The margin is constant in angular space, which
      is the property that makes ArcFace the stable member of the
      SphereFace/CosFace/ArcFace family.

The model's per-identity `prototypes` matrix `(num_classes, dim)` plays the role
of ArcFace's class-weight matrix `W`: each prototype is the proxy/anchor for one
training identity. Both features and prototypes are L2-normalized internally, so
classification depends only on the angle between an embedding and each prototype.

See dev-docs/COMPARISONS_TO_OTHERS.md for the conceptual comparison against the
PBR-ReID baseline.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Softmax loss.

    For each sample with embedding f and ground-truth identity y, with
    L2-normalized prototypes p_j:

        cosθ_j = p̂_jᵀ f̂                              (cosine logits)
        logit_y  = s · cos(θ_y + m)                   (margin on the target)
        logit_j  = s · cosθ_j           (j ≠ y)       (others unchanged)
        L        = CrossEntropy(logits, y)

    The `cos(θ_y + m)` term is computed without an explicit `arccos` via the
    angle-addition identity `cos(θ + m) = cosθ·cos m − sinθ·sin m`, which is
    differentiable and numerically stable.

    Args:
        scale (float): Logit scale `s` (a.k.a. temperature⁻¹). Normalized
            cosines live in [−1, 1]; `s` rescales them so the softmax can
            saturate. ArcFace literature default: 64.0.
        margin (float): Additive angular margin `m` in radians. ArcFace
            literature default: 0.5.
        easy_margin (bool): If True, only apply the margin where `cosθ_y > 0`
            (the easy-margin variant from the ArcFace reference code). If False
            (default), use the Deng et al. threshold trick that keeps the target
            logit monotonically decreasing in θ even past θ = π − m.
        label_smoothing (float): Cross-entropy label smoothing. ArcFace papers
            use 0.0; ReID strong-baselines (BoT) often use 0.1. Default 0.0.
        learnable_scale (bool): If True, `s` is a learnable parameter. Default
            False (fixed buffer), matching the standard recipe.
    """

    def __init__(
        self,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
        label_smoothing: float = 0.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if not 0.0 <= margin < (math.pi / 2):
            raise ValueError(f"margin must be in [0, π/2), got {margin}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

        self.margin = float(margin)
        self.easy_margin = easy_margin
        self.label_smoothing = float(label_smoothing)

        # Precompute the constants of the angle-addition identity.
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        # Threshold trick (non-easy-margin): for θ > π − m, cos(θ + m) starts to
        # increase again, which would *reward* large angles. Beyond the
        # threshold cosθ = cos(π − m) = −cos m we fall back to a linear
        # penalty `cosθ − m·sin m` that stays monotonic.
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = self.sin_m * self.margin

    def forward(self, features, prototypes, targets, prototypes_similarity=None):
        """
        Args:
            features: (batch_size, feature_dim) — embeddings (normalized
                internally; need not be pre-normalized).
            prototypes: (num_classes, feature_dim) — class prototypes = ArcFace
                weight matrix W (normalized internally).
            targets: (batch_size,) — ground-truth identity indices.
            prototypes_similarity: unused; accepted for call-signature parity
                with CustomCosineLoss / CosineSimilarityLoss.
        """
        # Guard against NaN/Inf upstream (mirrors CustomCosineLoss behavior).
        if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
            print("Warning: NaN/Inf detected in ArcFace inputs; returning zero loss")
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        eps = 1e-7
        features_n = F.normalize(features, p=2, dim=1, eps=eps)
        prototypes_n = F.normalize(prototypes, p=2, dim=1, eps=eps)

        # Cosine logits: (batch_size, num_classes). Clamp to a valid cos range
        # so sqrt(1 − cos²) below is real and gradients stay finite.
        cosine = features_n @ prototypes_n.T
        cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)

        # cos(θ + m) = cosθ·cos m − sinθ·sin m, with sinθ = sqrt(1 − cos²θ).
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(min=eps))
        phi = cosine * self.cos_m - sine * self.sin_m  # (B, C)

        if self.easy_margin:
            # Only push the angle where the sample is already on the correct
            # side (cosθ > 0); otherwise leave the cosine untouched.
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Monotonicity guard for θ > π − m (Deng et al. reference code).
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # Apply the margin only to the ground-truth class column.
        one_hot = F.one_hot(targets, num_classes=prototypes.shape[0]).to(phi.dtype)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        loss = F.cross_entropy(
            logits, targets, label_smoothing=self.label_smoothing
        )

        if not torch.isfinite(loss):
            print("Warning: non-finite ArcFace loss; returning zero loss")
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        return loss
