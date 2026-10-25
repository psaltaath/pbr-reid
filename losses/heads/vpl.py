"""
VPL — Variational Prototype Learning (Deng et al., CVPR 2021).

Standalone implementation alongside losses/arcface.py, losses/cosface.py,
losses/circle_loss.py. Plugs into the model's "classification slot" via
`loss.cosine_loss_type = "vpl"`, sees the same `(features, prototypes, targets)`
signature, and is weighted by `loss.cosine_weight`.

Idea (paper §4.1, Eqs. 3-7). Plain prototype learning compares each sample to a
single *point* prototype W_j (sample-to-prototype). VPL instead represents each
class as a *distribution* and compares the sample to a **variational prototype**
that is a linear combination of the learnable prototype and a memorized recent
feature of that class:

    W̃_j = λ₁·W_j + λ₂·M_j           (Eq. 4)

where M_j is a feature-memory bank holding the most recent embedding seen for
class j, kept "alive" for Δt steps via a per-class life indicator T_j. Injecting
the memorized feature simulates sample-to-sample comparison inside the
classification framework (the memorized feature has drifted only slightly — the
paper's slow-feature-drift observation). The variational prototype is then used
in a standard margin softmax; we use the VPL-ArcFace form (Eq. 7):

    L = −log [ e^{s·cos(θ̃_y + m)} / ( e^{s·cos(θ̃_y + m)} + Σ_{j≠y} e^{s·cos θ̃_j} ) ]

This module is stateful: it owns the memory bank `M` (num_classes × dim) and the
life indicator `T` (num_classes) as buffers, and updates them every training
forward — so it needs no trainer changes. The memory is non-parametric
(detached); gradients flow to the learnable prototypes W (and, through W̃, to the
features), exactly as in Eq. 6.

Paper hyperparameters (Tab. 2-3, §5.1): λ₂ (injection weight) = 0.15,
λ₁ (prototype weight) = 1.0, Δt (life span) = 100, ArcFace s = 64, m = 0.5. The
paper starts injection only after features stabilize (~epoch 4 of 24). Here the
memory starts empty (all classes dead → no injection), so injection ramps in
organically as classes are first seen; `start_step` optionally delays injection
further. With a strong pretrained DINOv3 backbone (and a frozen-backbone warmup),
early features are already stable, so `start_step = 0` is the default. This, and
the no-Bernoulli always-inject-when-alive behavior, are the only deviations from
the paper and are configurable.

NOTE: like the other margin heads this assumes the non-MixUp cosine path (one
forward per step); the ablations-batch-3 VPL arm disables MixUp accordingly.

See dev-docs/COMPARISONS_TO_OTHERS.md for the conceptual comparison.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class VPLArcFaceLoss(nn.Module):
    """
    Variational Prototype Learning with an ArcFace margin (VPL-ArcFace).

    Args:
        num_classes (int): Number of identities K (memory bank rows).
        feature_dim (int): Embedding dimension d (memory bank cols).
        scale (float): ArcFace logit scale s. VPL paper default: 64.0.
        margin (float): ArcFace additive angular margin m (radians). 0.5.
        injection_weight (float): λ₂, weight on the memorized feature. 0.15.
        prototype_weight (float): λ₁, weight on the learnable prototype. 1.0.
        life_span (int): Δt, steps a memorized feature stays valid. 100.
        start_step (int): delay (in training steps) before injection begins;
            memory still fills during the delay. Default 0.
        easy_margin (bool): ArcFace easy-margin variant. Default False.
        label_smoothing (float): CE label smoothing. Default 0.0.
        learnable_scale (bool): make s a learnable parameter. Default False.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        scale: float = 64.0,
        margin: float = 0.5,
        injection_weight: float = 0.15,
        prototype_weight: float = 1.0,
        life_span: int = 100,
        start_step: int = 0,
        easy_margin: bool = False,
        label_smoothing: float = 0.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        if num_classes <= 0 or feature_dim <= 0:
            raise ValueError("num_classes and feature_dim must be positive")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if not 0.0 <= margin < (math.pi / 2):
            raise ValueError(f"margin must be in [0, π/2), got {margin}")
        if life_span < 1:
            raise ValueError(f"life_span must be ≥ 1, got {life_span}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

        self.margin = float(margin)
        self.injection_weight = float(injection_weight)
        self.prototype_weight = float(prototype_weight)
        self.life_span = int(life_span)
        self.start_step = int(start_step)
        self.easy_margin = easy_margin
        self.label_smoothing = float(label_smoothing)

        # ArcFace angle-addition constants + monotonicity threshold.
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = self.sin_m * self.margin

        # Non-parametric feature memory bank and per-class life indicator.
        self.register_buffer("memory", torch.zeros(num_classes, feature_dim))
        self.register_buffer("life", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def _apply_arc_margin(self, cosine, targets, num_classes):
        """ArcFace additive angular margin on the GT column (see arcface.py)."""
        eps = 1e-7
        cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(min=eps))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = F.one_hot(targets, num_classes=num_classes).to(phi.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale

    def forward(self, features, prototypes, targets, prototypes_similarity=None):
        """
        Args:
            features: (batch_size, feature_dim) — embeddings (normalized
                internally).
            prototypes: (num_classes, feature_dim) — learnable prototypes W.
            targets: (batch_size,) — ground-truth identity indices.
            prototypes_similarity: unused; accepted for signature parity.
        """
        if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
            print("Warning: NaN/Inf detected in VPL inputs; returning zero loss")
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        eps = 1e-7
        num_classes = prototypes.shape[0]
        features_n = F.normalize(features, p=2, dim=1, eps=eps)
        prototypes_n = F.normalize(prototypes, p=2, dim=1, eps=eps)

        # Build the variational prototypes W̃ = normalize(λ₁·W + λ₂·M) for classes
        # whose memory is alive (T > 0); other classes keep the plain prototype.
        alive = self.life > 0
        inject = bool(alive.any().item()) and (
            self.num_updates.item() >= self.start_step
        )
        if inject:
            combined = (
                self.prototype_weight * prototypes_n
                + self.injection_weight * self.memory
            )
            combined_n = F.normalize(combined, p=2, dim=1, eps=eps)
            var_prototypes = torch.where(
                alive.unsqueeze(1), combined_n, prototypes_n
            )
        else:
            var_prototypes = prototypes_n

        cosine = features_n @ var_prototypes.T  # (B, C)
        logits = self._apply_arc_margin(cosine, targets, num_classes)
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        # Update the memory bank with the current (detached, normalized) features
        # and refresh / decrement the life indicators. Training only.
        if self.training:
            with torch.no_grad():
                self.memory[targets] = features_n.detach()
                self.life[targets] = self.life_span
                self.life.clamp_(min=0).sub_(1).clamp_(min=0)
                self.num_updates += 1

        if not torch.isfinite(loss):
            print("Warning: non-finite VPL loss; returning zero loss")
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        return loss
