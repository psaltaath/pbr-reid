"""
Circle Loss (Sun et al., CVPR 2020) — class-level / proxy formulation.

Standalone, parity implementation alongside losses/arcface.py and
losses/cosface.py. Plugs into the model's "classification slot" via
`loss.cosine_loss_type = "circleloss"`, sees the same `(features, prototypes,
targets)` signature, and is weighted by `loss.cosine_weight`.

Circle Loss unifies pair-wise (triplet-style) and class-level (softmax-style)
learning under one objective. Here we use the **class-level** form, which is the
proxy/prototype head analogue of ArcFace/CosFace: each sample's feature is scored
against the per-identity prototypes (= class weight matrix W). Its distinguishing
idea vs ArcFace/CosFace is a *per-similarity adaptive weight* — each within-class
score s_p and between-class score s_n is pushed harder the further it is from its
own optimum, giving a circular (rather than linear) decision boundary.

For a sample with within-class similarity s_p = cos(f, w_y) and between-class
similarities s_n^j = cos(f, w_j) (j ≠ y):

    α_p = [O_p − s_p]_+ ,  O_p = 1 + m ,  Δ_p = 1 − m
    α_n = [s_n − O_n]_+ ,  O_n = −m ,     Δ_n = m
    L   = softplus( logsumexp_j( γ·α_n^j·(s_n^j − Δ_n) )
                    − γ·α_p·(s_p − Δ_p) )

with scale γ and relaxation margin m. The α weights are detached (treated as
adaptive scalars), per the reference implementation.

See dev-docs/COMPARISONS_TO_OTHERS.md for the conceptual comparison against the
PBR-ReID baseline.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CircleLoss(nn.Module):
    """
    Class-level Circle Loss over prototypes (proxy head).

    Args:
        scale (float): Scale factor γ (analogous to the ArcFace/CosFace logit
            scale s, but Circle's α weights modulate the effective gradient, so
            its tuned value differs). ReID literature default: 128.0.
        margin (float): Relaxation margin m. ReID literature default: 0.25.
    """

    def __init__(self, scale: float = 128.0, margin: float = 0.25):
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if not 0.0 <= margin < 1.0:
            raise ValueError(f"margin must be in [0, 1), got {margin}")
        self.scale = float(scale)
        self.margin = float(margin)
        # Optimum and within/between targets (constants of the formulation).
        self.O_p = 1.0 + margin
        self.O_n = -margin
        self.delta_p = 1.0 - margin
        self.delta_n = margin

    def forward(self, features, prototypes, targets, prototypes_similarity=None):
        """
        Args:
            features: (batch_size, feature_dim) — embeddings (normalized
                internally; need not be pre-normalized).
            prototypes: (num_classes, feature_dim) — class prototypes = weight
                matrix W (normalized internally).
            targets: (batch_size,) — ground-truth identity indices.
            prototypes_similarity: unused; accepted for call-signature parity.
        """
        if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
            print("Warning: NaN/Inf detected in CircleLoss inputs; returning zero loss")
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        eps = 1e-7
        features_n = F.normalize(features, p=2, dim=1, eps=eps)
        prototypes_n = F.normalize(prototypes, p=2, dim=1, eps=eps)

        # Cosine similarities to every prototype: (batch_size, num_classes).
        sim = (features_n @ prototypes_n.T).clamp(-1.0 + eps, 1.0 - eps)

        B = features.shape[0]
        idx = torch.arange(B, device=sim.device)
        pos_mask = F.one_hot(targets, num_classes=prototypes.shape[0]).bool()

        # Within-class score (one per sample). Between-class scores are every
        # other column; we compute the negative term on the full matrix and mask
        # the GT columns out of the logsumexp *afterwards* (masking the score to
        # −inf first would give 0·−inf = NaN through the α weight).
        s_p = sim[idx, targets]  # (B,)

        # Adaptive, detached weighting factors.
        alpha_p = torch.clamp_min(self.O_p - s_p.detach(), min=0.0)  # (B,)
        alpha_n = torch.clamp_min(sim.detach() - self.O_n, min=0.0)  # (B, C)

        logit_p = -self.scale * alpha_p * (s_p - self.delta_p)  # (B,)
        logit_n = self.scale * alpha_n * (sim - self.delta_n)  # (B, C)
        logit_n = logit_n.masked_fill(pos_mask, float("-inf"))  # drop GT columns

        loss = F.softplus(torch.logsumexp(logit_n, dim=1) + logit_p).mean()

        if not torch.isfinite(loss):
            print("Warning: non-finite CircleLoss; returning zero loss")
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        return loss
