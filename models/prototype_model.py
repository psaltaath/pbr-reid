"""Encoder + learnable per-identity prototypes (paper Sec. 3.1).

Checkpoint layout (state_dict keys):
    model.*                   encoder weights
    _prototypes               (K, d) learnable prototype matrix P
    ema_prototypes            (K, d) EMA copy used during joint fine-tuning
    _prototypes_similarity    (K, K) last prototype similarity matrix S
"""

import torch
from torch import nn
from torch.nn import functional as F


def safe_normalize(x, dim=1, eps=1e-8):
    """L2-normalize along ``dim`` with a clamped norm."""
    norms = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=eps)
    return x / norms


class PrototypeReID(nn.Module):
    """Backbone CLS embedding on the unit sphere + a learnable prototype per identity.

    Args:
        model: encoder exposing ``embed_dim`` and ``forward(x) -> (B, embed_dim)``.
        num_classes: number of training identities K.
        ema_decay: gamma_ema of the prototype EMA (Eq. 14); 0 disables the EMA.
        init_gain: Xavier-uniform gain for the prototype matrix.
    """

    def __init__(self, model, num_classes, ema_decay=0.999, init_gain=0.5):
        super().__init__()
        self.model = model
        self.embed_dim = int(model.embed_dim)
        self.ema_decay = float(ema_decay)

        self._prototypes = nn.Parameter(torch.randn(num_classes, self.embed_dim))
        nn.init.xavier_uniform_(self._prototypes, gain=init_gain)

        self.register_buffer(
            "ema_prototypes", F.normalize(self._prototypes.detach().clone(), p=2, dim=1)
        )
        # S = P̂ P̂ᵀ with the diagonal set to -1 (self-similarity excluded).
        sim = torch.full((num_classes, num_classes), 0.1)
        sim.fill_diagonal_(-1.0)
        self.register_buffer("_prototypes_similarity", sim)

        self._use_ema_prototypes = True

    # ------------------------------------------------------------------ freeze
    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.model.parameters():
            p.requires_grad_(trainable)

    # --------------------------------------------------------------- prototypes
    @property
    def use_ema_prototypes(self):
        return self._use_ema_prototypes

    @use_ema_prototypes.setter
    def use_ema_prototypes(self, value: bool):
        """Switching modes re-seeds the EMA buffer from the current prototypes."""
        self._use_ema_prototypes = bool(value)
        self.ema_prototypes = F.normalize(self._prototypes.detach().clone(), p=2, dim=1)

    @torch.no_grad()
    def update_prototypes(self):
        """Post-step prototype update.

        Warmup (``use_ema_prototypes=False``): re-normalize in place (direct updates).
        Fine-tuning: Eq. 14, p <- gamma p̂_t + (1-gamma) p̂_{t+1}, renormalized.
        """
        normalized = safe_normalize(self._prototypes, dim=1)
        if not self._use_ema_prototypes:
            self._prototypes.data.copy_(normalized)
            return
        self.ema_prototypes = safe_normalize(
            self.ema_decay * self.ema_prototypes + (1.0 - self.ema_decay) * normalized, dim=1
        )
        self._prototypes.data.copy_(self.ema_prototypes)

    @property
    def prototypes(self):
        """Normalized prototypes P̂ (K, d)."""
        return safe_normalize(self._prototypes, dim=1)

    @property
    def unnormalized_prototypes(self):
        return self._prototypes

    @property
    def prototypes_similarity(self):
        """Prototype similarity matrix S (K, K), diagonal = -1."""
        return self._prototypes_similarity

    @torch.no_grad()
    def compute_prototypes_similarity(self):
        p = self.prototypes
        self._prototypes_similarity = p @ p.t()
        self._prototypes_similarity.fill_diagonal_(-1.0)

    # ------------------------------------------------------------------ forward
    def forward(self, x, return_unnormalized=False):
        """Return the L2-normalized embedding f̂; optionally also the raw f (for L_norm)."""
        features = self.model(x)
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Warning: NaN/Inf in backbone features; zeroing the offending entries")
            features = torch.where(
                torch.isnan(features) | torch.isinf(features), torch.zeros_like(features), features
            )
        normalized = safe_normalize(features, dim=1)
        if return_unnormalized:
            return normalized, features
        return normalized


def build_prototype_model(config, num_classes, backbone=None):
    """Build encoder (unless given) + PrototypeReID from a UnifiedConfig."""
    if backbone is None:
        from .backbones import build_backbone_from_config

        backbone, _ = build_backbone_from_config(config)
    return PrototypeReID(
        backbone,
        num_classes=num_classes,
        ema_decay=config.model.ema_decay,
        init_gain=config.model.prototype_init_gain,
    )
