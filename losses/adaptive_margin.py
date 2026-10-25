"""Adaptive-margin prototype separation (second term of paper Eq. 13).

For confusable class pairs (a, b) with S_ab > tau:

    [ gamma_sep (sigma_a + sigma_b) - (1 - cos(p_a, p_b)) ]_+

where sigma_k is an EMA (decay ``ema_decay``) of the intra-class angular spread
mean_i (1 - cos(f_i, mu_k)), held detached so it only sizes the margin.
Gradients flow into the prototypes only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sphere_mean(feats, mean_type="frechet", frechet_iters=2):
    """Mean of unit vectors on S^{d-1}: projected Euclidean mean, optionally refined
    by a few Riemannian (log/exp map) steps to the Fréchet mean. Returns (d,)."""
    mean = feats.mean(dim=0)
    norm = mean.norm()
    if norm < 1e-12:
        return F.normalize(feats[0], dim=0, eps=1e-12)
    mu = mean / norm
    if mean_type == "frechet" and frechet_iters > 0:
        for _ in range(int(frechet_iters)):
            dots = torch.clamp((feats * mu).sum(dim=1), -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(dots)
            coeff = (theta / torch.sin(theta).clamp_min(1e-7)).unsqueeze(1)
            v = (coeff * (feats - dots.unsqueeze(1) * mu)).mean(dim=0)
            vnorm = v.norm()
            if vnorm < 1e-9:
                break
            mu = F.normalize(torch.cos(vnorm) * mu + torch.sin(vnorm) * (v / vnorm), dim=0, eps=1e-12)
    return mu


class AdaptiveMarginSeparation(nn.Module):
    """Tracks sigma_k and computes the adaptive-margin separation over confusable pairs.

    Args:
        num_classes: K.
        gamma: gamma_sep (required separation per unit of summed spread).
        tau: confusability threshold on S.
        ema_decay: EMA decay of sigma_k.
        pair_scope: "global" (all identities with S_ab > tau) or "batch"
            (only identities present in the current batch, P_S of Eq. 13).
        mean_type / frechet_iters: how the class centroid mu_k is computed.
    """

    def __init__(
        self,
        num_classes,
        gamma=1.5,
        tau=0.7,
        ema_decay=0.9,
        pair_scope="global",
        mean_type="frechet",
        frechet_iters=2,
    ):
        super().__init__()
        if pair_scope not in ("global", "batch"):
            raise ValueError(f"pair_scope must be 'global' or 'batch', got {pair_scope!r}")
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.ema_decay = float(ema_decay)
        self.pair_scope = pair_scope
        self.mean_type = mean_type
        self.frechet_iters = int(frechet_iters)
        self.register_buffer("sigma_hat", torch.zeros(num_classes))
        self.register_buffer("filled", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update_spread(self, features, targets):
        """EMA-update sigma_k from the (detached) batch features."""
        feats = F.normalize(features.detach(), dim=1, eps=1e-12)
        for c in torch.unique(targets).tolist():
            fc = feats[targets == c]
            if fc.shape[0] == 0:
                continue
            mu = sphere_mean(fc, self.mean_type, self.frechet_iters)
            spread = (1.0 - (fc * mu).sum(dim=1)).mean()
            if not bool(self.filled[c]):
                self.sigma_hat[c] = spread
                self.filled[c] = True
            else:
                self.sigma_hat[c] = self.ema_decay * self.sigma_hat[c] + (1.0 - self.ema_decay) * spread

    def confusable_pairs(self, prototypes_similarity, targets=None):
        """Upper-triangular (a, b) index tensors with S_ab > tau (within scope)."""
        S = prototypes_similarity
        if self.pair_scope == "batch":
            ids = torch.unique(targets)
            if ids.numel() < 2:
                empty = torch.empty(0, dtype=torch.long, device=S.device)
                return empty, empty
            iu = torch.triu_indices(ids.numel(), ids.numel(), offset=1, device=S.device)
            a_idx, b_idx = ids[iu[0]], ids[iu[1]]
        else:
            K = S.shape[0]
            iu = torch.triu_indices(K, K, offset=1, device=S.device)
            a_idx, b_idx = iu[0], iu[1]
        keep = S[a_idx, b_idx] > self.tau
        return a_idx[keep], b_idx[keep]

    def forward(self, prototypes, prototypes_similarity, targets=None):
        """Returns (loss, num_pairs). Call ``update_spread`` on the batch first."""
        a_idx, b_idx = self.confusable_pairs(prototypes_similarity, targets)
        if a_idx.numel() == 0:
            return prototypes.new_zeros(()), 0
        if self.sigma_hat.device != prototypes.device:
            self.to(prototypes.device)
        dist_ab = 1.0 - (prototypes[a_idx] * prototypes[b_idx]).sum(dim=1)
        required = torch.clamp(self.gamma * (self.sigma_hat[a_idx] + self.sigma_hat[b_idx]), max=1.0)
        return torch.clamp(required - dist_ab, min=0.0).mean(), int(a_idx.numel())
