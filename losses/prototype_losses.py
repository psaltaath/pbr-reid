"""Prototype-based losses of PBR-ReID (paper Sec. 3.2.1).

* FeaturePrototypeTripletLoss  L_fpt, Eq. 2-4
* PrototypeCenterLoss          L_cen, Eq. 7-11
* CombinedLoss                 assembles L_fpt + L_cen + L_norm (+ optional head)
                               and the externally computed L_BRM (Eq. 1 / 13).
"""

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_margin import sphere_mean


def exp_weight(similarity, strength):
    """w = exp(strength * similarity) — the adaptive weight of Eq. 4 and Eq. 9."""
    return torch.exp(strength * similarity)


class FeaturePrototypeTripletLoss(nn.Module):
    """Prototype-anchored batch-hard triplet (Eq. 2-4).

    For every identity k present in the batch the anchor is p_k, the positive
    is the farthest same-class feature and the negative the closest
    different-class feature (cosine distance). The per-anchor loss
    [d(p_k, f+) - d(p_k, f-) + m]_+ is scaled by w_k = exp(gamma * p_k^T f-)
    when ``use_similarity_weighting`` is on.
    """

    def __init__(self, margin=0.3, use_similarity_weighting=True, similarity_weight_strength=2.0):
        super().__init__()
        self.margin = float(margin)
        self.use_similarity_weighting = use_similarity_weighting
        self.similarity_weight_strength = float(similarity_weight_strength)

    def forward(self, features, prototypes, targets):
        """features (B, d) and prototypes (K, d) L2-normalized; targets (B,)."""
        present = torch.unique(targets)
        anchors = prototypes[present]  # (K', d)
        dist = 1.0 - anchors @ features.t()  # (K', B) cosine distance
        same = present.unsqueeze(1) == targets.unsqueeze(0)  # (K', B)

        has_neg = (~same).any(dim=1)
        if not has_neg.any():
            return features.new_zeros(())
        neg_inf = torch.finfo(dist.dtype).min
        pos_inf = torch.finfo(dist.dtype).max
        d_pos = torch.where(same, dist, dist.new_full((), neg_inf)).max(dim=1).values
        d_neg = torch.where(~same, dist, dist.new_full((), pos_inf)).min(dim=1).values
        d_pos, d_neg = d_pos[has_neg], d_neg[has_neg]

        loss = F.relu(d_pos - d_neg + self.margin)
        if self.use_similarity_weighting:
            # similarity between the anchor prototype and its hardest negative feature
            loss = loss * exp_weight(1.0 - d_neg, self.similarity_weight_strength)
        return loss.mean()


class PrototypeCenterLoss(nn.Module):
    """Prototype center loss (Eq. 7-11).

    For each identity in the batch: mu_k = (Fréchet) mean of its features on
    the sphere, l_k = 1 - p_k^T mu_k, weighted by
    w_sim = exp(gamma_sim * mean_{k'!=k} S_{k k'}) and
    w_var = 1 + gamma_var * mean_i ||f_i - mu_k||^2.

    ``stop_gradient_on_prototype`` detaches p_k (ablation in Sec. 4.3). During
    the warmup stage the backbone is frozen, so the prototype gradient is the
    only learning signal; the detach is therefore never applied in warmup.
    """

    def __init__(
        self,
        mean_type="frechet",
        frechet_iterations=2,
        use_similarity_weighting=True,
        similarity_weight_strength=1.0,
        use_variance_weighting=True,
        variance_weight_strength=2.5,
        stop_gradient_on_prototype=False,
    ):
        super().__init__()
        if mean_type not in ("frechet", "euclidean_projected"):
            raise ValueError(f"Invalid mean_type {mean_type!r}")
        self.mean_type = mean_type
        self.frechet_iterations = int(frechet_iterations)
        self.use_similarity_weighting = use_similarity_weighting
        self.similarity_weight_strength = float(similarity_weight_strength)
        self.use_variance_weighting = use_variance_weighting
        self.variance_weight_strength = float(variance_weight_strength)
        self.stop_gradient_on_prototype = stop_gradient_on_prototype

    def forward(self, features, prototypes, targets, prototypes_similarity=None, is_warmup_phase=False):
        if self.use_similarity_weighting and prototypes_similarity is None:
            raise ValueError("prototypes_similarity is required when use_similarity_weighting=True")
        detach = self.stop_gradient_on_prototype and not is_warmup_phase
        num_classes = prototypes.shape[0]

        losses = []
        for class_idx in torch.unique(targets):
            class_features = features[targets == class_idx]  # (n, d)
            if class_features.shape[0] > 1:
                mu = sphere_mean(class_features, self.mean_type, self.frechet_iterations)
            else:
                mu = F.normalize(class_features[0], dim=0)

            p = prototypes[class_idx]
            if detach:
                p = p.detach()
            l_k = 1.0 - (p * mu).sum()

            if self.use_similarity_weighting:
                sims = prototypes_similarity[class_idx]
                mask = torch.ones(num_classes, dtype=torch.bool, device=sims.device)
                mask[class_idx] = False
                s_bar = sims[mask].mean() if mask.any() else sims.new_zeros(())
                l_k = l_k * exp_weight(s_bar, self.similarity_weight_strength)

            if self.use_variance_weighting and class_features.shape[0] > 1:
                var = ((class_features - mu.unsqueeze(0)) ** 2).sum(dim=1).mean()
                l_k = l_k * (1.0 + self.variance_weight_strength * var)

            losses.append(l_k)

        if not losses:
            return features.new_zeros(())
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    """Training objective (Eq. 1): weighted sum of the enabled terms.

    ``forward`` handles three cases used by the trainer:
      * clean batch (``is_mixup=False``);
      * BRM-engaged batch: clean terms on the original batch + lambda_BRM * L_BRM
        (``brm_loss`` given);
      * standard / neighbour-guided MixUp (``is_mixup=True``): head and center
        terms on the mixed features with hard targets lambda L(y_a) + (1-lambda)
        L(y_b); L_fpt always on the clean batch (``original_features``).
    """

    def __init__(
        self,
        fpt_loss=None,
        fpt_weight=1.0,
        center_loss=None,
        center_weight=0.5,
        norm_loss=None,
        norm_weight=0.005,
        head_loss=None,
        head_weight=1.0,
        brm_lambda=0.4,
    ):
        super().__init__()
        self.fpt_loss = fpt_loss
        self.fpt_weight = float(fpt_weight)
        self.center_loss = center_loss
        self.center_weight = float(center_weight)
        self.norm_loss = norm_loss
        self.norm_weight = float(norm_weight)
        self.head_loss = head_loss
        self.head_weight = float(head_weight)
        self.brm_lambda = float(brm_lambda)

    def forward(
        self,
        features,
        prototypes,
        targets,
        unnorm_features=None,
        unnorm_prototypes=None,
        prototypes_similarity=None,
        is_mixup=False,
        targets_b=None,
        lambda_mix=None,
        original_features=None,
        original_targets=None,
        brm_loss=None,
        is_warmup_phase=False,
    ):
        total = features.new_zeros(())
        loss_dict = {}

        if self.head_loss is not None:
            if is_mixup:
                h_a = self.head_loss(features, prototypes, targets, prototypes_similarity)
                h_b = self.head_loss(features, prototypes, targets_b, prototypes_similarity)
                head = (lambda_mix * h_a + (1 - lambda_mix) * h_b).mean()
            else:
                head = self.head_loss(features, prototypes, targets, prototypes_similarity)
            total = total + self.head_weight * head
            loss_dict["head_loss"] = head.item()

        if self.fpt_loss is not None:
            f, t = (original_features, original_targets) if is_mixup else (features, targets)
            if f is not None and t is not None:
                fpt = self.fpt_loss(f, prototypes, t)
                total = total + self.fpt_weight * fpt
                loss_dict["fpt_loss"] = fpt.item()

        if self.center_loss is not None:
            if is_mixup:
                c_a = self.center_loss(features, prototypes, targets, prototypes_similarity, is_warmup_phase)
                c_b = self.center_loss(features, prototypes, targets_b, prototypes_similarity, is_warmup_phase)
                cen = (lambda_mix * c_a + (1 - lambda_mix) * c_b).mean()
            else:
                cen = self.center_loss(features, prototypes, targets, prototypes_similarity, is_warmup_phase)
            total = total + self.center_weight * cen
            loss_dict["center_loss"] = cen.item()

        if self.norm_loss is not None and unnorm_features is not None and unnorm_prototypes is not None:
            norm = self.norm_loss(unnorm_features) + self.norm_loss(unnorm_prototypes)
            total = total + self.norm_weight * norm
            loss_dict["norm_loss"] = norm.item()

        if brm_loss is not None:
            total = total + self.brm_lambda * brm_loss
            loss_dict["brm_loss"] = brm_loss.item()

        return total, loss_dict
