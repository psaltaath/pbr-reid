"""Feature-level MixUp and Boundary Repulsion MixUp (paper Sec. 3.2.2, Eq. 7-13).

Three mixing modes, all applied in embedding space during joint fine-tuning:

* ``hard_target``          standard MixUp with random pairing; the loss on the
                           mixed feature is lambda L(y_i) + (1-lambda) L(y_j).
* ``neighbor_hard_target`` neighbour-guided MixUp: each feature is mixed with the
                           prototype of one of its k most similar identities
                           (Table 5 "NG").
* BRM (``enable_brm``)     confusable pairs (S_{y_i,y_j} > tau) inside the batch
                           are mixed feature-to-feature, a boundary proxy
                           b_ij = lambda p_i + (1-lambda) p_j is built, and
                           L_BRM = mean_(i,j) [ l_att + beta l_rep ] is returned
                           (Eq. 10-13). Falls back to ``hard_target`` MixUp when
                           the batch holds no confusable pair.
"""

import numpy as np
import torch
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# neighbour sampling (neighbour-guided MixUp)                                 #
# --------------------------------------------------------------------------- #
def top_k_neighbours(targets, prototypes_similarity, k):
    """Top-k most similar identities (excluding self) for each target: (B, k) indices and sims."""
    num_classes = prototypes_similarity.shape[0]
    k = min(k, num_classes - 1)
    sims = prototypes_similarity[targets].clone()  # (B, K)
    sims[torch.arange(targets.shape[0], device=targets.device), targets] = float("-inf")
    values, indices = torch.topk(sims, k=k, dim=1, largest=True)
    return indices, values


def sample_neighbour_class(top_k_classes, top_k_similarities, temperature=1.0):
    """Sample one of the top-k identities per row with softmax(sim / T) weights."""
    weights = F.softmax(top_k_similarities / temperature, dim=1)
    picked = torch.multinomial(weights, num_samples=1).squeeze(1)
    return top_k_classes[torch.arange(top_k_classes.shape[0], device=top_k_classes.device), picked]


# --------------------------------------------------------------------------- #
# MixUp / BRM                                                                 #
# --------------------------------------------------------------------------- #
class FeatureMixUp:
    """Feature-level MixUp with optional Boundary Repulsion."""

    def __init__(self, alpha=0.4, mode="hard_target", neighbor_k=5, neighbor_temperature=1.0):
        if mode not in ("hard_target", "neighbor_hard_target"):
            raise ValueError(f"Invalid mixup mode {mode!r}: expected 'hard_target' or 'neighbor_hard_target'")
        self.alpha = float(alpha)
        self.mode = mode
        self.neighbor_k = int(neighbor_k)
        self.neighbor_temperature = float(neighbor_temperature)

    def _sample_lambda(self, n, device):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha, size=n)
            return torch.from_numpy(lam).float().to(device)
        return torch.ones(n, device=device)

    def mix_features(self, features, labels):
        """Random-pair MixUp: returns (mixed, labels_a, labels_b, lambda)."""
        lam = self._sample_lambda(features.size(0), features.device)
        perm = torch.randperm(features.size(0), device=features.device)
        lam_e = lam.unsqueeze(1)
        mixed = lam_e * features + (1 - lam_e) * features[perm]
        return mixed, labels, labels[perm], lam

    def mix_features_with_neighbors(self, features, labels, prototypes, prototypes_similarity):
        """Neighbour-guided MixUp: mix each feature with a similar identity's prototype."""
        top_classes, top_sims = top_k_neighbours(labels, prototypes_similarity, self.neighbor_k)
        neighbours = sample_neighbour_class(top_classes, top_sims, self.neighbor_temperature)
        lam = self._sample_lambda(features.size(0), features.device)
        lam_e = lam.unsqueeze(1)
        mixed = lam_e * features + (1 - lam_e) * prototypes[neighbours]
        return mixed, labels, neighbours, lam

    @staticmethod
    def find_confusable_pairs(labels, prototypes_similarity, tau):
        """Index pairs (i, j), i < j, y_i != y_j, S_{y_i, y_j} > tau. Returns (idx_i, idx_j) LongTensors."""
        B = labels.shape[0]
        iu = torch.triu_indices(B, B, offset=1, device=labels.device)
        li, lj = labels[iu[0]], labels[iu[1]]
        keep = (li != lj) & (prototypes_similarity[li, lj] > tau)
        return iu[0][keep], iu[1][keep]

    def mix_features_batch_with_boundary(self, features, labels, prototypes, prototypes_similarity, tau=0.7):
        """BRM mixing (Eq. 8-9).

        Returns a dict with keys mixed, labels_a, labels_b, lam, boundary, idx_i,
        idx_j when confusable pairs exist; otherwise falls back to
        ``mix_features`` and returns its 4-tuple.
        """
        idx_i, idx_j = self.find_confusable_pairs(labels, prototypes_similarity, tau)
        if idx_i.numel() == 0:
            return self.mix_features(features, labels)
        lam = self._sample_lambda(idx_i.numel(), features.device)
        lam_e = lam.unsqueeze(1)
        labels_a, labels_b = labels[idx_i], labels[idx_j]
        mixed = lam_e * features[idx_i] + (1 - lam_e) * features[idx_j]
        boundary = lam_e * prototypes[labels_a] + (1 - lam_e) * prototypes[labels_b]
        return {
            "mixed": mixed,
            "labels_a": labels_a,
            "labels_b": labels_b,
            "lam": lam,
            "boundary": boundary,
            "idx_i": idx_i,
            "idx_j": idx_j,
        }

    @staticmethod
    def compute_brm_loss(features, prototypes, pairs, margin=0.5, beta=0.7):
        """L_BRM (first term of Eq. 13) = mean over pairs of l_att + beta * l_rep.

        l_att = lam (1 - cos(f~, p_a)) + (1 - lam)(1 - cos(f~, p_b))         (Eq. 10)
        l_rep = [m - ||f_i - b||]_+ + [m - ||f_j - b||]_+                        (Eq. 11)
        """
        mixed, boundary, lam = pairs["mixed"], pairs["boundary"], pairs["lam"]
        proto_a, proto_b = prototypes[pairs["labels_a"]], prototypes[pairs["labels_b"]]
        f_i, f_j = features[pairs["idx_i"]], features[pairs["idx_j"]]

        cos_a = F.cosine_similarity(mixed, proto_a, dim=1)
        cos_b = F.cosine_similarity(mixed, proto_b, dim=1)
        l_att = lam * (1.0 - cos_a) + (1.0 - lam) * (1.0 - cos_b)

        d_i = torch.norm(f_i - boundary, p=2, dim=1)
        d_j = torch.norm(f_j - boundary, p=2, dim=1)
        l_rep = torch.clamp(margin - d_i, min=0.0) + torch.clamp(margin - d_j, min=0.0)

        return (l_att + beta * l_rep).mean()


def should_apply_mixup(config, is_warmup_phase, training=True):
    """MixUp/BRM is applied only in joint fine-tuning, with probability mixup_prob."""
    if not training or not config.training.enable_mixup or is_warmup_phase:
        return False
    return np.random.rand() <= config.training.mixup_prob
